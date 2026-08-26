from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

RUN_STORE: Dict[str, Dict[str, Any]] = {}
RUN_INPUT_STORE: Dict[str, Dict[str, Any]] = {}
ALIAS_STORE: Dict[str, str] = {}

ISO_8601_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$"
)
CANONICAL_VERSION_REGEX = re.compile(r"^(0|[1-9]\d*)$")


def parse_iso8601_utc(ts_str: Any) -> Optional[float]:
    if not isinstance(ts_str, str) or not ISO_8601_REGEX.match(ts_str):
        return None
    try:
        formatted_ts = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(formatted_ts)
        return dt.timestamp()
    except Exception:
        return None


def round_12(val: float) -> float:
    return round(val, 12)


def is_non_negative_safe_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and 0 <= val <= (2**53 - 1)


def is_positive_safe_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and 1 <= val <= (2**53 - 1)


def is_canonical_version_str(val: Any) -> bool:
    if not isinstance(val, str) or not CANONICAL_VERSION_REGEX.match(val):
        return False
    try:
        n = int(val)
        return 1 <= n <= (2**53 - 1)
    except Exception:
        return False


def is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


@app.post("/bqml")
async def handle_bqml(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict) or "phase" not in body:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    phase = body.get("phase")
    if phase == "select":
        return process_select_phase(body)
    elif phase == "evaluate":
        return process_evaluate_phase(body)
    else:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})


def process_select_phase(body: Dict[str, Any]) -> JSONResponse:
    run_id = body.get("runId")
    if not isinstance(run_id, str) or not (1 <= len(run_id) <= 128):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if run_id in RUN_STORE:
        if RUN_INPUT_STORE.get(run_id) == body:
            return JSONResponse(status_code=200, content=RUN_STORE[run_id])
        else:
            return JSONResponse(status_code=409, content={"error": "RUN_ID_CONFLICT"})

    reason_codes: Set[str] = set()

    num_trials_limit = body.get("numTrialsLimit")
    forbidden_features = body.get("forbiddenFeatures")
    rows = body.get("rows")
    trials = body.get("trials")

    valid_structure = True
    if not is_positive_safe_int(num_trials_limit):
        valid_structure = False
    if not isinstance(forbidden_features, list) or not all(isinstance(x, str) for x in forbidden_features):
        valid_structure = False
    if not isinstance(rows, list) or len(rows) == 0:
        valid_structure = False
    if not isinstance(trials, list):
        valid_structure = False

    if not valid_structure:
        reason_codes.add("INVALID_INPUT")

    if isinstance(trials, list) and is_positive_safe_int(num_trials_limit):
        if len(trials) > num_trials_limit:
            reason_codes.add("TRIAL_LIMIT_EXCEEDED")

    valid_rows = []
    seen_row_ids = set()
    if isinstance(rows, list) and "INVALID_INPUT" not in reason_codes:
        for r in rows:
            if not isinstance(r, dict):
                reason_codes.add("INVALID_INPUT")
                break
            r_id = r.get("id")
            entity = r.get("entity")
            event_time = r.get("eventTime")
            pred_time = r.get("predictionTime")
            version = r.get("version")
            split = r.get("split")
            features = r.get("features")

            if (
                not isinstance(r_id, str) or r_id in seen_row_ids or
                not isinstance(entity, str) or
                not isinstance(event_time, str) or
                not isinstance(pred_time, str) or
                not is_non_negative_safe_int(version) or
                split not in ("TRAIN", "EVAL") or
                not isinstance(features, dict)
            ):
                reason_codes.add("INVALID_INPUT")
                break

            seen_row_ids.add(r_id)

            evt_ts = parse_iso8601_utc(event_time)
            pred_ts = parse_iso8601_utc(pred_time)
            if evt_ts is None or pred_ts is None:
                reason_codes.add("INVALID_INPUT")
                break

            valid_rows.append({
                "id": r_id,
                "entity": entity,
                "eventTime": event_time,
                "evt_ts": evt_ts,
                "pred_ts": pred_ts,
                "version": version,
                "split": split,
                "features": features
            })

    retained_rows = []
    if "INVALID_INPUT" not in reason_codes:
        dedup_map: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
        for r in valid_rows:
            key = (r["entity"], r["evt_ts"])
            dedup_map.setdefault(key, []).append(r)

        for key, group in dedup_map.items():
            group.sort(key=lambda x: (-x["version"], x["id"].encode("utf-8")))
            retained_rows.append(group[0])

    eligible_features = []
    if "INVALID_INPUT" not in reason_codes and retained_rows:
        forbidden_set = set(forbidden_features)
        candidate_feats = set(retained_rows[0]["features"].keys())

        for r in retained_rows[1:]:
            candidate_feats = candidate_feats.intersection(set(r["features"].keys()))

        for feat in candidate_feats:
            if feat in forbidden_set:
                continue

            feat_eligible = True
            for r in retained_rows:
                feat_obj = r["features"].get(feat)
                if not isinstance(feat_obj, dict) or "availableAt" not in feat_obj:
                    feat_eligible = False
                    break
                avail_time = feat_obj["availableAt"]
                avail_ts = parse_iso8601_utc(avail_time)
                if avail_ts is None or avail_ts > r["pred_ts"]:
                    feat_eligible = False
                    break

            if feat_eligible:
                eligible_features.append(feat)

        eligible_features.sort(key=lambda x: x.encode("utf-8"))

    train_row_ids = []
    eval_row_ids = []
    if "INVALID_INPUT" not in reason_codes:
        for r in retained_rows:
            if r["split"] == "TRAIN":
                train_row_ids.append(r["id"])
            elif r["split"] == "EVAL":
                eval_row_ids.append(r["id"])

        train_row_ids.sort(key=lambda x: x.encode("utf-8"))
        eval_row_ids.sort(key=lambda x: x.encode("utf-8"))

    dataset_digest = None
    if "INVALID_INPUT" not in reason_codes:
        digest_obj = {
            "trainRowIds": train_row_ids,
            "evalRowIds": eval_row_ids,
            "featureNames": eligible_features
        }
        compact_json = json.dumps(digest_obj, separators=(',', ':'))
        dataset_digest = hashlib.sha256(compact_json.encode('utf-8')).hexdigest()

    selected_trial_id = None
    eligible_trials = []
    seen_trial_ids = set()

    if isinstance(trials, list):
        for t in trials:
            if not isinstance(t, dict):
                reason_codes.add("INVALID_INPUT")
                break
            t_id = t.get("trialId")
            status = t.get("status")
            metric = t.get("evalMetric")

            if not is_non_negative_safe_int(t_id) or t_id in seen_trial_ids or status not in ("SUCCEEDED", "FAILED"):
                reason_codes.add("INVALID_INPUT")
                break

            seen_trial_ids.add(t_id)

            if status == "SUCCEEDED":
                if isinstance(metric, (int, float)) and not isinstance(metric, bool) and math.isfinite(metric):
                    eligible_trials.append({"trialId": t_id, "evalMetric": float(metric)})

    if "INVALID_INPUT" not in reason_codes:
        if not eligible_trials:
            reason_codes.add("NO_SUCCESSFUL_TRIAL")
        else:
            eligible_trials.sort(key=lambda x: (-x["evalMetric"], x["trialId"]))
            selected_trial_id = eligible_trials[0]["trialId"]

    if reason_codes or selected_trial_id is None:
        selected_trial_id = None

    if "INVALID_INPUT" in reason_codes:
        dataset_digest = None
        train_row_ids = []
        eval_row_ids = []
        eligible_features = []

    sorted_reasons = sorted(list(reason_codes), key=lambda x: x.encode("utf-8"))

    response_data = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_row_ids if "INVALID_INPUT" not in reason_codes else [],
        "evalRowIds": eval_row_ids if "INVALID_INPUT" not in reason_codes else [],
        "featureNames": eligible_features if "INVALID_INPUT" not in reason_codes else [],
        "datasetDigest": dataset_digest,
        "reasonCodes": sorted_reasons
    }

    RUN_STORE[run_id] = response_data
    RUN_INPUT_STORE[run_id] = body

    return JSONResponse(status_code=200, content=response_data)


def process_evaluate_phase(body: Dict[str, Any]) -> JSONResponse:
    run_id = body.get("runId")
    selected_trial_id = body.get("selectedTrialId")
    dataset_digest = body.get("datasetDigest")
    metric_floor = body.get("metricFloor")
    req_slices = body.get("requiredSlices")
    rows = body.get("rows")
    bytes_processed = body.get("bytesProcessed")
    max_bytes = body.get("maxBytes")

    reason_codes: Set[str] = set()

    if not isinstance(run_id, str) or not is_non_negative_safe_int(selected_trial_id) or not isinstance(dataset_digest, str):
        reason_codes.add("INVALID_INPUT")
    if not isinstance(metric_floor, (int, float)) or isinstance(metric_floor, bool) or not (0.0 <= metric_floor <= 1.0) or not math.isfinite(metric_floor):
        reason_codes.add("INVALID_INPUT")
    if not isinstance(req_slices, dict):
        reason_codes.add("INVALID_INPUT")
    else:
        for k, v in req_slices.items():
            if not isinstance(k, str) or not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 <= v <= 1.0) or not math.isfinite(v):
                reason_codes.add("INVALID_INPUT")
                break
    if not is_non_negative_safe_int(bytes_processed) or not is_non_negative_safe_int(max_bytes):
        reason_codes.add("INVALID_INPUT")
    if not isinstance(rows, list):
        reason_codes.add("INVALID_INPUT")

    stored_run = RUN_STORE.get(run_id)
    lineage_valid = True
    if stored_run is None or stored_run.get("selectedTrialId") is None:
        lineage_valid = False
    elif stored_run.get("selectedTrialId") != selected_trial_id or stored_run.get("datasetDigest") != dataset_digest:
        lineage_valid = False

    if not lineage_valid:
        reason_codes.add("INVALID_LINEAGE")

    invalid_test_row = False
    if isinstance(rows, list):
        if len(rows) > 0:
            for r in rows:
                if not isinstance(r, dict):
                    invalid_test_row = True
                    break
                lbl = r.get("label")
                pred = r.get("prediction")
                slice_name = r.get("slice")

                if lbl not in (0, 1) or pred not in (0, 1) or type(lbl) is bool or type(pred) is bool:
                    invalid_test_row = True
                    break
                if not isinstance(slice_name, str) or len(slice_name) == 0:
                    invalid_test_row = True
                    break

    if invalid_test_row:
        reason_codes.add("INVALID_TEST_ROW")

    if is_non_negative_safe_int(bytes_processed) and is_non_negative_safe_int(max_bytes):
        if bytes_processed > max_bytes:
            reason_codes.add("BYTE_LIMIT")

    test_metric = None
    critical_slice_pass = True

    if isinstance(rows, list) and len(rows) > 0 and not invalid_test_row and "INVALID_INPUT" not in reason_codes:
        correct_count = sum(1 for r in rows if r["label"] == r["prediction"])
        test_metric = round_12(correct_count / len(rows))

        if test_metric < metric_floor:
            reason_codes.add("AGGREGATE_FLOOR")

        slice_map: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            slice_map.setdefault(r["slice"], []).append(r)

        if isinstance(req_slices, dict):
            for s_name, s_floor in req_slices.items():
                if s_name not in slice_map:
                    reason_codes.add(f"MISSING_SLICE:{s_name}")
                    critical_slice_pass = False
                else:
                    s_rows = slice_map[s_name]
                    s_correct = sum(1 for r in s_rows if r["label"] == r["prediction"])
                    s_acc = round_12(s_correct / len(s_rows))
                    if s_acc < s_floor:
                        reason_codes.add(f"SLICE_FLOOR:{s_name}")
                        critical_slice_pass = False

    if (
        "INVALID_INPUT" in reason_codes
        or "INVALID_LINEAGE" in reason_codes
        or "INVALID_TEST_ROW" in reason_codes
        or any(code.startswith("MISSING_SLICE:") or code.startswith("SLICE_FLOOR:") for code in reason_codes)
    ):
        critical_slice_pass = False

    decision = "admit" if len(reason_codes) == 0 else "reject"
    sorted_reasons = sorted(list(reason_codes), key=lambda x: x.encode("utf-8"))

    return JSONResponse(status_code=200, content={
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed if is_non_negative_safe_int(bytes_processed) else None,
        "reasonCodes": sorted_reasons
    })


@app.post("/promote")
async def handle_promote(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    as_of_str = body.get("asOf")
    champion_version_raw = body.get("championVersion")
    champion_version_in = str(champion_version_raw) if champion_version_raw is not None else ""
    policy = body.get("policy")
    versions_list = body.get("versions")

    if not isinstance(as_of_str, str) or policy is None or not isinstance(policy, dict) or not isinstance(versions_list, list):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    as_of_ts = parse_iso8601_utc(as_of_str)
    if as_of_ts is None:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    dataset_digest = policy.get("datasetDigest")
    schema_digest = policy.get("schemaDigest")
    max_age_seconds = policy.get("maxAgeSeconds")
    accuracy_floor = policy.get("accuracyFloor")
    required_slices = policy.get("requiredSlices", {})
    max_latency_ms = policy.get("maxLatencyMs")
    max_size_bytes = policy.get("maxSizeBytes")
    min_improvement = policy.get("minImprovement", 0.0)

    valid_policy = True
    if not isinstance(dataset_digest, str) or not dataset_digest:
        valid_policy = False
    if not isinstance(schema_digest, str) or not schema_digest:
        valid_policy = False
    if not is_non_negative_safe_int(max_age_seconds):
        valid_policy = False
    if not is_finite_number(accuracy_floor) or not (0.0 <= accuracy_floor <= 1.0):
        valid_policy = False
    if not isinstance(required_slices, dict):
        valid_policy = False
    else:
        for sk, sv in required_slices.items():
            if not isinstance(sk, str) or not is_finite_number(sv) or not (0.0 <= sv <= 1.0):
                valid_policy = False
                break
    if not is_finite_number(max_latency_ms) or max_latency_ms < 0:
        valid_policy = False
    if not is_non_negative_safe_int(max_size_bytes):
        valid_policy = False
    if not is_finite_number(min_improvement):
        valid_policy = False

    if not valid_policy:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    # Always respect championVersion provided in the request payload if available
    current_champion_version = champion_version_in if champion_version_in else ALIAS_STORE.get("champion", "")

    seen_version_ids: Set[str] = set()
    failed_gates: Dict[str, List[str]] = {}
    valid_version_objects: List[Dict[str, Any]] = []

    for v_item in versions_list:
        if not isinstance(v_item, dict):
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

        v_id_raw = v_item.get("version")
        v_id = str(v_id_raw) if v_id_raw is not None else ""
        v_codes: Set[str] = set()

        if not v_id or not is_canonical_version_str(v_id):
            v_codes.add("INVALID_VERSION")

        if v_id in seen_version_ids:
            v_codes.add("DUPLICATE_VERSION")

        seen_version_ids.add(v_id)

        artifact_digest = v_item.get("artifactDigest")
        if not isinstance(artifact_digest, str):
            v_codes.add("INVALID_VERSION")

        eval_obj = v_item.get("evaluation")
        if eval_obj is None or not isinstance(eval_obj, dict):
            v_codes.add("MISSING_EVALUATION")
        else:
            created_at_str = eval_obj.get("createdAt")
            created_at_ts = parse_iso8601_utc(created_at_str)
            if created_at_ts is None:
                v_codes.add("INVALID_TIMESTAMP")
            else:
                if created_at_ts > as_of_ts:
                    v_codes.add("FUTURE_EVALUATION")
                elif created_at_ts < (as_of_ts - max_age_seconds):
                    v_codes.add("STALE_EVALUATION")

            e_artifact = eval_obj.get("artifactDigest")
            e_dataset = eval_obj.get("datasetDigest")
            e_schema = eval_obj.get("schemaDigest")

            if not isinstance(e_artifact, str) or e_artifact != artifact_digest:
                v_codes.add("ARTIFACT_MISMATCH")
            if not isinstance(e_dataset, str) or e_dataset != dataset_digest:
                v_codes.add("DATASET_MISMATCH")
            if not isinstance(e_schema, str) or e_schema != schema_digest:
                v_codes.add("SCHEMA_MISMATCH")

            acc = eval_obj.get("accuracy")
            lat = eval_obj.get("latencyMs")
            sz = eval_obj.get("sizeBytes")

            if not is_finite_number(acc) or not is_finite_number(lat) or not is_finite_number(sz):
                v_codes.add("NON_FINITE")
            else:
                if not (0.0 <= acc <= 1.0) or lat < 0 or sz < 0:
                    v_codes.add("METRIC_RANGE")

            if is_finite_number(acc) and 0.0 <= acc <= 1.0:
                if round_12(acc) < round_12(accuracy_floor):
                    v_codes.add("ACCURACY_FLOOR")

            if is_finite_number(lat) and lat >= 0:
                if round_12(lat) > round_12(max_latency_ms):
                    v_codes.add("LATENCY_LIMIT")

            if is_non_negative_safe_int(sz):
                if sz > max_size_bytes:
                    v_codes.add("SIZE_LIMIT")

            slices_obj = eval_obj.get("slices")
            if not isinstance(slices_obj, dict):
                v_codes.add("NON_FINITE")
            else:
                for req_name, req_floor in required_slices.items():
                    if req_name not in slices_obj:
                        v_codes.add(f"MISSING_SLICE:{req_name}")
                    else:
                        s_val = slices_obj[req_name]
                        if not is_finite_number(s_val):
                            v_codes.add("NON_FINITE")
                        elif not (0.0 <= s_val <= 1.0):
                            v_codes.add(f"SLICE_RANGE:{req_name}")
                        elif round_12(s_val) < round_12(req_floor):
                            v_codes.add(f"SLICE_FLOOR:{req_name}")

        sorted_v_codes = sorted(list(v_codes), key=lambda x: x.encode("utf-8"))
        failed_gates[v_id] = sorted_v_codes

        if len(v_codes) == 0:
            valid_version_objects.append(v_item)

    eligible_versions = [str(v["version"]) for v in valid_version_objects]
    eligible_versions.sort(key=lambda x: int(x))

    if not valid_version_objects:
        return JSONResponse(status_code=200, content={
            "action": "block",
            "championVersion": current_champion_version,
            "selectedVersion": None,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None
        })

    def rank_key(v):
        e = v["evaluation"]
        return (
            -round_12(e["accuracy"]),
            round_12(e["latencyMs"]),
            e["sizeBytes"],
            -int(v["version"])  # Standard higher-version tie-breaking for equal performance
        )

    ranked_eligible = sorted(valid_version_objects, key=rank_key)
    best_challenger = ranked_eligible[0]
    best_challenger_version = str(best_challenger["version"])

    champion_item = next((v for v in valid_version_objects if str(v["version"]) == current_champion_version), None)

    if champion_item is None:
        ALIAS_STORE["champion"] = best_challenger_version
        return JSONResponse(status_code=200, content={
            "action": "promote",
            "championVersion": current_champion_version,
            "selectedVersion": best_challenger_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": {
                "alias": "champion",
                "version": best_challenger_version
            },
            "evidence": best_challenger["evaluation"]
        })

    champion_acc = round_12(champion_item["evaluation"]["accuracy"])
    challenger_acc = round_12(best_challenger["evaluation"]["accuracy"])
    acc_diff = round_12(challenger_acc - champion_acc)

    if best_challenger_version != current_champion_version and acc_diff >= round_12(min_improvement):
        ALIAS_STORE["champion"] = best_challenger_version
        return JSONResponse(status_code=200, content={
            "action": "promote",
            "championVersion": current_champion_version,
            "selectedVersion": best_challenger_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": {
                "alias": "champion",
                "version": best_challenger_version
            },
            "evidence": best_challenger["evaluation"]
        })
    else:
        return JSONResponse(status_code=200, content={
            "action": "retain",
            "championVersion": current_champion_version,
            "selectedVersion": current_champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": champion_item["evaluation"]
        })
