from datetime import datetime
import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Stateful store to retain mutated champion alias state for replay tests
# Maps champion_key (or global alias state) -> current champion version string
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


def is_canonical_version_str(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    if not CANONICAL_VERSION_REGEX.match(val):
        return False
    # Safe integer check
    try:
        n = int(val)
        return 1 <= n <= (2**53 - 1)
    except Exception:
        return False


def is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


@app.post("/promote")
async def handle_promote(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    as_of_str = body.get("asOf")
    champion_version_in = body.get("championVersion")
    policy = body.get("policy")
    versions_list = body.get("versions")

    # High-level HTTP 400 validation per spec
    if not isinstance(as_of_str, str) or not isinstance(champion_version_in, str) or not isinstance(policy, dict) or not isinstance(versions_list, list):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    as_of_ts = parse_iso8601_utc(as_of_str)
    if as_of_ts is None:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    # Validate Policy structure
    dataset_digest = policy.get("datasetDigest")
    schema_digest = policy.get("schemaDigest")
    max_age_seconds = policy.get("maxAgeSeconds")
    accuracy_floor = policy.get("accuracyFloor")
    required_slices = policy.get("requiredSlices")
    max_latency_ms = policy.get("maxLatencyMs")
    max_size_bytes = policy.get("maxSizeBytes")
    min_improvement = policy.get("minImprovement")

    valid_policy = True
    if not isinstance(dataset_digest, str) or len(dataset_digest) == 0:
        valid_policy = False
    if not isinstance(schema_digest, str) or len(schema_digest) == 0:
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
    if not is_finite_number(min_improvement) or not (0.0 <= min_improvement <= 1.0):
        valid_policy = False

    if not valid_policy:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    # Check stateful champion override from prior alias mutations
    current_champion_version = ALIAS_STORE.get("champion", champion_version_in)

    # Process versions sequentially to catch non-canonical / duplicate versions
    seen_version_ids: Set[str] = set()
    failed_gates: Dict[str, List[str]] = {}
    valid_version_objects: List[Dict[str, Any]] = []

    for v_item in versions_list:
        if not isinstance(v_item, dict):
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

        v_id = v_item.get("version")
        v_codes: Set[str] = set()

        if not is_canonical_version_str(v_id):
            v_codes.add("INVALID_VERSION")

        if isinstance(v_id, str) and v_id in seen_version_ids:
            v_codes.add("DUPLICATE_VERSION")

        if isinstance(v_id, str):
            seen_version_ids.add(v_id)

        # Evaluate individual version gates
        artifact_digest = v_item.get("artifactDigest")
        if not isinstance(artifact_digest, str):
            v_codes.add("INVALID_VERSION")

        eval_obj = v_item.get("evaluation")
        if eval_obj is None or not isinstance(eval_obj, dict):
            v_codes.add("MISSING_EVALUATION")
        else:
            # Timestamp checks
            created_at_str = eval_obj.get("createdAt")
            created_at_ts = parse_iso8601_utc(created_at_str)
            if created_at_ts is None:
                v_codes.add("INVALID_TIMESTAMP")
            else:
                if created_at_ts > as_of_ts:
                    v_codes.add("FUTURE_EVALUATION")
                elif created_at_ts < (as_of_ts - max_age_seconds):
                    v_codes.add("STALE_EVALUATION")

            # Digest binding checks
            e_artifact = eval_obj.get("artifactDigest")
            e_dataset = eval_obj.get("datasetDigest")
            e_schema = eval_obj.get("schemaDigest")

            if not isinstance(e_artifact, str) or e_artifact != artifact_digest:
                v_codes.add("ARTIFACT_MISMATCH")
            if not isinstance(e_dataset, str) or e_dataset != dataset_digest:
                v_codes.add("DATASET_MISMATCH")
            if not isinstance(e_schema, str) or e_schema != schema_digest:
                v_codes.add("SCHEMA_MISMATCH")

            # Numeric & metric range checks
            acc = eval_obj.get("accuracy")
            lat = eval_obj.get("latencyMs")
            sz = eval_obj.get("sizeBytes")

            # NON_FINITE vs METRIC_RANGE
            if not is_finite_number(acc) or not is_finite_number(lat) or not is_finite_number(sz):
                v_codes.add("NON_FINITE")
            else:
                if not (0.0 <= acc <= 1.0):
                    v_codes.add("METRIC_RANGE")

            if isinstance(lat, (int, float)) and not isinstance(lat, bool) and math.isfinite(lat) and lat < 0:
                v_codes.add("METRIC_RANGE")

            if isinstance(sz, (int, float)) and not isinstance(sz, bool) and math.isfinite(sz) and sz < 0:
                v_codes.add("METRIC_RANGE")

            # Gate floor limits
            if is_finite_number(acc) and 0.0 <= acc <= 1.0:
                if acc < accuracy_floor:
                    v_codes.add("ACCURACY_FLOOR")

            if is_finite_number(lat) and lat >= 0:
                if lat > max_latency_ms:
                    v_codes.add("LATENCY_LIMIT")

            if is_non_negative_safe_int(sz):
                if sz > max_size_bytes:
                    v_codes.add("SIZE_LIMIT")

            # Slices evaluation
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
                        elif s_val < req_floor:
                            v_codes.add(f"SLICE_FLOOR:{req_name}")

        # Store failed gates sorted by UTF-8 bytes
        if isinstance(v_id, str):
            sorted_v_codes = sorted(list(v_codes), key=lambda x: x.encode("utf-8"))
            failed_gates[v_id] = sorted_v_codes

            if len(v_codes) == 0:
                valid_version_objects.append(v_item)

    # Determine eligible version strings
    eligible_versions = [v["version"] for v in valid_version_objects]
    eligible_versions.sort(key=lambda x: x.encode("utf-8"))

    # Validate Champion Eligibility
    champion_item = next((v for v in valid_version_objects if v["version"] == current_champion_version), None)

    if champion_item is None:
        # Champion evidence is invalid or champion not present -> BLOCK
        return JSONResponse(status_code=200, content={
            "action": "block",
            "championVersion": current_champion_version,
            "selectedVersion": None,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None
        })

    # Rank all eligible versions
    # Sorting key: accuracy descending, latency ascending, size ascending, numeric version ascending
    def rank_key(v):
        e = v["evaluation"]
        return (
            -e["accuracy"],
            e["latencyMs"],
            e["sizeBytes"],
            int(v["version"])
        )

    ranked_eligible = sorted(valid_version_objects, key=rank_key)
    best_challenger = ranked_eligible[0]

    champion_acc = champion_item["evaluation"]["accuracy"]
    challenger_acc = best_challenger["evaluation"]["accuracy"]

    acc_diff = round_12(challenger_acc - champion_acc)

    if best_challenger["version"] != current_champion_version and acc_diff >= min_improvement:
        # PROMOTE
        new_champion_version = best_challenger["version"]
        ALIAS_STORE["champion"] = new_champion_version

        return JSONResponse(status_code=200, content={
            "action": "promote",
            "championVersion": current_champion_version,
            "selectedVersion": new_champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": {
                "alias": "champion",
                "version": new_champion_version
            },
            "evidence": best_challenger["evaluation"]
        })
    else:
        # RETAIN
        return JSONResponse(status_code=200, content={
            "action": "retain",
            "championVersion": current_champion_version,
            "selectedVersion": current_champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": champion_item["evaluation"]
        })from datetime import datetime
import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Stateful store to retain mutated champion alias state for replay tests
# Maps champion_key (or global alias state) -> current champion version string
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


def is_canonical_version_str(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    if not CANONICAL_VERSION_REGEX.match(val):
        return False
    # Safe integer check
    try:
        n = int(val)
        return 1 <= n <= (2**53 - 1)
    except Exception:
        return False


def is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


@app.post("/promote")
async def handle_promote(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    as_of_str = body.get("asOf")
    champion_version_in = body.get("championVersion")
    policy = body.get("policy")
    versions_list = body.get("versions")

    # High-level HTTP 400 validation per spec
    if not isinstance(as_of_str, str) or not isinstance(champion_version_in, str) or not isinstance(policy, dict) or not isinstance(versions_list, list):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    as_of_ts = parse_iso8601_utc(as_of_str)
    if as_of_ts is None:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    # Validate Policy structure
    dataset_digest = policy.get("datasetDigest")
    schema_digest = policy.get("schemaDigest")
    max_age_seconds = policy.get("maxAgeSeconds")
    accuracy_floor = policy.get("accuracyFloor")
    required_slices = policy.get("requiredSlices")
    max_latency_ms = policy.get("maxLatencyMs")
    max_size_bytes = policy.get("maxSizeBytes")
    min_improvement = policy.get("minImprovement")

    valid_policy = True
    if not isinstance(dataset_digest, str) or len(dataset_digest) == 0:
        valid_policy = False
    if not isinstance(schema_digest, str) or len(schema_digest) == 0:
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
    if not is_finite_number(min_improvement) or not (0.0 <= min_improvement <= 1.0):
        valid_policy = False

    if not valid_policy:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    # Check stateful champion override from prior alias mutations
    current_champion_version = ALIAS_STORE.get("champion", champion_version_in)

    # Process versions sequentially to catch non-canonical / duplicate versions
    seen_version_ids: Set[str] = set()
    failed_gates: Dict[str, List[str]] = {}
    valid_version_objects: List[Dict[str, Any]] = []

    for v_item in versions_list:
        if not isinstance(v_item, dict):
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

        v_id = v_item.get("version")
        v_codes: Set[str] = set()

        if not is_canonical_version_str(v_id):
            v_codes.add("INVALID_VERSION")

        if isinstance(v_id, str) and v_id in seen_version_ids:
            v_codes.add("DUPLICATE_VERSION")

        if isinstance(v_id, str):
            seen_version_ids.add(v_id)

        # Evaluate individual version gates
        artifact_digest = v_item.get("artifactDigest")
        if not isinstance(artifact_digest, str):
            v_codes.add("INVALID_VERSION")

        eval_obj = v_item.get("evaluation")
        if eval_obj is None or not isinstance(eval_obj, dict):
            v_codes.add("MISSING_EVALUATION")
        else:
            # Timestamp checks
            created_at_str = eval_obj.get("createdAt")
            created_at_ts = parse_iso8601_utc(created_at_str)
            if created_at_ts is None:
                v_codes.add("INVALID_TIMESTAMP")
            else:
                if created_at_ts > as_of_ts:
                    v_codes.add("FUTURE_EVALUATION")
                elif created_at_ts < (as_of_ts - max_age_seconds):
                    v_codes.add("STALE_EVALUATION")

            # Digest binding checks
            e_artifact = eval_obj.get("artifactDigest")
            e_dataset = eval_obj.get("datasetDigest")
            e_schema = eval_obj.get("schemaDigest")

            if not isinstance(e_artifact, str) or e_artifact != artifact_digest:
                v_codes.add("ARTIFACT_MISMATCH")
            if not isinstance(e_dataset, str) or e_dataset != dataset_digest:
                v_codes.add("DATASET_MISMATCH")
            if not isinstance(e_schema, str) or e_schema != schema_digest:
                v_codes.add("SCHEMA_MISMATCH")

            # Numeric & metric range checks
            acc = eval_obj.get("accuracy")
            lat = eval_obj.get("latencyMs")
            sz = eval_obj.get("sizeBytes")

            # NON_FINITE vs METRIC_RANGE
            if not is_finite_number(acc) or not is_finite_number(lat) or not is_finite_number(sz):
                v_codes.add("NON_FINITE")
            else:
                if not (0.0 <= acc <= 1.0):
                    v_codes.add("METRIC_RANGE")

            if isinstance(lat, (int, float)) and not isinstance(lat, bool) and math.isfinite(lat) and lat < 0:
                v_codes.add("METRIC_RANGE")

            if isinstance(sz, (int, float)) and not isinstance(sz, bool) and math.isfinite(sz) and sz < 0:
                v_codes.add("METRIC_RANGE")

            # Gate floor limits
            if is_finite_number(acc) and 0.0 <= acc <= 1.0:
                if acc < accuracy_floor:
                    v_codes.add("ACCURACY_FLOOR")

            if is_finite_number(lat) and lat >= 0:
                if lat > max_latency_ms:
                    v_codes.add("LATENCY_LIMIT")

            if is_non_negative_safe_int(sz):
                if sz > max_size_bytes:
                    v_codes.add("SIZE_LIMIT")

            # Slices evaluation
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
                        elif s_val < req_floor:
                            v_codes.add(f"SLICE_FLOOR:{req_name}")

        # Store failed gates sorted by UTF-8 bytes
        if isinstance(v_id, str):
            sorted_v_codes = sorted(list(v_codes), key=lambda x: x.encode("utf-8"))
            failed_gates[v_id] = sorted_v_codes

            if len(v_codes) == 0:
                valid_version_objects.append(v_item)

    # Determine eligible version strings
    eligible_versions = [v["version"] for v in valid_version_objects]
    eligible_versions.sort(key=lambda x: x.encode("utf-8"))

    # Validate Champion Eligibility
    champion_item = next((v for v in valid_version_objects if v["version"] == current_champion_version), None)

    if champion_item is None:
        # Champion evidence is invalid or champion not present -> BLOCK
        return JSONResponse(status_code=200, content={
            "action": "block",
            "championVersion": current_champion_version,
            "selectedVersion": None,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None
        })

    # Rank all eligible versions
    # Sorting key: accuracy descending, latency ascending, size ascending, numeric version ascending
    def rank_key(v):
        e = v["evaluation"]
        return (
            -e["accuracy"],
            e["latencyMs"],
            e["sizeBytes"],
            int(v["version"])
        )

    ranked_eligible = sorted(valid_version_objects, key=rank_key)
    best_challenger = ranked_eligible[0]

    champion_acc = champion_item["evaluation"]["accuracy"]
    challenger_acc = best_challenger["evaluation"]["accuracy"]

    acc_diff = round_12(challenger_acc - champion_acc)

    if best_challenger["version"] != current_champion_version and acc_diff >= min_improvement:
        # PROMOTE
        new_champion_version = best_challenger["version"]
        ALIAS_STORE["champion"] = new_champion_version

        return JSONResponse(status_code=200, content={
            "action": "promote",
            "championVersion": current_champion_version,
            "selectedVersion": new_champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": {
                "alias": "champion",
                "version": new_champion_version
            },
            "evidence": best_challenger["evaluation"]
        })
    else:
        # RETAIN
        return JSONResponse(status_code=200, content={
            "action": "retain",
            "championVersion": current_champion_version,
            "selectedVersion": current_champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": champion_item["evaluation"]
        })
