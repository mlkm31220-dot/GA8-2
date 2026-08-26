from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI()


# ============================================================
# IN-MEMORY STORES
# ============================================================

RUN_STORE: Dict[str, Dict[str, Any]] = {}
RUN_INPUT_STORE: Dict[str, Dict[str, Any]] = {}

# Persistent for the lifetime of the running service.
ALIAS_STORE: Dict[str, str] = {}


# ============================================================
# REGEX / CONSTANTS
# ============================================================

ISO_8601_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$"
)

CANONICAL_VERSION_REGEX = re.compile(
    r"^(0|[1-9]\d*)$"
)


# ============================================================
# HELPERS
# ============================================================

def parse_iso8601_utc(ts_str: Any) -> Optional[float]:
    """
    Accept:
      YYYY-MM-DDTHH:mm:ssZ
      YYYY-MM-DDTHH:mm:ss.sssZ
      YYYY-MM-DDTHH:mm:ss+05:30
      YYYY-MM-DDTHH:mm:ss.sss+05:30

    Fractional seconds: 1-3 digits.
    """

    if not isinstance(ts_str, str):
        return None

    if not ISO_8601_REGEX.match(ts_str):
        return None

    try:
        formatted = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(formatted)
        return dt.timestamp()
    except Exception:
        return None


def round_12(value: float) -> float:
    return round(value, 12)


def is_non_negative_safe_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= (2**53 - 1)
    )


def is_positive_safe_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= (2**53 - 1)
    )


def is_canonical_version_str(value: Any) -> bool:
    """
    Valid:
        "1"
        "2"
        "100"

    Invalid:
        "01"
        "001"
        "0"
        "-1"
        1
    """

    if not isinstance(value, str):
        return False

    if not CANONICAL_VERSION_REGEX.match(value):
        return False

    try:
        number = int(value)

        return (
            1 <= number <= (2**53 - 1)
        )

    except Exception:
        return False


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


# ============================================================
# BQML ENDPOINT
# ============================================================

@app.post("/bqml")
async def handle_bqml(
    body: Dict[str, Any]
):

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if "phase" not in body:
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    phase = body.get("phase")

    if phase == "select":
        return process_select_phase(body)

    if phase == "evaluate":
        return process_evaluate_phase(body)

    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_INPUT"
        }
    )


# ============================================================
# BQML SELECT
# ============================================================

def process_select_phase(
    body: Dict[str, Any]
) -> JSONResponse:

    run_id = body.get("runId")

    if (
        not isinstance(run_id, str)
        or not (1 <= len(run_id) <= 128)
    ):
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # --------------------------------------------------------
    # Idempotency / conflict
    # --------------------------------------------------------

    if run_id in RUN_STORE:

        if RUN_INPUT_STORE.get(run_id) == body:

            return JSONResponse(
                status_code=200,
                content=RUN_STORE[run_id]
            )

        return JSONResponse(
            status_code=409,
            content={
                "error": "RUN_ID_CONFLICT"
            }
        )

    reason_codes: Set[str] = set()

    num_trials_limit = body.get(
        "numTrialsLimit"
    )

    forbidden_features = body.get(
        "forbiddenFeatures"
    )

    rows = body.get("rows")

    trials = body.get("trials")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    valid_structure = True

    if not is_positive_safe_int(
        num_trials_limit
    ):
        valid_structure = False

    if (
        not isinstance(
            forbidden_features,
            list
        )
        or not all(
            isinstance(x, str)
            for x in forbidden_features
        )
    ):
        valid_structure = False

    if (
        not isinstance(rows, list)
        or len(rows) == 0
    ):
        valid_structure = False

    if not isinstance(trials, list):
        valid_structure = False

    if not valid_structure:
        reason_codes.add(
            "INVALID_INPUT"
        )

    # --------------------------------------------------------
    # Trial limit
    # --------------------------------------------------------

    if (
        isinstance(trials, list)
        and is_positive_safe_int(
            num_trials_limit
        )
    ):

        if len(trials) > num_trials_limit:

            reason_codes.add(
                "TRIAL_LIMIT_EXCEEDED"
            )

    # --------------------------------------------------------
    # Validate rows
    # --------------------------------------------------------

    valid_rows = []

    seen_row_ids = set()

    if (
        isinstance(rows, list)
        and "INVALID_INPUT"
        not in reason_codes
    ):

        for row in rows:

            if not isinstance(row, dict):

                reason_codes.add(
                    "INVALID_INPUT"
                )

                break

            row_id = row.get("id")
            entity = row.get("entity")
            event_time = row.get(
                "eventTime"
            )
            prediction_time = row.get(
                "predictionTime"
            )
            version = row.get("version")
            split = row.get("split")
            features = row.get("features")

            if (
                not isinstance(
                    row_id,
                    str
                )
                or row_id in seen_row_ids
                or not isinstance(
                    entity,
                    str
                )
                or not isinstance(
                    event_time,
                    str
                )
                or not isinstance(
                    prediction_time,
                    str
                )
                or not is_non_negative_safe_int(
                    version
                )
                or split not in (
                    "TRAIN",
                    "EVAL"
                )
                or not isinstance(
                    features,
                    dict
                )
            ):

                reason_codes.add(
                    "INVALID_INPUT"
                )

                break

            seen_row_ids.add(row_id)

            event_ts = parse_iso8601_utc(
                event_time
            )

            prediction_ts = parse_iso8601_utc(
                prediction_time
            )

            if (
                event_ts is None
                or prediction_ts is None
            ):

                reason_codes.add(
                    "INVALID_INPUT"
                )

                break

            valid_rows.append({
                "id": row_id,
                "entity": entity,
                "eventTime": event_time,
                "evt_ts": event_ts,
                "pred_ts": prediction_ts,
                "version": version,
                "split": split,
                "features": features
            })

    # --------------------------------------------------------
    # Deduplication
    # --------------------------------------------------------

    retained_rows = []

    if (
        "INVALID_INPUT"
        not in reason_codes
    ):

        dedup_map: Dict[
            Tuple[str, float],
            List[Dict[str, Any]]
        ] = {}

        for row in valid_rows:

            key = (
                row["entity"],
                row["evt_ts"]
            )

            dedup_map.setdefault(
                key,
                []
            ).append(row)

        for group in dedup_map.values():

            group.sort(
                key=lambda x: (
                    -x["version"],
                    x["id"].encode(
                        "utf-8"
                    )
                )
            )

            retained_rows.append(
                group[0]
            )

    # --------------------------------------------------------
    # Feature eligibility
    # --------------------------------------------------------

    eligible_features = []

    if (
        "INVALID_INPUT"
        not in reason_codes
        and retained_rows
    ):

        forbidden_set = set(
            forbidden_features
        )

        candidate_features = set(
            retained_rows[0][
                "features"
            ].keys()
        )

        for row in retained_rows[1:]:

            candidate_features &= set(
                row["features"].keys()
            )

        for feature in candidate_features:

            if feature in forbidden_set:
                continue

            eligible = True

            for row in retained_rows:

                feature_object = row[
                    "features"
                ].get(feature)

                if (
                    not isinstance(
                        feature_object,
                        dict
                    )
                    or "availableAt"
                    not in feature_object
                ):

                    eligible = False
                    break

                available_ts = parse_iso8601_utc(
                    feature_object[
                        "availableAt"
                    ]
                )

                if (
                    available_ts is None
                    or available_ts
                    > row["pred_ts"]
                ):

                    eligible = False
                    break

            if eligible:
                eligible_features.append(
                    feature
                )

        eligible_features.sort(
            key=lambda x: x.encode(
                "utf-8"
            )
        )

    # --------------------------------------------------------
    # Train / Eval IDs
    # --------------------------------------------------------

    train_row_ids = []
    eval_row_ids = []

    if (
        "INVALID_INPUT"
        not in reason_codes
    ):

        for row in retained_rows:

            if row["split"] == "TRAIN":
                train_row_ids.append(
                    row["id"]
                )

            elif row["split"] == "EVAL":
                eval_row_ids.append(
                    row["id"]
                )

        train_row_ids.sort(
            key=lambda x: x.encode(
                "utf-8"
            )
        )

        eval_row_ids.sort(
            key=lambda x: x.encode(
                "utf-8"
            )
        )

    # --------------------------------------------------------
    # Dataset digest
    # --------------------------------------------------------

    dataset_digest = None

    if (
        "INVALID_INPUT"
        not in reason_codes
    ):

        digest_object = {
            "trainRowIds":
                train_row_ids,
            "evalRowIds":
                eval_row_ids,
            "featureNames":
                eligible_features
        }

        compact_json = json.dumps(
            digest_object,
            separators=(
                ",",
                ":"
            )
        )

        dataset_digest = hashlib.sha256(
            compact_json.encode(
                "utf-8"
            )
        ).hexdigest()

    # --------------------------------------------------------
    # Trials
    # --------------------------------------------------------

    selected_trial_id = None

    eligible_trials = []

    seen_trial_ids = set()

    if isinstance(trials, list):

        for trial in trials:

            if not isinstance(
                trial,
                dict
            ):

                reason_codes.add(
                    "INVALID_INPUT"
                )

                break

            trial_id = trial.get(
                "trialId"
            )

            status = trial.get(
                "status"
            )

            metric = trial.get(
                "evalMetric"
            )

            if (
                not is_non_negative_safe_int(
                    trial_id
                )
                or trial_id
                in seen_trial_ids
                or status
                not in (
                    "SUCCEEDED",
                    "FAILED"
                )
            ):

                reason_codes.add(
                    "INVALID_INPUT"
                )

                break

            seen_trial_ids.add(
                trial_id
            )

            if status == "SUCCEEDED":

                if (
                    isinstance(
                        metric,
                        (int, float)
                    )
                    and not isinstance(
                        metric,
                        bool
                    )
                    and math.isfinite(
                        metric
                    )
                ):

                    eligible_trials.append({
                        "trialId":
                            trial_id,
                        "evalMetric":
                            float(metric)
                    })

    if (
        "INVALID_INPUT"
        not in reason_codes
    ):

        if not eligible_trials:

            reason_codes.add(
                "NO_SUCCESSFUL_TRIAL"
            )

        else:

            eligible_trials.sort(
                key=lambda x: (
                    -x["evalMetric"],
                    x["trialId"]
                )
            )

            selected_trial_id = (
                eligible_trials[0][
                    "trialId"
                ]
            )

    if reason_codes:
        selected_trial_id = None

    if "INVALID_INPUT" in reason_codes:

        dataset_digest = None
        train_row_ids = []
        eval_row_ids = []
        eligible_features = []

    sorted_reasons = sorted(
        list(reason_codes),
        key=lambda x: x.encode(
            "utf-8"
        )
    )

    response = {
        "runId": run_id,
        "selectedTrialId":
            selected_trial_id,
        "trainRowIds":
            train_row_ids,
        "evalRowIds":
            eval_row_ids,
        "featureNames":
            eligible_features,
        "datasetDigest":
            dataset_digest,
        "reasonCodes":
            sorted_reasons
    }

    RUN_STORE[run_id] = response
    RUN_INPUT_STORE[run_id] = body

    return JSONResponse(
        status_code=200,
        content=response
    )


# ============================================================
# BQML EVALUATE
# ============================================================

def process_evaluate_phase(
    body: Dict[str, Any]
) -> JSONResponse:

    run_id = body.get("runId")
    selected_trial_id = body.get(
        "selectedTrialId"
    )
    dataset_digest = body.get(
        "datasetDigest"
    )
    metric_floor = body.get(
        "metricFloor"
    )
    required_slices = body.get(
        "requiredSlices"
    )
    rows = body.get("rows")
    bytes_processed = body.get(
        "bytesProcessed"
    )
    max_bytes = body.get(
        "maxBytes"
    )

    reason_codes: Set[str] = set()

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if (
        not isinstance(run_id, str)
        or not is_non_negative_safe_int(
            selected_trial_id
        )
        or not isinstance(
            dataset_digest,
            str
        )
    ):

        reason_codes.add(
            "INVALID_INPUT"
        )

    if (
        not isinstance(
            metric_floor,
            (int, float)
        )
        or isinstance(
            metric_floor,
            bool
        )
        or not (
            0.0
            <= metric_floor
            <= 1.0
        )
        or not math.isfinite(
            metric_floor
        )
    ):

        reason_codes.add(
            "INVALID_INPUT"
        )

    if not isinstance(
        required_slices,
        dict
    ):

        reason_codes.add(
            "INVALID_INPUT"
        )

    else:

        for name, floor in (
            required_slices.items()
        ):

            if (
                not isinstance(
                    name,
                    str
                )
                or not is_finite_number(
                    floor
                )
                or not (
                    0.0
                    <= floor
                    <= 1.0
                )
            ):

                reason_codes.add(
                    "INVALID_INPUT"
                )

                break

    if (
        not is_non_negative_safe_int(
            bytes_processed
        )
        or not is_non_negative_safe_int(
            max_bytes
        )
    ):

        reason_codes.add(
            "INVALID_INPUT"
        )

    if not isinstance(
        rows,
        list
    ):

        reason_codes.add(
            "INVALID_INPUT"
        )

    # --------------------------------------------------------
    # Lineage
    # --------------------------------------------------------

    stored_run = RUN_STORE.get(
        run_id
    )

    lineage_valid = True

    if (
        stored_run is None
        or stored_run.get(
            "selectedTrialId"
        ) is None
    ):

        lineage_valid = False

    elif (
        stored_run.get(
            "selectedTrialId"
        )
        != selected_trial_id
        or stored_run.get(
            "datasetDigest"
        )
        != dataset_digest
    ):

        lineage_valid = False

    if not lineage_valid:

        reason_codes.add(
            "INVALID_LINEAGE"
        )

    # --------------------------------------------------------
    # Test rows
    # --------------------------------------------------------

    invalid_test_row = False

    if (
        isinstance(rows, list)
        and len(rows) > 0
    ):

        for row in rows:

            if not isinstance(
                row,
                dict
            ):

                invalid_test_row = True
                break

            label = row.get(
                "label"
            )

            prediction = row.get(
                "prediction"
            )

            slice_name = row.get(
                "slice"
            )

            if (
                label not in (0, 1)
                or prediction not in (
                    0,
                    1
                )
                or type(label) is bool
                or type(prediction) is bool
            ):

                invalid_test_row = True
                break

            if (
                not isinstance(
                    slice_name,
                    str
                )
                or len(slice_name) == 0
            ):

                invalid_test_row = True
                break

    if invalid_test_row:

        reason_codes.add(
            "INVALID_TEST_ROW"
        )

    # --------------------------------------------------------
    # Bytes
    # --------------------------------------------------------

    if (
        is_non_negative_safe_int(
            bytes_processed
        )
        and is_non_negative_safe_int(
            max_bytes
        )
        and bytes_processed > max_bytes
    ):

        reason_codes.add(
            "BYTE_LIMIT"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    test_metric = None

    critical_slice_pass = True

    if (
        isinstance(rows, list)
        and len(rows) > 0
        and not invalid_test_row
        and "INVALID_INPUT"
        not in reason_codes
    ):

        correct_count = sum(
            1
            for row in rows
            if row["label"]
            == row["prediction"]
        )

        test_metric = round_12(
            correct_count
            / len(rows)
        )

        if (
            test_metric
            < metric_floor
        ):

            reason_codes.add(
                "AGGREGATE_FLOOR"
            )

        slice_map: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        for row in rows:

            slice_map.setdefault(
                row["slice"],
                []
            ).append(row)

        for (
            slice_name,
            slice_floor
        ) in required_slices.items():

            if (
                slice_name
                not in slice_map
            ):

                reason_codes.add(
                    f"MISSING_SLICE:{slice_name}"
                )

                critical_slice_pass = False

            else:

                slice_rows = slice_map[
                    slice_name
                ]

                correct = sum(
                    1
                    for row in slice_rows
                    if row["label"]
                    == row["prediction"]
                )

                accuracy = round_12(
                    correct
                    / len(slice_rows)
                )

                if (
                    accuracy
                    < slice_floor
                ):

                    reason_codes.add(
                        f"SLICE_FLOOR:{slice_name}"
                    )

                    critical_slice_pass = False

    if (
        "INVALID_INPUT"
        in reason_codes
        or "INVALID_LINEAGE"
        in reason_codes
        or "INVALID_TEST_ROW"
        in reason_codes
        or any(
            code.startswith(
                "MISSING_SLICE:"
            )
            or code.startswith(
                "SLICE_FLOOR:"
            )
            for code in reason_codes
        )
    ):

        critical_slice_pass = False

    decision = (
        "admit"
        if len(reason_codes) == 0
        else "reject"
    )

    sorted_reasons = sorted(
        list(reason_codes),
        key=lambda x: x.encode(
            "utf-8"
        )
    )

    return JSONResponse(
        status_code=200,
        content={
            "runId": run_id,
            "selectedTrialId":
                selected_trial_id,
            "datasetDigest":
                dataset_digest,
            "testMetric":
                test_metric,
            "criticalSlicePass":
                critical_slice_pass,
            "decision":
                decision,
            "bytesProcessed":
                (
                    bytes_processed
                    if is_non_negative_safe_int(
                        bytes_processed
                    )
                    else None
                ),
            "reasonCodes":
                sorted_reasons
        }
    )


# ============================================================
# MODEL PROMOTION ENDPOINT
# ============================================================

@app.post("/promote")
async def handle_promote(
    body: Dict[str, Any]
):

    # ========================================================
    # TOP-LEVEL VALIDATION
    # ========================================================

    if not isinstance(
        body,
        dict
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    as_of = body.get(
        "asOf"
    )

    champion_version = body.get(
        "championVersion"
    )

    policy = body.get(
        "policy"
    )

    versions = body.get(
        "versions"
    )

    # asOf must be a string
    if not isinstance(
        as_of,
        str
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # championVersion MUST be string.
    #
    # IMPORTANT:
    # Do not do str(champion_version).
    #
    # 1 is invalid.
    # "1" is valid.

    if not isinstance(
        champion_version,
        str
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if not isinstance(
        policy,
        dict
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if not isinstance(
        versions,
        list
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # ========================================================
    # AS-OF
    # ========================================================

    as_of_ts = parse_iso8601_utc(
        as_of
    )

    if as_of_ts is None:

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # ========================================================
    # POLICY
    # ========================================================

    dataset_digest = policy.get(
        "datasetDigest"
    )

    schema_digest = policy.get(
        "schemaDigest"
    )

    max_age_seconds = policy.get(
        "maxAgeSeconds"
    )

    accuracy_floor = policy.get(
        "accuracyFloor"
    )

    required_slices = policy.get(
        "requiredSlices"
    )

    max_latency_ms = policy.get(
        "maxLatencyMs"
    )

    max_size_bytes = policy.get(
        "maxSizeBytes"
    )

    min_improvement = policy.get(
        "minImprovement",
        0.0
    )

    # --------------------------------------------------------
    # Dataset digest
    # --------------------------------------------------------

    if (
        not isinstance(
            dataset_digest,
            str
        )
        or not dataset_digest
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # --------------------------------------------------------
    # Schema digest
    # --------------------------------------------------------

    if (
        not isinstance(
            schema_digest,
            str
        )
        or not schema_digest
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # --------------------------------------------------------
    # Max age
    # --------------------------------------------------------

    if not is_non_negative_safe_int(
        max_age_seconds
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # --------------------------------------------------------
    # Accuracy floor
    # --------------------------------------------------------

    if (
        not is_finite_number(
            accuracy_floor
        )
        or not (
            0.0
            <= accuracy_floor
            <= 1.0
        )
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # --------------------------------------------------------
    # Required slices
    # --------------------------------------------------------

    if not isinstance(
        required_slices,
        dict
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    for (
        slice_name,
        slice_floor
    ) in required_slices.items():

        if (
            not isinstance(
                slice_name,
                str
            )
            or not is_finite_number(
                slice_floor
            )
            or not (
                0.0
                <= slice_floor
                <= 1.0
            )
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

    # --------------------------------------------------------
    # Max latency
    # --------------------------------------------------------

    if (
        not is_finite_number(
            max_latency_ms
        )
        or max_latency_ms < 0
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # --------------------------------------------------------
    # Max size
    # --------------------------------------------------------

    if not is_non_negative_safe_int(
        max_size_bytes
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # --------------------------------------------------------
    # Min improvement
    # --------------------------------------------------------

    if not is_finite_number(
        min_improvement
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if not (
        0.0
        <= min_improvement
        <= 1.0
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    # ========================================================
    # FIRST PASS:
    #
    # Detect duplicate / noncanonical versions BEFORE
    # constructing any version lookup map.
    # ========================================================

    seen_versions: Set[str] = set()

    failed_gates: Dict[
        str,
        Set[str]
    ] = {}

    # We also track the actual listed version objects.
    listed_versions: List[
        Tuple[str, Dict[str, Any]]
    ] = []

    for item in versions:

        if not isinstance(
            item,
            dict
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

        version = item.get(
            "version"
        )

        # Version must be a string.
        if not isinstance(
            version,
            str
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

        codes = failed_gates.setdefault(
            version,
            set()
        )

        # Canonical version check
        if not is_canonical_version_str(
            version
        ):

            codes.add(
                "INVALID_VERSION"
            )

        # Duplicate check
        if version in seen_versions:

            codes.add(
                "DUPLICATE_VERSION"
            )

            # Also mark the first occurrence.
            failed_gates.setdefault(
                version,
                set()
            ).add(
                "DUPLICATE_VERSION"
            )

        seen_versions.add(
            version
        )

        listed_versions.append(
            (
                version,
                item
            )
        )

    # ========================================================
    # SECOND PASS:
    # EVALUATE EVERY VERSION
    # ========================================================

    eligible_items: List[
        Dict[str, Any]
    ] = []

    for version, item in listed_versions:

        codes = failed_gates[
            version
        ]

        # Invalid or duplicate versions
        # can never become eligible.

        if (
            "INVALID_VERSION"
            in codes
            or "DUPLICATE_VERSION"
            in codes
        ):
            continue

        # ----------------------------------------------------
        # Registered artifact digest
        # ----------------------------------------------------

        artifact_digest = item.get(
            "artifactDigest"
        )

        if not isinstance(
            artifact_digest,
            str
        ):

            codes.add(
                "INVALID_VERSION"
            )

            continue

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        evaluation = item.get(
            "evaluation"
        )

        if not isinstance(
            evaluation,
            dict
        ):

            codes.add(
                "MISSING_EVALUATION"
            )

            continue

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        created_at = evaluation.get(
            "createdAt"
        )

        created_at_ts = parse_iso8601_utc(
            created_at
        )

        if created_at_ts is None:

            codes.add(
                "INVALID_TIMESTAMP"
            )

        else:

            # Future
            if created_at_ts > as_of_ts:

                codes.add(
                    "FUTURE_EVALUATION"
                )

            # Stale
            elif created_at_ts < (
                as_of_ts
                - max_age_seconds
            ):

                codes.add(
                    "STALE_EVALUATION"
                )

        # ----------------------------------------------------
        # Artifact lineage
        # ----------------------------------------------------

        evaluation_artifact = evaluation.get(
            "artifactDigest"
        )

        if (
            not isinstance(
                evaluation_artifact,
                str
            )
            or evaluation_artifact
            != artifact_digest
        ):

            codes.add(
                "ARTIFACT_MISMATCH"
            )

        # ----------------------------------------------------
        # Dataset lineage
        # ----------------------------------------------------

        evaluation_dataset = evaluation.get(
            "datasetDigest"
        )

        if (
            not isinstance(
                evaluation_dataset,
                str
            )
            or evaluation_dataset
            != dataset_digest
        ):

            codes.add(
                "DATASET_MISMATCH"
            )

        # ----------------------------------------------------
        # Schema lineage
        # ----------------------------------------------------

        evaluation_schema = evaluation.get(
            "schemaDigest"
        )

        if (
            not isinstance(
                evaluation_schema,
                str
            )
            or evaluation_schema
            != schema_digest
        ):

            codes.add(
                "SCHEMA_MISMATCH"
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = evaluation.get(
            "accuracy"
        )

        latency = evaluation.get(
            "latencyMs"
        )

        size = evaluation.get(
            "sizeBytes"
        )

        metrics_are_finite = (
            is_finite_number(
                accuracy
            )
            and is_finite_number(
                latency
            )
            and is_finite_number(
                size
            )
        )

        if not metrics_are_finite:

            codes.add(
                "NON_FINITE"
            )

        else:

            if not (
                0.0
                <= accuracy
                <= 1.0
            ):

                codes.add(
                    "METRIC_RANGE"
                )

            if latency < 0:

                codes.add(
                    "METRIC_RANGE"
                )

            if size < 0:

                codes.add(
                    "METRIC_RANGE"
                )

        # ----------------------------------------------------
        # Accuracy floor
        # ----------------------------------------------------

        if (
            is_finite_number(
                accuracy
            )
            and 0.0
            <= accuracy
            <= 1.0
        ):

            if (
                round_12(
                    accuracy
                )
                < round_12(
                    accuracy_floor
                )
            ):

                codes.add(
                    "ACCURACY_FLOOR"
                )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        if (
            is_finite_number(
                latency
            )
            and latency >= 0
        ):

            if (
                round_12(
                    latency
                )
                > round_12(
                    max_latency_ms
                )
            ):

                codes.add(
                    "LATENCY_LIMIT"
                )

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------

        if (
            is_non_negative_safe_int(
                size
            )
            and size > max_size_bytes
        ):

            codes.add(
                "SIZE_LIMIT"
            )

        # ----------------------------------------------------
        # Required slices
        # ----------------------------------------------------

        slices = evaluation.get(
            "slices"
        )

        if not isinstance(
            slices,
            dict
        ):

            for slice_name in (
                required_slices.keys()
            ):

                codes.add(
                    f"MISSING_SLICE:{slice_name}"
                )

        else:

            for (
                slice_name,
                slice_floor
            ) in required_slices.items():

                if (
                    slice_name
                    not in slices
                ):

                    codes.add(
                        f"MISSING_SLICE:{slice_name}"
                    )

                    continue

                slice_value = slices[
                    slice_name
                ]

                if not is_finite_number(
                    slice_value
                ):

                    codes.add(
                        "NON_FINITE"
                    )

                    continue

                if not (
                    0.0
                    <= slice_value
                    <= 1.0
                ):

                    codes.add(
                        f"SLICE_RANGE:{slice_name}"
                    )

                    continue

                if (
                    round_12(
                        slice_value
                    )
                    < round_12(
                        slice_floor
                    )
                ):

                    codes.add(
                        f"SLICE_FLOOR:{slice_name}"
                    )

        # ----------------------------------------------------
        # Eligible
        # ----------------------------------------------------

        if len(codes) == 0:

            eligible_items.append(
                item
            )

    # ========================================================
    # SORT FAILED GATES
    # ========================================================

    final_failed_gates: Dict[
        str,
        List[str]
    ] = {}

    for version, codes in (
        failed_gates.items()
    ):

        final_failed_gates[
            version
        ] = sorted(
            list(codes),
            key=lambda x: x.encode(
                "utf-8"
            )
        )

    # ========================================================
    # ELIGIBLE VERSIONS
    # ========================================================

    eligible_versions = sorted(
        [
            str(
                item["version"]
            )
            for item in eligible_items
        ],
        key=lambda x: int(x)
    )

    # ========================================================
    # FIND CHAMPION
    # ========================================================

    champion_item = None

    for version, item in listed_versions:

        if version == champion_version:

            champion_item = item
            break

    # Champion is not listed.
    if champion_item is None:

        return JSONResponse(
            status_code=200,
            content={
                "action": "block",
                "championVersion":
                    champion_version,
                "selectedVersion":
                    None,
                "eligibleVersions":
                    eligible_versions,
                "failedGates":
                    final_failed_gates,
                "aliasMutation":
                    None,
                "evidence":
                    None
            }
        )

    # ========================================================
    # CHAMPION MUST PASS ALL GATES
    # ========================================================

    champion_failed_gates = (
        final_failed_gates.get(
            champion_version,
            []
        )
    )

    if len(champion_failed_gates) > 0:

        return JSONResponse(
            status_code=200,
            content={
                "action": "block",
                "championVersion":
                    champion_version,
                "selectedVersion":
                    None,
                "eligibleVersions":
                    eligible_versions,
                "failedGates":
                    final_failed_gates,
                "aliasMutation":
                    None,
                "evidence":
                    None
            }
        )

    # ========================================================
    # FIND WINNER
    #
    # accuracy DESC
    # latency ASC
    # size ASC
    # numeric version ASC
    # ========================================================

    def ranking_key(
        item: Dict[str, Any]
    ):

        evaluation = item[
            "evaluation"
        ]

        return (
            -round_12(
                evaluation[
                    "accuracy"
                ]
            ),
            round_12(
                evaluation[
                    "latencyMs"
                ]
            ),
            evaluation[
                "sizeBytes"
            ],
            int(
                item["version"]
            )
        )

    ranked_items = sorted(
        eligible_items,
        key=ranking_key
    )

    # Champion is valid, therefore there
    # should be at least one eligible item.
    if not ranked_items:

        return JSONResponse(
            status_code=200,
            content={
                "action": "block",
                "championVersion":
                    champion_version,
                "selectedVersion":
                    None,
                "eligibleVersions":
                    eligible_versions,
                "failedGates":
                    final_failed_gates,
                "aliasMutation":
                    None,
                "evidence":
                    None
            }
        )

    winner = ranked_items[0]

    winner_version = str(
        winner["version"]
    )

    # ========================================================
    # CHAMPION ACCURACY
    # ========================================================

    champion_accuracy = round_12(
        champion_item[
            "evaluation"
        ][
            "accuracy"
        ]
    )

    winner_accuracy = round_12(
        winner[
            "evaluation"
        ][
            "accuracy"
        ]
    )

    improvement = round_12(
        winner_accuracy
        - champion_accuracy
    )

    # ========================================================
    # PROMOTE
    # ========================================================

    if (
        winner_version
        != champion_version
        and improvement
        >= round_12(
            min_improvement
        )
    ):

        # Persist alias.
        ALIAS_STORE[
            "champion"
        ] = winner_version

        return JSONResponse(
            status_code=200,
            content={
                "action":
                    "promote",
                "championVersion":
                    champion_version,
                "selectedVersion":
                    winner_version,
                "eligibleVersions":
                    eligible_versions,
                "failedGates":
                    final_failed_gates,
                "aliasMutation": {
                    "alias":
                        "champion",
                    "version":
                        winner_version
                },
                "evidence":
                    winner[
                        "evaluation"
                    ]
            }
        )

    # ========================================================
    # RETAIN
    # ========================================================

    return JSONResponse(
        status_code=200,
        content={
            "action":
                "retain",
            "championVersion":
                champion_version,
            "selectedVersion":
                champion_version,
            "eligibleVersions":
                eligible_versions,
            "failedGates":
                final_failed_gates,
            "aliasMutation":
                None,
            "evidence":
                champion_item[
                    "evaluation"
                ]
        }
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "ok",
        "service": "model-registry-promotion-gate"
    }
