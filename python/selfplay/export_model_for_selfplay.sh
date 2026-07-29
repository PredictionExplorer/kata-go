#!/bin/bash -eu
set -o pipefail
{
#Takes any models in torchmodels_toexport/ and outputs a cuda-runnable model file to modelstobetested/
#Takes any models in torchmodels_toexport_extra/ and outputs a cuda-runnable model file to models_extra/
#Should be run periodically.

if [[ $# -ne 3 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR USEGATING"
    echo "Currently expects to be run from within the 'python' directory of the KataGo repo, or otherwise in the same dir as export_model.py."
    echo "NAMEPREFIX string prefix for this training run, try to pick something globally unique. Will be displayed to users when KataGo loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "USEGATING = 1 to use gatekeeper, 0 to not use gatekeeper and output directly to models/"
    exit 0
fi
NAMEPREFIX="$1"
shift
BASEDIR="$1"
shift
USEGATING="$1"
shift

#------------------------------------------------------------------------------

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

mkdir -p "$BASEDIR"/torchmodels_toexport
mkdir -p "$BASEDIR"/torchmodels_exported
mkdir -p "$BASEDIR"/torchmodels_toexport_extra
mkdir -p "$BASEDIR"/modelstobetested
mkdir -p "$BASEDIR"/models_extra
mkdir -p "$BASEDIR"/models

function exportStuff() {
    FROMDIR="$1"
    TODIR="$2"

    # Sort by timestamp so that we process in order of oldest to newest if there are multiple
    $PYTHON -W ignore "$(dirname "$0")/list_by_mtime.py" "$BASEDIR/$FROMDIR" | while read -r FILEPATH
    do
        #Make sure to skip tmp directories that are transiently there by the training,
        #they are probably in the process of being written
        if [ -z "$FILEPATH" ]
        then
            continue
        fi
        if [ "${FILEPATH: -4}" == ".tmp" ]
        then
            echo "Skipping tmp file:" "$FILEPATH"
        elif [ "${FILEPATH: -9}" == ".exported" ]
        then
            echo "Skipping self tmp file:" "$FILEPATH"
        else
            echo "Found model to export:" "$FILEPATH"
            NAME="$(basename "$FILEPATH")"

            SRC="$BASEDIR/$FROMDIR/$NAME"
            TMPDST="$BASEDIR/$FROMDIR/$NAME.exported"
            TARGET="$BASEDIR/$TODIR/$NAME"

            if [ -d "$BASEDIR"/modelstobetested/"$NAME" ] ||  \
               [ -d "$BASEDIR"/rejectedmodels/"$NAME" ] || \
               [ -d "$BASEDIR"/models/"$NAME" ] || \
               [ -d "$BASEDIR"/models_extra/"$NAME" ] || \
               [ -d "$BASEDIR"/modelsuploaded/"$NAME" ]
            then
                echo "Model with same name already exists, so skipping:" "$SRC"
            else
                rm -rf "$TMPDST"
                mkdir "$TMPDST"

                set -x
                $PYTHON ./export_model_pytorch.py \
                        -checkpoint "$SRC/model.ckpt" \
                        -export-dir "$TMPDST" \
                        -model-name "$NAMEPREFIX-$NAME" \
                        -filename-prefix model \
                        -use-swa

                $PYTHON ./clean_checkpoint.py \
                        -checkpoint "$SRC/model.ckpt" \
                        -output "$TMPDST/model.ckpt"
                set +x

                rm -r "$SRC"
                gzip "$TMPDST"/model.bin

                #Make a bunch of the directories that selfplay will need so that there isn't a race on the selfplay
                #machines to concurrently make it, since sometimes concurrent making of the same directory can corrupt
                #a filesystem
                #Only when not gating. When gating, gatekeeper is responsible.
                if [ "$USEGATING" -eq 0 ]
                then
                    if [ "$TODIR" != "models_extra" ]
                    then
                        mkdir -p "$BASEDIR/selfplay/$NAME/sgfs"
                        mkdir -p "$BASEDIR/selfplay/$NAME/tdata"
                    fi
                fi

                #Sleep a little to allow some tolerance on the filesystem
                sleep 5

                mv "$TMPDST" "$TARGET"
                echo "Done exporting:" "$NAME" "to" "$TARGET"
            fi
        fi
    done
}

function exportGatedStuffHardened() {
    FROMDIR="$1"
    if [ -n "${KATAGO_PROMOTION_BACKPRESSURE_FILE:-}" ]
    then
        if [ -z "${KATAGO_PROMOTION_POLICY_HASH:-}" ]
        then
            echo "KATAGO_PROMOTION_POLICY_HASH is required with KATAGO_PROMOTION_BACKPRESSURE_FILE." >&2
            return 1
        fi
        BACKPRESSURE_MAX_AGE="${KATAGO_PROMOTION_BACKPRESSURE_MAX_AGE_SECONDS:-120}"
        if ! BACKPRESSURE_STATUS="$("$PYTHON" - \
            "$KATAGO_PROMOTION_BACKPRESSURE_FILE" \
            "$KATAGO_PROMOTION_POLICY_HASH" \
            "$BACKPRESSURE_MAX_AGE" <<'PY'
import datetime
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
expected_policy_hash = sys.argv[2]
try:
    maximum_age = float(sys.argv[3])
except ValueError as exc:
    raise SystemExit(f"invalid backpressure maximum age: {exc}")
if maximum_age <= 0:
    raise SystemExit("backpressure maximum age must be positive")
if not re.fullmatch(r"[0-9a-f]{64}", expected_policy_hash):
    raise SystemExit("promotion policy hash must be lowercase SHA-256")
if path.is_symlink() or not path.is_file():
    raise SystemExit(f"backpressure status is not a regular file: {path}")
data = path.read_bytes()
try:
    value = json.loads(data)
except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid backpressure JSON: {exc}")
canonical = (
    json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    + "\n"
).encode("utf-8")
if data != canonical:
    raise SystemExit("backpressure status is not canonical JSON")
if (
    not isinstance(value, dict)
    or value.get("schema_version") != 1
    or value.get("policy_hash") != expected_policy_hash
    or type(value.get("allowExport")) is not bool
):
    raise SystemExit("backpressure status binding is invalid")
timestamp = value.get("updated_at_utc")
if not isinstance(timestamp, str):
    raise SystemExit("backpressure status has no update timestamp")
try:
    updated = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
except ValueError as exc:
    raise SystemExit(f"invalid backpressure timestamp: {exc}")
now = datetime.datetime.now(datetime.timezone.utc)
age = (now - updated).total_seconds()
if age < -5 or age > maximum_age:
    raise SystemExit(f"backpressure status is stale or future-dated: age={age:.3f}s")
print("ALLOW" if value["allowExport"] else "PAUSE")
PY
        )"
        then
            echo "Cannot verify promotion export backpressure; failing closed." >&2
            return 1
        fi
        if [ "$BACKPRESSURE_STATUS" = "PAUSE" ]
        then
            echo "Promotion export paused by controller backpressure."
            return 0
        fi
        if [ "$BACKPRESSURE_STATUS" != "ALLOW" ]
        then
            echo "Unknown promotion export backpressure status: $BACKPRESSURE_STATUS" >&2
            return 1
        fi
    fi
    HARDENED_EXPORTER="${KATAGO_HARDENED_EXPORTER:-$(dirname "$0")/../risk_score/hardened_exporter.py}"
    if [ ! -f "$HARDENED_EXPORTER" ]
    then
        echo "Hardened exporter not found: $HARDENED_EXPORTER" >&2
        echo "Set KATAGO_HARDENED_EXPORTER to its absolute path." >&2
        return 1
    fi
    if [ -z "${KATAGO_MODEL_PROBE_COMMAND_JSON:-}" ]
    then
        echo "KATAGO_MODEL_PROBE_COMMAND_JSON is required for gated publication." >&2
        echo "Set it to a JSON argv array that loads the model and checks finite output." >&2
        return 1
    fi

    # Keep the source checkpoint intact. Publication and duplicate handling
    # are performed transactionally by the hardened Python exporter.
    $PYTHON -W ignore "$(dirname "$0")/list_by_mtime.py" "$BASEDIR/$FROMDIR" | while read -r FILEPATH
    do
        if [ -z "$FILEPATH" ]
        then
            continue
        fi
        if [ "${FILEPATH: -4}" == ".tmp" ]
        then
            echo "Skipping tmp file:" "$FILEPATH"
        elif [ "${FILEPATH: -9}" == ".exported" ]
        then
            echo "Skipping legacy self tmp file:" "$FILEPATH"
        elif [ "${FILEPATH: -8}" == ".partial" ]
        then
            echo "Skipping partial file:" "$FILEPATH"
        else
            NAME="$(basename "$FILEPATH")"
            SRC="$BASEDIR/$FROMDIR/$NAME"
            echo "Found gated model to publish safely:" "$SRC"
            if ! "$PYTHON" "$HARDENED_EXPORTER" \
                --source-dir "$SRC" \
                --destination-root "$BASEDIR/modelstobetested" \
                --candidate-name "$NAME" \
                --model-name "$NAMEPREFIX-$NAME" \
                --python-executable "$PYTHON" \
                --model-probe-command-json "$KATAGO_MODEL_PROBE_COMMAND_JSON"
            then
                exit 1
            fi
            PUBLISHED="$BASEDIR/modelstobetested/$NAME"
            if [ ! -d "$PUBLISHED" ] || [ ! -f "$PUBLISHED/manifest.json" ]
            then
                echo "Hardened exporter returned without a complete publication: $PUBLISHED" >&2
                exit 1
            fi
            ARCHIVE="$BASEDIR/torchmodels_exported/$NAME"
            if [ -e "$ARCHIVE" ]
            then
                echo "Export archive already exists; refusing overwrite: $ARCHIVE" >&2
                exit 1
            fi
            # Publication is complete and verified. Move the intact source
            # directory out of the producer queue for retention-managed
            # archival; never delete the checkpoint here.
            if ! mv "$SRC" "$ARCHIVE"
            then
                echo "Failed to archive intact source checkpoint: $SRC" >&2
                exit 1
            fi
            echo "Archived intact source checkpoint at:" "$ARCHIVE"
        fi
    done
}

if [ "$USEGATING" -eq 0 ]
then
    exportStuff "torchmodels_toexport" "models"
else
    exportGatedStuffHardened "torchmodels_toexport"
    GATED_EXPORT_STATUS="$?"
    if [ "$GATED_EXPORT_STATUS" -ne 0 ]
    then
        exit "$GATED_EXPORT_STATUS"
    fi
fi
exportStuff "torchmodels_toexport_extra" "models_extra"

exit 0
}
