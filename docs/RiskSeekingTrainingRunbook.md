# Risk-Seeking KataGo Training Runbook

## Safety gate

**Do not start the expensive self-play fine-tune until the Phase 1 go/no-go
report in [RiskSeekingScoreUtility.md](RiskSeekingScoreUtility.md) has passed
and been approved.** Host readiness, successful compilation, or an available
GPU allocation does not satisfy that gate.

This runbook separates commands by execution status:

- **VERIFIED 2026-07-26** means the command or fact was checked during project
  bootstrap.
- **TEMPLATE — NOT YET HOST-VERIFIED** means the command is based on the
  current repository's [SelfplayTraining.md](../SelfplayTraining.md),
  [Compiling.md](../Compiling.md), and scripts, but must be reviewed, filled in,
  and exercised with a bounded smoke test before production use.

No setup, build, Phase 1, or Phase 2 launch command in this document should be
treated as already executed merely because it appears in a code block.

## Verified inventory

The approved host inventory is:

```text
SSH target:       set privately as $TRAIN_HOST
checkout path:    /home/ubuntu/kata-go
resolved path:    /lambda/nfs/kata-go
GPUs:             8 x NVIDIA H100, 80 GB each
logical CPUs:     208
RAM:              approximately 1.7 TiB
available disk:   approximately 22 TB at inventory time
CUDA:             12.8
PyTorch:          2.7
```

The local repository was also verified on branch `main` with:

```text
upstream  https://github.com/lightvector/KataGo.git
origin    git@github.com:PredictionExplorer/kata-go.git
```

`main` currently follows the official source line represented locally by
`upstream/master`. Capacity figures are observations, not reservations; recheck
free space and GPU occupancy before every run.

### Read-only inventory recheck

**TEMPLATE — NOT YET HOST-VERIFIED AS A SINGLE BLOCK**

```bash
ssh "$TRAIN_HOST"
readlink -f /home/ubuntu/kata-go
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader
nproc
free -h
df -h /lambda/nfs
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("GPU count", torch.cuda.device_count())
PY
```

Stop if the checkout resolves somewhere unexpected, fewer than eight GPUs are
available, the source revision differs from the run manifest, or free storage
is below the run's reserved budget.

The source checkout, writable artifact root, dependency installation, CUDA
build, and core test run were verified during bootstrap. Model-load and search
smokes remain pending until the score-utility implementation is built.

## Git and source workflow

The remote names are fixed:

- `upstream` is the official `lightvector/KataGo` repository;
- `origin` is the writable `PredictionExplorer/kata-go` fork; and
- development occurs on local branch `main`.

Never change these meanings, force-push, or rewrite official history. Do not
push from the training host. Source publication is performed from the reviewed
development checkout only and only when separately authorized.

### Verify source identity

**VERIFIED 2026-07-26 locally**

```bash
cd "${LOCAL_REPO:-/path/to/kata-go}"
git status --short --branch
git remote -v
git branch --show-current
git rev-parse HEAD
```

Before a run, the manifest must record both `git rev-parse HEAD` and
`git status --porcelain=v1`. A dirty checkout is acceptable only for a bounded
development experiment whose complete diff is copied into the external run
manifest. It is not acceptable for Phase 2.

### Synchronize without rewriting history

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
cd "${LOCAL_REPO:-/path/to/kata-go}"
git fetch upstream
git switch main
git merge --no-edit upstream/master
git status --short --branch
```

Before the project branch has custom commits, this may fast-forward. After it
has diverged, the ordinary merge preserves both histories and may require a
reviewed merge commit; do not rebase published commits merely to avoid that
commit.

When publication is explicitly authorized, use an ordinary push:

**TEMPLATE — DO NOT RUN WITHOUT PUBLICATION AUTHORIZATION**

```bash
git push -u origin main
```

There is no `--force`, `--force-with-lease`, rebase of shared history, or remote
replacement in this workflow.

### Training-host checkout

The host path already exists as `/home/ubuntu/kata-go`, resolving to
`/lambda/nfs/kata-go`. Do not reclone over it. After an authorized source push:

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
cd /home/ubuntu/kata-go
test "$(git remote get-url origin)" = \
  "git@github.com:PredictionExplorer/kata-go.git"
test "$(git remote get-url upstream)" = \
  "https://github.com/lightvector/KataGo.git"
git fetch origin
git switch main
git merge --ff-only origin/main
git status --short --branch
git rev-parse HEAD
```

Compare the resulting SHA byte-for-byte with the approved source SHA before
building.

## Artifact prohibition

Never commit or stage:

- inference models;
- raw or generated checkpoints;
- self-play or shuffled training data;
- logs;
- SGFs;
- generated CUDA, cuDNN, OpenCL, or TensorRT tuning plans and caches;
- run manifests containing private infrastructure details beyond what is
  approved for publication; or
- credentials, tokens, SSH keys, signed URLs, or cloud configuration.

All generated artifacts live outside the Git checkout. Do not rely on the
repository `.gitignore`; inspect `git status` before every source commit.

## External run-directory layout

Use an immutable run ID and a sibling NFS tree, never a directory under
`/home/ubuntu/kata-go`.

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
export REPO=/home/ubuntu/kata-go
export ARTIFACTS_ROOT=/home/ubuntu/kata-go-artifacts
export RUNS_ROOT="$ARTIFACTS_ROOT/runs/risk-score"
export RUN_ID=rs-YYYYMMDD-HHMM-<short-source-sha>
export RUN_DIR="$RUNS_ROOT/$RUN_ID"
export HOME_DATA="$RUN_DIR/cache/katago-home"
export PHASE1_DIR="$RUN_DIR/phase1"
export TRAIN_BASE="$RUN_DIR/phase2/training"
export SHUFFLE_SCRATCH="$RUN_DIR/phase2/shuffle-scratch"

test "$(readlink -f "$REPO")" = /lambda/nfs/kata-go
test -d "$ARTIFACTS_ROOT"
test ! -e "$RUN_DIR"
mkdir -p \
  "$RUN_DIR"/{manifest,artifacts/inference,artifacts/checkpoints,configs,schedules,recovery,pids} \
  "$HOME_DATA" \
  "$PHASE1_DIR"/{logs,sgfs,jsonl,summaries} \
  "$TRAIN_BASE"/{logs,models,modelstobetested,rejectedmodels,selfplay,shuffleddata,train} \
  "$SHUFFLE_SCRATCH"
chmod 0700 "$RUN_DIR" "$RUN_DIR/manifest"
```

Expected layout:

```text
RUN_DIR/
  manifest/                  source, hardware, provenance, hashes, decisions
  artifacts/
    inference/               downloaded .bin.gz files
    checkpoints/             downloaded trusted raw .ckpt files
  cache/katago-home/         generated backend tuning caches and plans
  configs/                   frozen copies actually used
  schedules/                 frozen positions, pairings, colors, and seeds
  phase1/
    logs/ sgfs/ jsonl/ summaries/
  phase2/
    shuffle-scratch/
    training/                KataGo's asynchronous BASEDIR
      models/                controller-accepted models only
      modelstobetested/      staged, unevaluated exports
      rejectedmodels/
      selfplay/
      shuffleddata/
      train/
      torchmodels_toexport/
      logs/
  recovery/                  quarantined artifacts; never silently deleted
  pids/                      process-group IDs for this run only
```

The scripts will create additional directories below `TRAIN_BASE`. Preserve
their names; `train.sh`, `shuffle.sh`, and the exporter depend on them.

## Run manifest

Create the manifest before downloading or launching anything. At minimum,
record:

- run ID and UTC creation time;
- source SHA, branch, remotes, clean/dirty status, and diff if dirty;
- build flags and compiler/CMake versions;
- host, GPU, driver, CUDA, Python, and PyTorch versions;
- exact config and deterministic schedule SHA-256s;
- model/checkpoint provenance, applicable license notice, and SHA-256s;
- Phase 1 gate report and approval;
- selected utility and search parameters;
- process launch commands and environment;
- checkpoint promotion/rejection decisions; and
- stop, restart, recovery, and rollback events.

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
cd "$REPO"
{
  date -u +"created_utc=%Y-%m-%dT%H:%M:%SZ"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'repo=%s\n' "$(readlink -f "$REPO")"
  printf 'source_sha=%s\n' "$(git rev-parse HEAD)"
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'origin=%s\n' "$(git remote get-url origin)"
  printf 'upstream=%s\n' "$(git remote get-url upstream)"
} > "$RUN_DIR/manifest/source.env"
git status --porcelain=v1 > "$RUN_DIR/manifest/git-status.txt"
git diff --no-color > "$RUN_DIR/manifest/source.diff"
git diff --staged --no-color > "$RUN_DIR/manifest/source-staged.diff"
cmake --version > "$RUN_DIR/manifest/cmake-version.txt"
g++ --version > "$RUN_DIR/manifest/compiler-version.txt"
nvidia-smi -q > "$RUN_DIR/manifest/nvidia-smi.txt"
python3 - <<'PY' > "$RUN_DIR/manifest/python-torch.txt"
import platform
import torch
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

After freezing configs and schedules:

```bash
sha256sum "$RUN_DIR"/configs/* "$RUN_DIR"/schedules/* \
  > "$RUN_DIR/manifest/config-schedule.sha256"
```

If a glob has no matches, stop and fix the manifest rather than accepting an
incomplete hash file.

## Model and checkpoint provenance

Use official artifacts from
<https://katagotraining.org/networks/kata1/>:

- primary b40:
  `kata1-zhizi-b40c768nbt-s11272M-d5935M`;
- control b28:
  `kata1-b28c512nbt-s13255194368-d5935380940`.

For each identifier, acquire both the C++ inference network and its matching
raw training checkpoint. Do not infer a checkpoint URL from the filename. Copy
the exact links exposed by the official network page, record the final resolved
URLs, and verify that model and checkpoint identifiers match. Save the license
or usage notice linked by the artifact source, its URL, and its acquisition
date; do not assume that a model has the same license as the source repository.

The following public artifacts were downloaded and hashed on 2026-07-26:

```text
b40 inference:
  URL: https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-zhizi-b40c768nbt-s11272M-d5935M.bin.gz
  bytes: 863210287
  SHA-256: 15fb3baf85cdb6578e6c19b65e6201ca906cec3ba5fee19039d05221d57eb0e8
b40 checkpoint archive:
  URL: https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-zhizi-b40c768nbt-s11272M-d5935M.zip
  bytes: 1733920920
  SHA-256: 93460fb0ca90b642b2ee939ef7a0277a8ea7ebbfad524f04b2ce0c8cc0d904d9
b40 extracted checkpoint:
  bytes: 1870061832
  SHA-256: 23e3a65fc8f5b505e89a8e5646a9eb17044c9a9ddf2c326bbbcda251625b7abc

b28 inference:
  URL: https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz
  bytes: 271440852
  SHA-256: c5bca453d7b08ea8df6546439325d4dd681e77d975e1da3f7593d771147b73bc
b28 checkpoint archive:
  URL: https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b28c512nbt-s13255194368-d5935380940.zip
  bytes: 545063328
  SHA-256: 224d626f8e065ad4afc192d2f44a9b0048ec8a508efccac1aebd986af08bd6e9
b28 extracted checkpoint:
  bytes: 585785026
  SHA-256: a53b1ad80cae56af25f13b20497af222620fef11b76a584fb9c40404e8a5c93d
```

The verified shared copies are under `/home/ubuntu/kata-go-artifacts`; each run
must copy or reflink them into its immutable artifact directory and recheck the
hashes.

### Atomic download and hashing

**VERIFIED URLS; TEMPLATE FOR A NEW IMMUTABLE RUN**

```bash
export B40_MODEL_URL='https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-zhizi-b40c768nbt-s11272M-d5935M.bin.gz'
export B40_CKPT_URL='https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-zhizi-b40c768nbt-s11272M-d5935M.zip'
export B28_MODEL_URL='https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz'
export B28_CKPT_URL='https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b28c512nbt-s13255194368-d5935380940.zip'

download_one() {
  url="$1"
  dest="$2"
  case "$url" in
    https://*) ;;
    *) printf 'Refusing non-HTTPS URL: %s\n' "$url" >&2; return 1 ;;
  esac
  test ! -e "$dest"
  curl --fail --location --proto '=https' --proto-redir '=https' \
    "$url" --output "$dest.partial"
  test -s "$dest.partial"
  mv "$dest.partial" "$dest"
}

download_one "$B40_MODEL_URL" \
  "$RUN_DIR/artifacts/inference/kata1-zhizi-b40c768nbt-s11272M-d5935M.bin.gz"
download_one "$B40_CKPT_URL" \
  "$RUN_DIR/artifacts/checkpoints/kata1-zhizi-b40c768nbt-s11272M-d5935M.zip"
download_one "$B28_MODEL_URL" \
  "$RUN_DIR/artifacts/inference/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz"
download_one "$B28_CKPT_URL" \
  "$RUN_DIR/artifacts/checkpoints/kata1-b28c512nbt-s13255194368-d5935380940.zip"

unzip -q "$RUN_DIR/artifacts/checkpoints/kata1-zhizi-b40c768nbt-s11272M-d5935M.zip" \
  -d "$RUN_DIR/artifacts/checkpoints"
unzip -q "$RUN_DIR/artifacts/checkpoints/kata1-b28c512nbt-s13255194368-d5935380940.zip" \
  -d "$RUN_DIR/artifacts/checkpoints"

sha256sum \
  "$RUN_DIR"/artifacts/inference/* \
  "$RUN_DIR"/artifacts/checkpoints/* \
  | tee "$RUN_DIR/manifest/artifacts.sha256"
stat --printf='%n %s bytes\n' \
  "$RUN_DIR"/artifacts/inference/* \
  "$RUN_DIR"/artifacts/checkpoints/* \
  > "$RUN_DIR/manifest/artifact-sizes.txt"
```

Do not place signed URLs, cookies, or authorization headers in the manifest or
shell history. The approved artifacts should be public.

### Checkpoint inspection

Only load checkpoints from the recorded trusted source. PyTorch checkpoints are
not safe to deserialize when obtained from an untrusted party.

Inspect the embedded model config, optimizer name/state, SWA state, and sample
counters before deciding how to resume:

**VERIFIED 2026-07-26**

```bash
export CHECKPOINT="$RUN_DIR/artifacts/checkpoints/kata1-zhizi-b40c768nbt-s11272M-d5935M.ckpt"
PYTHONPATH="$REPO/python" python3 - <<'PY' \
  > "$RUN_DIR/manifest/b40-checkpoint-summary.txt"
import os
from katago.train.load_model import load_checkpoint

path = os.environ["CHECKPOINT"]
state = load_checkpoint(path, map_location="cpu")
print("top_level_keys", sorted(state.keys()))
print("config", state.get("config"))
train_state = state.get("train_state", {})
print("train_state", train_state)
print("optimizer_name", train_state.get("optimizer_name"))
print("has_optimizer_state", "optimizer" in state)
print("has_swa_state", "swa_model" in state)
print("global_step_samples", train_state.get("global_step_samples"))
PY
```

Repeat for b28. If the checkpoint lacks optimizer state, or its optimizer name
does not match the selected training flags, treat the optimizer as fresh. Use a
conservative learning rate and a bounded overfit/validation smoke; do not claim
an optimizer resume occurred. KataGo's current `train.py` warns and discards an
incompatible optimizer state.

Trusted checkpoint inspection on 2026-07-26 found:

```text
b40: 40 blocks, 768 trunk channels, Muon, 11271985984 samples,
     SWA present, optimizer state absent
b28: 28 blocks, 512 trunk channels, SGD, 13255194368 samples,
     SWA present, optimizer state absent
```

Therefore Phase 2 cannot resume optimizer moments from either public
checkpoint. The b40 fine-tune should start a fresh Muon optimizer unless a
separate measured experiment justifies another optimizer; it must use a
conservative learning-rate smoke before persistent training.

### Round-trip export smoke

Current repository scripts export PyTorch checkpoints with
`export_model_pytorch.py`.

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
export EXPORT_SMOKE="$RUN_DIR/recovery/export-smoke-b40"
mkdir -p "$EXPORT_SMOKE"
cd "$REPO/python"
python3 ./export_model_pytorch.py \
  -checkpoint "$CHECKPOINT" \
  -export-dir "$EXPORT_SMOKE" \
  -model-name kata1-zhizi-b40c768nbt-s11272M-d5935M-roundtrip \
  -filename-prefix model \
  -use-swa
gzip -n "$EXPORT_SMOKE/model.bin"
sha256sum "$EXPORT_SMOKE/model.bin.gz" \
  > "$RUN_DIR/manifest/b40-roundtrip.sha256"
```

Load the resulting network with the compiled C++ engine and compare bounded
analysis output against the official inference network. Export success alone
does not prove numerical agreement. Repeat for b28 before Phase 1.

## Host setup and CUDA build

`Compiling.md` requires CMake 3.18.2 or newer, a C++14-capable compiler, CUDA
and compatible cuDNN, zlib, and libzip. For high-thread-count self-play it also
recommends TCMalloc to avoid glibc allocator fragmentation.

### Dependency check

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
cmake --version
g++ --version
nvcc --version
pkg-config --modversion libzip
ldconfig -p | awk '/libcudnn|libtcmalloc|libzip|libz\.so/'
python3 -c 'import numpy, torch; print(numpy.__version__, torch.__version__)'
```

If packages are missing, use the host's approved package-management procedure.
Do not blindly replace the verified CUDA or PyTorch installation. A typical
Ubuntu source-build dependency set includes `cmake`, `g++`, `zlib1g-dev`,
`libzip-dev`, and `libgoogle-perftools-dev`, but installation remains a
template requiring administrator review.

### Build

The current upstream CUDA instructions configure from `cpp` and use
`-DUSE_BACKEND=CUDA`. TCMalloc is enabled explicitly for this workload.

**VERIFIED 2026-07-26**

```bash
cmake -S "$REPO/cpp" -B "$REPO/cpp/build-cuda" -G Ninja \
  -DUSE_BACKEND=CUDA \
  -DUSE_TCMALLOC=1 \
  -DCUDNN_INCLUDE_DIR=/usr/lib/python3/dist-packages/torch/include \
  -DCUDNN_LIBRARY=/usr/lib/python3/dist-packages/torch/lib/libcudnn.so
cmake --build "$REPO/cpp/build-cuda" --parallel 24
LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:${LD_LIBRARY_PATH:-} \
  "$REPO/cpp/build-cuda/katago" version \
  | tee "$RUN_DIR/manifest/katago-version.txt"
```

The 24-way Ninja build was verified on this host; it is not a claim that all
208 logical CPUs should compile simultaneously. Record the complete CMake output and
`CMakeCache.txt` outside Git. Backend tuning data must use
`homeDataDir = $HOME_DATA` in frozen runtime configs so generated plans stay
outside the checkout.

### Verification before experiments

**CORE TEST VERIFIED 2026-07-26; OTHER SUITES PENDING IMPLEMENTATION BUILD**

```bash
export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}
"$REPO/cpp/build-cuda/katago" runtests
"$REPO/cpp/build-cuda/katago" runconfigtests
"$REPO/cpp/build-cuda/katago" runoutputtests
```

Then run bounded CUDA model-load/search smoke tests for both networks using the
frozen 19x19 config. Record commands, stdout, stderr, exit status, runtime, and
GPU assignment. The exact GPU test command must be selected from
`./katago -help` and the current test targets after the utility implementation
lands; do not copy a stale command from another branch.

## Phase 1 operations

Phase 1 is search-only. It uses the b40 primary and b28 control with no weight
updates.

### Freeze configs and schedules

Each Phase 1 config must enforce:

```text
bSizes = 19
bSizeRelProbs = 1
allowRectangleProb = 0
koRules = POSITIONAL
scoringRules = AREA
taxRules = NONE
multiStoneSuicideLegals = true
hasButtons = false
komiMean = 7.5
komiStdev = 0
komiBigStdevProb = 0
komiBiggerStdevProb = 0
handicapProb = 0
allowResignation = false
maxVisits = 800
numSearchThreads = 1
rootNoiseEnabled = false
chosenMoveTemperatureEarly = 0
chosenMoveTemperature = 0
nnRandomize = false
```

Use `homeDataDir = <RUN_DIR>/cache/katago-home`. Freeze one deterministic
manifest containing every start position, seed, Black/White assignment, and
color-reversed pair. Utility variants differ only in the opt-in flag,
`winWeight`, and later `scorePower`; all other settings remain byte-identical
until the dedicated search-scaling exercise.

The sweep order is:

1. `winWeight = 1, 2, 4` at `scorePower = 1.5`,
   `scoreScale = 20`;
2. search-scaling checks and selection of stable settings; then
3. `scorePower = 1.2, 1.35, 1.5` at the selected win weight.

### Bounded launch

The repository's current `match` interface accepts `-config`, `-log-file`, and
`-sgf-output-dir`. The deterministic schedule option is part of this project's
evaluation harness and must be confirmed from `./katago match -help` after that
harness lands.

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
cd "$REPO"
export CFG="$RUN_DIR/configs/phase1-b40-winweight2.cfg"
export OUT="$PHASE1_DIR/b40-winweight2"
mkdir -p "$OUT"/{logs,sgfs,jsonl}

"$REPO/cpp/katago" match \
  -config "$CFG" \
  -log-file "$OUT/logs/match.log" \
  -sgf-output-dir "$OUT/sgfs"
```

Do not start the full matrix until a small paired smoke demonstrates:

- identical schedule rows across variants;
- exact color reversal;
- deterministic rerun agreement;
- correct white/player perspective;
- expected 19x19 legal bounds `[-353.5,368.5]`;
- populated decomposed utility and tail diagnostics; and
- clean process termination with complete output.

Resume Phase 1 only from the last completely written schedule row. Never
silently regenerate seeds. A resumed output directory must record the original
schedule hash and the first row resumed.

### Summaries and gate

Run the project summarizer and move-comparison tooling only against immutable
JSONL/SGF sets. Preserve raw data and write a new summary directory per tool
version; do not edit results in place.

The report must include all metrics, catastrophic-loss labels, search-scaling
checks, exploitability tests, and go/no-go criteria defined in the utility
specification. The b40 result is primary and b28 is a required control.

Write the decision to:

```text
RUN_DIR/manifest/phase1-gate.md
```

It must contain `GO` or `NO-GO`, approver, UTC time, source/config/schedule
hashes, and unresolved risks. The absence of this file, an ambiguous decision,
or a changed source/config after approval means Phase 2 remains blocked.

## Phase 2 configuration

Phase 2 uses the existing asynchronous five-part KataGo pipeline: self-play,
shuffle, training, export, and checkpoint promotion. Unlike the stock pipeline,
promotion is not delegated solely to the conventional win-rate gatekeeper.

### Required game settings

Build the frozen self-play config from the current training template, but
override all broad-distribution features:

```text
dataBoardLen = 19
bSizes = 19
bSizeRelProbs = 1
allowRectangleProb = 0

koRules = POSITIONAL
scoringRules = AREA
taxRules = NONE
multiStoneSuicideLegals = true
hasButtons = false
komiMean = 7.5
komiStdev = 0
komiBigStdevProb = 0
komiBiggerStdevProb = 0
komiAllowIntegerProb = 0
handicapProb = 0

allowResignation = false
reduceVisits = false
cheapSearchProb = 0
policySurpriseDataWeight = 0
valueSurpriseDataWeight = 0

earlyForkGameProb = 0
forkGameProb = 0
sekiForkHackProb = 0
initGamesWithPolicy = false
policyInitAreaProp = 0
compensateAfterPolicyInitProb = 0
forkSidePositionProb = 0
fancyKomiVarying = false
handicapAsymmetricPlayoutProb = 0
normalAsymmetricPlayoutProb = 0
```

Also remove alternate `bSizesXY`, automatic or randomized komi, tax/rules
lists, `numExtraBlackFixed`, every `startPoses*`/`hintPoses*` source, and any
nonzero initialization mixture inherited from the template.
`reduceVisits = false` is mandatory. Already-decided games are central to this
objective and must retain full search visits and normal target weight.

The same frozen config must enable the approved Phase 1 winner:

```text
useScoreMaximizingUtility = true
scorePower = <approved value from 1.2, 1.35, or 1.5>
scoreScale = 20.0
winWeight = <approved value from 1, 2, or 4>
```

Replace both placeholders with numeric values before parsing the config and
record the Phase 1 gate that selected them.

The neural-network heads and losses remain unchanged.

### GPU mapping

The fixed allocation is:

```text
physical GPUs 0-6: one self-play NN server thread per GPU
physical GPU 7:   PyTorch training
CPUs and RAM:     game threads, shuffle, export, summaries
```

The frozen self-play config must contain:

```text
numNNServerThreadsPerModel = 7
cudaDeviceToUseModel0Thread0 = 0
cudaDeviceToUseModel0Thread1 = 1
cudaDeviceToUseModel0Thread2 = 2
cudaDeviceToUseModel0Thread3 = 3
cudaDeviceToUseModel0Thread4 = 4
cudaDeviceToUseModel0Thread5 = 5
cudaDeviceToUseModel0Thread6 = 6
```

Run training with `CUDA_VISIBLE_DEVICES=7`; inside that process physical GPU 7
appears as logical device 0. Measure `numGameThreads`, `nnMaxBatchSize`, and the
per-GPU training batch size with bounded H100 throughput/OOM tests. Do not copy
legacy b18 values blindly. The shuffler's batch size and training batch size
must match.

There is no spare evaluation GPU in this allocation. Evaluate checkpoints by
pausing training and using GPU 7, or by deliberately pausing one self-play GPU.
Do not silently oversubscribe a GPU and then compare timing-sensitive results.

## Phase 2 preflight

All items must be true:

- Phase 1 gate is `GO`;
- checkout is clean and at the approved SHA;
- CUDA build and required tests passed;
- official b40 and b28 inference models load;
- both raw checkpoints were inspected and round-trip exported;
- model, checkpoint, config, and schedule hashes are in the manifest;
- the policy-v3 suite transitively binds a machine-consensus
  `risk-score-reviewed-position-bank-v2` curation manifest;
- frozen self-play config is exactly 19x19 Tromp-Taylor 7.5;
- `reduceVisits=false`, no resignation, no handicap, no rectangles, and no
  exotic initialization are confirmed from parsed config output;
- the run directory and storage budget exist;
- process supervision and graceful stop were tested;
- a bounded self-play -> shuffle -> train -> export cycle completed; and
- checkpoint evaluation can run without relying solely on win rate.

If b40 training memory or throughput is impractical after measured batch/AMP
tests, record the evidence and make an explicit decision before using b28 as a
fallback.

## Seed the accepted-model directory

Self-play uses the newest model under `TRAIN_BASE/models`. An empty directory
causes random self-play, which is not an acceptable fine-tuning start. Seed it
before launching any process.

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
export SEED_NAME=kata1-zhizi-b40c768nbt-s11272M-d5935M
export SEED_MODEL="$RUN_DIR/artifacts/inference/$SEED_NAME.bin.gz"
export SEED_CKPT="$RUN_DIR/artifacts/checkpoints/$SEED_NAME.ckpt"
export SEED_DIR="$TRAIN_BASE/models/$SEED_NAME"

test -f "$SEED_MODEL"
test -f "$SEED_CKPT"
test ! -e "$SEED_DIR"
mkdir -p "$SEED_DIR"
cp --reflink=auto "$SEED_MODEL" "$SEED_DIR/model.bin.gz"
cp --reflink=auto "$SEED_CKPT" "$SEED_DIR/model.ckpt"
sha256sum "$SEED_DIR/model.bin.gz" "$SEED_DIR/model.ckpt" \
  > "$RUN_DIR/manifest/seeded-model.sha256"
mkdir -p "$TRAIN_BASE/selfplay/$SEED_NAME"/{sgfs,tdata,vadata}
```

Verify the copied hashes equal the acquisition hashes. Do not modify model
timestamps after self-play begins except through a documented promotion.

## Bounded end-to-end smoke

Before persistent launch, run a small finite self-play using the production
config and `-max-games-total`, then one bounded shuffle/train/export cycle.
The synchronous example in `python/selfplay/synchronous_loop.sh` demonstrates
the five stages, but its defaults are explicitly described upstream as small
and not heavily tested; do not use them as production tuning.

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
cd "$REPO"
"$REPO/cpp/katago" selfplay \
  -max-games-total 8 \
  -output-dir "$TRAIN_BASE/selfplay" \
  -models-dir "$TRAIN_BASE/models" \
  -config "$RUN_DIR/configs/selfplay-tromp-taylor-19.cfg"
```

Confirm all eight games used the intended model, rules, komi, full visits, and
GPUs 0-6. Then run a deliberately bounded shuffle and training invocation using
`-max-epochs-this-instance` or `-stop-when-train-bucket-limited`. Do not proceed
if any generated row has another board size/ruleset or if decided positions are
downweighted.

## Persistent asynchronous launch

The source documentation recommends 4x-40x more GPU power for self-play than
training. Seven H100s to one H100 is within that broad range, but actual balance
must be measured from data-production rate and train-bucket waiting.

Use a supervisor that records one process group per component. The templates
below disable interactive job control and use `setsid`, then verify that the
recorded PID is also the new process-group ID. Review this mechanism on the host
first. All paths point outside Git.

### 1. Self-play on GPUs 0-6

**TEMPLATE — BLOCKED UNTIL PHASE 1 GO**

```bash
cd "$REPO"
set +m
setsid "$REPO/cpp/katago" selfplay \
  -output-dir "$TRAIN_BASE/selfplay" \
  -models-dir "$TRAIN_BASE/models" \
  -config "$RUN_DIR/configs/selfplay-tromp-taylor-19.cfg" \
  >> "$TRAIN_BASE/logs/selfplay.console.log" 2>&1 &
selfplay_pid="$!"
sleep 1
test "$(ps -o pgid= -p "$selfplay_pid" | tr -d ' ')" = "$selfplay_pid"
echo "$selfplay_pid" > "$RUN_DIR/pids/selfplay.pgid"
```

The config, not `CUDA_VISIBLE_DEVICES`, maps the seven NN server threads to
physical GPUs 0-6.

### 2. Shuffler

This foreground loop is the same core operation used by
`shuffle_and_export_loop.sh`, without its `disown`, so its process group can be
stopped reliably.

**TEMPLATE — BLOCKED UNTIL PHASE 1 GO**

```bash
export SHUFFLE_THREADS='SET_ME'
export BATCH_SIZE='SET_ME'
export KATAGO_SHUFFLE_FORCE_AFTER_SECONDS=3600
export KATAGO_DATA_WATERMARK="$TRAIN_BASE/promotion/watermarks/data.json"
export KATAGO_STRICT_SHUFFLE_PROVENANCE=1
test -f "$KATAGO_DATA_WATERMARK"
case "$SHUFFLE_THREADS" in
  ''|*[!0-9]*) echo "Set integer SHUFFLE_THREADS" >&2; exit 1 ;;
esac
case "$BATCH_SIZE" in
  ''|*[!0-9]*) echo "Set integer BATCH_SIZE" >&2; exit 1 ;;
esac

set +m
setsid bash -c '
  cd "$1/python"
  while true; do
    ./selfplay/shuffle.sh "$2" "$3" "$4"
    sleep 20
  done
' _ "$REPO" "$TRAIN_BASE" "$SHUFFLE_SCRATCH" "$SHUFFLE_THREADS" \
  >> "$TRAIN_BASE/logs/outshuffle.txt" 2>&1 &
shuffle_pid="$!"
sleep 1
test "$(ps -o pgid= -p "$shuffle_pid" | tr -d ' ')" = "$shuffle_pid"
echo "$shuffle_pid" > "$RUN_DIR/pids/shuffle.pgid"
```

Fill the placeholders with integers before execution. Choose window and
retention parameters only after reading `python/shuffle.py -help`; retain a
validation split for checkpoint evaluation unless an approved experiment says
otherwise. `shuffle.sh` fingerprints complete self-play inputs and skips an
unchanged shuffle. The force interval is a safety valve that provides a fresh
random sample when a trainer still has bucket credit but no new self-play file
has completed; tune it from measured trainer consumption time rather than
returning to an unconditional tight loop. Automatic promotion additionally
requires the feedback watcher to initialize the data watermark first. Strict
mode publishes `generation-provenance.json` beside every new shuffle and fails
closed on unbound or changed self-play input.

### 3. Trainer on GPU 7

`train.sh` stores the persistent checkpoint at
`TRAIN_BASE/train/$TRAINING_NAME/checkpoint.ckpt`. `-initial-checkpoint` is used
only when that persistent checkpoint does not yet exist.

**TEMPLATE — BLOCKED UNTIL PHASE 1 GO**

```bash
export TRAINING_NAME=riskb40
export MODEL_KIND=b40c768nbt
export INITIAL_CHECKPOINT="$RUN_DIR/artifacts/checkpoints/kata1-zhizi-b40c768nbt-s11272M-d5935M.ckpt"
export LR_SCALE='SET_ME'
export FIXED_VAL_DIR='SET_ME'
export FIXED_VAL_MANIFEST='SET_ME'
export MAX_BUCKET_PER_DATA=4
export MAX_BUCKET_SIZE=5000000
test "$LR_SCALE" != SET_ME
test "$FIXED_VAL_DIR" != SET_ME
test "$FIXED_VAL_MANIFEST" != SET_ME
case "$BATCH_SIZE" in *[!0-9]*|'') echo "Set BATCH_SIZE" >&2; exit 1 ;; esac

set +m
setsid bash -c '
  cd "$1/python"
  export CUDA_VISIBLE_DEVICES=7
  while true; do
    if ! ./selfplay/train.sh \
      "$2" "$3" "$4" "$5" main \
      -initial-checkpoint "$6" \
      -lr-scale "$7" \
      -fixed-val-datadir "$8" \
      -fixed-val-manifest "$9" \
      -max-train-bucket-per-new-data "${10}" \
      -max-train-bucket-size "${11}" \
      -samples-per-epoch 100000 \
      -swa-period-samples 1000000 \
      -use-muon -use-bf16 \
      -epochs-per-export 5 \
      -export-min-sample-interval 500000 \
      -max-val-samples 12288 \
      -generation-provenance-dir "$2/promotion/provenance/trainer" \
      -require-shuffle-provenance \
      -no-repeat-files -stop-when-train-bucket-limited
    then
      exit 1
    fi
    sleep 30
  done
' _ \
  "$REPO" "$TRAIN_BASE" "$TRAINING_NAME" "$MODEL_KIND" "$BATCH_SIZE" \
  "$INITIAL_CHECKPOINT" "$LR_SCALE" "$FIXED_VAL_DIR" "$FIXED_VAL_MANIFEST" \
  "$MAX_BUCKET_PER_DATA" "$MAX_BUCKET_SIZE" \
  >> "$TRAIN_BASE/logs/train-launch.log" 2>&1 &
train_pid="$!"
sleep 1
test "$(ps -o pgid= -p "$train_pid" | tr -d ' ')" = "$train_pid"
echo "$train_pid" > "$RUN_DIR/pids/train.pgid"
```

The flags above reflect the measured b40 Muon/BF16 production path; any batch
size change still requires a bounded smoke from a protected checkpoint copy.
Keep the fixed validation directory immutable and outside shuffler/trainer
inputs. Create its immutable inventory with
`python3 -m katago.train.training_controls freeze-validation --directory
"$FIXED_VAL_DIR" --output "$FIXED_VAL_MANIFEST"`. The trainer revalidates it
every epoch. The restart loop handles normal bucket-limited exits; once the
promotion host supervisor is installed, use that single-owner supervisor
instead of running both. If the source checkpoint's optimizer is not
resumable, start a fresh optimizer explicitly and document the conservative
schedule. The provenance options bind every consumed shuffle, checkpoint, and
export to admitted generation hashes. They are mandatory in automatic mode;
the runtime builder rejects an automatic trainer spec that omits them.

### 4. Exporter, staging only

Use `USEGATING=1` so exports go to `modelstobetested` rather than directly into
the live `models` directory. Do not start the stock win-rate-only gatekeeper.

**TEMPLATE — BLOCKED UNTIL PHASE 1 GO**

```bash
export NAME_PREFIX="$RUN_ID"
export KATAGO_MODEL_PROBE_COMMAND_JSON='SET_ME'
test "$KATAGO_MODEL_PROBE_COMMAND_JSON" != SET_ME
export KATAGO_PROMOTION_BACKPRESSURE_FILE="$TRAIN_BASE/promotion/operations/backpressure.json"
export KATAGO_PROMOTION_POLICY_HASH="0151ddcdee764b1e599eb5313f9dfae944e671ff8098dd471425f8d646ba3318"
export KATAGO_PROMOTION_BACKPRESSURE_MAX_AGE_SECONDS=120

python3 -m risk_score.promotion_preflight bootstrap-backpressure \
  --output "$KATAGO_PROMOTION_BACKPRESSURE_FILE" \
  --policy-hash "$KATAGO_PROMOTION_POLICY_HASH"

set +m
setsid bash -c '
  cd "$1/python"
  while true; do
    ./selfplay/export_model_for_selfplay.sh "$2" "$3" 1
    sleep 10
  done
' _ "$REPO" "$NAME_PREFIX" "$TRAIN_BASE" \
  >> "$TRAIN_BASE/logs/outexport.txt" 2>&1 &
export_pid="$!"
sleep 1
test "$(ps -o pgid= -p "$export_pid" | tr -d ' ')" = "$export_pid"
echo "$export_pid" > "$RUN_DIR/pids/export.pgid"
```

Set `KATAGO_MODEL_PROBE_COMMAND_JSON` to a reviewed JSON argv array that loads
`{model_file}` with the production CUDA binary and checks finite output. The
hardened gated exporter publishes through a unique `.partial` directory,
places completed candidates in `modelstobetested`, and archives the intact
source under `torchmodels_exported`. The controller backpressure file pauses
new gated exports when the evaluation queue is full. A stale denial remains a
safe pause; a stale allowance, missing file, malformed status, or policy
mismatch fails closed. Recommendation-only shadow mode deliberately leaves the
bootstrap denial in place; the first mutation-enabled reconciler tick replaces
it with the controller-owned live status after all activation gates pass.

### 5. Evaluation and controller promotion

#### Self-play-only margin mode

For runs whose primary objective is margin-seeking behavior rather than
preserving the original model's full playing strength, use the frozen
`selfplay_margin_policy_v1.json` policy and
`margin_safety_gatekeeper_19x19.cfg`.

This mode uses no external SGFs or curated position bank. Training remains
powered-utility self-play, while promotion is deliberately permissive:

- each candidate plays 100 fresh 19x19, 7.5-komi games against the current
  accepted model;
- the safety match uses conventional search at 400 visits;
- candidates scoring at least 35% are accepted automatically;
- model-load and finite-output probing remains mandatory before publication;
- the trainer, exporter/model probe, and gatekeeper share GPU 7 sequentially;
- prior accepted models and trainer checkpoints remain available for rollback.

The 35% floor is a guard against catastrophic playing-strength collapse, not a
requirement that margin-seeking candidates remain stronger than their parent.
The seven self-play GPUs continue generating training data while GPU 7
alternates between training and candidate gating.

#### Strict promotion mode

Build the source bank through the active `queries-consensus`,
`label-consensus`, optional `merge-labeling-consensus`, and
`finalize-consensus` flow in `RiskSeekingCheckpointPromotionRunbook.md`.
It freezes distinct original/champion hashes and standard/powered 2,000/8,000
visit results over all distinct shape-preserving symmetries. Ambiguous rows
remain permanently in `rejected.jsonl`; there is no decisions file. The legacy
human-review curation-v1 path is historical and cannot satisfy policy v3.

The final source must contain at least 3,200 ordinary, 2,080 Lead-40, and 4,128
Lead-80 positions. `build_evaluation_suites` requires `--curation-manifest` for
each source and emits the v3 suite manifest. Curation can require up to 64
analyses per non-symmetric square position, so execute the eight roles in
manifested shards.

For every important candidate:

1. yield GPU 7 through the exclusive trainer/evaluator lease;
2. hash the candidate directory;
3. run the frozen v3 look-specific ordinary and Lead suites plus fixed
   integrity probes;
4. compare against the original b40 and the current accepted champion;
5. execute the manifest-bound matrix with `risk_score.promotion_evaluator`,
   which finalizes paired statistics through `risk_score.promotion_evidence`;
   and
6. let the versioned gate record PASS, FAIL, or a prespecified continuation.

Required report fields include realized custom utility, score mean/median and
tails, win rate, every catastrophic-loss definition, utility decomposition,
endpoint-tail contribution, tactical stability, and exploitability results.
Win rate alone is insufficient. Start with mutation disabled:

```bash
cd "$REPO/python"
python3 -m risk_score.promotion_controller \
  --runtime-config "$RUN_DIR/configs/promotion-runtime.json" \
  --mode reconcile \
  --recommend-only
python3 -m risk_score.promotion_controller \
  --runtime-config "$RUN_DIR/configs/promotion-runtime.json" \
  --mode once \
  --recommend-only
```

Stage 0–2 screening is allowed before machine-review readiness. Direct and
automatic promotion are not: the controller must first validate transitive v3
provenance from the suite manifest to every machine-reviewed source, including
policy, source, rejected-artifact, original-model, and champion-model hashes.

Do not promote into the monolithic mtime-selected self-play process. Before
enabling mutation, migrate to seven generation-pinned workers, complete the
canary and rollback drills, and satisfy every repository-closure and live
validation item in `RiskSeekingCheckpointPromotionPlan.md`.

The controller performs the accepted-model move, canary quarantine, worker
acknowledgements, champion compare-and-swap, admission, and rollback
transactions. Policy v3 requires a 4,000-game canary plus 2,048 fresh audit
pairs before admission. Never manually copy a model into the live model
directory or touch an old file to influence mtime.

## Monitoring

Monitor health, not only process existence.

### Processes and GPUs

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
for f in "$RUN_DIR"/pids/*.pgid; do
  printf '%s: ' "$f"
  ps -o pid,pgid,stat,etime,%cpu,%mem,cmd -g "$(cat "$f")"
done
nvidia-smi
nvidia-smi dmon -s pucvmet
```

Expected GPU ownership is self-play on 0-6 and training on 7. Investigate
unexpected memory consumers, repeated CUDA initialization, or one idle
self-play GPU.

### Logs and data flow

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
tail -F \
  "$TRAIN_BASE/logs/selfplay.console.log" \
  "$TRAIN_BASE/logs/outshuffle.txt" \
  "$TRAIN_BASE/logs/outexport.txt" \
  "$TRAIN_BASE/logs/train-launch.log"
```

Track:

- games and rows generated per hour;
- NN batch size and utilization per self-play GPU;
- self-play errors and model switches;
- newest self-play, shuffle, checkpoint, and export timestamps;
- train-bucket occupancy and trainer wait time;
- training and validation losses by head;
- fixed-validation sample/batch count, wall time, and global sample anchor;
- gradient norm, non-finite values, and AMP scaler behavior;
- checkpoint/export cadence;
- shuffle gate `SHUFFLED` versus `SKIPPED_UNCHANGED` status;
- queued and evaluated candidates; and
- Phase 2 evaluation metrics over time.

If training waits for data, increase self-play throughput or reduce training
rate. If the train bucket grows without bound, training is not keeping up.
Change one measured control at a time and snapshot the new config.

### Disk and memory

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
df -h /lambda/nfs
du -sh "$RUN_DIR" "$TRAIN_BASE"/{selfplay,shuffleddata,train,models,modelstobetested}
free -h
```

Define warning and stop thresholds before launch. At the stop threshold, halt
producers before cleaning anything. Never delete the only copy of a checkpoint,
manifest, promotion report, or source model to recover space.

## Graceful stop

KataGo's self-play documentation states that the C++ self-play process handles
`SIGINT` gracefully and finishes pending data, which can take a minute or two.
Python and shell pipeline stages are designed to leave files in recoverable
states when interrupted, but a clean checkpoint/export boundary is preferred.

Stop the exporter first, then trainer, shuffler, and finally self-play. For each
recorded process group:

**TEMPLATE — NOT YET HOST-VERIFIED**

```bash
stop_group() {
  name="$1"
  file="$RUN_DIR/pids/$name.pgid"
  test -f "$file"
  pgid="$(cat "$file")"
  case "$pgid" in
    ''|*[!0-9]*) echo "Invalid PGID in $file" >&2; return 1 ;;
  esac
  test "$pgid" -gt 1
  ps -o pid,pgid,stat,etime,cmd -g "$pgid"
  kill -INT -- "-$pgid"
}

stop_group export
stop_group train
stop_group shuffle
stop_group selfplay
```

Wait for groups to exit and verify:

```bash
for f in "$RUN_DIR"/pids/*.pgid; do
  pgid="$(cat "$f")"
  while kill -0 -- "-$pgid" 2>/dev/null; do sleep 5; done
done
pgrep -af -- "$TRAIN_BASE" || true
date -u +"stopped_utc=%Y-%m-%dT%H:%M:%SZ" \
  >> "$RUN_DIR/manifest/events.log"
```

Do not send `SIGKILL` merely because self-play is flushing. If a process is
genuinely stuck, capture stack/log/GPU evidence first, record the escalation,
then kill only its verified process group.

## Resume

Resume with the same:

- `RUN_DIR` and `TRAIN_BASE`;
- source SHA and frozen configs;
- accepted model directory;
- training name, model kind, batch size, optimizer flags, and schedule; and
- external cache and scratch paths.

Before resuming:

1. confirm no old process group remains;
2. run artifact/config hash checks;
3. inspect disk and GPU availability;
4. verify the newest accepted model and candidate queue;
5. verify `TRAIN_BASE/train/$TRAINING_NAME/checkpoint.ckpt` is loadable; and
6. append a restart event to the manifest.

Rerun the same four persistent launch blocks. `train.py` automatically loads
its persistent checkpoint when present. Keep `-initial-checkpoint` in the
command for first-start compatibility, but **never add
`-always-initial-checkpoint` on resume**; that would overwrite the intended
resume path. `-no-repeat-files` preserves data-use tracking across restarts.

Self-play and shuffle continue from existing directories. The exporter skips
already exported model names. Inspect incomplete `.tmp` and `.exported`
directories rather than deleting them during startup.

## Recovery

### Training checkpoint fails to load

1. stop training and exporter;
2. copy the failed file and its SHA-256 to `RUN_DIR/recovery`;
3. test `checkpoint_prev0.ckpt`, then older rotated checkpoints, newest first;
4. copy the selected good checkpoint to a temporary filename in the training
   directory;
5. atomically rename it to `checkpoint.ckpt`;
6. record lost sample range and optimizer/SWA state; and
7. run a bounded training/export smoke before persistent resume.

Do not delete or modify the failed original until the incident report is
complete. Long-term checkpoints may omit optimizer state; if used, explicitly
record that optimizer continuity was lost.

### Exporter interruption

Stop the exporter, identify the last complete candidate, and inspect
`torchmodels_toexport`, `.tmp`, and `.exported` directories. Compare names and
hashes with `models`, `modelstobetested`, and `rejectedmodels`. Move ambiguous
artifacts to `recovery`; never make a partial export visible under `models`.
Restart the exporter and verify one candidate end to end.

### CUDA OOM or backend failure

- self-play OOM: reduce `nnMaxBatchSize` or measured game-thread pressure while
  retaining seven server threads and fixed search visits;
- training OOM: reduce per-GPU `BATCH_SIZE`, then regenerate shuffles with the
  same new batch size before resuming training;
- backend-plan corruption: stop all users, quarantine the specific external
  cache entry under `HOME_DATA`, and allow a bounded smoke to regenerate it;
- non-finite training: stop immediately, preserve checkpoint/data/config, and
  diagnose before resuming from a known-good checkpoint.

Every change creates a new frozen config snapshot and manifest event.

### Disk exhaustion

Stop self-play first to stop growth, then shuffler/training/exporter. Preserve
checkpoints, source artifacts, manifests, and evaluated model directories.
Reclaim only data covered by the predeclared retention policy, after listing and
hashing what will be removed. Resume with a bounded smoke.

### Bad promoted checkpoint

1. stop self-play and training;
2. move the bad candidate from `models` to a quarantine directory on the same
   filesystem;
3. verify the previous champion remains and is now the newest eligible model;
4. identify all self-play data generated with the bad model;
5. quarantine those data rather than silently mixing them into training;
6. restore a pre-promotion training checkpoint if the bad data were consumed;
7. rerun the promotion and exploitability reports; and
8. restart only after a written rollback decision.

Touching an old model to make it newest is not an acceptable undocumented
rollback.

## Completion and archival

At run completion:

- stop all process groups gracefully;
- verify no process still references `TRAIN_BASE`;
- hash final source/config/schedule/model/checkpoint/promotion artifacts;
- record final data and training sample counts;
- preserve Phase 1 and checkpoint-evaluation reports;
- write a final incident and deviation summary;
- mark whether the run is resumable or closed; and
- keep the entire run tree outside Git.

Only small, reviewed source code, tests, configs, scripts, and public
documentation may later be committed. Models, checkpoints, data, logs, SGFs,
backend plans, and credentials remain external permanently.
