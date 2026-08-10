# Risk-Seeking Checkpoint Promotion Runbook

## Scope and safety posture

This runbook operates the score-maximizing checkpoint promotion controller.
It supplements `RiskSeekingTrainingRunbook.md`; it does not replace the
training, shuffle, or host-provisioning procedures there.

The controller starts in recommendation-only mode. Enabling filesystem or
process mutation requires both:

1. `"mutationEnabled": true` in a frozen runtime configuration; and
2. an explicit automatic/mutation CLI flag.

Never enable mutation until the shadow gate, filesystem tests, GPU handoff,
canary, and rollback drills in this document have passed on the actual run
volume.

## Required artifacts

Freeze and hash these together:

- the source revision and local diff;
- `python/risk_score/promotion_policy_v3.json`;
- the unchanged v1 and v2 policies when retaining historical evidence;
- `cpp/configs/risk_score/promotion_powered_match.cfg`;
- `cpp/configs/risk_score/promotion_standard_match.cfg`;
- `cpp/configs/risk_score/promotion_curation_analysis.cfg`;
- `cpp/configs/risk_score/promotion_curation_lead_selfplay_19x19.cfg`;
- `cpp/configs/risk_score/promotion_selfplay_worker_19x19.cfg`;
- discovery, confirmation, and audit suite manifests and schedules;
- the immutable original model and its byte SHA-256;
- the frozen champion model used for curation and its byte SHA-256;
- the current trainer checkpoint;
- the frozen runtime configuration;
- `promotion/supervisor/trainer.json`, the dynamic consumer identity snapshot,
  and the live supervisor heartbeat;
- and `configs/deployment-manifest.json`, `trainer-launch.json`,
  `consumer-stop.json`, and `promotion-services.json`.

Keep the live runtime configuration outside Git. Start from
`python/risk_score/promotion_runtime.example.json` for controller paths and
rollout commands, and `python/risk_score/gpu_lease_runtime.example.json` for
trainer/evaluator handoff. Use absolute paths and argv arrays, not shell command
strings.

The policy identity is its canonical JSON hash, while models, configs, binaries,
and schedules use byte SHA-256. Print the policy identity with:

```bash
cd "$REPO/python"
python3 -c 'from risk_score.paired_stats import load_policy, canonical_sha256; print(canonical_sha256(load_policy()))'
```

The checked-in v3 policy prints
`0151ddcdee764b1e599eb5313f9dfae944e671ff8098dd471425f8d646ba3318`.
Any other value is a different policy and requires a new suite freeze. Policies
v1 and v2 are replay-only historical evidence; all new suite builds and
promotion plans use policy v3.

## Installation and preflight

From the repository root:

```bash
python3 -m json.tool python/risk_score/promotion_policy_v1.json >/dev/null
python3 -m json.tool python/risk_score/promotion_policy_v2.json >/dev/null
python3 -m json.tool python/risk_score/promotion_policy_v3.json >/dev/null
python3 -m json.tool python/risk_score/promotion_runtime.example.json >/dev/null
python3 -m json.tool python/risk_score/gpu_lease_runtime.example.json >/dev/null
cd python
python3 -m risk_score.promotion_controller --help
python3 -m risk_score.evaluation_runner --help
python3 -m risk_score.promotion_evaluator --help
python3 -m risk_score.promotion_evidence --help
python3 -m risk_score.curate_position_bank --help
python3 -m risk_score.build_evaluation_suites --help
python3 -m risk_score.model_probe --help
python3 -m risk_score.stage0_probe --help
python3 -m risk_score.promotion_host --help
python3 -m risk_score.promotion_preflight --help
python3 -m risk_score.build_live_runtime --help
```

Run the Python tests before deploying the source snapshot:

```bash
cd "$REPO/python"
pytest
```

On a host with the production KataGo binary, run bounded model-load and match
smokes separately. Unit tests do not establish CUDA compatibility.

Before touching live processes, deploy a clean versioned checkout rather than
updating the checkout used by existing loops. Capture an immutable snapshot
and prove run-volume rename/fsync semantics:

```bash
python3 -m risk_score.promotion_preflight snapshot \
  --run-root "$TRAIN_BASE" --repo "$DEPLOY_REPO" \
  --output "$RUN_DIR/manifest/promotion-deployment-snapshot.json"
python3 -m risk_score.promotion_preflight filesystem-test \
  --root "$TRAIN_BASE/promotion" \
  --output "$RUN_DIR/manifest/promotion-filesystem-test.json"
python3 -m risk_score.promotion_preflight candidate-inventory \
  --inbox "$TRAIN_BASE/modelstobetested" \
  --output "$RUN_DIR/manifest/promotion-candidate-inventory.json"
```

Materialize runtime JSON only after suites, trainer/consumer specs, the CUDA
binary, and the deployment Python environment are frozen. The builder rejects
a dirty or mismatched source revision and writes a deployment manifest that
automatic mode rechecks on every poll. Set `SHUFFLER_ARGV_JSON` and
`EXPORTER_ARGV_JSON` to reviewed JSON argv arrays for foreground loops; do not
embed shell command strings:

```bash
python3 -m risk_score.build_live_runtime \
  --repo "$DEPLOY_REPO" --run-root "$TRAIN_BASE" \
  --suite-dir "$RUN_DIR/evaluation/promotion-suites-v3" \
  --katago-binary "$DEPLOY_REPO/cpp/build-cuda/katago" \
  --python-executable "$DEPLOY_REPO/python/.venv/bin/python" \
  --trainer-spec "$RUN_DIR/configs/trainer-launch.json" \
  --consumer-spec "$RUN_DIR/configs/consumer-stop.json" \
  --original-model "$ORIGINAL_MODEL" \
  --trainer-checkpoint "$TRAIN_BASE/train/$TRAINING_NAME/checkpoint.ckpt" \
  --gpu-uuid "$GPU7_UUID" --actor "$CONTROLLER_ID" \
  --source-revision "$(git -C "$DEPLOY_REPO" rev-parse HEAD)" \
  --output-dir "$RUN_DIR/configs" \
  --service-user ubuntu \
  --shuffler-command-json "$SHUFFLER_ARGV_JSON" \
  --exporter-command-json "$EXPORTER_ARGV_JSON"
```

The host supervisor is safe to start with mutation disabled: it publishes
identity snapshots and a heartbeat but launches no process. After validated
runtime regeneration with mutation enabled, it adopts or starts exactly one
trainer, respects every GPU-lease phase, supervises rollout acknowledgements,
and keeps seven continuous workers synchronized to `champion.json`.

The generated `promotion-services.json` and `systemd/` units bind the host
supervisor, controller, rollout/deep-audit producer, feedback watcher, shuffler,
and exporter into the deployment manifest. Link and enable the generated
system-level units only in the activation phase; the host has no user-service
linger. A mutation-enabled build additionally requires the trainer spec to
include `-generation-provenance-dir
$TRAIN_BASE/promotion/provenance/trainer` and
`-require-shuffle-provenance`.

Plan and verify the hash-bound unit installation before applying it. The apply
step atomically installs only the exact generated KataGo units, reloads systemd,
starts the aggregate target, and refuses to publish a receipt unless every
configured service is active:

```bash
python3 -m risk_score.service_activation \
  --spec "$RUN_DIR/configs/promotion-services.json"

sudo -n "$DEPLOY_REPO/python/.venv/bin/python" \
  -m risk_score.service_activation \
  --spec "$RUN_DIR/configs/promotion-services.json" \
  --receipt "$RUN_DIR/manifest/systemd-activation.json" \
  --apply
```

Do not use `systemctl enable` as an unverified substitute. A target that is
active while a required child service is absent does not establish supervision.
Stop and reconcile if the activation receipt does not bind the exact generated
unit hashes.

Before strict shuffling or the automatic controller starts, initialize
historical baselines and rollback watermarks once with the mutation-disabled
runtime:

```bash
python3 -m risk_score.promotion_feedback \
  --runtime-config "$RUN_DIR/configs/promotion-runtime.json" \
  --run-root "$TRAIN_BASE" --mode once
```

## Promotion filesystem

Place controller state at `$TRAIN_BASE/promotion` on the same filesystem as
candidate and accepted-model storage:

```text
promotion/
  controller.lock
  champion.json
  events/
  candidates/
    discovered/
    claimed/
    superseded/
    rejected/
    quarantined/
  evaluations/
    partial/
    final/
  reports/
  rollouts/
  rollback/
  trash/
```

Event JSON files and finalized reports are authoritative. Any SQLite or
monitoring index must be rebuildable from them.

Confirm same-filesystem rename behavior and durable directory fsync on the live
volume before claiming a real candidate. Do not substitute copy-and-delete for
an atomic rename.

## Curate the source position bank

The active curation contract is machine-consensus v2. It emits only
machine-reviewed `ordinary`, `lead-40`, and `lead-80` positions for policy v3;
discovery/confirmation/audit assignment happens later in the suite builder.

Do not harvest live training self-play. Generate quarantined source SGFs outside
every shuffler/trainer input root, then publish and validate the immutable
harvest plan:

```bash
cd "$REPO/python"
python3 -m risk_score.curate_position_bank harvest-plan \
  --katago "$REPO/cpp/build-cuda/katago" \
  --sgfs-dir "$RUN_DIR/evaluation/curation/quarantined-sgfs" \
  --training-input-root "$TRAIN_BASE/selfplay" \
  --output-dir "$RUN_DIR/evaluation/curation/harvested" \
  --manifest "$RUN_DIR/evaluation/curation/harvest.json" \
  --threads 1
python3 -m risk_score.curate_position_bank harvest-execute \
  "$RUN_DIR/evaluation/curation/harvest.json"
```

The one parser thread makes harvest order reproducible. Normalize and
semantically deduplicate the harvested positions, then freeze distinct original
and champion model files and hashes in the consensus query manifest:

```bash
python3 -m risk_score.curate_position_bank normalize \
  "$RUN_DIR"/evaluation/curation/harvested/*.startposes.txt \
  --output "$RUN_DIR/evaluation/curation/normalized.jsonl" \
  --manifest "$RUN_DIR/evaluation/curation/normalized-manifest.json"

python3 -m risk_score.curate_position_bank queries-consensus \
  "$RUN_DIR/evaluation/curation/normalized.jsonl" \
  --output-dir "$RUN_DIR/evaluation/curation/query-bundle-v2" \
  --katago "$REPO/cpp/build-cuda/katago" \
  --analysis-config "$REPO/cpp/configs/risk_score/promotion_curation_analysis.cfg" \
  --original-model "$ORIGINAL_MODEL" \
  --champion-model "$CHAMPION_MODEL" \
  --policy "$REPO/python/risk_score/promotion_policy_v3.json"
```

`queries-consensus` creates exactly eight roles: original/champion ×
standard/powered × 2,000/8,000 visits. Every role covers every distinct
shape-preserving symmetry in each position's orbit. Square boards have at most
eight distinct members and rectangles at most four.

Resource warning: one non-symmetric square position requires up to 64 analyses
(2 models × 2 modes × 2 visit counts × 8 symmetries). Keep
`numAnalysisThreads=1` and shard each role; do not launch one monolithic query
file.

Before committing the full source pool to that cost, run a bounded consensus
pilot and measure acceptance by label. If model, top-move, or specialized-signal
disagreement rejects most pilot rows, use the advisory dual-model prefilter over
2,000-visit standard and powered analyses:

```bash
python3 -m risk_score.consensus_prefilter \
  "$NORMALIZED_SOURCE" \
  --analysis original/standard-2000="$ORIGINAL_STANDARD_2000" \
  --analysis original/powered-2000="$ORIGINAL_POWERED_2000" \
  --analysis champion/standard-2000="$CHAMPION_STANDARD_2000" \
  --analysis champion/powered-2000="$CHAMPION_POWERED_2000" \
  --label lead-80 \
  --output "$RUN_DIR/evaluation/curation/prefiltered-lead-80.jsonl" \
  --manifest "$RUN_DIR/evaluation/curation/prefiltered-lead-80.manifest.json"
```

The prefilter requires one top move, one buffered score band, no specialized
signal, and a bounded model/mode score spread. It deduplicates symmetry orbits
and binds every input analysis manifest. Its output is **not reviewed evidence**
and can never enter a suite directly; every selected row must still pass the
complete eight-role, all-symmetry `queries-consensus` and `label-consensus`
contract. Use it only to avoid spending 64 analyses on obviously unstable
sources.

Use the restartable orchestrator for production. It validates all eight roles,
assigns bounded per-GPU work, resumes only missing shards, merges each role, and
atomically publishes progress and ETA to `WORK_DIR/status.json`:

```bash
python3 -m risk_score.curation_orchestrator watch \
  --query-manifest "$RUN_DIR/evaluation/curation/query-bundle-v2/manifest.json" \
  --work-dir "$RUN_DIR/evaluation/curation/consensus-work-v2" \
  --shards-per-role 8 \
  --gpus 0 1 2 3 4 5 6 7 \
  --per-gpu-parallelism 4 \
  --poll-interval 30
```

Benchmark per-GPU parallelism on a bounded pilot before using the example
value. The manual commands below remain useful for diagnosis, but do not use
detached SSH children as the production scheduler.

```bash
ROLE=original/standard-2000
python3 -m risk_score.curate_position_bank split-queries \
  "$RUN_DIR/evaluation/curation/query-bundle-v2/queries/$ROLE.jsonl" \
  --output-dir "$RUN_DIR/evaluation/curation/query-shards/$ROLE" \
  --shards 8

# Run once per shard, using ORIGINAL_MODEL for original/* and CHAMPION_MODEL
# for champion/*.
python3 -m risk_score.curate_position_bank run-analysis \
  --katago "$REPO/cpp/build-cuda/katago" \
  --config "$REPO/cpp/configs/risk_score/promotion_curation_analysis.cfg" \
  --model "$ORIGINAL_MODEL" \
  --queries "$RUN_DIR/evaluation/curation/query-shards/$ROLE/shard-000.jsonl" \
  --output "$RUN_DIR/evaluation/curation/result-shards/$ROLE/shard-000.jsonl"

# Repeat --shard-output for every shard in the split manifest.
python3 -m risk_score.curate_position_bank merge-analysis \
  --queries "$RUN_DIR/evaluation/curation/query-bundle-v2/queries/$ROLE.jsonl" \
  --split-manifest "$RUN_DIR/evaluation/curation/query-shards/$ROLE/manifest.json" \
  --shard-output "$RUN_DIR/evaluation/curation/result-shards/$ROLE/shard-000.jsonl" \
  --output "$RUN_DIR/evaluation/curation/results/$ROLE.jsonl"
```

Every analysis output has a sibling `.manifest.json` binding the model, binary,
config, query, and result hashes. After merging all eight roles:

```bash
python3 -m risk_score.curate_position_bank label-consensus \
  "$RUN_DIR/evaluation/curation/normalized.jsonl" \
  --query-manifest "$RUN_DIR/evaluation/curation/query-bundle-v2/manifest.json" \
  --analysis original/standard-2000="$RUN_DIR/evaluation/curation/results/original/standard-2000.jsonl" \
  --analysis original/standard-8000="$RUN_DIR/evaluation/curation/results/original/standard-8000.jsonl" \
  --analysis original/powered-2000="$RUN_DIR/evaluation/curation/results/original/powered-2000.jsonl" \
  --analysis original/powered-8000="$RUN_DIR/evaluation/curation/results/original/powered-8000.jsonl" \
  --analysis champion/standard-2000="$RUN_DIR/evaluation/curation/results/champion/standard-2000.jsonl" \
  --analysis champion/standard-8000="$RUN_DIR/evaluation/curation/results/champion/standard-8000.jsonl" \
  --analysis champion/powered-2000="$RUN_DIR/evaluation/curation/results/champion/powered-2000.jsonl" \
  --analysis champion/powered-8000="$RUN_DIR/evaluation/curation/results/champion/powered-8000.jsonl" \
  --output-dir "$RUN_DIR/evaluation/curation/labeling-v2"
```

Acceptance is deliberately narrow. All standard scores must agree on one
buffered band: `abs(score) < 25` for ordinary, `45 <= score < 75` for Lead-40,
or `score >= 85` for Lead-80. Visit, model, symmetry, or canonical top-move
disagreement; a threshold boundary; a specialized signal; or an unclassifiable
label sends the position permanently to `rejected.jsonl`. Only rows in
`machine-labeled.jsonl` have classification `machine-reviewed`. Policy v3
freezes the global score-stability margin at 5.0; it is not an operator-tunable
promotion parameter.

### Supplement missing Lead pools

Policy v3 requires at least 3,200 ordinary, 2,080 Lead-40, and 4,128 Lead-80
source positions. If either Lead pool is short, generate a separate quarantined
corpus with `promotion_curation_lead_selfplay_19x19.cfg`. A terminal-margin SGFS
filter may reduce obvious misses, but it does not assign labels. Normalize the
supplement and run the same complete `queries-consensus` and `label-consensus`
workflow with the same frozen policy, original hash, champion hash, and
stability margin.

Merge disjoint consensus bundles without weakening their provenance:

```bash
python3 -m risk_score.curate_position_bank merge-labeling-consensus \
  "$RUN_DIR/evaluation/curation/labeling-v2" \
  "$RUN_DIR/evaluation/curation/lead-v2/labeling-v2" \
  --output-dir "$RUN_DIR/evaluation/curation/labeling-combined-v2"
```

The merge carries both accepted and rejected rows and rejects semantic
duplicates or policy/model/stability mismatches. Finalization has no decisions
file and cannot rescue a rejected row:

```bash
python3 -m risk_score.curate_position_bank finalize-consensus \
  --machine-labeled "$RUN_DIR/evaluation/curation/labeling-combined-v2/machine-labeled.jsonl" \
  --rejected "$RUN_DIR/evaluation/curation/labeling-combined-v2/rejected.jsonl" \
  --labeling-manifest "$RUN_DIR/evaluation/curation/labeling-combined-v2/manifest.json" \
  --policy "$REPO/python/risk_score/promotion_policy_v3.json" \
  --output "$RUN_DIR/evaluation/source-positions.jsonl" \
  --manifest "$RUN_DIR/evaluation/source-positions.manifest.json"
```

Preserve `rejected.jsonl` permanently with the final bank. Do not use promotion
match output to curate the suite that evaluates those candidates.

### Historical curation v1

The older `queries`, `label`, `merge-labeling`, and `finalize` commands produced
an `auto-labeled.jsonl`/`review-queue.jsonl` bundle and required a human
`--decisions` file. That path remains available only to replay historical
evidence. It does not satisfy policy-v3 machine-review provenance and must not
be used for a new promotion suite.

## Freeze evaluation suites

Suite build v3 requires exactly one `--curation-manifest` for each source JSONL
and binds the `risk-score-reviewed-position-bank-v2` contract transitively:

```bash
cd "$REPO/python"
python3 -m risk_score.build_evaluation_suites \
  "$RUN_DIR/evaluation/source-positions.jsonl" \
  --output-dir "$RUN_DIR/evaluation/promotion-suites-v3" \
  --seed risk-score-promotion-v3 \
  --policy "$REPO/python/risk_score/promotion_policy_v3.json" \
  --curation-manifest "$RUN_DIR/evaluation/source-positions.manifest.json"
```

Validate the emitted `risk-score-authoritative-evaluation-manifest-v3`,
including `machineReviewOnly=true`, curation sources, model hashes, source
hashes, labels, quotas, and schedule hashes. The source minima are 3,200
ordinary, 2,080 Lead-40, and 4,128 Lead-80 positions. Deep audit reserves 2,048
ordinary, 1,024 Lead-40, and 2,048 Lead-80 pairs, each at both 2,000 and 8,000
visits. Its v2 request/report matrix binds all 24 bank/visit/control cells to
runner and statistics artifact hashes. Before scheduling, freeze the b28
control at `$PROMOTION_ROOT/controls/b28/model.bin.gz`; its content hash is
bound into the request with candidate, champion, and original hashes. Every
runner manifest must bind existing result and move JSONL files, and each
statistics artifact must bind those exact output hashes. Cell PASS/FAIL is
recomputed from the validated games against the frozen win-rate threshold;
report authors cannot declare it independently. Publish a new version rather
than overwriting a frozen suite.

## Bootstrap and inventory

Before controller startup:

1. stop the stock gatekeeper and verify it is not supervised;
2. pin the original model, current champion, and trainer checkpoint;
3. record the original and champion model SHA-256 values;
4. inventory `modelstobetested` without moving entries;
5. investigate incomplete `.tmp`, `.partial`, or `.exported` directories;
6. verify duplicate names have identical content or quarantine the conflict;
7. verify available space against the configured reserve; and
8. bootstrap `champion.json` exactly once.

The gated exporter invokes `risk_score/hardened_exporter.py`. If deployment
uses an archived scripts directory that does not contain the module, set
`KATAGO_HARDENED_EXPORTER` to the reviewed absolute path before starting the
export loop. After verified publication, the shell loop atomically moves the
intact source export from `torchmodels_toexport` to `torchmodels_exported`;
that archive is retention-managed and is never deleted by the exporter.

Gated export also requires `KATAGO_MODEL_PROBE_COMMAND_JSON`, a JSON argv array
that loads `{model_file}` with the production CUDA binary and rejects
non-finite or incompatible output. Publication fails closed when this variable
is missing or the probe exits nonzero.

When the promotion controller owns the run, also set:

```bash
export KATAGO_PROMOTION_BACKPRESSURE_FILE="$TRAIN_BASE/promotion/operations/backpressure.json"
export KATAGO_PROMOTION_POLICY_HASH="0151ddcdee764b1e599eb5313f9dfae944e671ff8098dd471425f8d646ba3318"
export KATAGO_PROMOTION_BACKPRESSURE_MAX_AGE_SECONDS=120
python3 -m risk_score.promotion_preflight bootstrap-backpressure \
  --output "$KATAGO_PROMOTION_BACKPRESSURE_FILE" \
  --policy-hash "$KATAGO_PROMOTION_POLICY_HASH"
```

The gated exporter validates canonical JSON, policy identity, and freshness,
pauses cleanly when `allowExport=false`, honors a stale denial, and fails
closed on a stale allowance or malformed status. Re-running the bootstrap
command safely replaces a controller allowance with a denial during an
intentional maintenance restart. Shadow mode remains paused; the first
mutation-enabled controller reconciliation replaces the bootstrap record.
Leave these variables unset only when the controller is deliberately not
supervising export cadence.

The first inventory should identify the original, earliest, newest,
approximately 500k-sample anchors, and checkpoints adjacent to known training
anomalies. Unselected candidates are `SUPERSEDED`, not statistically rejected.

## Shadow operation

Keep runtime mutation disabled and reconcile state:

```bash
cd "$REPO/python"
python3 -m risk_score.promotion_controller \
  --runtime-config "$RUN_DIR/configs/promotion-runtime.json" \
  --mode reconcile \
  --recommend-only
```

Then execute one bounded controller iteration:

```bash
python3 -m risk_score.promotion_controller \
  --runtime-config "$RUN_DIR/configs/promotion-runtime.json" \
  --mode once \
  --recommend-only
```

For every recommendation, independently verify:

- candidate, champion, original, policy, config, and schedule hashes;
- complete color pairs and position clusters;
- separate discovery and confirmation schedules;
- powered utility weight 4 on both bots;
- all ordinary-strength and catastrophe bounds;
- no-results, resignations, turn limits, duplicates, or missing traces;
- report decision and machine-readable reasons; and
- the expected champion SHA used by confirmation.

Run enough shadow cycles to demonstrate that restarts and repeated invocations
produce the same events, evaluation keys, and reports.

Stage 0–2 screening is allowed before machine-review readiness is complete.
That permission does not extend to activation: both direct promotion and
automatic advancement remain blocked unless the controller can trace the v3
suite manifest through every `risk-score-reviewed-position-bank-v2` curation
manifest, source hash, policy hash, and original/champion model binding.

### Configured evaluator adapter

Automatic controller mode invokes the runtime `commands.evaluator` argv
template. Use the in-repository `risk_score.promotion_evaluator`
unless an equivalent replacement has been audited. It runs the exact
manifest-bound match cells through `EvaluationRunner`, then uses
`risk_score.promotion_evidence` to atomically publish canonical
`evidence.json` at the provided `{evidence_output}` path. Its envelope must
bind:

- controller stage;
- candidate, tested champion, and original hashes;
- evaluation key;
- aggregate config and schedule hashes;
- canonical policy hash; and
- `finalized=true`.

For integrity, screen, and finalist stages, the evaluator derives a finalized
decision from validated runner/statistics artifacts. For confirmation, it
includes `promotion_evidence`; the controller runs the in-repo versioned
promotion gate and does not trust an external PASS. Stage 0 is a separate
configured argv probe that must atomically publish a canonical, request-bound
result before any later stage may run.

The adapter owns environment-specific GPU lease invocation and construction of
the candidate/champion/original evaluation matrix. It must return nonzero
unless all requested shards and evidence publication complete.

## GPU 7 handoff

The trainer and evaluator may never overlap on GPU 7. Low utilization is not a
lease.

The configured trainer must run in bucket-limited bursts using
`-stop-when-train-bucket-limited`. A handoff is valid only after:

1. the trainer exits at a checkpoint boundary;
2. the recorded process group is gone;
3. PID start time, boot ID, command hash, and cgroup match the expected
   trainer;
4. the checkpoint hash at handoff is recorded;
5. repeated GPU observations show the expected UUID and no foreign compute
   process; and
6. the lease event is durably written.

`SIGSTOP` is prohibited. If evaluation fails, the lease recovery path drains
evaluator processes and attempts to restore the trainer. A failed restoration
is a safety halt, not a reason to continue promotion.

Inspect and dry-plan reconciliation first:

```bash
cd "$REPO/python"
python3 -m risk_score.gpu_lease \
  --config "$RUN_DIR/configs/gpu-lease-runtime.json" status
python3 -m risk_score.gpu_lease \
  --config "$RUN_DIR/configs/gpu-lease-runtime.json" reconcile
```

Apply only after the dry plan is valid:

```bash
python3 -m risk_score.gpu_lease \
  --config "$RUN_DIR/configs/gpu-lease-runtime.json" reconcile --apply
```

Benchmark 4, then 8, then 16 evaluator processes. Freeze the fastest topology
that is memory-safe and produces deterministic per-shard results.

## Manual promotion rehearsal

Before automatic promotion, rehearse a PASS transaction with a disposable
candidate and bounded dummy commands:

1. verify finalized PASS report and expected champion SHA;
2. write and fsync the promotion intent;
3. pin previous champion, checkpoint, and data/shuffle watermarks;
4. atomically move the exact candidate into accepted storage;
5. verify the destination manifest;
6. launch one generation-pinned worker against an isolated output root;
7. verify the worker loaded the expected model SHA;
8. stop the worker and reconcile the transaction; and
9. roll back to the previous champion.

Kill the controller after every filesystem step and rerun reconciliation.
Every kill point must converge to one unambiguous state without duplicate
events or a second champion.

## Canary and rollout

Canary output must be outside the shuffler-visible self-play root.

The rollout sequence is:

1. one of seven workers for 4,000 games;
2. fresh 2,048-pair audit against the previous champion;
3. schema, purity, throughput, crash-rate, behavioral, and catastrophe checks;
4. three of seven workers after canary PASS;
5. all seven workers after model-SHA acknowledgement; and
6. atomic admission of the generation data followed by champion commit.

Each worker uses an immutable one-model directory,
`switchNetsMidGame=false`, one GPU, and a generation-specific output root.
Model name or modification time is not proof of identity.

The runtime `commands.selfplay` entry must call a bounded supervisor/launcher
that verifies process startup and then exits. Do not put a persistent foreground
`katago selfplay` process directly in this argv template.

Supervisors publish canonical, hashed worker reports to the configured
`workerAckInbox`; reports bind generation, worker, model, self-play config,
thread count, verified process identity, and a stable closed-output manifest.
Canary and intermediate auditors publish finalized reports to
`rolloutReportInbox`. The owning controller ingests these IPC files while
holding the process-lifetime writer lock; bare acknowledgement or PASS marker
files are rejected. A v3 canary report must bind a canonical
`risk-score-canary-fresh-audit-v1` manifest containing the
candidate/reference, suite, policy, schedule, pair-identity, and
statistics-artifact hashes. The 2,048 pair IDs must equal the complete
two-member pair set in the frozen audit schedule.

Run `risk_score.promotion_auditor` as the generated auditor service. It derives
canary/intermediate decisions from closed worker trees and manifest-bound game
outputs, and executes queued deep-audit matrices under the GPU 7 lease. It
publishes both PASS and FAIL rollout reports; the controller advances PASS,
quarantines failed pre-activation generations, and routes failed active deep
audits through replay-safe rollback.

After commit, record the first game, tdata file, admitted directory, shuffle,
and trainer consumption that reference the new generation. The generated
feedback service maintains generation-indexed rollback watermarks and immutable
trainer receipts. Inspect the combined live view with:

```bash
python3 -m risk_score.promotion_status --run-root "$TRAIN_BASE"
```

## Enabling automatic promotion

Automatic mode is allowed only after:

- recommendation-only decisions match the frozen gate and expected fixtures;
- event replay and report finalization are idempotent;
- GPU lease handoff and trainer restoration pass on GPU 7;
- evaluator topology is frozen;
- seven worker launch/acknowledgement is verified;
- canary data remains invisible to the shuffler before admission;
- all rollback drills pass; and
- monitoring exposes every invariant and latency target.

Policies v1 and v2 remain immutable historical evidence and must not be used
for new automatic promotion. Policy v3 uses cumulative confirmation looks:

- look 1: 512 powered ordinary pairs per matchup, 128 standard pairs, 512
  Lead-40 pairs, and 1,024 Lead-80 pairs;
- look 2: 1,024 powered ordinary pairs per matchup, 128 standard pairs, 1,024
  Lead-40 pairs, and 2,048 Lead-80 pairs.

At the final catastrophe alpha, zero events across 1,024 independent clusters
have an exact upper bound of about 0.470%; 2,048 clusters give about 0.235%.
The first look may therefore continue rather than promote when its tightest
zero-event margins remain inconclusive. A full final confirmation is 5,248
color pairs, or 10,496 games, before canary audit work; benchmark this workload
before accepting the four-hour target.

Set `"mutationEnabled": true` in both controller and GPU-lease runtime configs
only after all activation checks pass, then use the controller's explicit
automatic flag. A PASS report alone cannot bypass machine-review readiness.

## Rollback

### Before canary admission

Stop candidate workers, restore previous champion workers, leave canary output
quarantined, and record `CANARY_FAILED`. The trainer checkpoint is unchanged.

### After admission, before trainer consumption

Stop shuffler/trainer intake, quarantine the candidate generation and any
derived shuffle, restore previous champion workers, and resume from the
existing trainer checkpoint only after reconciliation.

### After trainer consumption

Stop self-play, shuffle, trainer, and exporter. Quarantine candidate-derived
self-play and shuffle data, restore the pinned pre-promotion checkpoint, restore
the previous generation, run bounded recovery smokes, and write the rollback
event before resume.

Never touch an old model to manipulate mtime. Never delete quarantined evidence
during incident response.

## Restart and reconciliation

After any controller or host restart:

1. verify no stale controller owns the advisory lock;
2. run recommendation-only reconciliation;
3. validate the event hash chain and champion file;
4. inspect partial exports, evaluation shards, reports, promotion intents, and
   rollout directories;
5. verify process identities and GPU ownership;
6. verify candidate and accepted manifests; and
7. resume mutation only after reconciliation reports no contradiction.

If a promotion source is absent but the exact manifest exists at its intended
destination, complete the recorded intent. A different destination hash is a
quarantine-and-halt condition.

## Manual override

A manual actor may halt automation or reject/quarantine a candidate with a
written reason. Manual promotion must not bypass the finalized gate, expected
champion compare-and-swap, canary, or worker acknowledgement requirements.

Record actor, UTC time, source revision, affected hashes, reason, and incident
reference in the immutable event stream. Do not edit old events or reports.

## Storage and retention

Move eligible data to `promotion/trash` with a deletion manifest and grace
deadline. Delete only after confirming no references from:

- champion lineage;
- rollback pins;
- active or partial evaluation;
- finalized reports;
- suite/audit provenance;
- trainer recovery; or
- incident investigation.

At the configured reserve threshold, halt producers before cleanup.

## Incident escalation

Immediately halt mutation for:

- event hash-chain failure;
- duplicate candidate name with different content;
- candidate hash/path contradiction;
- stale champion during confirmation or promotion;
- incomplete confirmation matrix or report;
- trainer/evaluator GPU overlap;
- foreign GPU 7 process;
- model-SHA acknowledgement mismatch;
- canary data visible to the shuffler before admission;
- destination manifest mismatch; or
- failed trainer restoration after evaluator cleanup.

Preserve logs, process/GPU observations, partial artifacts, manifests, events,
and checkpoints. Resume only from a written reconciliation and recovery
decision.
