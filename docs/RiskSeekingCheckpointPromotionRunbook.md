# Risk-Seeking Checkpoint Promotion Runbook

## Scope and safety posture

This runbook operates the score-maximizing checkpoint promotion controller.
It supplements `RiskSeekingTrainingRunbook.md`; it does not replace the
training, shuffle, or host-provisioning procedures there.

The controller starts in recommendation-only mode. Enabling filesystem or
process mutation requires both:

1. `"mutationEnabled": true` in a reviewed runtime configuration; and
2. an explicit automatic/mutation CLI flag.

Never enable mutation until the shadow gate, filesystem tests, GPU handoff,
canary, and rollback drills in this document have passed on the actual run
volume.

## Required artifacts

Freeze and hash these together:

- the source revision and local diff;
- `python/risk_score/promotion_policy_v2.json`;
- the unchanged v1 policy and its hash when retaining historical v1 evidence;
- `cpp/configs/risk_score/promotion_powered_match.cfg`;
- `cpp/configs/risk_score/promotion_standard_match.cfg`;
- `cpp/configs/risk_score/promotion_curation_analysis.cfg`;
- `cpp/configs/risk_score/promotion_selfplay_worker_19x19.cfg`;
- discovery, confirmation, and audit suite manifests and schedules;
- the immutable original model;
- the current champion model;
- the current trainer checkpoint; and
- the reviewed runtime configuration.

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

The checked-in v2 policy prints
`8562bcd7b835ae0cfcfe517a290748258da229b3fcf588dc99b3703c2b8f6023`.
Any other value is a different policy and requires a new review and suite
freeze.

Policy v1 is replay-only historical evidence. Its runner-v2 bundles remain
readable without a synthetic visit field, but no new evaluation may be
launched under v1; every new promotion plan uses policy v2 and runner v3.

## Installation and preflight

From the repository root:

```bash
python3 -m json.tool python/risk_score/promotion_policy_v1.json >/dev/null
python3 -m json.tool python/risk_score/promotion_policy_v2.json >/dev/null
python3 -m json.tool python/risk_score/promotion_runtime.example.json >/dev/null
python3 -m json.tool python/risk_score/gpu_lease_runtime.example.json >/dev/null
cd python
python3 -m risk_score.promotion_controller --help
python3 -m risk_score.evaluation_runner --help
python3 -m risk_score.promotion_evaluator --help
python3 -m risk_score.promotion_evidence --help
python3 -m risk_score.curate_position_bank --help
python3 -m risk_score.build_evaluation_suites --help
```

Run the Python tests before deploying the source snapshot:

```bash
cd "$REPO/python"
pytest
```

On a host with the production KataGo binary, run bounded model-load and match
smokes separately. Unit tests do not establish CUDA compatibility.

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

`risk_score.curate_position_bank` creates the reviewed input consumed by the
suite builder. It is staged and fail-closed: discovery/confirmation/audit
assignment does not happen during curation.

Do not harvest the live training self-play corpus: those positions have already
trained the candidates and would contaminate confirmation holdouts. Generate a
separate original-model SGF corpus under
`$RUN_DIR/evaluation/curation/quarantined-sgfs`, keep it outside every
shuffler/trainer input root, and never admit those games to training.

First publish an immutable harvest plan with the complete source inventory and
training-root exclusions:

```bash
cd "$REPO/python"
python3 -m risk_score.curate_position_bank harvest-plan \
  --katago "$REPO/cpp/build-cuda/katago" \
  --sgfs-dir "$RUN_DIR/evaluation/curation/quarantined-sgfs" \
  --training-input-root "$TRAIN_BASE/selfplay" \
  --output-dir "$RUN_DIR/evaluation/curation/harvested" \
  --manifest "$RUN_DIR/evaluation/curation/harvest.json" \
  --threads 1

# After independent review of harvest.json:
python3 -m risk_score.curate_position_bank harvest-execute \
  "$RUN_DIR/evaluation/curation/harvest.json"
```

The curation harvester deliberately requires one parser thread so that SGF
output order and source provenance are reproducible. Execution revalidates the
complete source inventory before and after `samplesgfs`, writes into a
temporary directory, and atomically publishes the output with a receipt.
Query analysis is the GPU-bound stage and may be batched separately.

Normalize and semantically deduplicate the harvested `*.startposes.txt`
outputs, then create analysis queries bound to the immutable original model
and frozen policy:

```bash
python3 -m risk_score.curate_position_bank normalize \
  "$RUN_DIR"/evaluation/curation/harvested/*.startposes.txt \
  --output "$RUN_DIR/evaluation/curation/normalized.jsonl" \
  --manifest "$RUN_DIR/evaluation/curation/normalized-manifest.json"

python3 -m risk_score.curate_position_bank queries \
  "$RUN_DIR/evaluation/curation/normalized.jsonl" \
  --output-dir "$RUN_DIR/evaluation/curation/query-bundle" \
  --katago "$REPO/cpp/build-cuda/katago" \
  --analysis-config "$REPO/cpp/configs/risk_score/promotion_curation_analysis.cfg" \
  --reference-model "$ORIGINAL_MODEL" \
  --policy "$REPO/python/risk_score/promotion_policy_v2.json"
```

Run each query role with the reviewed analysis config by using the
`run-analysis` subcommand. The query bundle currently contains standard
200/800/2,000-visit and powered 800/2,000-visit files. Pass every resulting
`ROLE=PATH` file to `label`; missing, duplicate, or misbound IDs fail.
Each run also publishes `<result>.manifest.json`, binding the query, model,
binary, config, and result hashes, and `label` requires that sidecar:

```bash
python3 -m risk_score.curate_position_bank run-analysis \
  --katago "$REPO/cpp/build-cuda/katago" \
  --config "$REPO/cpp/configs/risk_score/promotion_curation_analysis.cfg" \
  --model "$ORIGINAL_MODEL" \
  --queries "$RUN_DIR/evaluation/curation/query-bundle/queries/standard-800.jsonl" \
  --output "$RUN_DIR/evaluation/curation/results/standard-800.jsonl"

# Repeat run-analysis for every role, then bind all five results:
python3 -m risk_score.curate_position_bank label \
  "$RUN_DIR/evaluation/curation/normalized.jsonl" \
  --query-manifest "$RUN_DIR/evaluation/curation/query-bundle/manifest.json" \
  --analysis standard-200="$RUN_DIR/evaluation/curation/results/standard-200.jsonl" \
  --analysis standard-800="$RUN_DIR/evaluation/curation/results/standard-800.jsonl" \
  --analysis standard-2000="$RUN_DIR/evaluation/curation/results/standard-2000.jsonl" \
  --analysis powered-800="$RUN_DIR/evaluation/curation/results/powered-800.jsonl" \
  --analysis powered-2000="$RUN_DIR/evaluation/curation/results/powered-2000.jsonl" \
  --output-dir "$RUN_DIR/evaluation/curation/labeling"
```

Only stable ordinary/Lead-40/Lead-80 cases are auto-labeled. Every tactical,
exploitability, bait, tail, sacrifice, small-gain/large-lead, adversarial, or
unstable case remains in `review-queue.jsonl`. A reviewer must provide exactly
one decision per queued semantic SHA:

```json
{"approved":true,"labels":["exploitability","baits"],"semantic_sha256":"..."}
```

Finalize only after review. The command rejects semantic duplicates,
unreviewed rows, changed inputs, and pools below policy-v2 minima:

```bash
python3 -m risk_score.curate_position_bank finalize \
  --auto "$RUN_DIR/evaluation/curation/labeling/auto-labeled.jsonl" \
  --review-queue "$RUN_DIR/evaluation/curation/labeling/review-queue.jsonl" \
  --decisions "$RUN_DIR/evaluation/curation/review-decisions.jsonl" \
  --labeling-manifest "$RUN_DIR/evaluation/curation/labeling/manifest.json" \
  --policy "$REPO/python/risk_score/promotion_policy_v2.json" \
  --output "$RUN_DIR/evaluation/source-positions.jsonl" \
  --manifest "$RUN_DIR/evaluation/source-positions.manifest.json"
```

Do not use promotion match output to curate the suite that evaluates those
candidates. That would leak discovery or confirmation results into the frozen
holdouts.

## Freeze evaluation suites

The suite builder consumes reviewed, labeled `PositionSample` JSONL. It does
not synthesize tactical or exploitability positions.

Create a new, previously absent output directory:

```bash
cd "$REPO/python"
python3 -m risk_score.build_evaluation_suites \
  "$RUN_DIR/evaluation/source-positions.jsonl" \
  --output-dir "$RUN_DIR/evaluation/promotion-suites-v2" \
  --seed risk-score-promotion-v2 \
  --policy "$REPO/python/risk_score/promotion_policy_v2.json"
```

Review row counts, exclusions, labels, source hashes, bank hashes, and schedule
hashes in the emitted manifest. Confirm that every risk-bearing pair has a
distinct independent position cluster, that look 1 is an exact complete-pair
prefix of look 2, and that Lead discovery positions are disjoint from Lead
confirmation positions. Confirmation data must not have been used to rank
candidates. Publish a new version rather than overwriting a frozen suite.

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
export KATAGO_PROMOTION_POLICY_HASH="8562bcd7b835ae0cfcfe517a290748258da229b3fcf588dc99b3703c2b8f6023"
export KATAGO_PROMOTION_BACKPRESSURE_MAX_AGE_SECONDS=120
```

The gated exporter validates canonical JSON, policy identity, and freshness,
pauses cleanly when `allowExport=false`, and fails closed on stale or malformed
status. Leave these variables unset only when the controller is deliberately
not supervising export cadence.

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

### Configured evaluator adapter

Automatic controller mode invokes the runtime `commands.evaluator` argv
template. Use the reviewed in-repository `risk_score.promotion_evaluator`
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

Only after reviewing the plan, apply it explicitly:

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

1. one of seven workers for 2,000 games;
2. fresh 1,024-pair audit against the previous champion;
3. schema, purity, throughput, crash-rate, tactical, and catastrophe checks;
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
files are rejected.

After commit, record the first game, tdata file, admitted directory, shuffle,
and trainer consumption that reference the new generation.

## Enabling automatic promotion

Automatic mode is allowed only after:

- recommendation-only decisions match independent review;
- event replay and report finalization are idempotent;
- GPU lease handoff and trainer restoration pass on GPU 7;
- evaluator topology is frozen;
- seven worker launch/acknowledgement is verified;
- canary data remains invisible to the shuffler before admission;
- all rollback drills pass; and
- monitoring exposes every invariant and latency target.

Policy v1 remains immutable historical evidence and must not be used for new
automatic promotion. Policy v2 uses cumulative confirmation looks:

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

Review and set `"mutationEnabled": true` in both controller and GPU-lease
runtime configs, then use the controller's explicit automatic flag. Keep a
human present for the first promotion.

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
