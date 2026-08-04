# Risk-score experiment harness

This directory contains a deterministic paired Phase 1 match template and a
fixed-rules Phase 2 self-play template. The experiment changes terminal utility
only; it does not add style-specific move rewards.

## Phase 1 workflow

Start with one or more JSONL files whose rows use the existing
`Sgf::PositionSample` schema. Empty-board samples, played-out opening samples,
and curated positions with 40/80-point leads use the same schema. Put labels
such as `ordinary`, `lead-40`, and `lead-80` in the optional `metadata` field;
the harness preserves the position but does not infer a lead from that label.

Generate a color-paired schedule:

```sh
python3 python/risk_score/generate_schedule.py \
  positions.jsonl \
  --output phase1_schedule.jsonl \
  --base-seed phase1-2026-07 \
  --bot-a-index 0 \
  --bot-b-index 1
```

Each input position produces two adjacent rows. Bot A is black in the first and
white in the second. The pair shares the exact `startPosition`; each game gets
its own stable search `seed`. `--pairs-per-position N` creates N independently
seeded color pairs for each position.

Run the match, supplying the model outside the checked-in config:

```sh
./cpp/katago match \
  -config cpp/configs/risk_score/phase1_match.cfg \
  -override-config \
  "nnModelFile=/path/to/model.bin.gz,deterministicScheduleFile=phase1_schedule.jsonl,matchResultJsonlFile=phase1_results.jsonl" \
  -sgf-output-dir phase1_sgfs
```

The template uses the same model for both bots and varies only their search
utility configuration. Override visits or utility parameters explicitly when
defining an experiment.

### Schedule JSONL contract

Every nonblank row is a JSON object with:

- `schemaVersion`: currently `1`
- `scheduleId`: the same nonempty string on every row
- `gameId`: a nonempty ID unique within the file
- `seed`: the exact base seed passed to `GameRunner`; distinct bots receive the
  existing `@B` and `@W` search-seed suffixes
- `blackBot` and `whiteBot`: zero-based config bot indices
- `startPosition`: a complete `Sgf::PositionSample` JSON object
- optional `pairId` and `positionId`, which are copied into results

The generator derives IDs and seeds from canonicalized position content, bot
indices, repetition count, and `--base-seed`. Re-running it with the same
inputs and options produces byte-identical JSONL.

When `deterministicScheduleFile` is set, `katago match`:

1. consumes rows strictly in file order;
2. bypasses `MatchPairer` and all random start-position sampling;
3. uses each row's fixed colors, start position, game ID, and search seed;
4. seeds `GameInitializer` from `scheduleId`;
5. requires `numGameThreads=1`, one search thread for every scheduled bot,
   visit/playout limits rather than `maxTime`, and
   `initGamesWithPolicy=false`;
6. requires each scheduled board size to be listed in `bSizes`/`bSizesXY`;
7. rejects duplicate game IDs, mixed schedule IDs, and incompletely replayed
   start positions.

If `numGamesTotal` is present, it must equal the schedule row count. If it is
absent, the schedule length is the game count. Without
`deterministicScheduleFile`, the existing randomized match path is unchanged.

For reproducible rules and komi as well as reproducible schedule inputs, keep
the Phase 1 template's single ruleset, zero komi noise, zero compensation, and
no policy initialization. GPU kernels or backend/library changes can still
cause cross-machine floating-point differences; the contract fixes experiment
inputs and KataGo RNG streams, not bit-identical arithmetic across hardware.

### Result JSONL

Set `matchResultJsonlFile` (or override it) to write one JSON object per
completed game while retaining normal SGF output. Set `matchMoveJsonlFile` to
write the corresponding pre-move root trace. The game record includes schedule,
game, pair, and position IDs; seed; bot names and indices; board dimensions;
rules and komi; SGF-style final result; numeric white-minus-black score when the
game was scored; winner; move counts; game hash; and explicit turn-limit,
resignation, no-result, and scored flags. Move records include mover-perspective
win/loss probabilities, score mean/lead/stdev, result/score/total utility,
endpoint-tail probabilities, visits, and the selected move.

Turn-limit, no-result, and resignation records have a null numeric final score.
Resignations still count in ordinary win/loss statistics, but cannot contribute
to final-margin or powered-score statistics.

Summarize one or more result files:

```sh
python3 python/risk_score/summarize_matches.py \
  phase1_results.jsonl \
  --moves phase1_moves.jsonl \
  --target-bot powered-utility \
  --catastrophe-thresholds 20 50 \
  --score-power 1.5 \
  --score-scale 20 \
  --win-weight 4
```

Machine-readable JSON is written to stdout and a readable summary to stderr.
The realized terminal utility is:

```text
win_weight * outcome
  + sign(target_margin)
    * ((1 + abs(target_margin) / score_scale) ** score_power - 1)
```

where target wins/draws/losses are `1/0/-1`. Final-margin catastrophe rates use
20 and 50 points. When move traces are supplied, the report separately computes
“predicted lead reached 40/80, then lost” and “95% win probability, then lost.”
Labels and denominators remain explicit.

To compare standard and powered analysis responses for the same query IDs:

```sh
python3 python/risk_score/compare_moves.py \
  baseline_analysis.jsonl powered_analysis.jsonl \
  --output move_comparison.json
```

This reports top-move disagreement and, when both candidates are present,
powered-utility and predicted-score differences between the selected moves.

## Checkpoint promotion

New promotion evaluation uses the immutable v2 policy while v1 remains
available only for replaying historical evidence. V2 publishes separate
cumulative look-1/look-2 schedule artifacts and requires one independent
position cluster per risk-bearing pair.

Promotion evaluation uses:

- `promotion_powered_match.cfg`, where candidate and reference both use power
  1.5, scale 20, and win weight 4; and
- `promotion_standard_match.cfg`, where both use standard utility while all
  objective parameters remain explicit.

Always provide different `nnModelFile0` and `nnModelFile1` paths. The
pair-safe Python runner verifies model, policy, config, and schedule hashes,
splits only at complete `pairId` boundaries, and publishes a final bundle only
after every shard validates:

```sh
cd python
python3 -m risk_score.evaluation_runner --help
python3 -m risk_score.promotion_evaluator --help
python3 -m risk_score.promotion_evidence --help
```

The promotion evaluator executes the exact manifest-bound cells; the evidence
assembler then binds finalized runner output to paired statistics and the
five-cell gate. An external PASS marker is not promotion evidence.

`promotion_curation_analysis.cfg` is the deterministic one-GPU analysis
template for `risk_score.curate_position_bank`. Query files override visits
and powered/standard utility per position; the curation manifest binds this
config, the KataGo binary, and the immutable original-model hash. It enables
KataGo's deterministic test seed and one analysis worker. Run it only under
the same exclusive GPU lease used for other promotion analysis.

`promotion_curation_lead_selfplay_19x19.cfg` generates a quarantined
supplemental corpus for Lead-40/Lead-80 discovery. It keeps the fixed 19x19
rules and immutable original model, but deliberately gives one color an
uncompensated random playout advantage. This makes large-lead positions common
without asserting any labels: the manifest-bound five-tier analysis remains
the only source of automatic Lead labels. Its output must never enter a
shuffler or trainer input root.

`promotion_selfplay_worker_19x19.cfg` is the generation-pinned one-GPU
self-play template. Launch seven separate workers through the promotion
controller/supervisor, each with an immutable one-model directory, distinct
output root, and explicit `cudaDeviceToUseModel0Thread0`. It fixes
`switchNetsMidGame=false`.

## Phase 2 self-play

`phase2_selfplay_19x19.cfg` fixes 19x19 positional-superko area scoring with
7.5 komi, powered utility, full visits, no resignation, no handicap,
rectangles, alternate rules, forks, cheap searches, visit reduction, surprise
downweighting, or fancy komi. Standard self-play root noise and root policy
temperature remain enabled.

Run it with model and output directories supplied on the command line:

```sh
./cpp/katago selfplay \
  -config cpp/configs/risk_score/phase2_selfplay_19x19.cfg \
  -models-dir /path/to/models \
  -output-dir /path/to/selfplay-output
```

The config creates seven CUDA inference workers pinned to devices 0 through 6.
It intentionally has no worker on device 7, reserving that GPU for training.
The powered-utility keys assume core support for
`useScoreMaximizingUtility`, `scorePower`, `scoreScale`, and `winWeight`.
