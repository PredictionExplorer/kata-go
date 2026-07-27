# Risk-Seeking, Score-Maximizing Utility

## Status and scope

This document is the normative specification for the opt-in score-maximizing
KataGo project. The first release is intentionally narrow:

- standard KataGo behavior remains the default;
- the new behavior is enabled only with `useScoreMaximizingUtility = true`;
- games are square 19x19 Tromp-Taylor area-scored games with 7.5 komi;
- there is no handicap, button, rectangle, resignation, or alternate ruleset in
  the first training run; and
- the neural-network architecture, exported outputs, heads, and losses do not
  change.

Search-only experiments must pass the Phase 1 gate below before any expensive
self-play training begins. Operational procedures are in
[RiskSeekingTrainingRunbook.md](RiskSeekingTrainingRunbook.md).

## Objective and non-goals

The objective is to make each player prefer a larger final score margin while
retaining a direct incentive to win. A player that is far ahead may therefore
accept a controlled increase in loss probability when the score upside is large
enough. Any aggressive, unusual, or complicated play must emerge from that
terminal objective and ordinary search.

There are **no explicit rewards or bonuses** for:

- invasions;
- captures;
- attacks;
- contact moves;
- fighting;
- tactical or strategic complexity; or
- any named style of play.

Such terms would confound the experiment by encoding a style rather than a
terminal preference. They are out of scope for both search and training.

## Exact utility

All quantities in this section use KataGo's internal white-positive convention.
Let:

- \(p_W\), \(p_L\), and \(p_{NR}\) be White win, loss, and no-result
  probabilities;
- \(X\) be the inferred final White score before legal clamping;
- \(L\) and \(H\) be the lowest and highest legally possible White scores for
  the current board and rules;
- \(S = \operatorname{clamp}(X,L,H)\);
- \(a=\mathtt{scorePower}\);
- \(c=\mathtt{scoreScale}\); and
- \(w=\mathtt{winWeight}\).

The signed point utility is

\[
u(s)=\operatorname{sign}(s)
     \left[\left(1+\frac{|s|}{c}\right)^a-1\right],
\qquad u(0)=0.
\]

The total White-positive leaf utility is

\[
U_W =
 w(p_W-p_L)
 + p_{NR}U_{NR,W}
 + \mathbb{E}[u(S)].
\]

\(U_{NR,W}\) is KataGo's existing configured
`noResultUtilityForWhite`; its existing default is zero. With that default, the
formula is exactly

\[
U_W=w(p_W-p_L)+\mathbb{E}[u(S)].
\]

The approved defaults are:

```text
useScoreMaximizingUtility = false
scorePower = 1.5
scoreScale = 20.0
winWeight = 2.0
```

`scoreScale` is measured in points. In opt-in mode the powered expectation
replaces, rather than augments, KataGo's existing static and dynamic arctangent
score utilities. Mixing both score utilities would create a different
objective.

The first implementation accepts `scorePower` from 1.0 through 2.0 and
`scoreScale` from 5 through 1000 points. These bounds keep the deterministic
lookup numerically resolved over legal 19x19 scores. The initial experiments
remain restricted to powers 1.2, 1.35, and 1.5 and do not start at 2.0.

For a completed, decisive game, the distribution is degenerate at the legal
final score and the win/loss term is \(+w\) or \(-w\). For a nonterminal leaf,
the probabilities and score moments are network estimates.

## Full-distribution expectation

KataGo's exported inference interface already supplies:

- `whiteScoreMean`, the first moment \(\mu\); and
- `whiteScoreMeanSq`, the second moment \(m_2\).

The search layer infers

\[
\sigma=\sqrt{\max(0,m_2-\mu^2)}
\]

and uses the normal approximation \(X\sim\mathcal{N}(\mu,\sigma^2)\). The
powered utility must be applied separately to every score represented by this
inferred Gaussian:

\[
\mathbb{E}[u(S)] =
 u(L)\Phi\left(\frac{L-\mu}{\sigma}\right)
{}+\int_L^H u(s)\frac{1}{\sigma}
   \phi\left(\frac{s-\mu}{\sigma}\right)\,ds
{}+u(H)\left[1-\Phi\left(\frac{H-\mu}{\sigma}\right)\right].
\]

Here \(\phi\) and \(\Phi\) are the standard normal density and CDF. The
\(\sigma=0\) case is \(u(\operatorname{clamp}(\mu,L,H))\).

The implementation must use deterministic numerical integration or a
deterministic cached lookup with bounded error. It must not:

- evaluate only `u(whiteScoreMean)`;
- draw Monte Carlo score samples;
- discard or renormalize Gaussian tail mass outside the legal range; or
- perform expensive brute-force quadrature independently at every leaf.

In particular,

\[
\mathbb{E}[u(X)]\ne u(\mathbb{E}[X])
\]

in general. Replacing the expectation with utility of the mean would erase the
Jensen effect that makes a spread of positive outcomes preferable to a certain
outcome with the same mean when `scorePower > 1`.

This design does **not** add a neural-network head. The training code has richer
score-belief targets internally, but exposing a full score histogram through
the inference model would require architecture and export changes. That is
deliberately outside this project.

## Legal score bounds

Every integration sample is clamped before applying \(u\). Probability below
\(L\) receives exactly \(u(L)\), and probability above \(H\) receives exactly
\(u(H)\). This both respects the game and prevents an unbounded Gaussian tail
from dominating search.

For ordinary 19x19 Tromp-Taylor area scoring with 7.5 komi and no handicap or
button, board area is 361 and White's final score is White area minus Black area
plus komi. Therefore:

```text
minimum White score = -361 + 7.5 = -353.5
maximum White score = +361 + 7.5 = +368.5
legal range         = [-353.5, 368.5]
```

The general area-scoring calculation must account for board area, komi, the
current history's White handicap/bonus adjustment, and any possible button
adjustment. The first version must fail clearly if score-maximizing mode is
requested with territory scoring; it must not advertise an approximate bound
as a legal one.

At the approved defaults, the score component at the 19x19 endpoints is about
`-79.7` and `+84.6`, before the win term. This much larger and asymmetric range
is why search constants and utility bounds require explicit validation.

## Perspective and player-to-move behavior

KataGo's raw network value is initially in the player-to-move perspective.
Inference post-processing converts win/loss probabilities, score mean, lead,
ownership, and related values to White-positive form. Score second moment does
not change sign. Search nodes continue to store White-positive utility.

Selection then converts the stored value to self utility:

- White to move maximizes \(U_W\);
- Black to move maximizes \(-U_W\), equivalently minimizing \(U_W\).

Thus both colors maximize their own win probability and score margin. The sign
handling must not cause Black to seek large White wins. User-facing analysis
may report from the root player or a requested perspective, but that output
conversion does not change internal storage.

Perspective tests must cover identical White/Black positions, color reversal,
terminal scores, nonzero variance, and move selection at both colors.

## Intended qualitative behavior

The utility is odd and strictly increasing. Consequently:

- winning by more is always preferred to winning by less, all else equal;
- losing by one is preferred to losing by one hundred;
- the direct win term keeps close games from becoming pure score gambles;
- for positive scores and `scorePower > 1`, upside dispersion can increase
  expected utility; and
- legal clamping limits the value of extremely unlikely predicted tails.

These properties do not guarantee sound aggressive play. Score-head
miscalibration, insufficient visits, or inappropriate search scaling can still
make the bot chase illusory upside. Phase 1 is designed to detect those cases.

## Phase 1: deterministic paired experiments

Phase 1 changes search utility only. No neural-network weights are trained.

### Networks

Use both official networks:

- primary: `kata1-zhizi-b40c768nbt-s11272M-d5935M`;
- control: `kata1-b28c512nbt-s13255194368-d5935380940`.

The b40 result is the primary decision input. The b28 control helps distinguish
utility behavior from a peculiarity or calibration error in one score head.
Every model file and raw checkpoint must have source URL, acquisition time,
exact filename, byte size, and SHA-256 recorded before use.

### Fixed game conditions

Every comparison must use:

- 19x19 only;
- Tromp-Taylor rules: positional superko, area scoring, no tax, multi-stone
  suicide legal, no button;
- 7.5 komi;
- no handicap or rectangular boards;
- no resignation;
- equal fixed visits for both bots, initially 800 after a throughput smoke
  test;
- one search thread per game;
- no root Dirichlet noise;
- zero chosen-move temperature;
- identical non-utility search parameters; and
- a recorded deterministic schedule of position, seed, color assignment, and
  color-reversed partner.

Parallel execution is allowed only if the schedule makes results independent of
worker ordering. Each utility setting must see the same openings, positions,
seeds, and colors.

Use two suites:

1. ordinary fixed openings, covering normal whole-game behavior; and
2. fixed real positions where baseline analysis already shows approximately
   40-point and 80-point leads, covering decisions in won games.

Save SGFs and per-game/per-move JSONL outside Git.

### Sweep order

Do not tune all dimensions at once.

1. At `scorePower=1.5` and `scoreScale=20`, compare normal KataGo against
   powered variants with `winWeight=1`, `2`, and `4`.
2. Use those results to select a stable win weight and calibrate
   scale-sensitive search settings.
3. Only then compare `scorePower=1.2`, `1.35`, and `1.5`, keeping the selected
   win weight, score scale, and search settings fixed.

The initial power sweep intentionally excludes `2.0`.

### Required metrics

Report paired results overall and stratified by network, suite, opening or
position, color, utility setting, and visit count:

- wins, losses, no-results, and win rate with confidence intervals;
- final score mean, standard deviation, median, selected quantiles, and
  confidence intervals;
- largest wins and losses, with links to the corresponding external SGFs;
- realized custom utility from the final outcome and score;
- predicted win, powered-score, no-result, and total utility components;
- predicted score mean, second moment or standard deviation, and endpoint-tail
  contribution;
- top-candidate utility range and gap;
- move disagreement rate against standard KataGo;
- visit distribution and entropy;
- policy-prior influence and top-move stability as visits increase; and
- score upside and loss-risk change for moves rejected by standard KataGo.

Do not present win rate as the sole success metric. A bot can meet this
project's objective while sacrificing some win rate, but a large or unstable
loss in safety is still a failure.

### Catastrophic-loss definitions

Keep the following labels separate; never collapse them into one undocumented
rate:

- **Final-20:** the experimental bot's final score margin is at most -20.
- **Final-50:** the experimental bot's final score margin is at most -50.
- **Lead-40 loss:** at an earlier own turn its recorded predicted score lead
  was at least +40, but it ultimately lost.
- **Lead-80 loss:** at an earlier own turn its recorded predicted score lead
  was at least +80, but it ultimately lost.
- **High-confidence loss:** at an earlier own turn its recorded win probability
  was at least 95%, but it ultimately lost.

For lead-based definitions, use the bot's own-turn pre-move analysis in its own
perspective, record the first and maximum threshold crossing, and retain the
underlying trace. Report paired counts, rates, and uncertainty for every
definition. The 40/80 labels refer to model-predicted lead and must not be
described as a proven historical lead.

### Search-scaling checks

Because the new utility is much larger than standard KataGo utility, a finite
result is not sufficient evidence of correct calibration. On the fixed position
suite:

- rerun at increasing visits around the 800-visit starting point;
- check top-move and candidate-rank stability;
- inspect visit entropy and concentration;
- verify that policy priors remain influential but do not override clear value
  differences;
- sweep only scale-sensitive controls such as `cpuctExploration`, root and
  non-root FPU reductions, LCB or utility-stdev settings, and affected
  value-weighting thresholds;
- retain tactical regression positions whose correct move is not optional;
- log observed minimum and maximum utility and candidate gaps; and
- rerun suspicious tail-driven choices with greater integration precision and
  more visits.

Select search settings from stability, tactical correctness, and sensible
prior/value balance, not from average score alone.

### Exploitability tests

Before training, test real and constructed positions containing:

- a low-probability, very-high-score bait;
- deliberately exaggerated or miscalibrated score tails;
- a whole-board sacrifice for speculative upside;
- a small possible point gain that risks a 40- or 80-point lead;
- ordinary tactical refutations; and
- adversarial opponents that decline the expected cooperative continuation.

Analyze each with standard and powered utility, both networks, color reversal,
and higher visits. Inspect endpoint-tail contribution explicitly. A failure
that disappears only on one network is evidence of score-head calibration risk,
not proof that the utility is safe.

### Phase 1 go/no-go gate

Freeze the paired schedule, sample count, confidence method, and acceptance
thresholds in the run manifest before viewing final results.

Training is a **go** only if all of the following hold:

- move divergence from standard KataGo is reproducible and not merely random
  temperature, root noise, color, or scheduling variation;
- representative divergent moves show measurable score upside under the custom
  objective;
- win-weight changes produce a tunable safety response, including the
  catastrophic-loss metrics;
- perspective, legal bounds, endpoint mass, and utility decomposition show no
  violations;
- selected moves and tactical answers are stable enough under increased visits;
- b40 primary and b28 control results are directionally coherent, with any
  material disagreement explained;
- exploitability tests show no unresolved whole-board sacrifice or tail-bait
  failure; and
- runtime and lookup error are acceptable for self-play.

Any perspective or clamp error, noisy/nonreproducible divergence, endpoint-tail
domination, catastrophic-loss response that cannot be controlled with
`winWeight`, or unresolved tactical/exploitability regression is a **no-go**.
The response is to fix or recalibrate Phase 1, not to hope self-play repairs it.

## Phase 2: self-play fine-tuning

Phase 2 starts only after the written Phase 1 gate is approved.

### Learning path

The neural network remains structurally unchanged. Existing policy,
win/loss/no-result, score, score-uncertainty, ownership, lead, Q-value, and
auxiliary heads and losses remain enabled. The only intended causal path is:

```text
powered terminal preference
  -> MCTS values and visits
  -> policy targets and game trajectories
  -> learning through the existing heads and losses
```

No powered-utility output head or style label is added.

### Fixed self-play domain

The initial fine-tune uses:

- 19x19 square boards only;
- positional superko, area scoring, no tax, multi-stone suicide legal;
- 7.5 fixed komi;
- no button, handicap, rectangles, alternate board sizes, or exotic rules;
- no resignation;
- no fork games, start-position mixtures, policy initialization, fancy komi,
  or other exotic initialization;
- `reduceVisits = false`;
- full decided-game search, with cheap search disabled initially; and
- surprise-based target downweighting disabled initially.

Keeping full visits and ordinary sample weight in decided games is essential:
those are precisely the positions in which powered score utility should alter
the policy target.

### Checkpoint evaluation and promotion

Each important exported checkpoint must be evaluated against both:

- the original pretrained network; and
- the current accepted score-maximizing champion.

Promotion is initially manual. Its report must include paired realized custom
utility, score center and tails, ordinary win rate, every catastrophic-loss
rate, search stability, and exploitability-suite results. A checkpoint must
not be promoted solely because it wins a conventional gatekeeper match.

## Numerical and experimental risks

- **Gaussian approximation:** mean and second moment do not capture multimodal
  score beliefs. Use b40/b28 controls and inspect predicted versus realized
  tails.
- **Endpoint mass:** large inferred variance can place material mass on legal
  endpoints. Log the contribution and test integration precision.
- **Utility scale:** powered endpoint values are far outside standard utility
  scale. Audit FPU, virtual loss, LCB, uncertainty weighting, and any assumed
  symmetric utility radius.
- **Nonstationarity:** after fine-tuning, score predictions and policy targets
  change together. Keep immutable checkpoint evaluations and compare to the
  original model.
- **Selection bias:** do not choose thresholds after seeing results. Freeze the
  manifest first.
- **Throughput bias:** fixed-visit paired experiments avoid time-control
  interference between concurrently batched games.

## Reproducibility and artifact policy

Every result must identify:

- source commit SHA and uncommitted diff status;
- exact config snapshots and their SHA-256s;
- model and checkpoint filenames, provenance URLs, byte sizes, and SHA-256s;
- deterministic schedule and seed-manifest SHA-256;
- build backend and flags;
- GPU and software inventory; and
- start/end times and process exit status.

Models, raw checkpoints, self-play data, shuffled data, logs, SGFs, generated
CUDA or TensorRT plans, and credentials must never be committed. Keep them in
the external run directory described by the runbook. Do not rely on
`.gitignore` as the primary safeguard.
