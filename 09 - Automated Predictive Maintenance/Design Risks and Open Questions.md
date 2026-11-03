# Design Risks and Open Questions

Written for: the repo owner, early design planning. This is the running list of problems that are hard
specifically because the system is meant to be generic and largely autonomous, plus the cross-cutting
decisions that several components depend on but that no other document commits to yet. See
[README.md](README.md) for the stage legend and the other design docs.

## Cross-cutting questions (resolved 2026-11-02)

These shaped multiple components at once. All three are now decided; the follow-on risks each decision
creates are logged in the next section.

### 1. How broad is "generalized" for now? — Resolved

The incoming stream is assumed pre-curated: cleaning and feature engineering happen in a
machine-type-specific layer, built and owned per machine type, and everything downstream of it (anomaly
detection through maintenance optimization) is generic over a clean, consistent, feature-rich stream. This
is now drawn as an explicit boundary in [Block Diagram.md](Block%20Diagram.md#1-logical-components) —
"Machine-type feature engineering" is the one box in that diagram that is not generic. See risk 12 below
for the problem this creates (training/serving skew).

### 2. Deployment topology — Resolved

Inference always runs at the edge, per machine. Training usually runs centrally, pooling data across
machines, but the design also has to accept a model trained on the edge itself (poor connectivity, or a
machine that needs a specialized, non-pooled model). This is drawn out in
[Block Diagram.md](Block%20Diagram.md#2-deployment-topology-edge-and-central). It confirms the plan's ONNX
mention as the right direction for a packaging format portable between the two training locations. See
risks 13 and 14 below for what this introduces.

### 3. Fleet-level learning vs. per-machine cold start — Resolved

New machines of a known type warm-start from an existing fleet model instead of starting at "Threshold
Detection." This is now the `WarmStart` entry path in the
[Detection Maturity diagram](State%20Diagram.md#1-detection-maturity), decided by the Asset Registry at
onboarding. It also means RUL estimation (Stage 4) can become reachable for a single well-maintained
machine that never fails on its own, by pooling failure events across machines of its type. See risks
15-17 below for what warm-starting requires to be trustworthy.

## Problems that are hard because the system should be autonomous

### 4. Alert precision and reviewer workload

Active learning assumes a human reviewer can keep up with the queue. The early tiers (threshold,
statistical) are intentionally conservative or noisy by design — if they flag too much, the one human in
the loop (Stage 3) becomes the bottleneck or starts rubber-stamping, which defeats the point of having
them confirm labels. No document yet states a target precision or an alert budget per machine.

**Needs:** an explicit false-positive budget (e.g. per machine per week), and a policy for how
conservative the default threshold tier should be before any statistics exist.

### 5. Operating-regime ambiguity (concept drift vs. real degradation)

A statistically anomalous reading can be a genuine fault, a legitimate regime change (load, ambient
temperature, duty cycle, a planned process change), or sensor drift/miscalibration. Conflating these is
probably the single hardest modeling problem in the system — it drives both false positives (regime
change flagged as a fault) and false negatives (a real fault masked by a regime change). The State
Diagram currently lists this only as an open question under "Healthy / Monitoring" sub-states; it is
closer to a core design requirement than a footnote.

**Needs:** detectors conditioned on known operating-context variables where available, not just raw
sensor values.

### 6. Rare classes and evaluation integrity

Early in Stage 3, each fault type may have only a handful of confirmed labels. A classifier trained and
evaluated on single-digit counts will look more or less reliable than it actually is.

**Needs:** a minimum-label threshold before the classifier's output is trusted over clustering/novelty
detection, and clustering/novelty detection kept running permanently — new, unseen fault types keep
appearing, so there is no state where clustering is "done."

### 7. RUL trust and safety margin

RUL models are notoriously overconfident on small run-to-failure datasets. A maintenance-optimization
engine that schedules purely against a point estimate risks scheduling too late.

**Needs:** bias scheduling toward the conservative (lower) bound of the uncertainty interval, not the
point estimate, and keep the human override path already sketched in Use Case 3's alternate flows.

### 8. Data quality in the wild — reframed, not closed

Resolving risk 1 moves primary responsibility for this into the machine-type feature engineering layer —
cleaning, alignment, and consistency are now explicitly that layer's job, per the user's own framing.
That does not eliminate the risk; it relocates it. Real deployments still have sensor dropout, clock
skew, relabeled or rewired channels, and unit mismatches across vendors, and a bug in one machine type's
feature-engineering code is now invisible to every generic component downstream of it.

**Needs:** the generic side should still do lightweight sanity gating as defense in depth (e.g. reject a
window with an impossible number of missing features) even though it is not responsible for cleaning, and
data-quality problems should surface as their own alert type rather than silently dropped or — worse —
silently scored as an anomaly.

## Components that are named but not designed

### 9. Asset/configuration registry — now planned, not yet detailed

Resolving risk 3 turned this from a hypothetical into a real component,
`Asset / Machine-Type Registry`, drawn into both diagrams in
[Block Diagram.md](Block%20Diagram.md). What it still lacks: a schema (what exactly it stores per
machine), and an answer to who/what assigns a machine's type at onboarding — see risk 16.

### 10. Maintenance Optimization engine internals

Turning an RUL estimate into a schedule needs a real cost model (downtime cost vs. premature-replacement
cost vs. failure cost) and constraint handling (crew, parts availability). This is effectively its own
optimization sub-project, not a thin wrapper around the RUL output, and nothing in the current docs
specifies it beyond naming it.

### 11. No business-level success metric

Per-model metrics are planned (precision/recall, macro F1, RMSE and the NASA asymmetric score). Nothing
yet defines what "working" means at the business level — e.g. reduction in unplanned downtime, false
alarms per machine-month, or cost avoided. The optimization engine needs a target to optimize toward, and
the manager dashboard needs something to report.

## Risks introduced by the 2026-11-02 decisions

### 12. Training/serving skew in machine-type feature engineering

The same feature-engineering logic has to run twice: in real time at the edge (feeding inference) and in
batch, centrally (feeding training on historical data). If the two implementations drift apart — even a
rounding difference, a different window alignment, or a library version mismatch — the model sees
different features at training time than at inference time, and degrades in ways that are hard to
diagnose because every number still looks reasonable.

**Needs:** the feature-engineering code should be one versioned artifact, deployed to both the edge
and the central training pipeline, not two independent implementations that happen to agree today.

### 13. Edge hardware constraints on model size

Inference for every detection-maturity tier — including the ML detector, the classifier, and the RUL
model — now has to run on edge hardware, which may be far less capable than a training machine. A model
trained centrally may simply not fit, or not run fast enough, on a given edge device.

**Needs:** a defined resource budget per edge target, and a fallback (e.g. stay on a smaller/older model
version, or fall back a tier) when a newly trained model doesn't fit the device it would be deployed to.

### 14. Edge/central synchronization and offline operation

The store-and-forward buffer assumes a machine can run disconnected for some period and reconcile later.
How long that period can safely be, what happens to labeling and drift monitoring while disconnected, and
how conflicts are resolved on reconnect (e.g. the registry pushed a new model while the edge was offline
and already made local decisions) are all undecided.

### 15. Fleet model update propagation

When the fleet model for a machine type improves (new data, new confirmed labels), it's undecided whether
every machine of that type gets the update immediately, on a schedule, or only on request. Push-everywhere
risks instability across many machines from one bad fleet update; slow propagation means machines run
stale models longer than necessary.

### 16. Machine-type taxonomy ownership

Warm-starting requires deciding what counts as "the same type" — exact make/model, a broader equipment
class, or something in between. Too narrow, and warm-starting rarely triggers; too broad, and machines
with real behavioral differences get a misleading warm start (see risk 17). Nothing yet defines who sets
this taxonomy or how granular it is.

### 17. Negative transfer from a bad warm start

A fleet model built from machines installed under different conditions (different load profile, ambient
environment, or maintenance history) than a new unit could warm-start that unit into a confidently wrong
tier — e.g. skipping straight to "ML Anomaly Detection" with a baseline that doesn't actually fit. This
needs a way to detect that a warm start is wrong and fall back toward cold-start behavior, not just a way
to perform the warm start in the first place.

## Status

| # | Item | Status |
|---|---|---|
| 1 | Generalization scope boundary | **Resolved** 2026-11-02 — boundary drawn at machine-type feature engineering |
| 2 | Deployment topology | **Resolved** 2026-11-02 — edge inference, central-or-edge training |
| 3 | Fleet-level learning | **Resolved** 2026-11-02 — warm-start from fleet models by machine type |
| 4 | Alert precision / reviewer workload | Open |
| 5 | Operating-regime ambiguity | Open |
| 6 | Rare classes and evaluation | Open |
| 7 | RUL trust and safety margin | Open |
| 8 | Data quality in the wild | Open — reframed as the feature-engineering layer's job, plus generic-side sanity gating |
| 9 | Asset/configuration registry | In progress — component named and drawn, schema and ownership still open |
| 10 | Maintenance Optimization internals | Open — likely its own sub-project |
| 11 | Business-level success metric | Open |
| 12 | Training/serving skew (edge vs. central feature engineering) | Open — new, from decision 1 |
| 13 | Edge hardware constraints on model size | Open — new, from decision 2 |
| 14 | Edge/central sync and offline operation | Open — new, from decision 2 |
| 15 | Fleet model update propagation | Open — new, from decision 3 |
| 16 | Machine-type taxonomy ownership | Open — new, from decision 3 |
| 17 | Negative transfer from a bad warm start | Open — new, from decision 3 |
