# Automated Predictive Maintenance — Design Docs

Written for: the repo owner, early design planning. Status: design only, no code in this folder yet.

This folder holds the system-design documents for the predictive maintenance system described in
[`05 - Predictive Maintenance Roadmap/Predictive Maintenance Plan.md`](../05%20-%20Predictive%20Maintenance%20Roadmap/Predictive%20Maintenance%20Plan.md).
It is expected to be pulled out into its own project once the design stabilizes; everything here is
written to stand on its own when that happens.

## Documents

| Doc | Covers |
|---|---|
| [State Diagram.md](State%20Diagram.md) | Two state machines: (1) how detection matures from "dumb" thresholds through statistical and ML anomaly detection to semi-supervised classification and RUL estimation, as data and faults accumulate; (2) one machine's operational lifecycle, from commissioning through anomaly detection, fault classification, RUL estimation, and maintenance. |
| [Block Diagram.md](Block%20Diagram.md) | The system's components, from field sensors through the analytics pipeline to the MLOps platform and human consumers. |
| [Use Cases.md](Use%20Cases.md) | How each persona (technician, reliability engineer, plant manager) and the system itself interact with the pipeline. |
| [Design Risks and Open Questions.md](Design%20Risks%20and%20Open%20Questions.md) | Cross-cutting decisions (scope, deployment topology, fleet learning) and the specific problems that make a generic, autonomous system hard (alert fatigue, regime ambiguity, rare classes, RUL overconfidence, data quality, missing components). |

Diagrams are written in [Mermaid](https://mermaid.js.org/), which GitHub renders inline in Markdown.

## Stage legend

All three documents refer back to the stages defined in the plan:

| Stage | Name |
|---|---|
| 1 | Anomaly detection on streams |
| 2 | Discovering fault groups without labels |
| 3 | Growing labels with a person in the loop |
| 4 | Remaining useful life |
| 5 | Running it as a system |

## Open questions carried over from the plan

These design docs take a position on open questions from the plan so they can be concrete; treat these
as proposals, not final decisions:

- Fault labels are always human-confirmed before they train the classifier (per the plan's stage 3
  recommendation).
- Maintenance Optimization is treated as a distinct component that turns an RUL estimate plus business
  constraints (cost, parts, crew) into a scheduling recommendation — the plan does not detail this
  component yet, so its internal design is still open.
- The block diagram assumes a model registry and a retraining trigger exist (stage 5), even though no
  stage 1-4 notebook has been built with those in mind yet.
