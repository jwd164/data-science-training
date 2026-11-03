# Use Cases

Written for: the repo owner, early design planning. See [README.md](README.md) for the stage legend
these use cases refer to, and the [Block Diagram](Block%20Diagram.md) for the components named below.

## Personas

A generic set, consistent with the plan's goal of a system that "can be launched anywhere" rather than
one built for a specific industry:

| Persona | Who they are | What they care about |
|---|---|---|
| **Maintenance Technician** | Front-line worker who services the physical machine. | Clear alerts, trustworthy work orders, not being paged for false alarms. |
| **Reliability Engineer / Data Scientist** | Owns the modeling pipeline. | Model quality, label quality, knowing when to retrain. |
| **Plant / Operations Manager** | Schedules downtime and allocates crew and parts. | Enough lead time and confidence to plan around, not react to, failures. |
| **The System** (automated actor) | The software itself, acting without a person in the loop until Stage 3's review step. | Correctly separates normal variation from real anomalies; and does not change confirmed labels or models silently. |

The first three map to the human consumers in the [Block Diagram](Block%20Diagram.md) (technician alert
app, reviewer UI, manager dashboard). The fourth is written because several stages — ingestion, anomaly
detection, clustering, RUL estimation — happen with no person involved, and that behavior should be
specified like any other actor's.

---

## UC-1: System monitors a machine and escalates an anomaly

- **Actor:** The System
- **Stages:** 1, 2
- **Goal:** Detect a real deviation from normal behavior and route it toward a person, without
  overwhelming anyone with false alarms.
- **Preconditions:** The machine is in the "Healthy / Monitoring" state (see
  [State Diagram](State%20Diagram.md)); a baseline exists for it.
- **Trigger:** A new window of sensor data arrives through ingestion.

**Main flow**

1. Ingestion aligns and cleans the incoming sensor data and produces a feature window.
2. The Anomaly Detection service scores the window against the machine's baseline.
3. The score exceeds threshold; the machine's state moves to "Anomaly Detected."
4. The system extracts signal features for the anomalous window and passes them to the Fault
   Clustering service.
5. The clustering service compares the new anomaly to known clusters.
   - If it matches an existing cluster, it is tagged with that cluster's candidate label.
   - If it fits no cluster, it is tagged as "novel."
6. The anomaly (with its candidate label or "novel" tag) is placed in the active-learning queue and an
   alert is sent to the Technician alert app.
7. The machine's state moves to "Candidate Fault Clustering," then "Pending Human Review."

**Alternate flows**

- *3a. Score is below threshold:* no state change; the window is logged and monitoring continues.
- *5a. Clustering service is not yet trained (early in a machine's life, little data):* the anomaly is
  tagged "novel" by default and still queued for review.

**Postconditions:** An anomaly is recorded, a candidate label (or "novel" tag) is attached, and a
technician and reviewer both have visibility into it. No label has been confirmed yet.

---

## UC-2: Reliability engineer reviews a candidate fault and confirms a label

- **Actor:** Reliability Engineer / Data Scientist
- **Stages:** 3
- **Goal:** Turn a candidate fault cluster into a trustworthy label, without letting an incorrect
  cluster assignment reach the supervised classifier.
- **Preconditions:** At least one anomaly is sitting in "Pending Human Review" (from UC-1).
- **Trigger:** The reviewer opens the Reviewer UI, prompted by the active-learning queue (which
  prioritizes the anomalies the current models are least certain about).

**Main flow**

1. The reviewer opens the next item in the queue: the anomaly's raw signal, its extracted features, and
   the clustering service's candidate label or "novel" tag.
2. The reviewer inspects the evidence and decides the candidate label is correct.
3. The reviewer confirms the label in the Reviewer UI.
4. The confirmed label is written to the labeled dataset store.
5. The machine's state moves from "Pending Human Review" to "Classified Fault."
6. The confirmation is written to the audit log.

**Alternate flows**

- *2a. The reviewer judges the candidate label wrong, but the anomaly is still real:* the reviewer
  assigns a different (possibly new) fault label instead of the clustering service's suggestion. The
  rest of the flow proceeds as in the main flow using the reviewer's label.
- *2b. The reviewer judges this a false positive, not a real fault:* the reviewer rejects it; the
  machine's state returns to "Healthy / Monitoring," and the rejection is fed back to the Anomaly
  Detection service as a data point (per the Block Diagram's "rejected" path).
- *2c. The reviewer is unsure:* the item can be deferred, remaining in the queue for a second opinion or
  more data.

**Postconditions:** The anomaly has a human-confirmed label (or is dismissed), and the labeled dataset
used to train the supervised classifier has grown by zero or one examples.

**Related use case:** the same persona also owns a recurring, lower-frequency use case — reviewing the
drift monitor's alerts and deciding whether to approve a retraining run (Stage 5). That is not detailed
here because the plan has not yet specified retraining triggers in enough detail to describe the flow
precisely; see the open questions in [README.md](README.md).

---

## UC-3: Plant manager plans a maintenance window from an RUL estimate

- **Actor:** Plant / Operations Manager
- **Stages:** 4
- **Goal:** Schedule maintenance before failure, at a time that minimizes disruption, using an RUL
  estimate the manager trusts.
- **Preconditions:** A machine has a "Classified Fault" and the RUL Model has produced an estimate with
  an uncertainty interval.
- **Trigger:** The Maintenance Optimization engine posts a recommended maintenance window to the
  manager's dashboard.

**Main flow**

1. The manager sees the recommendation: the RUL point estimate, its uncertainty interval, the fault
   type driving it, and the engine's proposed window (chosen against cost, parts, and crew
   constraints).
2. The manager checks the proposed window against other planned downtime and crew availability.
3. The manager approves the window.
4. A work order is created in the CMMS for the proposed window.
5. The machine's state moves to "Maintenance Scheduled."

**Alternate flows**

- *2a. The proposed window conflicts with other plans:* the manager picks a different time within the
  RUL interval's safe margin and the system re-evaluates whether that window is still viable, given the
  uncertainty interval.
- *2b. The RUL interval is too wide to act on confidently:* the manager can flag the estimate for the
  reliability engineer, rather than scheduling against it.
- *2c. No safe window exists before the RUL estimate's lower bound:* this is treated as an urgent case;
  the manager is prompted to schedule immediately rather than optimize for cost or convenience.

**Postconditions:** A maintenance work order exists with a scheduled window, visible to the technician,
and the machine's state reflects that scheduling has happened.

---

## UC-4: Technician performs scheduled maintenance

- **Actor:** Maintenance Technician
- **Stages:** 4 (consumes its output), transition back to Stage 1
- **Goal:** Complete the scheduled repair and let the system know the machine is healthy again.
- **Preconditions:** A work order exists from UC-3 (or, in the reactive case, a failure has already
  occurred).
- **Trigger:** The scheduled maintenance window arrives, or (in the unplanned case) the machine fails.

**Main flow**

1. The technician receives the work order, including the fault type and any diagnostic detail carried
   over from the classified fault.
2. The technician performs the repair.
3. The technician closes the work order and records whether the component was replaced or repaired in
   place.
4. The action is written to the audit log.
5. The machine's state moves to "Maintenance Performed," then automatically to either "Baselining"
   (component replaced) or "Healthy / Monitoring" (repaired in place), per the
   [State Diagram](State%20Diagram.md).

**Alternate flows**

- *Trigger is an unplanned failure (no prior work order):* the technician creates the work order
  themselves at the point of repair; steps 2 onward proceed the same way, with the machine's state
  coming from "Unplanned Failure" instead of "Maintenance Scheduled."
- *3a. The technician finds a different fault than the one classified:* this is recorded as a
  correction and routed back to the reliability engineer, since it means a label was likely wrong (an
  input to UC-2, not a new use case).

**Postconditions:** The work order is closed, the audit log reflects what was actually done (which may
differ from what was predicted), and the machine resumes normal monitoring, possibly after a new
baseline is established.

## Open questions

- Should a technician be able to reject a work order (e.g. "not actually needed") the way a reviewer
  can reject a candidate label in UC-2? The current flows assume the technician always acts on an
  approved work order.
- UC-3's "urgent case" (2c) assumes the system can detect that no safe window exists before the RUL
  lower bound. Whether that check lives in the Maintenance Optimization engine or the dashboard is not
  yet decided.
