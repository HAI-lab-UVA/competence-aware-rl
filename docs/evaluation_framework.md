# Evaluation framework

Hold all policy parameters fixed. Pair an adaptive agent with a fixed partner (or partner team), then vary the competence belief supplied to the adaptive agent. Belief stays constant within each episode. Evaluate behavioral changes and performance separately.

## Overcooked

`evaluate_agents.py --robustness-sweep` evaluates beliefs 1 through 7 against the held-out partners. `--save-trial-logs` records actions and seeds. The metric CLI reconstructs inventory/task events and pre-resolution movement conflicts, preserving the final transition rather than treating an automatic environment reset as episode evidence.

The evaluator compares Baseline, CF, CCM, and CCM+CF. CCM and CCM+CF specialists are evaluated through their competence-conditioned selectors.

`trial_constructs.csv` contains architecture, layout, partner, role, competence, and trial identifiers, together with the paper metrics:

| Column | Definition |
|---|---|
| `workload_allocation` | Adaptive productive actions divided by both teammates' productive actions |
| `task_distinctiveness` | Task distinctiveness: half the summed absolute differences between onion/plate/soup action proportions |
| `coordination_interference` | Directional bumps and same-destination collisions per 100 observed steps |
| `team_performance` | Both teammates' soup deliveries |
| `belief_error` | Absolute belief–label difference divided by 6 |
| `signed_belief_error` | Signed belief–label difference divided by 6 |

Workload is undefined if neither teammate performs productive work. Distinctiveness is undefined if either teammate has no productive work. These cases remain missing, rather than becoming zero.

Productive events are onion pickup and pot placement, plate pickup, soup plating, and delivery. Counter Circuit additionally counts onion-counter placement; Forced Coordination additionally counts onion- and plate-counter placement. `productive_actions.json` documents these rules, implemented in the extracted metric modules.

Coordination interference counts directional bumps and same-destination collisions per 100 observed steps. The implementation excludes attempted swaps from this primary measure, matching the original main-results derivation. Task distinctiveness is total variation distance between task distributions.

With `--n-trials 10`, the main evaluator runs ten trials per starting role in symmetric layouts and twenty in Asymmetric Advantages and Forced Coordination. Role is retained in the outputs. Belief and reference competence are separate columns; the frozen partner may react to a changed environment even though its policy does not change.

## STAT

The partner's task-assignment action is replaced by an alternative feasible task with probability epsilon: 0 (high reliability), 0.5 (medium), or 1 (low). Movement, completion, forced-idle, and single-feasible-task choices are unaffected. Both partners in the paper scenario share the reliability condition. Belief varies independently across 0, 0.5, and 1.

CSV metrics include `reward` (team performance), `adaptive_workload_share`, `adaptive_workload_excess`, `adaptive_assignment_share`, and `adaptive_assignment_excess`. Workload excess is the adaptive agent’s share of completed tasks minus 1/3 in the paper scenario; assignment excess is the corresponding assignment share minus 1/3. Other diagnostics record coordination and corruption opportunities. STAT does not instantiate every Overcooked construct.

## Interpretation and scope

A response to belief establishes behavioral sensitivity. It does not establish better teamwork; compare team performance and belief alignment separately. Short examples use reduced budgets and cannot be interpreted as paper results. Newly trained policies are new experiments, not replacements for the released historical checkpoints.

The released implementations preserve historical training algorithms and cohort choices. This focused packaging does not resolve methodological limitations or guarantee every statistical conclusion. This repository does not contain the paper's model-fitting or figure notebooks.
