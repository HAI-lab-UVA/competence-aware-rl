# Competence-aware configurations

| Configuration | Overcooked model names | STAT configuration |
|---|---|---|
| CF | `<Layout>_CF` | `CF` |
| CCM | `<Layout>_CCM_{low,middle,high}` | `CCM` |
| CCM+CF | `<Layout>_CCM+CF_{low,middle,high}` | `CCM+CF` |
| Baseline | `artifacts/baselines/<layout>.npz` | `Baseline` |

CF supplies believed competence as an input to one policy. CCM selects a specialist by competence group. CCM+CF selects a specialist and also supplies believed competence as an input. Evaluation outputs use CF, CCM, CCM+CF, and Baseline.

Overcooked uses PPO with population-based training. The baseline is trained through self-play; the competence-aware policies are initialized from it. Competence labels range from 1 to 7 and are associated with partner role. The supplied configurations retain the original competence-based partner pools and train/validation/test assignments.

STAT uses a factorized DQN with one adaptive agent and frozen partner agents. Belief is 0, 0.5, or 1. Each CCM or CCM+CF specialist is trained separately. The paper's main scenario has three agents, twelve tasks, and a 10 × 6 grid.
