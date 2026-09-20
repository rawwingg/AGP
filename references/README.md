# Reference papers

Background reading for the AGP reinforcement-learning work. BibTeX entries for everything
here live in [`docs/latex/refs.bib`](../docs/latex/refs.bib).

## `locomotion/` — staged RL for legged / humanoid control

These three papers were read for their *training methodology*, not their robotics content.
They are the source of the multi-stage curriculum design in
[`docs/latex/agp_staged_ppo_curriculum.tex`](../docs/latex/agp_staged_ppo_curriculum.tex);
Table 1 of that document compares them side by side.

| File | Paper | What we took from it |
| --- | --- | --- |
| `wang2025_dribble_master_arXiv-2505.12679v1.pdf` | Wang, Zhou & Wu, *Dribble Master: Learning Agile Humanoid Dribbling Through Legged Locomotion*, [arXiv:2505.12679](https://arxiv.org/abs/2505.12679) | The two-stage template: one reward basis with a weight column per stage; the eventual task reward held at **exactly** 0.0 through stage 1; scaffolding rewards deleted rather than decayed; making the learning signal artificially abundant early (their 2× camera FOV). Their single-stage ablation (86.7% → 33.0% success) is the empirical case for staging. |
| `wang2025_vision_driven_soccer_arXiv-2511.03996v1.pdf` | Wang et al., *Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots*, [arXiv:2511.03996](https://arxiv.org/abs/2511.03996) | Potential-based reward shaping as an alternative to annealing (it provably cannot change the optimum, so it never has to be removed); the multi-critic split with weighted advantages; the mirror-symmetry consistency loss. Notably this paper is deliberately **single-stage** — it is the counter-argument to the curriculum, and the reason ablation A1 should be run early. |
| `xu2025_agile_striker_skills_arXiv-2512.06571v3.pdf` | Xu et al., *Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input*, [arXiv:2512.06571](https://arxiv.org/abs/2512.06571) | The four-stage reward-weight schedule (their Table II is the format ours copies); deriving the control target from geometry rather than taking it as input; DAgger distillation from a privileged teacher; and state-wise constrained RL as the fix for inhomogeneous credit assignment, with the constraint threshold calibrated against a known-good reference behaviour. |

Neither multi-stage paper gives a quantitative criterion for advancing a stage — both say
only "once the robot consistently demonstrates …". The performance gates in Section 7 of
the curriculum document are ours.

## `ppo/` — algorithm background

The original PPO paper plus two multi-agent / target-localisation papers that were
already in the repository.
