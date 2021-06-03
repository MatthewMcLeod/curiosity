# Continual Auxiliary Task Learning


For the review process we provide non-working code for spot checking. The code does not include package information to run due to anonymity requirements. Full code and package information will be released upon publication.

While the code is not runnable, there are several places the reviewers can spot check our algorithms to confirm correctness. We also provide a list of configuration files mapped to what experimental data they generated. These configs act as another complete specification of a large set of hyperparameters.

If one was able to run the code as is, the `parallel/toml_parallel.jl` file is the entry point. This parallelizes our sweeps and works with all the listed config files.


### Code locations for spot checking:
- Algorithms: `src/updates`
  - TB: `TB.jl`
  - ESARSA: `ESARSA.jl`
  - ETB: `ETB.jl`
  - InterestTB: `InterestTB.jl`
  - Auto Optimizer: `optimizers/Auto.jl`
- Learners: `src/learners`
  - Q: `value.jl`
  - LSTD: `LSTD.jl`
  - SR: `SR.jl`
  - GPI: `GPI.jl`
- Agent: `src/agent/agent.jl` and `src/agent/agent_er.jl`
- Policies: `src/agent/exploration.jl`
- Environments: `src/environments/`
  - Tabular T-Maze: `tabular_tmaze.jl`
  - 1D T-Maze: `1d-tmaze.jl`
  - 2D Open World: `cont_2d_worlds.jl` and `cont_2d_worlds_spec.jl`
  - Mountain Car: `mountain_car.jl`
  - Cumulant schedules for T-Mazes and 2d Open World: `tmaze_cumulants.jl`
- Replay: `src/utils/replay.jl`
- TileCoder: `src/agent/tile_coder.jl`



### Config Files Mapped to Experiments:
- Figure 2:
- Figure 3:
- Figure 4:

### Appendix empirical results:
- Figure 5:
- Figure 6:
- Figure 7:
- Figure 8:
