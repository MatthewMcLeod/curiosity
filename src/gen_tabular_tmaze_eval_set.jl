module TabularTMazeEvalSet

using Curiosity
using MinimalRLCore
const TTMU = Curiosity.TabularTMazeUtils
include("../experiments/tabular_tmaze.jl")
using Random


function gen_dataset()
    parsed = TabularTMazeExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0

    horde = TabularTMazeExperiment.get_horde(parsed)

    cumulant_schedule = TTMU.get_cumulant_schedule(parsed)

    exploring_starts = parsed["exploring_starts"]
    env = TabularTMaze(exploring_starts, cumulant_schedule)

    start_states = Curiosity.valid_state_mask()
    num_returns = 10
    γ_thresh=1e-6
    rets = monte_carlo_returns(env, horde.gvfs[1], start_states, num_returns, γ_thresh)

    return rets
end

end
