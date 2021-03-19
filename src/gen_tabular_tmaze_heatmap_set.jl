module TabularTMazeHeatmapSet

using Curiosity
using MinimalRLCore
const TTMU = Curiosity.TabularTMazeUtils
include("../experiments/tabular_tmaze.jl")
using Random
using StatsBase
using GVFHordes
using Statistics
using JLD2


StatsBase.sample(p::GVFHordes.GVFParamFuncs.FunctionalPolicy, s, actions) =
    sample(Weights([p.func(s, a) for a in actions]))


function gen_dataset()
    parsed = TabularTMazeExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0

    cumulant_schedule = TTMU.get_cumulant_schedule(parsed)

    exploring_starts = parsed["exploring_starts"]
    env = TabularTMaze(exploring_starts, cumulant_schedule)
    start_states = Curiosity.valid_state_mask()

    function del!(arr, ind_to_del)
        arr = arr[1:end .!= ind_to_del]
        return arr
    end
    #NOTE: This lets the agent start in terminal states. This is not normally possible.
    # Work backwards with index so you dont have to reindex due to shifting
    ind_to_delete = [21, 17, 5, 1]
    for i in ind_to_delete
        start_states = del!(start_states, i)
    end
    observations = []
    for s in start_states
        MinimalRLCore.reset!(env,s)
        push!(observations, MinimalRLCore.get_state(env))
    end

    num_actions = 4

    total_obs = repeat(observations, inner=num_actions)
    total_actions = repeat(collect(1:4),outer=length(observations))

    return total_obs, total_actions
end

function save_data(obs,actions)
    ValueSet = Dict()
    ValueSet["actions"] = actions
    ValueSet["states"] = obs
    @save "./src/data/TTMazeValueSet.jld2" ValueSet
end

end
