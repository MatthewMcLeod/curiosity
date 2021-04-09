module OneDTMazeEvalSet

using Curiosity
using MinimalRLCore
const ODTMU = Curiosity.OneDTMazeUtils
include("../experiments/1d-tmaze.jl")
using Random
using StatsBase
using GVFHordes
using Statistics
using JLD2

StatsBase.sample(p::GVFHordes.GVFParamFuncs.FunctionalPolicy, s, actions) =
    sample(Weights([p.func(;state_t = s, action_t = a) for a in actions]))


StatsBase.sample(rng, p::GVFHordes.GVFParamFuncs.AbstractPolicy, s, actions) =
    sample(Weights([get(p;state_t = s, action_t = a) for a in actions]))

function gen_dataset()
    parsed = OneDTmazeExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0
    parsed["demon_learner"] = "Q"

    horde = Horde(
        [GVF(GVFParamFuncs.FeatureCumulant(i+2),
             ODTMU.GoalTermination(0.9),
             ODTMU.GoalPolicy(i)) for i in 1:4])

    cumulant_schedule = ODTMU.get_cumulant_schedule(parsed)



    parsed["exploring_starts"] = "whole"
    env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"])

    start_states = []
    num_start_states = 100
    @show env.starts
    for i in 1:num_start_states
        MinimalRLCore.reset!(env)
        # s = MinimalRLCore.get_state(env)
        # Should use get state but state also contains cumulant observations.
        # This makes it tricky to use out of the box for resetting the env back to
        s = deepcopy(env.pos)
        push!(start_states,s)
    end

    num_returns = 2000
    γ_thresh=1e-6
    horde_rets = zeros(length(horde.gvfs), length(start_states))
    actions = rand(1:4, length(start_states))
    for (gvf_i,gvf) in enumerate(horde.gvfs)
        rets = monte_carlo_returns(env, gvf, start_states, actions, num_returns, γ_thresh)
        rets_avg = mean(hcat(rets...),dims=1) # stack horizontally and then average over runs
        horde_rets[gvf_i,:] = rets_avg
    end

    # NOTE: Need about ~2000 returns to be within 1% of estimates for the cumulants from beginning section
    # max_difference = max((horde_rets .- mean(horde_rets,dims=1)) ./ horde_rets)
    return horde_rets, start_states, actions
end

function save_data(rets,obs,actions)
    OneDTMazeEvalSet = Dict()
    OneDTMazeEvalSet["ests"] = rets
    OneDTMazeEvalSet["actions"] = actions
    OneDTMazeEvalSet["states"] = obs

    @save "./src/data/OneDTMazeEvalSet.jld2" OneDTMazeEvalSet
end

end
