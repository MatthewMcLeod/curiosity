module TwoDGridWorldEvalSet

using Curiosity
using Curiosity: OpenWorld
using MinimalRLCore
const TDGWU = Curiosity.TwoDGridWorldUtils
include("../experiments/2d-gridworld.jl")
using Random
using StatsBase
using GVFHordes
using Statistics
using JLD2

using ProgressMeter

StatsBase.sample(p::GVFHordes.GVFParamFuncs.FunctionalPolicy, s, actions) =
    sample(Weights([p.func(;state_t = s, action_t = a) for a in actions]))

StatsBase.sample(rng, p::GVFHordes.GVFParamFuncs.AbstractPolicy, s, actions) =
    sample(Weights([get(p;state_t = s, action_t = a) for a in actions]))

function gen_dataset(num_start_states=500; action_noise = 0.0)
    parsed = TwoDGridWorldExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0
    parsed["demon_learner"] = "Q"

    horde = Horde(
        [GVF(GVFParamFuncs.FeatureCumulant(i+2),
             TDGWU.GoalTermination(0.95),
             TDGWU.GoalPolicy(i, true)) for i in 1:4])

    cumulant_schedule = TDGWU.get_cumulant_schedule(parsed)

    parsed["exploring_starts"] = "whole"
    parsed["env_step_penalty"] = -0.005
    env = OpenWorld(10, 10,
                    cumulant_schedule=cumulant_schedule,
                    start_type=:none, normalized=true)

    start_states = []
    # @show env.starts
    # for x ∈ 0.05:0.05:0.95
    #     for y ∈ 0.95
    #         push!(start_states, [y, x])
    #     end
    # end

    # push!(start_states, [0.9, 0.05])
    # push!(start_states, [0.85, 0.05])
    # push!(start_states, [0.85, 0.05])
    for i in 1:num_start_states
        MinimalRLCore.reset!(env)
        s = MinimalRLCore.get_state(env)
        # Should use get state but state also contains cumulant observations.
        # This makes it tricky to use out of the box for resetting the env back to
        push!(start_states, s[1:2])
    end
    # @show start_states
    num_returns = 1000
    γ_thresh=1e-6
    horde_rets = zeros(length(horde.gvfs), length(start_states))
    # actions = rand(1:4, length(start_states))
    actions = fill(1, length(start_states))
    for (gvf_i,gvf) in enumerate(horde.gvfs)
        rets = monte_carlo_returns(env, gvf, start_states, actions, num_returns, γ_thresh; agg=mean)
        # rets_avg = mean(hcat(rets...), dims=1) # stack horizontally and then average over runs
        horde_rets[gvf_i, :] = rets
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

function save_data_deterministic(rets,obs,actions)
    OneDTMazeDeterministicEvalSet = Dict()
    OneDTMazeDeterministicEvalSet["ests"] = rets
    OneDTMazeDeterministicEvalSet["actions"] = actions
    OneDTMazeDeterministicEvalSet["states"] = obs

    @save "./src/data/OneDTMazeDeterministicEvalSet.jld2" OneDTMazeDeterministicEvalSet
end

end
