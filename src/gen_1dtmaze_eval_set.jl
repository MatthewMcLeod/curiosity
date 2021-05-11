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

using ProgressMeter

StatsBase.sample(p::GVFHordes.GVFParamFuncs.FunctionalPolicy, s, actions) =
    sample(Weights([p.func(;state_t = s, action_t = a) for a in actions]))

StatsBase.sample(rng, p::GVFHordes.GVFParamFuncs.AbstractPolicy, s, actions) =
    sample(Weights([get(p;state_t = s, action_t = a) for a in actions]))

function gen_dataset(num_start_states=500; action_noise = 0.0)
    # Dataset from true uniform state distribution. This is not attainable with no action noise, so we deprecated it.
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
    parsed["env_step_penalty"] = -0.005
    env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"], parsed["env_step_penalty"], action_noise)

    start_states = []
    @show env.starts
    for i in 1:num_start_states
        MinimalRLCore.reset!(env)
        # s = MinimalRLCore.get_state(env)
        # Should use get state but state also contains cumulant observations.
        # This makes it tricky to use out of the box for resetting the env back to
        s = deepcopy(env.pos)
        push!(start_states,s)
    end

    num_returns = 1000
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


function gen_dataset_uniform_from_start(num_start_states=500, seed=1; action_noise = 0.0,eval_set_size_per_gvf = 150)
    # Generates dataset from based on uniform action weighting on the state distribution 
    # It is tricky to generate a uniform sampling without action noise as some states are not reachable. GVF policies cover the state space and randomizing action would make it uniform.
    _,states,_= gen_dataset_dpi(num_start_states, seed; action_noise = action_noise,eval_set_size_per_gvf = eval_set_size_per_gvf)
    all_states = vcat(states...)
    actions = rand(1:4,length(all_states))
    parsed = OneDTmazeExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0
    parsed["demon_learner"] = "Q"
    parsed["exploring_starts"] = "beg"
    cumulant_schedule = ODTMU.get_cumulant_schedule(parsed)
    env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"], parsed["env_step_penalty"], action_noise)


    horde_rets = zeros(4,length(all_states))
    num_returns = 1000
    γ_thresh=1e-6

    for gvf_i ∈ 1:4
        GVF_of_interest = GVF(GVFParamFuncs.FeatureCumulant(gvf_i+2),
             ODTMU.GoalTermination(parsed["demon_discounts"]),
             ODTMU.GoalPolicy(gvf_i))

        rets = monte_carlo_returns(env, GVF_of_interest, all_states, actions, num_returns, γ_thresh)
        rets_avg = mean(hcat(rets...),dims=1) # stack horizontally and then average over runs
        horde_rets[gvf_i,:] = rets_avg
    end
    return horde_rets,all_states,actions
end

function gen_dataset_dpi(num_start_states=500, seed=1; action_noise = 0.0,eval_set_size_per_gvf = 150)
    # Generates dataset based on dpi for each GVF
    Random.seed!(seed)
    parsed = OneDTmazeExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0
    parsed["demon_learner"] = "Q"
    parsed["exploring_starts"] = "beg"

    rets = Vector{Float64}[]
    start_states = Vector{Vector{Float64}}[]
    actions = Vector{Int}[]

    for gvf_i ∈ 1:4
        GVF_of_interest = GVF(GVFParamFuncs.FeatureCumulant(gvf_i+2),
             ODTMU.GoalTermination(parsed["demon_discounts"]),
             ODTMU.GoalPolicy(gvf_i))

        cumulant_schedule = ODTMU.get_cumulant_schedule(parsed)

        env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"], parsed["env_step_penalty"], action_noise)

        ss = Vector{Float64}[]
        as = Int[]
        policy = ODTMU.GoalPolicy(gvf_i)

        for i in 1:num_start_states
            s = MinimalRLCore.start!(env)
            a = policy(s)
            push!(ss, s[1:2])
            push!(as, a)

            term = false

            while term == false
                s, r, term = MinimalRLCore.step!(env, a)
                if term
                    break
                end
                a = policy(s)
                push!(ss, s[1:2])
                push!(as, a)
            end
        end
        #subsample all states in ss for evaluation subset
        indices = sample(1:size(ss)[1], eval_set_size_per_gvf, replace=false)
        ss_eval = ss[indices]
        as_eval = as[indices]

        # @show start_states
        num_returns = 10
        γ_thresh=1e-6

        gvf_rets = Curiosity.monte_carlo_returns_agg(env, GVF_of_interest, ss_eval, as_eval, num_returns, γ_thresh; agg=mean)

        push!(rets, gvf_rets)
        push!(start_states, ss_eval)
        push!(actions, as_eval)
    end
    return rets, start_states, actions
end

function save_data_dpi(rets,states,actions)
    eval_set = Dict(
        "ests" => rets,
        "actions" => actions,
        "states" => states
    )
    @save "./src/data/OneDTMazeEvalSet_d_pi.jld2" eval_set
end

function save_data_uniform(rets,states,actions)
    OneDTMazeEvalSetUniform = Dict()
    OneDTMazeEvalSetUniform["ests"] = rets
    OneDTMazeEvalSetUniform["actions"] = actions
    OneDTMazeEvalSetUniform["states"] = states
    @save "./src/data/OneDTMazeEvalSet_Uniform.jld2" OneDTMazeEvalSetUniform

end

end
