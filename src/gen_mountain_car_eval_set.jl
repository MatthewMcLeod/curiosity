using Curiosity
using MinimalRLCore
using Statistics
using GVFHordes
using StatsBase
using JLD2
using Plots
const MCU = Curiosity.MountainCarUtils
const MCC = Curiosity.MountainCarConst

import ..get_normalized_state

include("../experiments/mountain_car.jl")

#Hacky this function should be called from environment
function get_normalized_state(env::MountainCar,state)
    pos_limit = MCC.pos_limit
    vel_limit = MCC.vel_limit
    return Float32[(state[1] - pos_limit[1])/(pos_limit[2] - pos_limit[1]),
                   (state[2] - vel_limit[1])/(vel_limit[2] - vel_limit[1])]
end

StatsBase.sample(p::GVFHordes.GVFParamFuncs.FunctionalPolicy, s, actions) =
    sample(Weights([p.func(;state_t = s, action_t = a) for a in actions]))

StatsBase.sample(rng, p::GVFHordes.GVFParamFuncs.AbstractPolicy, s, actions) =
    sample(Weights([get(p;state_t = s, action_t = a) for a in actions]))

function gen_start_state_eval_set()
    parsed = MountainCarExperiment.default_args()
    # horde = MountainCarExperiment.get_horde(parsed)
    # gvfs = [MCU.steps_to_wall_gvf(), MCU.steps_to_goal_gvf()]
    policies = MCU.get_policies(Dict("learned_policy_names" => ["Wall","Goal"] , "learned_policy" => true))
    gvfs = [MCU.steps_to_wall_gvf(policies[1]),MCU.steps_to_goal_gvf(policies[2])]

    normalized = true
    env = MountainCar(0.0,0.0, normalized)
    start_states = []

    pos_step_interval = 0.025
    vel_step_interval = 0.005
    # possible_pos = collect(0:0.05:1)
    # possible_vels = collect(0:0.1:1)
    possible_pos = collect(MCC.pos_initial_range[1]:pos_step_interval:MCC.pos_initial_range[2])
    possible_vels = collect(MCC.vel_initial_range[1]:vel_step_interval:MCC.vel_initial_range[2])

    states = []
    for p in possible_pos
        for v in possible_vels
            push!(states,get_normalized_state(env,[p,v]))
        end
    end
    # states = [[0.45,0.3]]
    actions = rand(1:3,length(states))
    gvf_rets = Array{Float64, 2}(undef, length(gvfs),length(states))

    num_returns = 1
    γ_thresh=1e-6

    for (gvf_i,gvf) in enumerate(gvfs)
        println("here in gvf loop")
        rets = monte_carlo_returns(env, gvf, states, actions,num_returns, γ_thresh)
        @show size(rets)
        @show rets[1],rets[2],rets[3]
        episode_lengths = [log.(0.99,r) for r in rets]
        println("Mean Episode length: ", mean(rets))
        rets = mean(rets, dims = 2)
        rets = collect(Iterators.flatten(rets))
        # scatter(x,rets, legend=false, ylabel="Cumulant Val", xlabel="Starting X Pos", title = "GVF: $( gvf_i)")
        # savefig("./MC_gvf_$(gvf_i).png")
        gvf_rets[gvf_i,:] = rets
    end
    return gvf_rets,states,actions
end


function gen_learned_eval_set()
    parsed = MountainCarExperiment.default_args()
    # horde = MountainCarExperiment.get_horde(parsed)
    # gvfs = [MCU.steps_to_wall_gvf(), MCU.steps_to_goal_gvf()]
    policies = MCU.get_policies(Dict("learned_policy_names" => ["Wall","Goal"] , "learned_policy" => true))
    gvfs = [MCU.steps_to_wall_gvf(policies[1]),MCU.steps_to_goal_gvf(policies[2])]

    normalized = true
    env = MountainCar(0.0,0.0, normalized)
    start_states = []

    pos_step_interval = 0.05
    vel_step_interval = 0.01
    possible_pos = collect(0:0.05:1)
    possible_vels = collect(0:0.1:1)
    # possible_pos = collect(MCC.pos_limit[1]:pos_step_interval:MCC.pos_limit[2])
    # possible_vels = collect(MCC.vel_limit[1]:vel_step_interval,MCC.vel_limit[2])

    states = []
    for p in possible_pos
        for v in possible_vels
            push!(states,[p,v])
        end
    end
    # states = [[0.45,0.3]]
    actions = rand(1:3,length(states))
    gvf_rets = Array{Float64, 2}(undef, length(gvfs),length(states))

    num_returns = 1
    γ_thresh=1e-6

    for (gvf_i,gvf) in enumerate(gvfs)
        println("here in gvf loop")
        rets = monte_carlo_returns(env, gvf, states, actions,num_returns, γ_thresh)
        @show size(rets)
        @show rets[1],rets[2],rets[3]
        episode_lengths = [log.(0.99,r) for r in rets]
        println("Mean Episode length: ", mean(rets))
        rets = mean(rets, dims = 2)
        rets = collect(Iterators.flatten(rets))
        # scatter(x,rets, legend=false, ylabel="Cumulant Val", xlabel="Starting X Pos", title = "GVF: $( gvf_i)")
        # savefig("./MC_gvf_$(gvf_i).png")
        gvf_rets[gvf_i,:] = rets
    end
    return gvf_rets,states,actions
end

function plot_eval_set(states,rets)
    x = [xy[1] for xy in states]
    y = [xy[2] for xy in states]

    s1 = scatter(x,y,rets[1,:], xlabel="Position", ylabel="Velocity", title = "To Back Wall")
    s2 = scatter(x,y,rets[2,:], xlabel="Position", ylabel="Velocity", title = "To Top Hill")
    plot([s1,s2]...)
end

function save_learned_policy(rets,states,actions)
    MCLearnedEvalSet = Dict()
    MCLearnedEvalSet["ests"] = rets
    MCLearnedEvalSet["actions"] = actions
    MCLearnedEvalSet["states"] = states
    @save "./src/data/MCLearnedEvalSet.jld2" MCLearnedEvalSet
end

function gen_eval_set()
    parsed = MountainCarExperiment.default_args()
    # horde = MountainCarExperiment.get_horde(parsed)
    gvfs = [MCU.steps_to_wall_gvf(), MCU.steps_to_goal_gvf()]

    normalized = true
    env = MountainCar(0.0,0.0, normalized)

    num_returns = 1
    γ_thresh=1e-6



    task_gvf = Curiosity.MountainCarUtils.task_gvf()
    # gvfs = [horde.gvfs..., task_gvf]
    # gvfs = [horde.gvfs..., task_gvf]


    num_start_states = 400
    gvf_rets = Array{Float64, 2}(undef, length(gvfs),num_start_states)
    start_states = []

    for i in 1:num_start_states
        MinimalRLCore.reset!(env)
        s = MinimalRLCore.get_state(env)
        push!(start_states,s)
    end
    possible_actions = MinimalRLCore.get_actions(env)
    action_probs = ones(length(possible_actions)) / length(possible_actions)
    start_actions = [sample(possible_actions, Weights(action_probs)) for i in 1:num_start_states]

    for (gvf_i,gvf) in enumerate(gvfs)
        rets = monte_carlo_returns(env, gvf, start_states, start_actions,num_returns, γ_thresh)

        x = [x for (x,y) in start_states]
        y = [y for (x,y) in start_states]

        rets = mean(rets, dims = 2)
        rets = collect(Iterators.flatten(rets))
        # scatter(x,rets, legend=false, ylabel="Cumulant Val", xlabel="Starting X Pos", title = "GVF: $( gvf_i)")
        # savefig("./MC_gvf_$(gvf_i).png")
        gvf_rets[gvf_i,:] = rets

    end
    return start_states, start_actions, gvf_rets
end

function save_data(rets,obs,actions)
    MCEvalSet = Dict()
    MCEvalSet["ests"] = rets
    MCEvalSet["actions"] = actions
    MCEvalSet["states"] = obs
    @save "./src/data/MCEvalSet.jld2" MCEvalSet
end

# using Plots
# pyplot()
# for (gvf_i,gvf) in enumerate(gvfs)
# # Create heatmap version
# increment = 0.01
# vel_limit = (0, 1)
# pos_initial_range = (0.4, 0.9)
#
# ys = collect(vel_limit[1]:increment:vel_limit[2])
# xs = collect(pos_initial_range[1]:increment:pos_initial_range[2])
#
# start_states = [[x,y] for x in xs for y in ys]
# rets = monte_carlo_returns(env, gvf, start_states, num_returns, γ_thresh)
# rets = collect(Iterators.flatten(rets))
# ret_heatmap = reshape(rets,length(xs),length(ys))
#
# surface(xs,ys,ret_heatmap', xlabel=" Starting X position", ylabel = "Starting Velocity", title = "GVF: $( gvf_i)")
# savefig("./MC_gvf_$(gvf_i).png")
#
# end
