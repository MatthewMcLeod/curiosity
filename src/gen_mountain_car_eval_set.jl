using Curiosity
using MinimalRLCore
using Statistics
using GVFHordes
using StatsBase
using JLD2
const MCU = Curiosity.MountainCarUtils

include("../experiments/mountain_car.jl")

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
