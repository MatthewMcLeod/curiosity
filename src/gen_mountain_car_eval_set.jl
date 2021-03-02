using Curiosity
using MinimalRLCore
using Statistics
using GVFHordes

include("../experiments/mountain_car.jl")

parsed = MountainCarExperiment.default_args()
horde = MountainCarExperiment.get_horde(parsed)

normalized = true
env = MountainCar(0.0,0.0, normalized)

num_returns = 1
γ_thresh=1e-6

using Plots
pyplot()
task_gvf = Curiosity.MountainCarUtils.task_gvf()

gvfs = [horde.gvfs..., task_gvf]
#NOTE: GENERATE SCATTER
# num_start_states = 500
#
# start_states = []
#
# for i in 1:num_start_states
#     MinimalRLCore.reset!(env)
#     s = MinimalRLCore.get_state(env)
#     push!(start_states,s)
# end
# for (gvf_i,gvf) in enumerate(horde.gvfs)
#     rets = monte_carlo_returns(env, gvf, start_states, num_returns, γ_thresh)
#
#     x = [x for (x,y) in start_states]
#     y = [y for (x,y) in start_states]
#
#     rets = mean(rets, dims = 2)
#     rets = collect(Iterators.flatten(rets))
#     scatter(x,rets, legend=false, ylabel="Cumulant Val", xlabel="Starting X Pos", title = "GVF: $( gvf_i)")
#     savefig("./MC_gvf_$(gvf_i).png")
# end


for (gvf_i,gvf) in enumerate(gvfs)
# Create heatmap version
increment = 0.01
vel_limit = (0, 1)
pos_initial_range = (0.4, 0.9)

ys = collect(vel_limit[1]:increment:vel_limit[2])
xs = collect(pos_initial_range[1]:increment:pos_initial_range[2])

start_states = [[x,y] for x in xs for y in ys]
rets = monte_carlo_returns(env, gvf, start_states, num_returns, γ_thresh)
rets = collect(Iterators.flatten(rets))
ret_heatmap = reshape(rets,length(xs),length(ys))

surface(xs,ys,ret_heatmap', xlabel=" Starting X position", ylabel = "Starting Velocity", title = "GVF: $( gvf_i)")
savefig("./MC_gvf_$(gvf_i).png")

end
