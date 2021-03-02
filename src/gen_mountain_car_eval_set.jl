using Curiosity
using MinimalRLCore
using Statistics

include("../experiments/mountain_car.jl")

parsed = MountainCarExperiment.default_args()
horde = MountainCarExperiment.get_horde(parsed)

normalized = true
env = MountainCar(0.0,0.0, normalized)

num_start_states = 500

start_states = []

for i in 1:num_start_states
    MinimalRLCore.reset!(env)
    s = MinimalRLCore.get_state(env)
    push!(start_states,s)
end
num_returns = 1
γ_thresh=1e-6
rets = monte_carlo_returns(env, horde.gvfs[1], start_states, num_returns, γ_thresh)

using Plots
pyplot()

x = [x for (x,y) in start_states]
y = [y for (x,y) in start_states]

rets = mean(rets, dims = 2)
rets = collect(Iterators.flatten(rets))
scatter(vec(x),vec(rets), legend=false, ylabel="Cumulant Val", xlabel="Starting X Pos")
