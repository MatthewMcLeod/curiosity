using Plots
using Reproduce
using JLD2

p = plot()
results_file = "TabularTMazeExperiment/RP_0_0xe47a21fa67c95c99/results.jld2"
@load results_file results
println(results)
error = results[:ttmaze_error]
print(size(error))

for i in 1:4
    println(size(error[i, :]))
    plot!(p, 1:400, error[i, :])
end

# print(error[3,10])
savefig("plotting/chunlok_plots/generated_plots/test_plot.svg")