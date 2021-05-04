using Plots
using Reproduce
using JLD2
using FileIO

p = plot()
folder = "experiment_data/TabularTMazeExperiment/RP_0_0xaa3ef5ee09768d83/"
save_file = "plotting/chunlok/generated_plots/test_plot.svg"


results_file = folder * "results.jld2"
@load results_file results
# println(results)

settings_file = folder * "settings.jld2"
println(settings_file)
settings = FileIO.load(settings_file)["parsed_args"]

print(settings)
# print(sett)

error = results[:ttmaze_error]
# print(size(error))

for i in 1:4
    # println(size(error[i, :]))
    plot!(p, 1:size(error)[2], error[i, :])
end

# print(error[3,10])
savefig(save_file)