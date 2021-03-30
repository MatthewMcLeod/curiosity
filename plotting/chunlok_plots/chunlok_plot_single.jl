using Plots
using Reproduce
using JLD2
using FileIO

p = plot()
# folder = "TabularTMazeExperiment/RP_0_0x278a9a4b9fa34c2b/"
folder = "TabularTMazeExperiment/RP_0_0xf0de2f3eb1b3bc1e/"
save_file = "plotting/chunlok_plots/generated_plots/test_plot.svg"


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
    # if i == 3
    #     continue
    # end
    println(size(error[i, :]))
    plot!(p, 1:400, error[i, :])
end

# print(error[3,10])
savefig(save_file)