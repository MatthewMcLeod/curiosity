using Plots
using Reproduce
using JLD2
using FileIO

include("plot_utils.jl")

p = plot()
# folder = "TabularTMazeExperiment/RP_0_0x278a9a4b9fa34c2b/"
folder = "TabularTMazeExperiment/RP_0_0xb6e5dbf217a7af96/"
save_file = "plotting/chunlok/generated_plots/test_plot_visitation.svg"


results_file = folder * "results.jld2"
@load results_file results
# println(results)

settings_file = folder * "settings.jld2"
println(settings_file)
settings = FileIO.load(settings_file)["parsed_args"]

print(settings)
# print(sett)

error = results[:ttmaze_uniform_error]
visits_arr = results[:visit_order]
episode_length_arr = results[:episode_length]


println(size(visits_arr))
println(size(episode_length_arr))


visit_perc = get_single_goal_percentage(visits_arr)

println(size(visit_perc))

for i in 1:4
    println(size(visit_perc[i, :]))
    plot!(p, 1:size(visit_perc)[2], visit_perc[i, :])
end

# print(error[3,10])
savefig(save_file)