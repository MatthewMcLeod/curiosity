using Plots
using Reproduce
using JLD2
using FileIO

include("plot_utils.jl")

function plot_single_goal_visitation(results; step_size=10)
    p = plot()

    save_file = "plotting/chunlok/generated_plots/single_goal_visitation.svg"

    visits_arr = results[:visit_order]
    episode_length_arr = results[:episode_length]

    visit_perc = get_single_goal_percentage(visits_arr;step_size=step_size)

    for i in 1:4
        plot!(p, 1:step_size:step_size*size(visit_perc)[2], visit_perc[i, :])
    end

    # print(error[3,10])
    savefig(save_file)
    println("Single goal visitation plot saved to $(save_file)")
end