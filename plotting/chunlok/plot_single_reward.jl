using Plots
using Reproduce
using JLD2
using FileIO

function plot_single_reward(results)
    save_file = "plotting/chunlok/generated_plots/single_reward.svg"

    p = plot()
    logger_key = :ttmaze_direct_error

    error = results[logger_key]

    for i in 1:4
        # println(size(error[i, :]))
        plot!(p, 1:size(error)[2], error[i, :])
    end

    # print(error[3,10])
    savefig(save_file)
    println("Single reward plot saved to $(save_file)")
end