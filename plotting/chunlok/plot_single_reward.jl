using Plots
using Reproduce
using JLD2
using FileIO

function plot_single_reward(results, log_interval, logger_key)
    previous_GKSwstype = get(ENV, "GKSwstype", "")
    ENV["GKSwstype"] = "100"

    save_file = "plotting/chunlok/generated_plots/single_reward.svg"

    p = plot()

    error = results[logger_key]

    for i in 1:4
        # println(size(error[i, :]))
        plot!(p, 1:log_interval:size(error)[2] * log_interval, error[i, :])
    end

    # print(error[3,10])
    savefig(save_file)
    println("Single reward plot saved to $(save_file)")

    ENV["GKSwstype"] = previous_GKSwstype 
end