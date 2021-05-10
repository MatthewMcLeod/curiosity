using Plots
using Reproduce
using JLD2
using FileIO
include("plot_utils.jl")

function plot_single_intrinsic_reward(results ; smooth_step = 300)
    save_file = "plotting/chunlok/generated_plots/ttmaze_intrinsic_reward.svg"

    p = plot()
    logger_key = :wc_per_demon
    
    intrinsic_rewards = results[logger_key]
    num_steps = size(intrinsic_rewards)[2]

    for i in 1:4
        line = smooth_lines(intrinsic_rewards[i, :], smooth_step)
        xs = 1:smooth_step:num_steps
        # println(line)
        # println(size(error[i, :]))
        plot!(p, xs, line)
    end
    
    # print(error[3,10])
    savefig(save_file)
    println("Single intrinsic reward plot saved to $(save_file)")
end

