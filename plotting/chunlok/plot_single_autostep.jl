using Plots
using Reproduce
using JLD2
using FileIO

function plot_single_autostep(results, log_interval)
    previous_GKSwstype = get(ENV, "GKSwstype", "")
    ENV["GKSwstype"] = "100"

    save_file = "plotting/chunlok/generated_plots/single_autostep_step_sizes.svg"

    p = plot()

    step_sizes = results[:autostep_stepsize]

    # println(findall(x->x!=1.0, step_sizes))
    # dasd
    # println(step_sizes[:, :, 1])
    # println(step_sizes[:, :, 1] - step_sizes[:, :, 100])

    # println(minimum(step_sizes))
    # println(maximum(step_sizes))

    function plot_gvf_step_sizes(range)
        gvf_stepsizes = minimum(step_sizes[range, :, :], dims=(1, 2))[1, 1, :]
        plot!(p, 1:log_interval:size(gvf_stepsizes)[1] * log_interval, gvf_stepsizes)
    end

    plot_gvf_step_sizes(1:4)
    plot_gvf_step_sizes(5:8)
    plot_gvf_step_sizes(9:12)
    plot_gvf_step_sizes(13:16)



    # for i in 1:4
    #     # println(size(error[i, :]))
    #     plot!(p, 1:size(error)[2], error[i, :])
    # end

    # # print(error[3,10])
    savefig(save_file)
    println("Single autostep stepsize plot saved to $(save_file)")

    ENV["GKSwstype"] = previous_GKSwstype 
end