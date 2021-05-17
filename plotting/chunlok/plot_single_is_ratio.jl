using Plots
using Reproduce
using JLD2
using FileIO

function plot_single_emphasis(results, log_interval)
    previous_GKSwstype = get(ENV, "GKSwstype", "")
    ENV["GKSwstype"] = "100"




    is_ratio = results[:is_ratio]


    println(size(is_ratio))
    # println(size(emphasis))
    # println(emphasis[:, 1:4])
    # println(maximum(emphasis))
    # println(minimum(emphasis))
    # println(mean(emphasis))
    # println(median(emphasis))

    # println(is_ratio[:,])

    p = plot()

    for gvf_i in 1:4
        plot!(p, 1:log_interval:size(is_ratio)[2] * log_interval, is_ratio[gvf_i, :])
    end
    save_file = "plotting/chunlok/generated_plots/single_is_ratio.svg"
    savefig(save_file)

    ylim = (0, 2.0)

    p = plot()
    plot!(p, 1:log_interval:size(is_ratio)[2] * log_interval, is_ratio[1, :], ylim=ylim, title="gvf 1")
    save_file = "plotting/chunlok/generated_plots/single_is_ratio_gvf_1.svg"
    savefig(save_file)


    p = plot()
    plot!(p, 1:log_interval:size(is_ratio)[2] * log_interval, is_ratio[2, :], ylim=ylim, title="gvf 2")
    save_file = "plotting/chunlok/generated_plots/single_is_ratio_gvf_2.svg"
    savefig(save_file)
    p = plot()
    plot!(p, 1:log_interval:size(is_ratio)[2] * log_interval, is_ratio[3, :], ylim=ylim, title="gvf 3")
    save_file = "plotting/chunlok/generated_plots/single_is_ratio_gvf_3.svg"
    savefig(save_file)
    p = plot()
    plot!(p, 1:log_interval:size(is_ratio)[2] * log_interval, is_ratio[4, :], ylim=ylim, title="gvf 4")
    save_file = "plotting/chunlok/generated_plots/single_is_ratio_gvf_4.svg"
    savefig(save_file)
    # asdasds

    # println(findall(x->x!=1.0, step_sizes))
    # dasd
    # println(step_sizes[:, :, 1])
    # println(step_sizes[:, :, 1] - step_sizes[:, :, 100])

    # println(minimum(step_sizes))
    # println(maximum(step_sizes))

    # function plot_gvf_step_sizes(range)
    #     gvf_stepsizes = minimum(step_sizes[range, :, :], dims=(1, 2))[1, 1, :]
    #     plot!(p, 1:log_interval:size(gvf_stepsizes)[1] * log_interval, gvf_stepsizes)
    # end

    # plot_gvf_step_sizes(1:4)
    # plot_gvf_step_sizes(5:8)
    # plot_gvf_step_sizes(9:12)
    # plot_gvf_step_sizes(13:16)



    # for i in 1:4
    #     # println(size(error[i, :]))
    #     plot!(p, 1:size(error)[2], error[i, :])
    # end

    # # print(error[3,10])

    println("Single autostep stepsize plot saved to $(save_file)")

    ENV["GKSwstype"] = previous_GKSwstype 
end