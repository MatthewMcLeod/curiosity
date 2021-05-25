# include("./plot_utils.jl")
# GPU = GeneralPlotUtils
# LU = LabelUtils
# include("./plot_utils.jl")
# GPU = GeneralPlotUtils
# LU = LabelUtils

module Plotting
using Plots; gr()
using Reproduce
using FileIO
using Statistics
using ProgressMeter
using JLD2
# import ..GeneralPlotUtils
# import ..LabelUtils
# include("./plot_utils.jl")
include("./plot_utils.jl")
GPU = GeneralPlotUtils
LU = LabelUtils


# data_key = :ttmaze_direct_error
# data_key = :ttmaze_uniform_error
# data_key = :ttmaze_round_robin_error
# data_key = :oned_tmaze_dpi_error
data_key = :oned_tmaze_uniform_error
# data_key = :oned_tmaze_dmu_error


folder_name = "oned_rr"
# data_home = "../data/Experiment1"
# data_home = "../data/Experiment2_d_pi"
# data_home = "../data/OneDTMaze_RR_dpi"
# data_home = "../data/OneDTMaze_Control"
data_home = "../data/OneDTMaze_RR"



function load_data()
    experiment_folders = [data_home]
    # folder_name = "oned_rr_dpi"
    # folder_name = "oned_control_dpi"
    ic = ItemCollection(joinpath(experiment_folders[1], "data"))
    return ic
end

function load_best(ic)
# data_key = :oned_tmaze_dpi_error

    algo_divisor_keys = ["behaviour_learner", "demon_learner", "demon_opt", "demon_update"]
    sweep_params = ["demon_alpha_init", "demon_eta", "alpha_init"]

    algo_specs_full = GPU.split_algo(ic, algo_divisor_keys)

    all_algos_ics = [search(ic,algo_spec) for algo_spec in algo_specs_full]
    @show length.(all_algos_ics)

    valid_algos_ind = findall(a -> length(a) != 0, all_algos_ics)

    algo_specs = algo_specs_full[valid_algos_ind]
    algo_ics = all_algos_ics[valid_algos_ind]
    best_per_algo_ics = []
    for (i,algo_ic) in enumerate(algo_ics)
        push!(best_per_algo_ics, GPU.get_best_final_perf(algo_ic,sweep_params, data_key, 0.1))
    end
    @show length.(best_per_algo_ics)
    return best_per_algo_ics
end

function plot_rmse_per_demon(algo_ics, inds_of_interest = nothing)
    best_per_algo_ics = if inds_of_interest isa Nothing
        deepcopy(algo_ics)
    else
        deepcopy(algo_ics)[inds_of_interest]
    end

    data_per_gvf = [GPU.get_stats(GPU.load_results(ic,data_key), per_gvf = true)[1] for ic in best_per_algo_ics]
    std_per_gvf = [GPU.get_stats(GPU.load_results(ic,data_key), per_gvf = true)[2] for ic in best_per_algo_ics]
    num_gvfs = 4
    ps = []
    gvf_labels = ["Distractor" "Constant" "Drifter" "Constant"]
    num_runs = length.(best_per_algo_ics)[1]
    for gvf_ind in 1:num_gvfs
        p = plot()
        for algo_ind in 1:length(data_per_gvf)
            smooth_gvf = GPU.smooth(data_per_gvf[algo_ind][gvf_ind,:],5)
            label = LU.get_label(best_per_algo_ics[algo_ind])[:label]
            plot!(p,smooth_gvf, palette=:tab10, ribbon = std_per_gvf[algo_ind][gvf_ind,:] / sqrt(num_runs), size = (500,500),label = label)
            plot!(xlabel="Steps", ylabel = "RMSE",  title = string(gvf_labels[gvf_ind], ""))
        end
        push!(ps,p)
    end
    plot(ps..., layout=(2,2), size = (1200,1200))
    savefig("./plots/$(folder_name)/RMSE_per_Demon_$(string(data_key)).png")

end

function plot_rmse(algo_ics, inds_of_interest = nothing)
    best_per_algo_ics = if inds_of_interest isa Nothing
        deepcopy(algo_ics)
    else
        deepcopy(algo_ics)[inds_of_interest]
    end

    data = [GPU.smooth(GPU.get_stats(GPU.load_results(ic,data_key))[1],10) for ic in best_per_algo_ics]
    std = [GPU.smooth(GPU.get_stats(GPU.load_results(ic,data_key))[2],10) for ic in best_per_algo_ics]

    ylabel = "RMSE"
    # title = "Hall Following in Tabular T-Maze"
    # title = "Control in Tabular T-Maze"
    title = "Fixed Behaviour in 1D T-Maze"
    # title = "Control in 1D T-Maze"

    xlabel = "Steps"
    step_increment=best_per_algo_ics[1].items[1].parsed_args["logger_interval"]
    num_samples = length(data[1])
    num_runs = length.(best_per_algo_ics)[1]
    plot_params = [LU.get_params(best_per_algo_ics[i]) for i in 1:length(best_per_algo_ics)]

    xticks=collect(1:step_increment:num_samples*step_increment)
    p = plot(ylabel=ylabel, grid=true, title=title, xlabel=xlabel)
    for i in 1:length(best_per_algo_ics)
        @show plot_params[i]
        plot!(p, xticks, data[i], ribbon = std[i]/sqrt(num_runs), legend=:topright; plot_params[i]...)
    end
    savefig("./plots/$(folder_name)/RMSE_$(string(data_key)).png")
end

function plot_goal_visitation(algo_ics, inds_of_interest = nothing)
    best_per_algo_ics = if inds_of_interest isa Nothing
        deepcopy(algo_ics)
    else
        deepcopy(algo_ics)[inds_of_interest]
    end
    episode_lengths = [GPU.load_results(ic,:episode_length, return_type = "array") for ic in best_per_algo_ics]
    visit_orders = [GPU.load_results(ic,:visit_order, return_type = "array") for ic in best_per_algo_ics]
    max_lengths = [GPU.get_min_length(arrs)-1 for arrs in episode_lengths]
    gvf_labels = ["Distractor" "Constant" "Drifter" "Constant"]

    ps = []
    for i in 1:length(visit_orders)
        @show max_lengths[i]
        @show length(visit_orders[i])
        visit_perc = GPU.goal_visits_per_episode(visit_orders[i], max_lengths[i])
        @show size(visit_perc)
        tmp = [GPU.smooth(visit_perc[i,:],20) for i in 1:4]
        title = LU.get_label(best_per_algo_ics[i])[:label]
        p = plot(tmp, labels = gvf_labels, xlabel="Episode Count", ylabel="Fraction of Goal Visits", ylim=(0.0,1.0), title = title, size = (800,1000), legend=:topleft)
        push!(ps,p)
    end
    println()
    @show length(visit_orders)
    plot(ps..., layout=(Int(length(best_per_algo_ics)/2),2))
    savefig("./plots/$(folder_name)/goal_visits_$(string(data_key)).png")
end

function load_and_plot_rmse(inds_of_interest=nothing)
    ic = load_data()
    algo_ic = load_best(ic)
    plot_rmse_per_demon(algo_ic,inds_of_interest)
    sweep_params = ["demon_alpha_init", "demon_eta", "alpha_init"]
    [GPU.print_params(bic,["demon_learner"],["demon_eta","alpha_init"]) for bic in algo_ic]
end

function load_and_plot_goal_visits(inds_of_interest=nothing)
    ic = load_data()
    algo_ic = load_best(ic)
    plot_goal_visitation(algo_ic,inds_of_interest)
end
end
