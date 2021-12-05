module Plotting

using Plots; pyplot()
using Reproduce
using FileIO
using Statistics
using ProgressMeter
using JLD2
using LaTeXStrings
using StatsPlots


include("./plot_utils.jl")
GPU = GeneralPlotUtils
LU = LabelUtils

 default(titlefont = (20, "times"), legendfontsize = 10, guidefont = (18, :black), tickfont = 12, foreground_color_legend = nothing)

function load_data(data_home)
    ic = ItemCollection(joinpath(data_home, "data"))
    return ic
end

function load_best(ic,data_key)
    algo_divisor_keys = ["behaviour_learner", "demon_learner", "demon_opt", "demon_update","exploration_param"]
    sweep_params = ["behaviour_eta", "demon_eta", "alpha_init", "eta","emphasis_clip_threshold"]

    algo_specs_full = GPU.split_algo(ic, algo_divisor_keys)
    all_algos_ics = [search(ic,algo_spec) for algo_spec in algo_specs_full]

    valid_algos_ind = findall(a -> length(a) != 0, all_algos_ics)

    algo_specs = algo_specs_full[valid_algos_ind]
    algo_ics = all_algos_ics[valid_algos_ind]
    best_per_algo_ics = []
    for (i,algo_ic) in enumerate(algo_ics)
        push!(best_per_algo_ics, GPU.get_best_final_perf(algo_ic,sweep_params, data_key, 0.1))
    end
    return best_per_algo_ics
end

function plot_rmse_per_demon(algo_ics, data_key; inds_of_interest = nothing, save_path = "")
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
            plot_params = LU.get_params(best_per_algo_ics[algo_ind])
            plot!(p,smooth_gvf, ribbon = std_per_gvf[algo_ind][gvf_ind,:] / sqrt(num_runs), size = (500,500); plot_params...)
            plot!(xlabel="Steps", ylabel = "RMSE",  title = string(gvf_labels[gvf_ind], ""))
        end
        push!(ps,p)
    end
    allp = plot(ps..., layout=(2,2), size = (1200,1200))
    figure_path = joinpath(save_path, "Demon_RMSE_$(string(data_key)).pdf")
    savefig(figure_path)
end

function plot_TE(data_path, data_key, save_path = "")
    ic = load_data(data_path)
    algo_ic = load_best(ic, data_key)
    plot_rmse(algo_ic, data_key, save_path = save_path)
end
function plot_TE_per_demon(data_path, data_key, save_path = "")
    ic = load_data(data_path)
    algo_ic = load_best(ic, data_key)
    plot_rmse_per_demon(algo_ic, data_key, save_path = save_path)
end

function plot_goal_visitations(data_path, data_key, save_path = "")
    ic = load_data(data_path)
    algo_ic = load_best(ic, data_key)
    _plot_goal_visitations(algo_ic, data_key, save_path = save_path)
end
function plot_rmse(algo_ics, data_key; inds_of_interest = nothing, save_path = "")
    best_per_algo_ics = if inds_of_interest isa Nothing
        deepcopy(algo_ics)
    else
        deepcopy(algo_ics)[inds_of_interest]
    end

    data = [GPU.smooth(GPU.get_stats(GPU.load_results(ic,data_key))[1],10) for ic in best_per_algo_ics]
    std = [GPU.smooth(GPU.get_stats(GPU.load_results(ic,data_key))[2],1) for ic in best_per_algo_ics]

    ylabel = "RMSE"
    title = ""
    tick_scaling = 1000
    xlabel = string("Steps", " (per $(tick_scaling))")

    step_increment = if "logger_interval" in keys(best_per_algo_ics[1].items[1].parsed_args)
        best_per_algo_ics[1].items[1].parsed_args["logger_interval"]
    else
        1000
    end
    num_samples = length(data[1])
    num_runs = length.(best_per_algo_ics)[1]
    plot_params = [LU.get_params(best_per_algo_ics[i]) for i in 1:length(best_per_algo_ics)]

    xticks=collect(1:step_increment:num_samples*step_increment) / tick_scaling
    p = plot(ylabel=ylabel, title=title, xlabel=xlabel, formatter=:plain, figsize=(800,800))
    for i in 1:length(best_per_algo_ics)
        plot!(p, xticks, data[i], ribbon = std[i]/sqrt(num_runs), legend=:none; plot_params[i]...)
    end
    figure_path = joinpath(save_path, "RMSE_$(string(data_key)).pdf")
    savefig(figure_path)
end

function _plot_goal_visitations(algo_ics, data_key; inds_of_interest = nothing, save_path="")
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
        visit_perc = GPU.goal_visits_per_episode(visit_orders[i], max_lengths[i])
        tmp = [GPU.smooth(visit_perc[i,:],20) for i in 1:4]
        title = LU.get_label(best_per_algo_ics[i])[:label]

        p = plot(tmp, labels = gvf_labels, xlabel="Episode Count", ylabel="Fraction of Goal Visits", ylim=(0.0,1.0), title = title, size = (800,1000), legend=:none)
        push!(ps,p)
    end
    plot(ps...)

    figure_path = joinpath(save_path, "Goal_Visitation_$(string(data_key)).pdf")
    savefig(figure_path)
end

end
