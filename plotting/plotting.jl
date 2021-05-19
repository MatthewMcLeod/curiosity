include("./plot_utils.jl")

module Plotting
GPU = GeneralPlotUtils
LU = LabelUtils
using Plots; gr()
using Reproduce
using FileIO
using Statistics
using ProgressMeter
using JLD2

function load_data()
    data_home = "../Experiment1_d_pi"
    experiment_folders = [data_home]
    # folder_name = "oned_rr_dpi"
    # folder_name = "oned_control_dpi"
    folder_name = "tmp"
    ic = ItemCollection(joinpath(experiment_folders[1], "data"))
    return ic
end

function load_best(ic, data_key = :ttmaze_direct_error)
# data_key = :oned_tmaze_dpi_error

    algo_divisor_keys = ["behaviour_learner", "demon_learner", "demon_opt", "demon_update"]
    sweep_params = ["demon_alpha_init", "demon_eta"]

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
    return algo_ics
end

function plot_rmse(ics; inds_of_interest)
    data = [GPU.smooth(GPU.get_stats(GPU.load_results(ic,data_key))[1],10) for ic in best_per_algo_ics]
    std = [GPU.smooth(GPU.get_stats(GPU.load_results(ic,data_key))[2],10) for ic in best_per_algo_ics]

    ylabel = "RMSE"
    title = "RMSE"
    xlabel = "Steps"
    step_increment=50
    num_samples = length(data[1])
    xticks=collect(1:step_increment:num_samples*step_increment)
    p = plot(ylabel=ylabel, grid=true, title=title, xlabel=xlabel)
    for i in 1:length(algo_ics)
        plot!(p, xticks, data[i], ribbon = std[i]/sqrt(num_runs), legend=:topright; plot_params[i]...)
    end
    savefig("./plots/$(folder_name)/RMSE_$(string(data_key)).png")
end
end
