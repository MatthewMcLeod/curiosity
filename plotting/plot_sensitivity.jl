using Plots; pyplot()
using Reproduce
using FileIO
using Statistics
using ProgressMeter
using JLD2
using LaTeXStrings
using Statistics


data_key = :oned_tmaze_old_error
folder_name = "oned_control"
data_home = "../data/OneDTMaze_Control"
experiment_folders = [data_home]
include("./plot_utils.jl")
GPU = GeneralPlotUtils
LU = LabelUtils


ic = ItemCollection(joinpath(experiment_folders[1], "data"))

algo_divisor_keys = ["behaviour_learner", "demon_learner"]
sweep_params = ["eta"]

algo_specs_full = GPU.split_algo(ic, algo_divisor_keys)

all_algos_ics = [search(ic,algo_spec) for algo_spec in algo_specs_full]


#Plot Sensitivity of eta.
p = plot()
for algo in all_algos_ics
    eta_vals = diff(algo)["eta"]

    line = []
    err_bars = []
    for eta in eta_vals
        algo_for_eta = search(algo, Dict("eta" => eta))
        best_algo_for_eta = GPU.get_best_final_perf(algo_for_eta, ["alpha_init"], data_key,0.1)

        data = GPU.load_results(best_algo_for_eta,data_key)

        # rmse,rmse_std = GPU.get_stats(data)
        rmse = mean(sum(data, dims = 1),dims = [2,3])[]
        rmse_std = std(mean(data, dims = [1,2]),dims = 3)[]

        num_runs = size(data)[3]

        CI = 1.97 * rmse_std / sqrt(num_runs)

        push!(line, mean(rmse))
        push!(err_bars, CI)
    end
    plot!(line, yerror=err_bars)
end
display(p)
