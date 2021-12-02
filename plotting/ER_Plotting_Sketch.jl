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
include("./plotting.jl")

function load_data(data_home)
    experiment_folders = [data_home]
    ic = ItemCollection(joinpath(experiment_folders[1], "data"))
    return ic
end

function load_best(ic)
    algo_divisor_keys = ["behaviour_learner", "demon_learner", "demon_opt", "demon_update"]
    sweep_params = ["alpha_init", "eta", "alpha_init","batch_size"]
    # data_key = :oned_tmaze_dmu_error
    data_key = :oned_tmaze_old_error

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

SR_ER_ic_all = load_data("../data/ER/OneDTMaze_RR_ER/")
SR_ER_ic_all = search(SR_ER_ic_all, Dict("batch_size" => 8))
SR_ER_ic = load_best(SR_ER_ic_all)

Q_ER_ic_all = load_data("../data/ER/OneDTMaze_RR_ER_2/")
Q_ER_ic_all = search(Q_ER_ic_all, Dict("batch_size" => 8))
Q_ER_ic = load_best(Q_ER_ic_all)

SR_ic_all = load_data("../data/OneDTMaze_RR/")
SR_ic_all = search(SR_ic_all, Dict("demon_learner" => "SR"))
SR_ic = load_best(SR_ic_all)

Q_ic_all = load_data("../data/OneDTMaze_RR/")
Q_ic_all = search(Q_ic_all, Dict("demon_learner" => "Q"))
Q_ic = load_best(Q_ic_all)

LSTD_ic_all = load_data("../data/OneDTMaze_RR/")
LSTD_ic_all = search(LSTD_ic_all, Dict("demon_learner" => "LSTD"))
LSTD_ic = load_best(LSTD_ic_all)

@show length(SR_ER_ic)
@show length(Q_ER_ic)
@show length(SR_ic)
@show length(Q_ic)
@show length(LSTD_ic)

all_ic = [
    SR_ER_ic[1],
    Q_ER_ic[1],
    SR_ic[1],
    Q_ic[1],
    LSTD_ic[1],
]
