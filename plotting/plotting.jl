# include("./plot_utils.jl")
# GPU = GeneralPlotUtils
# LU = LabelUtils
# include("./plot_utils.jl")
# GPU = GeneralPlotUtils
# LU = LabelUtils

module Plotting
# using Plots; gr()
using Plots; pyplot()
using Reproduce
using FileIO
using Statistics
using ProgressMeter
using JLD2
using LaTeXStrings
using StatsPlots
# Plots.scalefontsizes(2)
# import ..GeneralPlotUtils
# import ..LabelUtils
# include("./plot_utils.jl")
include("./plot_utils.jl")
GPU = GeneralPlotUtils
LU = LabelUtils

 # default(titlefont = (20, "times"), legendfontsize = 18, guidefont = (18, :darkgreen), tickfont = (12, :orange), guide = "x", framestyle = :zerolines, yminorgrid = true)
 default(titlefont = (20, "times"), legendfontsize = 10, guidefont = (18, :black), tickfont = 12, foreground_color_legend = nothing)


# data_key = :ttmaze_direct_error
# data_key = :ttmaze_uniform_error
# data_key = :ttmaze_round_robin_error
# data_key = :oned_tmaze_dpi_error
# data_key = :oned_tmaze_old_error
# data_key = :oned_tmaze_dmu_error
# data_key = :mc_uniform_error
# data_key = :mc_starts_error
data_key = :twod_grid_world_error_center_dpi

# folder_name = "oned_control"
# folder_name = "oned_rr"
# folder_name = "GPI_Sensitivity"
# folder_name = "NoLegend"
# folder_name = "tabular_control"
# folder_name = "MC_tmp"
# folder_name = "twod_gpi_tb"
folder_name = "twod_gpi_no_optimism"

# data_home = "../data/Experiment1"
# data_home = "../data/Experiment2_d_pi"
# data_home = "../data/OneDTMaze_Control"
# data_home = "../data/OneDTMaze_RR"
# data_home = "../data/GPI_Sensitivity"
# data_home = "../data/OneDTMaze_GPI_Sensitivity"
# data_home = "../data/MC_Experiments"
# data_home = "../data/MC_Experiments_Final"
# data_home = "../data/TwoDGridWorld_gpi"
# data_home = "../data/TwoDGridWorld_gpi_TB"
data_home = "../data/TwoDGridWorld_gpi_no_optimism"


function load_data()
    experiment_folders = [data_home]
    # folder_name = "oned_rr_dpi"
    # folder_name = "oned_control_dpi"
    ic = ItemCollection(joinpath(experiment_folders[1], "data"))
    # ic = search(ic,Dict("behaviour_learner" => "GPI", "demon_learner" => "SR"))
    return ic
end

function load_best(ic)
# data_key = :oned_tmaze_dpi_error

    algo_divisor_keys = ["behaviour_learner", "demon_learner", "demon_opt", "demon_update", "exploration_param"]
    # algo_divisor_keys = ["behaviour_learner", "demon_learner", "demon_opt", "demon_update","behaviour_reward_projector","behaviour_rp_tilings"]
    # sweep_params = ["demon_alpha_init", "demon_eta", "alpha_init"]

    sweep_params = ["behaviour_eta", "demon_eta", "alpha_init", "eta","emphasis_clip_threshold"]

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
            # label = LU.get_label(best_per_algo_ics[algo_ind])[:label]
            # label = LU.get_params(best_per_algo_ics[algo_ind])[:label]
            # color = LU.get_params(best_per_algo_ics[algo_ind])[:color]
            plot_params = LU.get_params(best_per_algo_ics[algo_ind])
            plot!(p,smooth_gvf, ribbon = std_per_gvf[algo_ind][gvf_ind,:] / sqrt(num_runs), size = (500,500); plot_params...)
            plot!(xlabel="Steps", ylabel = "RMSE",  title = string(gvf_labels[gvf_ind], ""))
        end
        push!(ps,p)
    end
    allp = plot(ps..., layout=(2,2), size = (1200,1200))
    display(allp)
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
    # std = [GPU.get_stats(GPU.load_results(ic,data_key))[2] for ic in best_per_algo_ics]

    ylabel = "RMSE"
    # title = "Hall Following in Tabular T-Maze"
    # title = "Control in Tabular T-Maze"
    # title = "Fixed Behaviour in 1D T-Maze"
    # title = "Control in 1D T-Maze"
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
        # @show plot_params[i]
        # if best_per_algo_ics[i].items[1].parsed_args["behaviour_reward_projector"] == "tilecoding"
        #     @show "SKETCHY LOOP. ONLY FOR GPI SENSTIVITY"
        #     if best_per_algo_ics[i].items[1].parsed_args["behaviour_rp_tilings"] == 8
        #         plot_params[i][:label] = string(plot_params[i][:label], " Tile Coding")
        #         plot_params[i][:color] =  colorant"#BBBBBB"
        #     elseif best_per_algo_ics[i].items[1].parsed_args["behaviour_rp_tilings"] == 2
        #         plot_params[i][:label] = string(plot_params[i][:label]," Low Tiling")
        #         plot_params[i][:color] =  colorant"#BBBBBB"
        #     end
        # end
        plot!(p, xticks, data[i], ribbon = std[i]/sqrt(num_runs), legend=:top; plot_params[i]...)
    end
    display(p)
    savefig("./plots/$(folder_name)/RMSE_$(string(data_key)).pdf")
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

        p = plot(tmp, labels = gvf_labels, xlabel="Episode Count", ylabel="Fraction of Goal Visits", ylim=(0.0,1.0), title = title, size = (800,1000), legend=:none)
        push!(ps,p)
    end
    println()
    @show length(visit_orders)
    # plot(ps..., layout=(Int(length(best_per_algo_ics)/2),2))
    plot(ps...)

    savefig("./plots/$(folder_name)/goal_visits_$(string(data_key)).png")
end

function load_and_plot_rmse(inds_of_interest=nothing)
    ic = load_data()
    # ic = search(ic, Dict("demon_learner" => "SR"))
    algo_ic = load_best(ic)
    # plot_rmse_per_demon(algo_ic,inds_of_interest)
    plot_rmse(algo_ic,inds_of_interest)
    # sweep_params = ["demon_alpha_init", "demon_eta", "alpha_init"]
    @show "here"
    [GPU.print_params(bic,["behaviour_learner"],["eta","demon_update","emphasis_clip_threshold"]) for bic in algo_ic]
end

function load_and_plot_rmse_per_demon(inds_of_interest=nothing)
    ic = load_data()
    # ic = search(ic, Dict("demon_learner" => "SR"))
    algo_ic = load_best(ic)
    # plot_rmse_per_demon(algo_ic,inds_of_interest)
    plot_rmse_per_demon(algo_ic,inds_of_interest)
    # sweep_params = ["demon_alpha_init", "demon_eta", "alpha_init"]
    [GPU.print_params(bic,["demon_update"],["demon_eta","alpha_init","exploration_param","emphasis_clip_threshold"]) for bic in algo_ic]
end

function load_and_plot_goal_visits(inds_of_interest=nothing)
    ic = load_data()
    algo_ic = load_best(ic)
    plot_goal_visitation(algo_ic,inds_of_interest)
end
function load_and_plot_states(inds_of_interest=nothing)
    ic = load_data()
    algo_ic = load_best(ic)
    plot_states(algo_ic,inds_of_interest)
end
function load_and_plot_episode_lengths(inds_of_interest=nothing)
    ic = load_data()
    algo_ic = load_best(ic)
    # [GPU.print_params(c,["demon_eta", "behaviour_eta"],[]) for c in algo_ic[inds_of_interest]]
    plot_episode_lengths(algo_ic,inds_of_interest)
end

function load_and_box_plot(inds_of_interest=nothing)
    ic = load_data()
    algo_ic = load_best(ic)
    # [GPU.print_params(c,["demon_eta", "behaviour_eta"],[]) for c in algo_ic[inds_of_interest]]
    boxplot(algo_ic;inds_of_interest=inds_of_interest)
end
function load_and_box_plot_back_wall(inds_of_interest=nothing)
    ic = load_data()
    algo_ic = load_best(ic)
    # [GPU.print_params(c,["demon_eta", "behaviour_eta"],[]) for c in algo_ic[inds_of_interest]]
    boxplot_backwall(algo_ic;inds_of_interest=inds_of_interest)
end
function load_and_plot_max_q(inds_of_interest=nothing)
    ic = load_data()
    algo_ic = load_best(ic)
    # [GPU.print_params(c,["demon_eta", "behaviour_eta"],[]) for c in algo_ic[inds_of_interest]]
    plot_max_q(algo_ic,inds_of_interest)
end
function plot_max_q(algo_ic, inds_of_interest)
    best_per_algo_ics = if inds_of_interest isa Nothing
        deepcopy(algo_ic)
    else
        deepcopy(algo_ic)[inds_of_interest]
    end

    max_qs = [GPU.load_results(ic,:max_behaviour_q, return_type = "array") for ic in best_per_algo_ics]
    # #### Generate
    p = plot(xlabel = "Episode Count", ylabel = "Max Q", ylim=(-3,3))
    for (ind,max_q) in enumerate(max_qs)
        l = hcat([q_run for q_run in max_q]...)
        # label = LU.get_label(best_per_algo_ics[i])[:label]
        plot_params = LU.get_params(best_per_algo_ics[ind])
        plot!(p,mean(l,dims=2); plot_params...)
    end
    savefig("./plots/$(folder_name)/max_qs_zoom$(string(data_key)).png")
    # title!("Step Length for Best Performing Algos Last 10 %")
end
function plot_episode_lengths(algo_ic, inds_of_interest)
    best_per_algo_ics = if inds_of_interest isa Nothing
        deepcopy(algo_ic)
    else
        deepcopy(algo_ic)[inds_of_interest]
    end

    episode_lengths = [GPU.load_results(ic,:episode_length, return_type = "array") for ic in best_per_algo_ics]
    # #### Generate
    max_lengths = [GPU.get_min_length(arrs)-1 for arrs in episode_lengths]
    p = plot(xlabel = "Episode Count", ylabel = "Step Length")
    for (ind,episode_lengths) in enumerate(episode_lengths)
        l = hcat([epi[1:max_lengths[ind]] for epi in episode_lengths]...)
        # label = LU.get_label(best_per_algo_ics[i])[:label]
        plot_params = LU.get_params(best_per_algo_ics[ind])
        plot!(p,mean(l,dims=2); plot_params...)
    end
    savefig("./plots/$(folder_name)/episode_lengths_$(string(data_key)).png")
    # title!("Step Length for Best Performing Algos Last 10 %")
end

function load_and_plot_meta_ss(inds_of_interest = nothing; sensitivity_param = "eta")

    ic = load_data()
    algo_divisor_keys = ["behaviour_learner", "demon_learner"]
    sweep_params = ["eta"]

    algo_specs_full = GPU.split_algo(ic, algo_divisor_keys)
    all_algos_ics = [search(ic,algo_spec) for algo_spec in algo_specs_full]
    all_algos_ics = if inds_of_interest isa Nothing
        all_algos_ics
    else
        all_algos_ics[inds_of_interest]
    end

    #Plot Sensitivity of eta.
    p = plot(ylabel="RMSE", xlabel = "Meta Step Size", xaxis=:log, legend=:topleft)
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
        plot_params = LU.get_params(algo)
        plot!(eta_vals, line, yerror=err_bars,;plot_params...)
    end
    display(p)
    savefig("./plots/$(folder_name)/meta_ss_$(string(data_key)).pdf")
end

function load_and_plot_init_ss(inds_of_interest=nothing; sensitivity_param = "alpha_init")

    ic = load_data()
    algo_divisor_keys = ["behaviour_learner", "demon_learner"]
    sweep_params = ["alpha_init"]

    algo_specs_full = GPU.split_algo(ic, algo_divisor_keys)
    all_algos_ics = [search(ic,algo_spec) for algo_spec in algo_specs_full]
    all_algos_ics = if inds_of_interest isa Nothing
        all_algos_ics
    else
        all_algos_ics[inds_of_interest]
    end    #Plot Sensitivity of eta.
    p = plot(ylabel="RMSE", xlabel = "Initial Step Size", legend=:topright)
    for algo in all_algos_ics
        eta_vals = diff(algo)["alpha_init"]
        line = []
        err_bars = []
        for eta in eta_vals
            algo_for_eta = search(algo, Dict("alpha_init" => eta))
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
        plot_params = LU.get_params(algo)
        plot!(eta_vals, line, yerror=err_bars,;plot_params...)
    end
    display(p)
    savefig("./plots/$(folder_name)/init_ss_$(string(data_key)).pdf")
end

function plot_states(algo_ic, inds_of_interest=nothing)
    best_per_algo_ics = if inds_of_interest isa Nothing
        deepcopy(algo_ic)
    else
        deepcopy(algo_ic)[inds_of_interest]
    end

    ps = []
    for algo in best_per_algo_ics
        states = hcat(GPU.load_results(algo,:states, return_type = "array")...)
        @show size(states)
        #states is 2x length of states
        p = marginalhist(states[1,:],states[2,:])
        push!(ps,p)
    end
    fullp = plot(ps...)
    display(fullp)
    savefig("./plots/$(folder_name)/marginal_hists_$(string(data_key)).png")
end

function boxplot(algo_ic; inds_of_interest = nothing)
    best_per_algo_ics = if inds_of_interest isa Nothing
        deepcopy(algo_ic)
    else
        deepcopy(algo_ic)[inds_of_interest]
    end
    p = plot(ylabel="Right Hill Pseudotermination")
    xticklabels = []
    for (i,algo) in enumerate(best_per_algo_ics)
        # episode_lengths = GPU.load_results(algo,:episode_length, return_type = "array")
        # num_episodes_per_run = [length(e) for e in episode_lengths]
        states_per_run = GPU.load_results(algo,:states, return_type = "array")
        pseudoterm_episodes_per_run = [length(findall(isone,states[1,:])) for states in states_per_run]
        num_episodes_per_run = pseudoterm_episodes_per_run
        plot_params = LU.get_params(algo)
        @show plot_params
        boxplot!(p,num_episodes_per_run; plot_params...)
        push!(xticklabels, plot_params[:label])
    end
    plot!(xticks=([1,2,3],xticklabels),legend=false,formatter=:plain)
    display(p)
    savefig("./plots/$(folder_name)/boxplot_$(string(data_key)).pdf")
end

function boxplot_backwall(algo_ic; inds_of_interest = nothing)
    best_per_algo_ics = if inds_of_interest isa Nothing
        deepcopy(algo_ic)
    else
        deepcopy(algo_ic)[inds_of_interest]
    end
    p = plot(ylabel="Left Hill Pseudotermination")
    xticklabels = []
    for (i,algo) in enumerate(best_per_algo_ics)
        states_per_run = GPU.load_results(algo,:states, return_type = "array")
        pseudoterm_episodes_per_run = [length(findall(iszero,states[1,:])) for states in states_per_run]
        @show pseudoterm_episodes_per_run
        plot_params = LU.get_params(algo)
        boxplot!(p,pseudoterm_episodes_per_run; plot_params...)
        push!(xticklabels, plot_params[:label])
    end
    plot!(xticks=([1,2,3],xticklabels),legend=false,formatter=:plain)
    display(p)
    savefig("./plots/$(folder_name)/boxplot_backwall_$(string(data_key)).pdf")
end
end
