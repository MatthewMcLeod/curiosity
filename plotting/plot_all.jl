using Plots; gr()
using Reproduce
using FileIO
using Statistics
using ProgressMeter
using JLD2

# data_home = "../Experiment2_d_pi"
# data_home = "../data/OneDTMaze_RR_dpi"
# data_home = "../data/OneDTMaze_Control_dpi"
# data_home = "../OneDTMaze_RR_dpi"
data_home = "../OneDTMaze_Control_dpi_new_method_for_reward_features"

include("./plot_utils.jl")
GPU = GeneralPlotUtils

experiment_folders = [data_home]
# folder_name = "oned_rr_dpi"
# folder_name = "oned_control_dpi"
folder_name = "tmp"

data_key = :oned_tmaze_dpi_error

ic = ItemCollection(joinpath(experiment_folders[1], "data"));


algo_divisor_keys = ["behaviour_learner", "demon_learner", "demon_opt"]
sweep_params = ["demon_alpha_init", "demon_eta"]

algo_specs_full = GPU.split_algo(ic, algo_divisor_keys)

all_algos_ics = [search(ic,algo_spec) for algo_spec in algo_specs_full]
@show length.(all_algos_ics)

valid_algos_ind = findall(a -> length(a) != 0, all_algos_ics)

algo_specs = algo_specs_full[valid_algos_ind]
algo_ics = all_algos_ics[valid_algos_ind]
println()
@show algo_specs

best_per_algo_ics = []
for (i,algo_ic) in enumerate(algo_ics)
#     push!(best_per_algo_ics, GPU.get_best(algo_ic,sweep_params, data_key))
    push!(best_per_algo_ics, GPU.get_best_final_perf(algo_ic,sweep_params, data_key, 0.1))

end
@show length.(best_per_algo_ics), "better be equal to num seeds"
num_runs = length.(best_per_algo_ics)[1]
@show num_runs

data = [GPU.smooth(GPU.get_stats(GPU.load_results(ic,data_key))[1],10) for ic in best_per_algo_ics]
std = [GPU.smooth(GPU.get_stats(GPU.load_results(ic,data_key))[2],10) for ic in best_per_algo_ics]

[GPU.print_params(algo, algo_divisor_keys, sweep_params) for algo in best_per_algo_ics]
label_keys = cat(algo_divisor_keys,sweep_params, dims = 1)
labels = [GPU.get_label(algo, label_keys) for algo in best_per_algo_ics]
labels = cat(labels..., dims=2)
string(data_key)

####
# Generate Average demon RMSE
ylabel = "RMSE"
# title = "SR Demons & Step Size Adaptation vs More Naive Approaches"
title = "RMSE"
step_increment=50
num_samples = length(data[1])
xticks=collect(1:step_increment:num_samples*step_increment)
plot(xticks, data, ylabel=ylabel, palette=:tab10, label= labels, grid=true, ribbon = std/sqrt(num_runs), legend=:topright, title=title)
savefig("./plots/$(folder_name)/RMSE_$(string(data_key)).png")

####
# Generate graph for RMSE per demon

data_per_gvf = [GPU.get_stats(GPU.load_results(ic,data_key), per_gvf = true)[1] for ic in best_per_algo_ics]
std_per_gvf = [GPU.get_stats(GPU.load_results(ic,data_key), per_gvf = true)[2] for ic in best_per_algo_ics]
num_gvfs = 4
ps = []
gvf_labels = ["Distractor" "Constant" "Drifter" "Constant"]
for gvf_ind in 1:num_gvfs
    p = plot()
    for algo_ind in 1:length(data_per_gvf)
        smooth_gvf = GPU.smooth(data_per_gvf[algo_ind][gvf_ind,:],5)
        plot!(p,smooth_gvf, palette=:tab10, ribbon = std_per_gvf[algo_ind][gvf_ind,:] / sqrt(num_runs), size = (500,500),label = labels[algo_ind])
        plot!(xlabel="Steps", ylabel = "RMSE",  title = string(gvf_labels[gvf_ind], ""))
    end
    push!(ps,p)
end
plot(ps..., layout=(2,2), size = (1200,1200))
savefig("./plots/$(folder_name)/RMSE_per_Demon_$(string(data_key)).png")

# Generate Goal visitation
episode_lengths = [GPU.load_results(ic,:episode_length, return_type = "array") for ic in best_per_algo_ics]
visit_orders = [GPU.load_results(ic,:visit_order, return_type = "array") for ic in best_per_algo_ics]
ps = []
for ind in 1:length(episode_lengths)
    num_episodes = [length(e) for e in episode_lengths[ind]]
    p = histogram(num_episodes; nbins=15, xlim=(0,3000))

    title!(labels[ind])
    push!(ps,p)
    xlabel!("Number of Episodes")
    ylabel!("Number of runs")
end

plot(ps..., size = (1000,1000))
savefig("./plots/$(folder_name)/RunSuccessRate_ALL_$(string(data_key)).png")

#### Generate
max_lengths = [GPU.get_min_length(arrs)-1 for arrs in episode_lengths]
p = plot(xlabel = "Episode Count", ylabel = "Step Length")
for (ind,episode_lengths) in enumerate(episode_lengths)
    l = hcat([epi[1:max_lengths[ind]] for epi in episode_lengths]...)
    plot!(p,mean(l,dims=2), label = labels[ind],yaxis=:log)
end
title!("Step Length for Best Performing Algos Last 10 %")
# p = plot(mean(episode_lengths,dims=2), xlabel="Episode Count", ylabel="Step Length", hline=5.5)
display(p)
savefig("./plots/$(folder_name)/step_length.png")

####
ps = []
for (algo_ind,algo) in enumerate(episode_lengths)
    p = plot()
    num_runs = length(algo)
    num_steps = cumsum(algo[1])[end]
    d = zeros(num_runs, num_steps)
    for (i,run) in enumerate(algo)
        cur_index = 1
        for (ep_num,ep) in enumerate(run)
            d[i,cur_index:cur_index + ep-1] .= ep
            cur_index += ep
        end
            plot!(d[i,:],color=:black, linealpha=0.4, label="")
    end
    plot!(mean(d,dims=1)', label = labels[algo_ind], yaxis=:log, size=(1000,1000), legend=:bottomleft)
    xlabel!("Step")
    ylabel!("Ave Ep Length at Step")
    push!(ps, p)

end
plot(ps...)
savefig("./plots/$(folder_name)/ep_length_per_step_$(string(data_key)).png")

#####
# Goal Visitation
ps = []
for i in 1:length(visit_orders)
    @show max_lengths[i]
    @show length(visit_orders[i])
    visit_perc = GPU.goal_visits_per_episode(visit_orders[i], max_lengths[i])
    @show size(visit_perc)
    tmp = [GPU.smooth(visit_perc[i,:],20) for i in 1:4]
    p = plot(tmp, labels = gvf_labels, xlabel="Episode Count", ylabel="Fraction of Goal Visits", ylim=(0.0,1.0), title = string(labels[i], " Last 10%"), xaxis=:log, size = (800,1000), legend=:topleft)
    push!(ps,p)
end
println()
@show length(visit_orders)
plot(ps..., layout=(Int(length(best_per_algo_ics)/2),2))
savefig("./plots/$(folder_name)/goal_visits_$(string(data_key)).png")

# Median Goal Visitation
tmp_lengths = [length.(visits) for visits in visit_orders]
med_visit_order_lengths = Int.(floor.([median(l) for l in tmp_lengths]))
med_visit_mask = [tmp_lengths[i] .> med_visit_order_lengths[i] for i in 1:length(visit_orders)]
ps = []

for i in 1:length(visit_orders)
    visit_perc = GPU.goal_visits_per_episode(visit_orders[i][med_visit_mask[i]], med_visit_order_lengths[i])
    tmp = [GPU.smooth(visit_perc[i,:],100) for i in 1:4]
    p = plot(tmp, labels = gvf_labels, xlabel="Episode Count", ylabel="Fraction of Goal Visits", ylim=(0.0,1.0), title = string("MEDIAN",labels[i], " Last 10%"), size = (800,1000), legend=:topleft)
    push!(ps,p)
end
println()
@show length(visit_orders)
display(plot(ps..., layout=(Int(length(best_per_algo_ics)/2),2)))
savefig("./plots/$(folder_name)/goal_visits_MEDIAN_$(string(data_key)).png")
