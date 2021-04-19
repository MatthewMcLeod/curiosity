using Plots
using Reproduce
using JLD2
using FileIO
using Statistics

include("plot_utils.jl")

function get_lines(data)
    means = []
    step = 10
    for start in 1:step:size(data)[2]
        chunk = data[:, start:min(start + step - 1, length(data)),: ]
        # println(size(chunk))
        chunk_mean = mean(chunk, dims=2)
        push!(means, chunk_mean)
    end

    smoothed = cat(means..., dims=2)[:,:, :]
    sum_lines = sum(smoothed, dims=1)
    num_runs = size(sum_lines)[3]
    mean_line = dropdims(mean(sum_lines, dims=3), dims=(1, 3))
    std_err_line = dropdims(std(sum_lines, dims=3), dims=(1, 3)) / sqrt(num_runs)


    xs = 1:100*step:30000
    return xs, mean_line, std_err_line
end


# There's duplicate copy of me in get_best, please replace me later.
function get_performance(ic, metric, percentage=(0, 1))
    res = load_results(ic, metric)

    if length(size(res)) == 3
        num_log = size(res)[2]
        start_index = floor(Int, num_log * percentage[1])
        if (start_index == 0)
            start_index = 1
        end
        end_index = ceil(Int, num_log * percentage[2])
        res = res[:, start_index:end_index, :]
    end

    means = dropdims(mean(sum(res, dims=1), dims=2), dims=(1, 2))
    std_err = std(means) / sqrt(length(means))
    return mean(means), std_err
end


num_steps = 30000
log_interval = 100


sweep_params = ["demon_eta", "exploration_param", "behaviour_eta"]


sens_key = "behaviour_eta"
sens_sweep_params = filter(x -> x != sens_key, sweep_params)
p = plot()

ic = ItemCollection("M:/globus/Experiment2/Experiment2_GPI/")

sens_values = diff(ic)[sens_key]
println(sens_values)

perfs = []
stds = []
for sens_val in sens_values
    sens_ic = search(ic, Dict(sens_key => sens_val))
    best_sens_ic = get_best(sens_ic, sens_sweep_params, :ttmaze_uniform_error)
    perf, std = get_performance(best_sens_ic, :ttmaze_uniform_error)
    push!(perfs, perf)
    push!(stds, std)
end

println(sens_values)
println(perfs)
println(stds)

# demon_eta
# perfs = [0.36069585419440175, 0.340265883367322, 0.32440065133391177, 0.43664044947316205]
# stds = [0.009840086433933035, 0.012308683479898372, 0.00966571250650264, 0.029534894799327426]


plot!(p, sens_values, perfs, yerror=stds, label="GPI", xlabel="behaviour step size")



ic = ItemCollection("M:/globus/Experiment2/Experiment2_ESarsa_NoLimit")
sens_values = diff(ic)[sens_key]
println(sens_values)
perfs = []
stds = []
for sens_val in sens_values
    sens_ic = search(ic, Dict(sens_key => sens_val))
    best_sens_ic = get_best(sens_ic, sens_sweep_params, :ttmaze_uniform_error)
    perf, std = get_performance(best_sens_ic, :ttmaze_uniform_error)
    push!(perfs, perf)
    push!(stds, std)
end

# println(sens_values)
# println(perfs)
# println(stds)

# demon eta
# perfs = [0.5190128725436324, 0.4815899839752351, 0.436221693493747, 0.42818578520772327, 0.40281394394567965, 0.38377415931803205, 0.7014830158663732]
# std = [0.01063201541960759, 0.010213129582727696, 0.009209626454773388, 0.009358865668267153, 0.007108921837893297, 0.009171313780882599, 0.23426893167605065]

plot!(p, sens_values, perfs, yerror=stds, label="ESARSA")





savefig("plotting/chunlok/generated_plots/sens_plot_reproduce_$(sens_key).svg")


asdasd

# best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error)
# print_params(best_ic, sweep_params, [])

# best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error, (0.9, 1))
# print_params(best_ic, sweep_params, [])

# best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.4, "behaviour_eta" => 0.5))
# best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.1, "behaviour_eta" => 0.0625))

print_params(best_ic, sweep_params, [])



data = load_results(best_ic, :ttmaze_uniform_error)

xs, mean_line, std_err_line = get_lines(data)
plot!(p, xs, mean_line, ribbons=std_err_line, label="GPI")




ic = ItemCollection("M:/globus/Experiment2/Experiment2_ESarsa_NoLimit")

println(diff(ic))

# best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error)
# print_params(best_ic, sweep_params, [])
# best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error, (0.9, 1))
# print_params(best_ic, sweep_params, [])


#AUC
# best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.4, "behaviour_eta" => 0.25))
#AUC last 10%
best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.3, "behaviour_eta" => 0.03125))


# Old Params
# best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.1, "behaviour_eta" => 0.0625))


data = load_results(best_ic, :ttmaze_uniform_error)

println(size(data))

xs, mean_line, std_err_line = get_lines(data)
plot!(p, xs, mean_line, ribbons=std_err_line, label="ESarsa")






savefig("plotting/chunlok/generated_plots/test_plot_reproduce.svg")