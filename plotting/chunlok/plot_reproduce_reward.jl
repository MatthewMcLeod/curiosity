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
        chunk = data[:, start:min(start + step - 1, size(data)[2]),: ]
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

num_steps = 30000
log_interval = 100


sweep_params = ["demon_eta", "exploration_param", "behaviour_eta"]


p = plot()


# ic = ItemCollection("M:/globus/Experiment2/Experiment2_GPI/")
ic = ItemCollection("./OneDTMaze_NEW_DYNAMICS_PLOT/")
best_ic = ic

print(diff(ic))

# best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error)
# print_params(best_ic, sweep_params, [])

# best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error, (0.9, 1))
# print_params(best_ic, sweep_params, [])

# best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.4, "behaviour_eta" => 0.5))
# best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.1, "behaviour_eta" => 0.0625))

# print_params(best_ic, sweep_params, [])



data = load_results(best_ic, :oned_tmaze_start_error)

xs, mean_line, std_err_line = get_lines(data)
plot!(p, mean_line, ribbons=std_err_line)



# # ic = ItemCollection("M:/globus/Experiment2/Experiment2_GPI/")
# ic = ItemCollection("./Test_Large_Distractor/")
# best_ic = ic

# print(diff(ic))

# # best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error)
# # print_params(best_ic, sweep_params, [])

# # best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error, (0.9, 1))
# # print_params(best_ic, sweep_params, [])

# # best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.4, "behaviour_eta" => 0.5))
# # best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.1, "behaviour_eta" => 0.0625))

# print_params(best_ic, sweep_params, [])



# data = load_results(best_ic, :ttmaze_uniform_error)

# xs, mean_line, std_err_line = get_lines(data)
# plot!(p, mean_line, ribbons=std_err_line, label="Naive")






savefig("plotting/chunlok/generated_plots/test_plot_reproduce_naive.svg")