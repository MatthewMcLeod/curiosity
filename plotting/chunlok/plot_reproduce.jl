using Plots
using Reproduce
using JLD2
using FileIO
using Statistics

include("plot_utils.jl")

num_steps = 4000
log_interval = 10


function get_lines(data)
    means = []
    step = 10
    for start in 1:step:size(data)[2]
        print()
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


    xs = 1:log_interval*step:num_steps + 1
    println(size(xs))
    println(size(mean_line))
    println(size(std_err_line))
    return xs, mean_line, std_err_line
end

sweep_params = ["demon_eta"]


p = plot()


ic = ItemCollection("experiment_data/EmphaticTest")

# EmphaticTest
print(diff(ic))


function plot_line(ic, demon_update_algo)
    filtered_ic = search(ic, Dict("demon_update" => "TD"))

    best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error)
    print_params(best_ic, sweep_params, [])
    
    data = load_results(best_ic, :ttmaze_uniform_error)
    
    xs, mean_line, std_err_line = get_lines(data)
    plot!(p, xs, mean_line, ribbons=std_err_line, label="TD")
end


savefig("plotting/chunlok/generated_plots/test_plot_reproduce.svg")

sdfsdf
# best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error)
# print_params(best_ic, sweep_params, [])

# best_ic = get_best(ic, sweep_params, :ttmaze_uniform_error, (0.9, 1))
# print_params(best_ic, sweep_params, [])

# best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.4, "behaviour_eta" => 0.5))
best_ic = search(ic, Dict("demon_eta" => 0.5, "exploration_param" => 0.1, "behaviour_eta" => 0.0625))

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