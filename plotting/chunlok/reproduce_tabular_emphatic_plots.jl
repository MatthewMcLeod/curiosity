using Plots
using Reproduce
using JLD2
using FileIO
using Statistics

include("plot_utils.jl")

num_steps = 60000
log_interval = 100


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

sweep_params = ["eta", "alpha_init"]


p = plot()


ic = ItemCollection("M:/globus/OneDTMaze_Emphatic_Control_Real")
# ic = ItemCollection("experiment_data/EmphaticTest")
# ic = search(ic, Dict("demon_learner" => "SR"))

demon_learner = "SR"
ic = search(ic, Dict("demon_learner" => demon_learner))
# EmphaticTest
print(diff(ic))

# sdfsdf

metric = :oned_tmaze_start_error
function plot_line(ic, demon_update_algo)
    println(diff(ic))
    filtered_ic = search(ic, Dict("demon_update" => demon_update_algo))
    println(diff(filtered_ic))

    best_ic = get_best(filtered_ic, sweep_params, metric)
    print_params(best_ic, sweep_params, [])
    
    data = load_results(best_ic, metric)
    println(size(data))

    xs, mean_line, std_err_line = get_lines(data)
    plot!(p, xs, mean_line, ribbons=std_err_line, label=demon_update_algo, ylabel="RMSE", xlabel="time step", ylim=(0,15), title=demon_learner)
end

plot_line(ic, "ESARSA")
plot_line(ic, "EmphESARSA")
plot_line(ic, "TB")
plot_line(ic, "ETB")
plot_line(ic, "PriorESARSA")
plot_line(ic, "PriorTB")

savefig("plotting/chunlok/generated_plots/test_plot_reproduce.svg")