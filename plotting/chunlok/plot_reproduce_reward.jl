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
ic = ItemCollection("./OneDTMaze_NEW_DYNAMICS_NEW_EVAL/")
ic = search(ic, Dict("steps" => 60000))

print(diff(ic))

etb_ic = search(ic, Dict("demon_update" => "ETB"))

data = load_results(etb_ic, :oned_tmaze_start_error)

xs, mean_line, std_err_line = get_lines(data)
plot!(p, mean_line, ribbons=std_err_line, label="ETB")



tb_ic = search(ic, Dict("demon_update" => "TB"))

data = load_results(tb_ic, :oned_tmaze_start_error)

xs, mean_line, std_err_line = get_lines(data)
plot!(p, mean_line, ribbons=std_err_line, label="TB")







savefig("plotting/chunlok/generated_plots/test_plot_reproduce.svg")