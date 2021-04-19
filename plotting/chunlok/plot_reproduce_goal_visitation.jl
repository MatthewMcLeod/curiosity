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

num_steps = 30000
log_interval = 100


save_file = "plotting/chunlok/generated_plots/reproduce_plot_visitation.svg"
p = plot()


ic = ItemCollection("./Test_Large_Distractor_GPI/")

println(diff(ic))

data = load_results(ic, :visit_order, return_type="array")

min_length = min([length(x) for x in data]...)

println(min_length)
# Trimming extra episodes for plotting sake
data = [x[1:min_length] for x in data]

data = cat(data..., dims = 2)
runs = size(data)[2]

percentages = []
for r in 1:runs
    percentage = get_single_goal_percentage(data[:,r]; step_size=2)
    push!(percentages, percentage)
end

percentages = cat(percentages..., dims = 3)
percentages = dropdims(mean(percentages, dims=3), dims=3)
println(size(percentages))




# sdfsdfsdf
for i in 1:4
    println(size(percentages[i, :]))
    plot!(p, 1:size(percentages)[2], percentages[i, :])
end

# print(error[3,10])
savefig(save_file)