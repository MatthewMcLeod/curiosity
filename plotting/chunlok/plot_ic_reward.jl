using Plots
using Reproduce
using JLD2
using FileIO
using Statistics

include("plot_utils.jl")

function plot_ic_reward!(p, ic, logger_key, log_interval; smooth_step=10, label=nothing)
    data = load_results(ic, logger_key)
    xs, mean_line, std_err_line = get_lines(data, log_interval, smooth_step)
    plot!(p, xs, mean_line, ribbons=std_err_line, label=label)
end

function plot_ic_reward_per_gvf!(p, ic, logger_key, log_interval, gvf; smooth_step=10, label=nothing)
    data = load_results(ic, logger_key)[gvf:gvf, :, :]
    # println(size(data))
    # sdfsd
    xs, mean_line, std_err_line = get_lines(data, log_interval, smooth_step)
    plot!(p, xs, mean_line, ribbons=std_err_line, label=label)
end

function get_lines(data, log_interval, smooth_step)
    means = []
    for start in 1:smooth_step:size(data)[2]
        chunk = data[:, start:min(start + smooth_step - 1, size(data)[2]),: ]
        # println(size(chunk))
        chunk_mean = mean(chunk, dims=2)
        push!(means, chunk_mean)
    end

    smoothed = cat(means..., dims=2)[:,:, :]
    sum_lines = sum(smoothed, dims=1)
    num_runs = size(sum_lines)[3]
    mean_line = dropdims(mean(sum_lines, dims=3), dims=(1, 3))
    std_err_line = dropdims(std(sum_lines, dims=3), dims=(1, 3)) / sqrt(num_runs)


    xs = 1:log_interval*smooth_step:log_interval*smooth_step*length(mean_line)
    return xs, mean_line, std_err_line
end