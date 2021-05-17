using Plots
using Reproduce
using JLD2
using FileIO
using Statistics

include("plot_utils.jl")

function plot_ic_goal_visitation!(p, ic; step_size=2)
    data = load_results(ic, :visit_order, return_type="array")

    # optimal filtering of the median. This is NOT YET COMPLETE. Still working on it.
    # median_length = median([length(x) for x in data])
    # min_length = min([length(x) for x in data]...)
    # println("min: $(min_length) median: $(median([length(x) for x in data]))")


    # Trimming extra episodes for plotting sake
    min_length = min([length(x) for x in data]...)
    data = [x[1:min_length] for x in data]
    
    data = cat(data..., dims = 2)
    runs = size(data)[2]
    
    percentages = []
    for r in 1:runs
        percentage = get_single_goal_percentage(data[:,r]; step_size=step_size)
        push!(percentages, percentage)
    end
    
    percentages = cat(percentages..., dims = 3)
    percentages = dropdims(mean(percentages, dims=3), dims=3)

    for i in 1:4
        # println(size(percentages[i, :]))
        plot!(p, 1:size(percentages)[2], percentages[i, :])
    end
end