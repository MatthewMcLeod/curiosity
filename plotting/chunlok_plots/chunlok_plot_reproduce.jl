using Plots
using Reproduce
using JLD2
using FileIO
using Statistics

ic = ItemCollection("./TMazeDrifterDIstractor")

println(diff(ic))

new_ic = search(ic, Dict(
    "demon_alpha" => 0.5,
    "lambda" => 0.9
))

println(diff(new_ic))

# algos_ics = [search(ic,algo_divisor) for algo_divisor in algo_divisors]
# sweep_params = ["demon_alpha", "lambda"]

function load_results(ic, logger_key; return_type = "tensor")
    num_results = length(ic)
    results = []
    for itm in ic.items
        data = FileIO.load(joinpath(itm.folder_str, "results.jld2"))["results"]
        push!(results,data[logger_key])
    end
    
    if return_type == "tensor"
        return cat(results..., dims = 3)
    elseif return_type == "array"
        return results
    end
end

data = load_results(new_ic, :ttmaze_error)

println(size(data))

p = plot()
for i in 1:4
    # if i == 3
    #     continue
    # end
    plot!(p, 1:400, data[i, :, 1])
end
savefig("plotting/chunlok_plots/generated_plots/test_plot_reproduce.svg")
# asdasdasd


# # After implementing, I think I saw there is a Iterators.product that does this?
# function combine(dict_arr, key, vals)
#     new_dict_arr = []
#     for dict in dict_arr
#         for val in vals
#             new_dict = deepcopy(Dict(dict))
#             new_dict[key] = val
#             push!(new_dict_arr, new_dict)
#         end
#     end
#     return new_dict_arr
# end

# function split_algo(algo_ic, swept_params)
#     println(swept_params)
#     dicts = [Dict()]
#     for k in swept_params
#         list_of_params = diff(algo_ic)[k]
#         dicts = combine(dicts, k, list_of_params)
#     end
#     return dicts
# end

# function get_best(ic, sweep_params, metric)
#     splits = split_algo(ic,sweep_params)
#     errors = ones(length(splits)) * Inf
#     for (ind, split) in enumerate(splits)
#         res = load_results(search(ic, split), metric)
#         error = mean(res)
#         errors[ind] = error
#     end
#     low_err, low_err_ind = findmin(errors)
    
#     print(errors)

#     return search(ic, splits[low_err_ind])
# end

# best_ic = get_best(ic, sweep_params, :ttmaze_error)

# function print_params(ic, algo_params, sweep_params)
#     println("For IC: ")
#     for p in algo_params
#         println(p,"  ", ic[1].parsed_args[p])
#     end
#     for p in sweep_params
#         println(p,"  ", ic[1].parsed_args[p])
#     end
#     println()
# end


# print(best_ic[1])

# data = load_results(best_ic,:ttmaze_error)
# # size should be (4,400,30)
# p = plot()

# for i in 1:2
#     plot!(p, 1:400, data[i, :, 1])
# end
# savefig("plotting/chunlok_plots/generated_plots/test_plot_reproduce.svg")

# print_params(best_ic, sweep_params, )


# best_per_algo_ics = [get_best(algo_ic,sweep_params, :mc_error) for algo_ic in algos_ics]


# p = plot()

# for i in ic
#     results_file = i.folder_str * "/results.jld2"
#     @load results_file results`
#     println(results)
#     error = results[:ttmaze_error]
    
#     print(size(error))

#     for i in 1:4
#         println(size(error[i, :]))
#         plot!(p, 1:40, error[i, :])
#     end
#     # 
#     savefig("plotting/chunlok_plots/generated_plots/test_plot.svg")
#     # println(i.key)
# end