module GeneralPlotUtils
using Reproduce
using FileIO
using Statistics
using ProgressMeter
using JLD2
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

function combine(dict_arr, key, vals)
    new_dict_arr = []
    for dict in dict_arr
        for val in vals
            new_dict = deepcopy(Dict(dict))
            new_dict[key] = val
            push!(new_dict_arr, new_dict)
        end
    end
    return new_dict_arr
end

function split_algo(algo_ic, swept_params)
    println(swept_params)
    dicts = [Dict()]
    for k in swept_params
        if k in keys(diff(algo_ic))
            list_of_params = diff(algo_ic)[k]
            dicts = combine(dicts, k, list_of_params)
        end
    end
    return dicts
end

function get_best(ic, sweep_params, metric)
    splits = split_algo(ic,sweep_params)
    errors = ones(length(splits)) * Inf
    for (ind, split) in enumerate(splits)
        res = load_results(search(ic, split), metric)
        error = mean(res)
        errors[ind] = error
    end
    low_err, low_err_ind = findmin(errors)
    println(errors)

    return search(ic, splits[low_err_ind])
end

function get_stats(data;per_gvf=false)
    mean_per_gvf, std_per_gvf = mean(data,dims=3)[:,:,1], std(data,dims=3)[:,:,1]
    return a,b = if per_gvf == true
        mean_per_gvf, std_per_gvf
    else
        mean(mean_per_gvf,dims=1)[1,:], mean(std_per_gvf,dims=1)[1,:]
    end
end


function print_params(ic, algo_params, sweep_params)
    println("For IC: ")
    for p in algo_params
        if p in keys(ic[1].parsed_args)
            println(p,"  ", ic[1].parsed_args[p])
        end
    end
    for p in sweep_params
        if p in keys(ic[1].parsed_args)
            println(p,"  ", ic[1].parsed_args[p])
        end
    end
    println()
end

function smooth(data, k)
    smoothed_data = zeros(size(data))
    for i = 1:size(data, 1)
        if i < k
            smoothed_data[i] = mean(data[1:i])
        else
            smoothed_data[i] = mean(data[i - k + 1:i])
        end
    end
    return smoothed_data
end

end
