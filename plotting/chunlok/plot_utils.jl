# # After implementing, I think I saw there is a Iterators.product that does this?
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
        list_of_params = diff(algo_ic)[k]
        dicts = combine(dicts, k, list_of_params)
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
    
    print(errors)

    return search(ic, splits[low_err_ind])
end

function print_params(ic, algo_params, sweep_params)
    println("For IC: ")
    for p in algo_params
        println(p,"  ", ic[1].parsed_args[p])
    end
    for p in sweep_params
        println(p,"  ", ic[1].parsed_args[p])
    end
    println()
end


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