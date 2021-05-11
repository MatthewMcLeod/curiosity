using Statistics

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

function get_best(ic, sweep_params, metric, percentage=(0, 1))
    splits = split_algo(ic,sweep_params)
    errors = ones(length(splits)) * Inf
    for (ind, split) in enumerate(splits)
        res = load_results(search(ic, split), metric)
        if length(size(res)) == 3
            num_log = size(res)[2]
            start_index = floor(Int, num_log * percentage[1])
            if (start_index == 0)
                start_index = 1
            end
            end_index = ceil(Int, num_log * percentage[2])
            
            # println("start_index $(start_index), end_index $(end_index)")
            # println(size(res))
            res = res[:, start_index:end_index, :]
        end

        error = mean(res)
        errors[ind] = error
    end
    low_err, low_err_ind = findmin(errors)

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
        file_path = joinpath(itm.folder_str, "results.jld2")
        if (isfile(file_path))
            try
                data = FileIO.load(file_path)["results"]
                push!(results,data[logger_key])
            catch e
                println("Error for file $(file_path)")
                println(e)
                continue
            end
            
        end
    end

    # println(resul)
    if (size(results)[1] == 0)
        println("empty collection")
        return [2^31]
    end
    
    if return_type == "tensor"
        return cat(results..., dims = 3)
    elseif return_type == "array"
        return results
    end
end

function single_onehot(data, num_gvfs)
    onehot_enc = zeros(num_gvfs)
    for gvf_i in 1:num_gvfs
        onehot_enc[gvf_i] += (data .== gvf_i)
    end
    return onehot_enc
end

function get_single_goal_percentage(visits; step_size=10)
    num_gvfs = 4
    m_length = length(visits)
    goal_visits = zeros(num_gvfs, ceil(Integer, m_length / step_size))

    for episode in 1:length(visits)
        index = ceil(Integer, episode / step_size)
        goal_visits[:, index] += single_onehot(visits[episode], num_gvfs)
    end
    goal_visit_percentage = goal_visits / step_size
end

function smooth_lines(data, step)
    # really simple smooth line function
    means = []
    for start in 1:step:size(data)[1]
        chunk = data[start:min(start + step - 1, size(data)[1])]
        chunk_mean = mean(chunk)
        push!(means, chunk_mean)
    end
    return means
end