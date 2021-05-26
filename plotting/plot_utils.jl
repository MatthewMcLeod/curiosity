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
function get_most_episodes(ic, sweep_params, metric)
    splits = split_algo(ic,sweep_params)
    lengths = -ones(length(splits)) * Inf
    for (ind, split) in enumerate(splits)
        candidate_best = search(ic, split)
        if length(candidate_best.items) == 0
            @warn "$(split) results in not a valid combination"
            continue
        end
        res = load_results(candidate_best, metric, return_type = "array")
        @show size(res)
        num_episodes = length.(res)
        lengths[ind] = mean(num_episodes)
    end
    mx,mx_ind = findmax(lengths)
    @show lengths
    return search(ic, splits[mx_ind])

end
function get_best_final_perf(ic, sweep_params, metric, cut_per)
    splits = split_algo(ic,sweep_params)
    errors = ones(length(splits)) * Inf
    for (ind, split) in enumerate(splits)
        candidate_best = search(ic, split)
        if length(candidate_best.items) == 0
            @warn "$(split) results in not a valid combination"
            continue
        end
        res = load_results(candidate_best, metric)
        num_steps = size(res)[2]
        cut_ind = Int(floor(num_steps * (1-cut_per)))
        error = mean(res[:,cut_ind:end,:])
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

        error_tot = sum(data,dims=1)[1,:,:]
        # sum(mean_per_gvf,dims=1)[1,:], sum(std_per_gvf,dims=1)[1,:]
        # @show size(std(error_tot,dims=2))
        # @show size(std(error_tot,dims=1))
        sum(mean_per_gvf,dims=1)[1,:], std(error_tot,dims=2)[:,1]
    end
end

function get_min_length(arrs)
    return minimum([length(arr) for arr in arrs])
end

function goal_visits_per_episode(arr_of_episodes, max_length; num_gvfs = 4)
    goal_visits = zeros(num_gvfs, max_length)
    for run in 1:length(arr_of_episodes)
        goal_visits += onehot(arr_of_episodes[run],num_gvfs)[:,1:max_length]
    end
    goal_visit_percentage = goal_visits / length(arr_of_episodes)
    return goal_visit_percentage
end

function onehot(data, num_gvfs = 4)
    onehot_enc = zeros(num_gvfs, length(data))
    for gvf_i in 1:num_gvfs
        onehot_enc[gvf_i,:] += (data .== gvf_i)
    end
    return onehot_enc
end

function get_label(ic, params)
    label = ""
    for p in params
        if p in keys(ic[1].parsed_args)
            label = string(label, " ", ic[1].parsed_args[p])
        end
    end
    return label
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

module LabelUtils
using Reproduce
using FileIO
using Statistics
using ProgressMeter
using JLD2
using Plots
# using LaTeXStrings

# Demon Algorithm results in different line styles!!
const val_matches = Dict(["GPI","TB","Q","TB"] => 1,
    ["GPI", "TB", "SR", "TB"] => 1,
    ["Q","ESARSA", "Q", "TB"] => 4,
    ["Q", "ESARSA", "SR", "TB"] => 4,
    ["Q", "TabularRoundRobin", "SR", "TB"] => 1,
    ["RoundRobin", "TB", "SR", "TB"] => 1,
    ["RoundRobin", "Q", "SR", "TB"] => 1,
    ["RoundRobin", "TB", "Q", "TB"] => 2,
    ["RoundRobin", "Q", "Q", "TB"] => 2,
    ["RoundRobin", "Q","LSTD","TB"] => 3,
    ["Q", "TabularRoundRobin", "Q", "TB"] => 2,
    ["Q", "TabularRoundRobin", "Q", "ESARSA"] => 2,
    ["Q", "TabularRoundRobin", "SR", "ESARSA"] => 1,
    ["Q", "TabularRoundRobin","LSTD","TB"] => 3,
    ["RandomDemons", "TB","Q","TB"] => 5,
    ["RandomDemons", "TB","SR","TB"] => 5,

    )

const algo_labels = Dict(["GPI","TB","Q","TB"] => "μ(Sarsa), π(TB)",
    ["Q","ESARSA", "Q", "TB"] => "μ(Sarsa), π(TB)",
    ["GPI", "TB", "SR", "TB"] => "μ(GPI), π(SR)",
    ["Q", "ESARSA", "SR", "TB"] => "μ(Sarsa), π(SR)",
    ["Q", "TabularRoundRobin", "SR", "TB"] => "μ(Fixed), π(SR)",
    ["Q", "TabularRoundRobin", "Q", "TB"] => "μ(Fixed), π(TB)",
    ["RoundRobin", "TB", "SR", "TB"] => "μ(Fixed), π(SR)",
    ["RoundRobin", "Q", "SR", "TB"] => "μ(Fixed), π(SR)",
    ["RoundRobin", "TB", "Q", "TB"] => "μ(Fixed), π(TB)",
    ["RoundRobin", "Q", "Q", "TB"] => "μ(Fixed), π(TB)",
    ["RoundRobin", "Q","LSTD","TB"] => "μ(Fixed), π(LSTD)",
    ["Q", "TabularRoundRobin","LSTD","TB"] => "μ(Fixed), π(LSTD)",
    ["Q", "TabularRoundRobin", "Q", "ESARSA"] => "μ(Fixed), π(TD)",
    ["Q", "TabularRoundRobin", "SR", "ESARSA"] => "μ(Fixed), π(SR + TD)",
    ["RandomDemons", "TB","Q","TB"] => "μ(Random), π(TB)",
    ["RandomDemons", "TB","SR","TB"] => "μ(Random), π(SR)",
    )

const algo_keys = ["behaviour_learner", "behaviour_update", "demon_learner", "demon_update"]
# TOL Muted Colour Scheme
# const color_scheme = [
#     colorant"#44AA99",
#     colorant"#332288",
#     colorant"#DDCC77",
    # colorant"#999933",
#     colorant"#CC6677",
#     colorant"#AA4499",
#     colorant"#DDDDDD",
# 	colorant"#117733",
# 	colorant"#882255",
# 	colorant"#88CCEE",
#     ]

const color_scheme = [
colorant"#0077BB",
colorant"#33BBEE",
colorant"#009988",
colorant"#EE7733",
colorant"#CC3311",
colorant"#EE3377",
colorant"#BBBBBB",
]

function _is_match(ic, keys, vals)
    tst = ([ic[1].parsed_args[keys[i]] == vals[i] for i in 1:length(keys)])
    return all(tst)
end
function _print_algo_keys(ic;algo_keys = algo_keys)
    println()
    [println(ic[1].parsed_args[k]) for k in algo_keys]
end
function _get_ic_ind(ic)
    ic_diff_keys = keys(diff(ic))
    for k in algo_keys
        if k in ic_diff_keys
            @show diff(ic)
            throw(ArgumentError("Multiple types of  $(k) in item collection. Unspecified what colour palette to use"))
        end
    end

    for algo_vals in keys(val_matches)
        if _is_match(ic, algo_keys, algo_vals)
            return val_matches[algo_vals]
        end
    end

    _print_algo_keys(ic)
    @warn string("No match found for: ", diff(ic))
end

function get_colour(ic)
    # Unique Colour given on a per {behaviour learner, behaviour update, demon learner, demon update} basis.
    ind = _get_ic_ind(ic)
    return Dict(:color => color_scheme[ind])
end

function get_linestyle(ic)
    #If we are in round robin  use solid linestyle
    if ic[1].parsed_args["behaviour_update"] == "TabularRoundRobin" || ic[1].parsed_args["behaviour_learner"] == "RoundRobin"
        return Dict(:linestyle => :solid)
    elseif ic[1].parsed_args["demon_learner"] == "SR"
        return Dict(:linestyle => :solid)
    elseif ic[1].parsed_args["demon_learner"] == "Q"
        return Dict(:linestyle => :dash)
    end
end

function get_label(ic)
    for algo_vals in keys(algo_labels)
        if _is_match(ic, algo_keys, algo_vals)
            return Dict(:label => algo_labels[algo_vals])
        end
    end
    throw(ArgumentError("No Label description!!"))
end

function get_params(ic)
    color = get_colour(ic)
    ls = get_linestyle(ic)
    label = get_label(ic)
    return merge(color,ls,label)
end

end
