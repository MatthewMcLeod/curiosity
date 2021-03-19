module PlotUtils

# import ..TabularTMazeUtils
using FileIO
using JLD2

function build_tabular_tmaze_heatmap(itm)
    values = load_result(itm, :value_map)
    map_set = @load "../src/data/TTMazeValueSet.jld2" ValueSet

    num_steps = size(values)[1]

    value_heatmap = zeros(num_steps, 9, 7)

    # position_mapping = TabularTMazeUtils.valid_state_mask()
    function valid_state_mask()
        """
        return a mask of valid states that is 9x7
        """
        world = [["G1", "0", "0", "0", "0", "0", "G3"],
                          ["1", "0", "0", "0", "0", "0", "1"],
                          ["1", "1", "1", "1", "1", "1", "1"],
                          ["1", "0", "0", "1", "0", "0", "1"],
                          ["G2", "0", "0", "1", "0", "0", "G4"],
                          ["0", "0", "0", "1", "0", "0", "0"],
                          ["0", "0", "0", "1", "0", "0", "0"],
                          ["0", "0", "0", "1", "0", "0", "0"],
                          ["0", "0", "0", "1", "0", "0", "0"]]

        world = permutedims(hcat(world...))
        valid_states = findall(x-> x!="0", world)
        return valid_states
    end
    position_mapping = valid_state_mask()

    for i in 1:num_steps

        for (set_ind,obs) in enumerate(ValueSet["states"])
            heatmap_ind = position_mapping[Int(obs[1])]
            value_heatmap[i,heatmap_ind] = values[i,set_ind,1]
        end
    end
    return value_heatmap
end

"""
    Given an ItemCollection, returns all logs at the logger_key
"""
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

function load_result(itm, logger_key)
    data = FileIO.load(joinpath(itm.folder_str, "results.jld2"))["results"]
    return data[logger_key]
end

end
