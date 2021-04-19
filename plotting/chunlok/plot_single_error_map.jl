using Plots
using Reproduce
using JLD2
using FileIO

p = plot()
# folder = "TabularTMazeExperiment/RP_0_0x278a9a4b9fa34c2b/"
folder = "TabularTMazeExperiment/RP_0_0xf91eb689cd1d55f6/"
save_file = "plotting/chunlok/generated_plots/test_plot_error_map"


results_file = folder * "results.jld2"
@load results_file results
# println(results)

settings_file = folder * "settings.jld2"
println(settings_file)
settings = FileIO.load(settings_file)["parsed_args"]

# print(settings)
# print(sett)

error_map_data = results[:ttmaze_error_map]
eval_set = results[:ttmaze_error_map_eval_set]
all_q_ests = results[:ttmaze_error_map_est]

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


state_map = valid_state_mask()

# @gif for time_step in 2:40
    # Last error map of the 3rd GVF (drifter)
    time_step = 40
    errors = error_map_data[time_step]
    q_ests = all_q_ests[time_step]

    # println(keys(eval_set))

    true_values = eval_set["ests"]
    actions = eval_set["actions"]

    imgs = []
    for action in 1:4
        error_map = zeros(9, 7)
        for (ind, obs) in enumerate(eval_set["states"])

            index_state = convert(Int32, obs[1])
            state = state_map[index_state]

            # println(size(ests))
            # println(size(data))


            error = errors[:, ind]
            true_value = true_values[:, ind]
            q_est = q_ests[:, ind]

            # println(size(q_ests))

            # println(size(error))
            gvf = 4
            # println(q_est)

            if (actions[ind] == action)
                error_map[state] += error[gvf]
                # error_map[state] += q_est[gvf]
                # error_map[state] += true_value[gvf]
            end
        end

        action_names = ["up", "right", "down", "left"]
        push!(imgs, heatmap(error_map, yflip=true, title="$(action_names[action]) $(time_step)"))
    end
    plot(imgs..., layout=4)
# end

# heatmap!(hcat(imgs...), yflip=true, layout=4)

savefig(save_file)