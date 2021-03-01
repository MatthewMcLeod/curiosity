
module TabularTMazeUtils
import ..TabularTMazeCumulantSchedules
const TTMCS = TabularTMazeCumulantSchedules

const NUM_DEMONS = 4
const NUM_ACTIONS = 4

DrifterDistractor(parsed) = begin
    if "drifter" ∈ keys(parsed)
        TTMCS.DrifterDistractor(
            parsed["constant_target"],
            parsed["drifter"][1],
            parsed["drifter"][2],
            parsed["distractor"][1],
            parsed["distractor"][2])
    else
        TTMCS.DrifterDistractor(
            parsed["constant_target"],
            parsed["drifter_init"],
            parsed["drifter_std"],
            parsed["distractor_mean"],
            parsed["distractor_std"])
    end
end

function get_cumulant_schedule(parsed)
    sched = parsed["cumulant_schedule"]
    if parsed["cumulant_schedule"] == "DrifterDistractor"
        DrifterDistractor(parsed)
    elseif parsed["cumulant_schedule"] == "Constant"
        if parsed["cumulant"] isa Number
            TTMCS.Constant(parsed["cumulant"])
        else
            TTMCS.Constant(parsed["cumulant"]...)
        end
            
    else
        throw("$(sched) Not Implemented")
    end
end

function pseudoterm(state)
    term = false
    if state[1] == 1 || state[1] == 5 || state[1] == 17 || state[1] == 21
        term = true
    end
    return term
end

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

function demon_target_policy(gvf_i, observation, action)
    state = convert(Int,observation[1])

    policy_1 =  [1 0 0 0 0 0 3;
                 1 0 0 0 0 0 3;
                 1 4 4 4 4 4 4;
                 1 0 0 1 0 0 1;
                 1 0 0 1 0 0 1;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0]

    policy_2  = [3 0 0 0 0 0 3;
                 3 0 0 0 0 0 3;
                 3 4 4 4 4 4 4;
                 3 0 0 1 0 0 1;
                 3 0 0 1 0 0 1;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0]
    policy_3 = [3 0 0 0 0 0 1;
                 3 0 0 0 0 0 1;
                 2 2 2 2 2 2 1;
                 1 0 0 1 0 0 1;
                 1 0 0 1 0 0 1;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0]
    policy_4 = [3 0 0 0 0 0 3;
                 3 0 0 0 0 0 3;
                 2 2 2 2 2 2 3;
                 1 0 0 1 0 0 3;
                 1 0 0 1 0 0 3;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0;
                 0 0 0 1 0 0 0]
    gvfs = [policy_1,policy_2,policy_3, policy_4]
    mask = valid_state_mask()

    action_prob = if gvfs[gvf_i][mask][state] == action
        1.0
    else
        0.0
    end
    return action_prob
end

end
