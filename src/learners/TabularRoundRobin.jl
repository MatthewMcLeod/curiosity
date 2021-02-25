# include("../utils/env_utils.jl")
mutable struct TabularRoundRobin <: Learner
end
function update!(self::TabularRoundRobin, reward, action, next_action, state, next_state, agent)
end
function get_action_probs(self::TabularRoundRobin, state, observation)
    action_probs = zeros(4)
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    LEFT_JUNCTION = 3
    MIDDLE_JUNCTION = 8
    RIGHT_JUNCTION = 19

    #This relies on being tabular. For continuous TMaze, need different round robin method
    state = if typeof(state) == Int
        state
    else
        #It is a sparse array encoding
        tups = findnz(state)
        tups[1][1]
    end

    if state == LEFT_JUNCTION
        action_probs[DOWN] = 0.5
        action_probs[UP] = 0.5
    elseif state == MIDDLE_JUNCTION
        action_probs[LEFT] = 0.5
        action_probs[RIGHT] = 0.5
    elseif state == RIGHT_JUNCTION
        action_probs[DOWN] = 0.5
        action_probs[UP] = 0.5
    elseif state < 3
        action_probs[UP] = 1
    elseif state < 5
        action_probs[DOWN] = 1
    elseif state < 8
        action_probs[LEFT] = 1
    elseif state < 15
        action_probs[UP] = 1
    elseif state < 17
        action_probs[RIGHT] = 1
    elseif state < RIGHT_JUNCTION
        action_probs[UP] = 1
    else
        action_probs[DOWN] = 1
    end

    return action_probs
end
