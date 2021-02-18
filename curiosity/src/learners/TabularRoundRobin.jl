# include("../utils/env_utils.jl")
mutable struct TabularRoundRobin <: Learner
end

# function init!(self::RoundRobinTMaze, feature_size, num_actions, w_init, alpha, lambda, gamma, exploration_strategy, demon_policies, round_robin_scheduler)
# # function init!(self::RoundRobin, feature_size=1, num_actions=1, w_init=0.0,args...)
#     self.demon_policies = demon_policies
#     self.active_gvf = 1
#     self.RoundRobinScheduler = round_robin_scheduler
#     self.exploration_strategy = exploration_strategy
#     self.w = zeros(feature_size, num_actions)
# end

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
