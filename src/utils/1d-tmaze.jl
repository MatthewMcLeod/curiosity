module OneDTMazeUtils

import ..TMazeCumulantSchedules
import ..OneDTmazeConst
import ..OneDTMaze
import ..check_goal
import ..range_check
const TMCS = TMazeCumulantSchedules
const ODTMC = OneDTmazeConst



#####
# GVF Parameter Functions
####

struct GoalTermination end

function Base.get(::GoalTermination; state_t, kwargs...)
    any([check_goal(OneDTMaze, i, state_t) for i in 1:4])
end

struct GoalPolicy
    goal::Int
end

function Base.get(π::GoalPolicy; state_t, action_t, kwargs...)
    cur_x = state_t[1]
    cur_y = state_t[2]
    if π.goal == 1
        if cur_x == 0.5
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.LEFT
            else
                ODTMC.UP
            end
        elseif range_check(cur_x, 0.0 - ODTMC.EPSILON, 0.0 + ODTMC.EPSILON)
            ODTMC.UP
        else
            ODTMC.LEFT
        end
    elseif π.goal == 2
        if cur_x == 0.5
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.LEFT
            else
                ODTMC.UP
            end
        elseif range_check(cur_x, 0.0 - ODTMC.EPSILON, 0.0 + ODTMC.EPSILON)
            ODTMC.DOWN
        else
            ODTMC.LEFT
        end
    elseif π.goal == 3
        if cur_x == 0.5
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.RIGHT
            else
                ODTMC.UP
            end
        elseif range_check(cur_x, 1.0 - ODTMC.EPSILON, 1.0 + ODTMC.EPSILON)
            ODTMC.UP
        else
            ODTMC.RIGHT
        end
    elseif π.goal == 4
        if cur_x == 0.5
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.RIGHT
            else
                ODTMC.UP
            end
        elseif range_check(cur_x, 1.0 - ODTMC.EPSILON, 1.0 + ODTMC.EPSILON)
            ODTMC.DOWN
        else
            ODTMC.RIGHT
        end
    end
end


####
# Behaviour policies
####

mutable struct RoundRobinPolicy
    cur_goal::Int
end

Base.get(π::RoundRobinPolicy; kwargs...) =
    Base.get(GoalPolicy(π.cur_goal); kwargs...)

get_action_probs(π::RoundRobinPolicy, state, obs) =
    [Base.get(π; state_t = state, action_t=a) for a ∈ 1:4]



####
# Cumulant Schedules
####
DrifterDistractor(parsed) = begin
    if "drifter" ∈ keys(parsed)
        TMCS.DrifterDistractor(
            parsed["constant_target"],
            parsed["drifter"][1],
            parsed["drifter"][2],
            parsed["distractor"][1],
            parsed["distractor"][2])
    else
        TMCS.DrifterDistractor(
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
            TMCS.Constant(parsed["cumulant"])
        else
            TMCS.Constant(parsed["cumulant"]...)
        end

    else
        throw("$(sched) Not Implemented")
    end
end


end
