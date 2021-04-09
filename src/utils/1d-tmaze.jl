module OneDTMazeUtils

import ..TMazeCumulantSchedules
import ..OneDTmazeConst
import ..OneDTMaze
import ..Learner
import ..check_goal
import ..range_check
import ..get_action_probs
import ..GVFHordes
import ..update
import ..Curiosity
import ..GVFSRHordes
import ..SRCreationUtils

const TMCS = TMazeCumulantSchedules
const ODTMC = OneDTmazeConst
const SRCU = Curiosity.SRCreationUtils


#####
# GVF Parameter Functions
####

struct GoalTermination <: GVFHordes.GVFParamFuncs.AbstractDiscount
    γ::Float64
end

function Base.get(gt::GoalTermination; state_tp1, kwargs...)
    any([check_goal(OneDTMaze, i, state_tp1) for i in 1:4]) ? 0.0 : gt.γ
end

struct GoalPolicy <: GVFHordes.GVFParamFuncs.AbstractPolicy
    goal::Int
end

function Base.get(π::GoalPolicy; state_t, action_t, kwargs...)
    cur_x = state_t[1]
    cur_y = state_t[2]
    if π.goal == 1
        if cur_x == 0.5
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.LEFT == action_t
            else
                ODTMC.UP == action_t
            end
        elseif range_check(cur_x, 0.0 - ODTMC.EPSILON, 0.0 + ODTMC.EPSILON)
            ODTMC.UP == action_t
        else
            ODTMC.LEFT == action_t
        end
    elseif π.goal == 2
        if cur_x == 0.5
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.LEFT == action_t
            else
                ODTMC.UP == action_t
            end
        elseif range_check(cur_x, 0.0 - ODTMC.EPSILON, 0.0 + ODTMC.EPSILON)
            ODTMC.DOWN == action_t
        else
            ODTMC.LEFT == action_t
        end
    elseif π.goal == 3
        if cur_x == 0.5
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.RIGHT == action_t
            else
                ODTMC.UP == action_t
            end
        elseif range_check(cur_x, 1.0 - ODTMC.EPSILON, 1.0 + ODTMC.EPSILON)
            ODTMC.UP == action_t
        else
            ODTMC.RIGHT == action_t
        end
    elseif π.goal == 4
        if cur_x == 0.5
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.RIGHT == action_t
            else
                ODTMC.UP == action_t
            end
        elseif range_check(cur_x, 1.0 - ODTMC.EPSILON, 1.0 + ODTMC.EPSILON)
            ODTMC.DOWN == action_t
        else
            ODTMC.RIGHT == action_t
        end
    end
end

####
# Demon Creation
####

function create_demons(parsed, demon_projected_fc = nothing)
    action_space = 4
    demons = if parsed["demon_learner"] != "SR"
        GVFHordes.Horde(
        [GVFHordes.GVF(GVFHordes.GVFParamFuncs.FeatureCumulant(i+2),
             GoalTermination(0.9),
             GoalPolicy(i)) for i in 1:4])
    elseif parsed["demon_learner"] == "SR"
        @assert demon_projected_fc != nothing
        pred_horde =  GVFHordes.Horde(
                [GVFHordes.GVF(GVFHordes.GVFParamFuncs.FeatureCumulant(i+2),
                     GVFHordes.GVFParamFuncs.ConstantDiscount(0.0),
                     GoalPolicy(i)) for i in 1:4])

        SF_policies = [GoalPolicy(i) for i in 1:4]
        SF_discounts = [GoalTermination(0.9) for i in 1:4]
        num_SFs = length(SF_policies)
        SF_horde = SRCU.create_SF_horde(SF_policies, SF_discounts, demon_projected_fc,1:action_space)
        GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, demon_projected_fc)
    else
        throw(ArgumentError("Cannot create demons"))
    end
    return demons
end


####
# Behaviour policies
####

Base.@kwdef struct RoundRobinPolicy <: Learner
    update = Nothing
end

Curiosity.update!(learner::RoundRobinPolicy, args...) = nothing

Base.get(π::RoundRobinPolicy; state_t, action_t, kwargs...) =
    get_action_probs(π, state_t, nothing)[action_t]



function Curiosity.get_action_probs(π::RoundRobinPolicy, features, state)
    cur_x = state[1]
    cur_y = state[2]
    ret = zeros(4)

    if cur_x == 0.5
        if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON) # Middle Junction
            ret[ODTMC.LEFT] = 0.5
            ret[ODTMC.RIGHT] = 0.5
        else # Middle Hallway
            ret[ODTMC.UP] = 1.0
        end
    elseif cur_y == 0.8 && range_check(cur_x, 0.0 - ODTMC.EPSILON, 0.0 + ODTMC.EPSILON) # Left Junction
        ret[ODTMC.UP] = 0.5
        ret[ODTMC.DOWN] = 0.5
    elseif cur_y == 0.8 && range_check(cur_x, 1.0 - ODTMC.EPSILON, 1.0 + ODTMC.EPSILON)
        ret[ODTMC.UP] = 0.5
        ret[ODTMC.DOWN] = 0.5
    elseif cur_x == 0.0
        if cur_y > 0.8
            ret[ODTMC.UP] = 1.0
        else
            ret[ODTMC.DOWN] = 1.0
        end
    elseif cur_x == 1.0
        if cur_y > 0.8
            ret[ODTMC.UP] = 1.0
        else
            ret[ODTMC.DOWN] = 1.0
        end
    elseif cur_x < 0.5
        ret[ODTMC.LEFT] = 1.0
    else
        ret[ODTMC.RIGHT] = 1.0
    end
    ret
end



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

function get_true_values(env::Curiosity.OneDTMaze, eval_set)
    copy_eval_est = deepcopy(eval_set)
    num_gvfs = 4
    goal_cumulants = TMCS.get_cumulant_eval_values(env.cumulant_schedule)
    for i in 1:num_gvfs
        copy_eval_est[i, :] .*= goal_cumulants[i]
    end
    return copy_eval_est
end

end
