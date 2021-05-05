module TwoDGridWorldUtils

using SparseArrays
using Distributions

import ..TMazeCumulantSchedules
import ..ContGridWorldParams
import ..ContGridWorld
import ..Learner
import ..check_goal
import ..range_check
import ..get_action_probs
import ..GVFHordes
import ..update
import ..Curiosity
import ..GVFSRHordes
import ..SRCreationUtils
import ..FeatureCreator
using ...StatsBase

const TMCS = TMazeCumulantSchedules
const CGWP = ContGridWorldParams
const SRCU = Curiosity.SRCreationUtils

struct GoalTermination <: GVFHordes.GVFParamFuncs.AbstractDiscount
    γ::Float64
end

function Base.get(gt::GoalTermination; state_tp1, kwargs...)
    any(state_tp1[3:end] .!= 0) ? 0.0 : gt.γ
end


struct GoalPolicy <: GVFHordes.GVFParamFuncs.AbstractPolicy
    goal::Int
    normalized::Bool
end

(π::GoalPolicy)(s) = sample(Weights([get(π, state_t=s, action_t=a) for a ∈ 1:4]))

function Base.get(π::GoalPolicy; state_t, action_t, kwargs...)
    cur_y = state_t[1]
    cur_x = state_t[2]

    if π.goal == 1
        boundry_y = π.normalized ? 0.9 : 9.0
        boundry_x = π.normalized ? 0.1 : 1.0
        if cur_x > boundry_x && cur_y < boundry_y
            if CGWP.UP == action_t || CGWP.LEFT == action_t
                0.5
            else
                0.0
            end
        elseif cur_x > boundry_x
            if CGWP.LEFT == action_t
                1.0
            else
                0.0
            end
        elseif cur_y < boundry_y
            if CGWP.UP == action_t
                1.0
            else
                0.0
            end
        else
            if CGWP.UP == action_t
                1.0
            else
                0.0
            end
            # @show action_t, state_t
            # throw("What?")
        end
    elseif π.goal == 2
        boundry_y = π.normalized ? 0.1 : 1.0
        boundry_x = π.normalized ? 0.1 : 1.0
        if cur_x > boundry_x && cur_y > boundry_y
            if CGWP.DOWN == action_t || CGWP.LEFT == action_t
                0.5
            else
                0.0
            end
        elseif cur_x > boundry_x
            if CGWP.LEFT == action_t
                1.0
            else
                0.0
            end
        elseif cur_y > boundry_y
            if CGWP.DOWN == action_t
                1.0
            else
                0.0
            end
        else
            if CGWP.DOWN == action_t
                1.0
            else
                0.0
            end
            # @show action_t, state_t
            # throw("What?")
        end
    elseif π.goal == 3
        boundry_y = π.normalized ? 0.9 : 9.0
        boundry_x = π.normalized ? 0.9 : 9.0
        if cur_x < boundry_x && cur_y < boundry_y
            if CGWP.UP == action_t || CGWP.RIGHT == action_t
                0.5
            else
                0.0
            end
        elseif cur_x < boundry_x
            if CGWP.RIGHT == action_t
                1.0
            else
                0.0
            end
        elseif cur_y < boundry_y
            if CGWP.UP == action_t
                1.0
            else
                0.0
            end
        else
            if CGWP.UP == action_t
                1.0
            else
                0.0
            end
            # @show action_t, state_t
            # throw("What?")
        end
    elseif π.goal == 4
        boundry_y = π.normalized ? 0.1 : 1.0
        boundry_x = π.normalized ? 0.9 : 9.0
        if cur_x < boundry_x && cur_y > boundry_y
            if CGWP.DOWN == action_t || CGWP.RIGHT == action_t
                0.5
            else
                0.0
            end
        elseif cur_x < boundry_x
            if CGWP.RIGHT == action_t
                1.0
            else
                0.0
            end
        elseif cur_y > boundry_y
            if CGWP.DOWN == action_t
                1.0
            else
                0.0
            end
        else
            if CGWP.DOWN == action_t
                1.0
            else
                0.0
            end
            # @show action_t, state_t
            # throw("What?")
        end
    end
end

struct NaiveGoalPolicy <: GVFHordes.GVFParamFuncs.AbstractPolicy
    goal::Int
end

(π::NaiveGoalPolicy)(s) = sample(Weights([Base.get(π, state_t=s, action_t=a) for a ∈ 1:4]))

function Base.get(π::NaiveGoalPolicy; state_t, action_t, kwargs...)
    if π.goal == 1
        if CGWP.UP == action_t || CGWP.LEFT == action_t
            0.5
        else
            0.0
        end
    elseif π.goal == 2
        if CGWP.DOWN == action_t || CGWP.LEFT == action_t
            0.5
        else
            0.0
        end
    elseif π.goal == 3
        if CGWP.UP == action_t || CGWP.RIGHT == action_t
            0.5
        else
            0.0
        end
    elseif π.goal == 4
        if CGWP.DOWN == action_t || CGWP.RIGHT == action_t
            0.5
        else
            0.0
        end
    end
end

function create_demons(parsed, demon_projected_fc = nothing)
    action_space = 4
    demons = if parsed["demon_learner"] != "SR"
        GVFHordes.Horde(
            [GVFHordes.GVF(GVFHordes.GVFParamFuncs.FeatureCumulant(i+2),
                           GoalTermination(0.9),
                           NaiveGoalPolicy(i)) for i in 1:4])
    elseif parsed["demon_learner"] == "SR"
        @assert demon_projected_fc != nothing
        pred_horde =  GVFHordes.Horde(
                [GVFHordes.GVF(GVFHordes.GVFParamFuncs.FeatureCumulant(i+2),
                     GVFHordes.GVFParamFuncs.ConstantDiscount(0.0),
                     NaiveGoalPolicy(i)) for i in 1:4])

        SF_policies = [NaiveGoalPolicy(i) for i in 1:4]
        SF_discounts = [GoalTermination(0.9) for i in 1:4]
        num_SFs = length(SF_policies)
        SF_horde = SRCU.create_SF_horde(SF_policies, SF_discounts, demon_projected_fc,1:action_space)
        GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, demon_projected_fc)
    else
        throw(ArgumentError("Cannot create demons"))
    end
    return demons
end

function b_π(state_constructor, learner, exploration_strategy; kwargs...)
    s = state_constructor(kwargs[:state_t])
    preds = learner(s)
    return exploration_strategy(preds)[kwargs[:action_t]]
end

function make_behaviour_gvf(learner, γ, state_constructor, expl_strat) #discount, state_constructor_func, learner, exploration_strategy)
    GVF_policy = GVFHordes.GVFParamFuncs.FunctionalPolicy((;kwargs...) -> b_π(state_constructor, learner, expl_strat; kwargs...))
    BehaviourGVF = GVFHordes.GVF(GVFHordes.GVFParamFuncs.RewardCumulant(),
                       GoalTermination(γ),
                       GVF_policy)
end


function check_goal(goal, state, epsilon=0.0)
    cur_y = state[1]
    cur_x = state[2]
    if goal == 1
        boundry_y = 0.9 - epsilon
        boundry_x = 0.1 + epsilon
        cur_y > boundry_y && cur_x < boundry_x
    elseif goal == 2
        boundry_y = 0.1 + epsilon
        boundry_x = 0.1 + epsilon
        cur_y < boundry_y && cur_x < boundry_x
    elseif goal == 3
        boundry_y = 0.9 - epsilon
        boundry_x = 0.9 - epsilon
        cur_y > boundry_y && cur_x > boundry_x
    elseif goal == 4
        boundry_y = 0.1 + epsilon
        boundry_x = 0.9 - epsilon
        cur_y < boundry_y && cur_x > boundry_x
    else
        false
    end
end

####
# Ideal Feature Creator
####
struct IdealDemonFeatures <: FeatureCreator
end

function project_features(fc::IdealDemonFeatures, state)
    new_state = sparsevec(convert(Array{Int,1}, [check_goal(i, state) for i in 1:4]))
    # if sum(new_state) != 0
    #     @show state, new_state
    # end
    return new_state
end

(FP::IdealDemonFeatures)(state) = project_features(FP, state)
Base.size(FP::IdealDemonFeatures) = 4

struct MarthaIdealDemonFeatures <: FeatureCreator
end

function project_features(fc::MarthaIdealDemonFeatures, state)
    new_state = sparsevec(convert(Array{Int,1}, [check_goal(i, state, 0.05) for i in 1:4]))
    return new_state
end

(FP::MarthaIdealDemonFeatures)(state) = project_features(FP, state)
Base.size(FP::MarthaIdealDemonFeatures) = 4


# Base.@kwdef struct RoundRobinPolicy <: Learner
#     update = Nothing
# end

# Curiosity.update!(learner::RoundRobinPolicy, args...) = nothing

# Base.get(π::RoundRobinPolicy; state_t, action_t, kwargs...) =
#     get_action_probs(π, state_t, nothing)[action_t]

# function Curiosity.get_action_probs(π::RoundRobinPolicy, features, state)
#     cur_x = state[2]
#     cur_y = state[1]
#     # ret = zeros(4)

#     # if cur_x == 0.5
#     #     if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON) # Middle Junction
#     #         ret[ODTMC.LEFT] = 0.5
#     #         ret[ODTMC.RIGHT] = 0.5
#     #     else # Middle Hallway
#     #         ret[ODTMC.UP] = 1.0
#     #     end
#     # elseif cur_y == 0.8 && range_check(cur_x, 0.0 - ODTMC.EPSILON, 0.0 + ODTMC.EPSILON) # Left Junction
#     #     ret[ODTMC.UP] = 0.5
#     #     ret[ODTMC.DOWN] = 0.5
#     # elseif cur_y == 0.8 && range_check(cur_x, 1.0 - ODTMC.EPSILON, 1.0 + ODTMC.EPSILON)
#     #     ret[ODTMC.UP] = 0.5
#     #     ret[ODTMC.DOWN] = 0.5
#     # elseif cur_x == 0.0
#     #     if cur_y > 0.8
#     #         ret[ODTMC.UP] = 1.0
#     #     else
#     #         ret[ODTMC.DOWN] = 1.0
#     #     end
#     # elseif cur_x == 1.0
#     #     if cur_y > 0.8
#     #         ret[ODTMC.UP] = 1.0
#     #     else
#     #         ret[ODTMC.DOWN] = 1.0
#     #     end
#     # elseif cur_x < 0.5
#     #     ret[ODTMC.LEFT] = 1.0
#     # else
#     #     ret[ODTMC.RIGHT] = 1.0
#     # end
#     # ret
# end
DrifterDistractor(parsed) = begin
    c_dist = Uniform(parsed["constant_target"][1],parsed["constant_target"][2])
    c1,c2 = rand(c_dist,2)
    if "drifter" ∈ keys(parsed)
        TMCS.DrifterDistractor(
            c1,
            c2,
            parsed["drifter"][1],
            parsed["drifter"][2],
            parsed["distractor"][1],
            parsed["distractor"][2])
    else
        TMCS.DrifterDistractor(
            c1,
            c2,
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


function get_true_values(env::Curiosity.ContGridWorld, eval_set)
    copy_eval_est = deepcopy(eval_set)
    num_gvfs = 4
    goal_cumulants = TMCS.get_cumulant_eval_values(env.cumulant_schedule)
    for i in 1:num_gvfs
        copy_eval_est[i, :] .*= goal_cumulants[i]
    end
    return copy_eval_est
end

function get_true_values(env::Curiosity.ContGridWorld, eval_set, gvf_idx)
    copy_eval_est = deepcopy(eval_set)
    goal_cumulants = TMCS.get_cumulant_eval_values(env.cumulant_schedule)
    copy_eval_est .*= goal_cumulants[gvf_idx]
    return copy_eval_est
end

end
