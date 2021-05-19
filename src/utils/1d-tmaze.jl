module OneDTMazeUtils

using SparseArrays
using Distributions

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
import ..FeatureCreator
using ...StatsBase


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
(π::GoalPolicy)(s) = sample(Weights([get(π, state_t=s, action_t=a) for a ∈ 1:4]))

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
        elseif cur_x == 1.0
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.LEFT == action_t
            elseif cur_y <= 0.8 - ODTMC.EPSILON
                ODTMC.UP == action_t
            elseif cur_y >= 0.8 + ODTMC.EPSILON
                ODTMC.DOWN == action_t
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
        elseif cur_x == 1.0
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.LEFT == action_t
            elseif cur_y <= 0.8 - ODTMC.EPSILON
                ODTMC.UP == action_t
            elseif cur_y >= 0.8 + ODTMC.EPSILON
                ODTMC.DOWN == action_t
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
        elseif cur_x == 0.0
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.RIGHT == action_t
            elseif cur_y <= 0.8 - ODTMC.EPSILON
                ODTMC.UP == action_t
            elseif cur_y >= 0.8 + ODTMC.EPSILON
                ODTMC.DOWN == action_t
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
        elseif cur_x == 0.0
            if range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON)
                ODTMC.RIGHT == action_t
            elseif cur_y <= 0.8 - ODTMC.EPSILON
                ODTMC.UP == action_t
            elseif cur_y >= 0.8 + ODTMC.EPSILON
                ODTMC.DOWN == action_t
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
        # SF_horde = if parsed["demon_reward_feature_type"] == "state"
        #     SRCU.create_state_SF_horde(SF_policies, SF_discounts, demon_projected_fc,1:action_space)
        # else
        # SF_horde = SRCU.create_SF_horde(SF_policies, SF_discounts, demon_projected_fc,1:action_space)
        SF_horde = SRCU.create_SF_horde(SF_policies, SF_discounts, demon_projected_fc,1:action_space)

        # end
        GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, demon_projected_fc)
    else
        throw(ArgumentError("Cannot create demons"))
    end
    return demons
end

function make_behaviour_gvf(learner, γ, state_constructor, expl_strat) #discount, state_constructor_func, learner, exploration_strategy)
    function b_π(state_constructor, learner, exploration_strategy; kwargs...)
        s = state_constructor(kwargs[:state_t])
        preds = learner(s)
        return exploration_strategy(preds)[kwargs[:action_t]]
    end
    GVF_policy = GVFHordes.GVFParamFuncs.FunctionalPolicy((;kwargs...) -> b_π(state_constructor, learner, expl_strat; kwargs...))
    BehaviourGVF = GVFHordes.GVF(GVFHordes.GVFParamFuncs.RewardCumulant(),
                       GoalTermination(γ),
                       GVF_policy)
end


# ####
struct IdealStateActionDemonFeatures <: FeatureCreator
    num_actions::Int
end
function project_features(fc::IdealStateActionDemonFeatures, s_t, a_t, s_tp1)
    # new_state = sparsevec(convert(Array{Int,1}, [check_goal(OneDTMaze, i, s_tp1) for i in 1:4]))
    goal_ind = findfirst([check_goal(OneDTMaze, i, s_tp1) for i in 1:4])
    reward_feature = zeros(Int(fc.num_actions*4))
    if !(goal_ind isa Nothing)
        reward_feature[Int((goal_ind-1)*fc.num_actions + a_t)] = 1
    end
    return sparsevec(reward_feature)
end
(FP::IdealStateActionDemonFeatures)(s_t,a_t,s_tp1) = project_features(FP, s_t,a_t,s_tp1)

Base.size(FP::IdealStateActionDemonFeatures) = 16
####
# Ideal Feature Creator
####
struct IdealDemonFeatures <: FeatureCreator
end

function project_features(fc::IdealDemonFeatures, state,action,state_tp1)
    new_state = sparsevec(convert(Array{Int,1}, [check_goal(OneDTMaze, i, state_tp1) for i in 1:4]))
    return new_state
end

(FP::IdealDemonFeatures)(state, action, next_state) = project_features(FP, state, action, next_state)
Base.size(FP::IdealDemonFeatures) = 4

struct MarthaIdealDemonFeatures <: FeatureCreator
    num_actions::Int
end

function project_features(fc::MarthaIdealDemonFeatures, state)
    new_state = sparsevec(convert(Array{Int,1}, [check_goal(OneDTMaze, i, state, ODTMC.ACTION_STEP + ODTMC.EPSILON + 0.00001) for i in 1:4]))
    # alt_state = sparsevec(convert(Array{Int,1}, [check_goal(OneDTMaze, i, state_tp1) for i in 1:4]))
    return new_state
end

(FP::MarthaIdealDemonFeatures)(state) = project_features(FP, state)
Base.size(FP::MarthaIdealDemonFeatures) = 4

####
# GPI Feature Creation
####
struct TMazeEncoding <: FeatureCreator
    num_segments::Int
    num_partitions::Int
    num_actions::Int
    function TMazeEncoding()
        new(7,3,4)
    end
end
Base.size(FP::TMazeEncoding) = FP.num_segments * FP.num_partitions * FP.num_actions
function project_features(fc::FeatureCreator, obs, a_t, obs_tp1)
    segment = Inf
    partition = Inf
    x,y = obs

    #Segment 1: Top Right
    if (x == 0.0 && y >= 0.8)
        segment_length = (1.0 + ODTMC.ACTION_STEP) - (0.8)
        segment = 1
        partition = (y - 0.8) / segment_length
    #Segment 2: Bottom Right
    elseif (x == 0.0 && y < 0.8)
        segment_length = (0.8) - (0.6 - ODTMC.ACTION_STEP)
        segment = 2
        partition = (y - 0.8) / segment_length
    #Segment 3: Middle Left Branch
    elseif (y == 0.8 && range_check(x,0.0,0.5))
        segment_length = (0.5 - 0.0)
        segment = 3
        partition = (x) / segment_length
    # Segment 4: Middle Right Branch
    elseif (y == 0.8 && range_check(x,0.5,1.0))
        segment_length = (1.0-0.5)
        segment = 4
        partition = (x - 0.5)/segment_length
    #Segment 5: Bottom Branch
    elseif (x == 0.5)
        segment_length = (0.8 - 0.0)
        segment = 5
        partition = (y - 0.0)/segment_length
    # #Segment 6: Top Left Branch
    elseif (x == 1.0 && y >= 0.8)
        segment_length = (1.0 + ODTMC.ACTION_STEP) - (0.8)
        segment = 6
        partition = (y - 0.8) / segment_length
        #Segment 7: Bottom Left Branch
    elseif (x == 1.0 && y < 0.8)
        segment_length = (0.8) - (0.6 - ODTMC.ACTION_STEP)
        segment = 7
        partition = (y - 0.8) / segment_length
    else
        @warn "Not A Valid State (x,y)"
    end

    state_action = zeros(fc.num_segments * fc.num_partitions * fc.num_actions)
    # Need maximum since if agent is right on boundary, ceil (0.0) is 0.
    partition_offset = maximum([Int(ceil(partition * fc.num_partitions)),1])
    if partition_offset == 0
        @show x,y,segment
    end
    state_ind = (segment-1)*fc.num_partitions + partition_offset
    state_action_ind = state_ind * fc.num_actions + a_t

    state_action[state_action_ind] = 1
    return sparsevec(state_action)
end
(FP::TMazeEncoding)(state, action, state_tp1) = project_features(FP, state, action, state_tp1)


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

function get_true_values(env::Curiosity.OneDTMaze, eval_set)
    copy_eval_est = deepcopy(eval_set)
    num_gvfs = 4
    goal_cumulants = TMCS.get_cumulant_eval_values(env.cumulant_schedule)
    for i in 1:num_gvfs
        copy_eval_est[i, :] .*= goal_cumulants[i]
    end
    return copy_eval_est
end
function get_true_values(env::Curiosity.OneDTMaze, eval_set, gvf_idx)
    copy_eval_est = deepcopy(eval_set)
    goal_cumulants = TMCS.get_cumulant_eval_values(env.cumulant_schedule)
    copy_eval_est .*= goal_cumulants[gvf_idx]
    return copy_eval_est
end

end
