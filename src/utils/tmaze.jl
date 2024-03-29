
module TabularTMazeUtils
using Curiosity
using GVFHordes
using SparseArrays
import ..TMazeCumulantSchedules
import ..GVFSRHordes
import ..FeatureCreator
import ..SRCreationUtils

using Distributions
using ...StatsBase
const TTMCS = TMazeCumulantSchedules

const NUM_DEMONS = 4
const NUM_ACTIONS = 4

DrifterDistractor(parsed) = begin
    c_dist = Uniform(parsed["constant_target"][1],parsed["constant_target"][2])
    c1,c2 = rand(c_dist,2)

    if "drifter" ∈ keys(parsed)
        TTMCS.DrifterDistractor(
            c1,
            c2,
            parsed["drifter"][1],
            parsed["drifter"][2],
            parsed["distractor"][1],
            parsed["distractor"][2])
    else

        TTMCS.DrifterDistractor(
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
            TTMCS.Constant(parsed["cumulant"])
        else
            TTMCS.Constant(parsed["cumulant"]...)
        end

    else
        throw("$(sched) Not Implemented")
    end
end

function pseudoterm(;kwargs...)
    obs = kwargs[:state_tp1]
    term = false
    if obs[1] == 1 || obs[1] == 5 || obs[1] == 17 || obs[1] == 21
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

struct TTMazeStateActionCumulant <: GVFParamFuncs.AbstractCumulant
    state_num::Int
    action::Int
end

function Base.get(cumulant::TTMazeStateActionCumulant; kwargs...)
    state = kwargs[:constructed_state_t]
    action = kwargs[:action_t]
    if state.nzind[1] == cumulant.state_num && action == cumulant.action
        return 1
    else
        return 0
    end
end

## Create Reward feature Projector
struct StateActionFeatures <: FeatureCreator
    num_states::Int
    num_actions::Int
end

function project_features(fc::StateActionFeatures, state, action, state_tp1)
    new_s = spzeros(fc.num_actions * fc.num_states)
    ind = (state[1] -1)* fc.num_actions + action
    new_s[convert(Int,ind)] = 1
    return new_s
end

(FP::StateActionFeatures)(state,action,state_tp1) = project_features(FP, state,action,state_tp1)
Base.size(FP::StateActionFeatures) = FP.num_states * FP.num_actions


function make_behaviour_gvf(discount, state_constructor_func, learner, exploration_strategy)
    function b_π(state_constructor_func, learner, exploration_strategy; kwargs...)
        s = state_constructor_func(kwargs[:state_t])
        preds = learner(s)
        return exploration_strategy(preds)[kwargs[:action_t]]
    end
    GVF_policy = GVFParamFuncs.FunctionalPolicy((;kwargs...) -> b_π(state_constructor_func, learner, exploration_strategy; kwargs...))
    BehaviourGVF = GVF(GVFParamFuncs.RewardCumulant(), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVF_policy)
end
function demon_target_policy(gvf_i; kwargs...)

    # state = convert(Int,observation[1])
    state = convert(Int,kwargs[:state_t][1])
    action = kwargs[:action_t]

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

struct GoalPolicy <: GVFHordes.GVFParamFuncs.AbstractPolicy
    goal::Int
end
(π::GoalPolicy)(s) = begin
    sample(Weights([demon_target_policy(π.goal, state_t=s, action_t=a) for a ∈ 1:4]))
end

function Base.get(π::GoalPolicy; state_t, action_t, kwargs...)
    s = state_t[1]
    demon_target_policy(π.goal; state_t=s, action_t)
end

function get_true_values(env::Curiosity.TabularTMaze, eval_set)
    copy_eval_est = deepcopy(eval_set)
    num_gvfs = 4
    goal_cumulants = TTMCS.get_cumulant_eval_values(env.cumulant_schedule)
    for i in 1:num_gvfs
        copy_eval_est[i, :] .*= goal_cumulants[i]
    end
    return copy_eval_est
end

end
