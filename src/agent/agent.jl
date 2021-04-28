using SparseArrays
using Distributions
using StatsBase
using GVFHordes
import GVFHordes: AbstractHorde
using MinimalRLCore

mutable struct Agent{#IR<:IntrinsicReward,
                     # H,
                     # BH,
                     # DL<:Learner,
                     # BL<:Learner,
                     O,
                     Φ,
                     SC} <: AbstractAgent


    behaviour_learner::Learner
    behaviour_demons::Union{AbstractHorde,Nothing} # Round Robin learners have behaviour demons set to nothing
    behaviour_gamma::Float64

    demons::AbstractHorde
    demon_learner::Learner

    last_obs::O
    last_state::Φ

    last_action::Int
    num_actions::Int

    intrinsic_reward::IntrinsicReward
    state_constructor::SC
    use_external_reward::Bool
    exploration::ExplorationStrategy

end

function Agent(horde,
               behaviour_feature_size::Int,
               behaviour_learner,
               behaviour_horde,
               behaviour_gamma,
               demon_learner,
               observation_size::Int,
               num_actions::Int,
               intrinsic_reward_type,
               state_constructor,
               use_external_reward,
               exploration_strategy)

    behaviour_weight_dims = (num_actions, behaviour_feature_size)

    intrinsic_reward = if intrinsic_reward_type == "weight_change"
        #TODO: The intrinsic reward is defined by how the components of the agent are put together. For example, an intrinsic reward
        # could be the model error, which would then require different components that are assembled in the agent
        # Not sure how the construction of intrinisic reward could be abstracted out of this constructor
        WeightChange(demon_learner)
    elseif intrinsic_reward_type == "no_reward"
        NoReward()
    else
        throw(ArgumentError("Not a valid intrinsic reward"))
    end

    Agent(behaviour_learner,
          behaviour_horde,
          behaviour_gamma,

          horde,
          demon_learner,

          zeros(observation_size),
          spzeros(behaviour_feature_size),

          0, # last_action
          num_actions,

          intrinsic_reward,
          state_constructor,

          use_external_reward,
          exploration_strategy)

end

function proc_input(agent, obs)
    return agent.state_constructor(obs)
end

function get_action(agent, state, obs)
    action_probs = if agent.behaviour_learner isa OneDTMazeUtils.RoundRobinPolicy
        get_action_probs(agent.behaviour_learner, state, obs)
    elseif agent.behaviour_learner.update isa TabularRoundRobin
        get_action_probs(agent.behaviour_learner.update, state, obs)
    elseif agent.behaviour_learner isa BaselineUtils.FollowDemon || agent.behaviour_learner isa BaselineUtils.RandomDemons
        agent.behaviour_learner(obs)
    else
        qs = agent.behaviour_learner(state)
        agent.exploration(qs)
    end
    action = sample(1:agent.num_actions, Weights(action_probs))
    return action, action_probs
end

function assign_horde!(agent, horde)
    agent.demons = horde
end

MinimalRLCore.end!(agent, obs, reward, is_terminal) =
    MinimalRLCore.step!(agent, obs, reward, is_terminal)


function MinimalRLCore.start!(agent::Agent, obs, args...)
    next_state = proc_input(agent, obs)
    #Always exploring starts
    step!(agent.exploration)
    next_action = sample(1:agent.num_actions, Weights(ones(agent.num_actions)))

    # NOTE: IS THIS RIGHT, seems wierd!?
    _, discounts, _ = get(agent.demons; state_t = obs, action_t = next_action, state_tp1 = obs, action_tp1 = next_action)

    # _, discounts, _ = get(agent.demons, obs, next_action, obs, next_action)
    agent.last_state = next_state
    agent.last_action = next_action
    agent.last_obs = obs
    zero_eligibility_traces!(agent.demon_learner)

    return next_action
end

function MinimalRLCore.step!(agent::Agent, obs, r, is_terminal, args...)
    next_state = proc_input(agent, obs)
    step!(agent.exploration)
    next_action, next_action_probs = get_action(agent, next_state, obs)

    update_demons!(agent,
                   agent.last_obs,
                   obs,
                   agent.last_state,
                   agent.last_action,
                   next_state,
                   next_action,
                   is_terminal,
                   r)

    #get intrinssic reward
    r_int = update_reward!(agent.intrinsic_reward, agent)
    total_reward = agent.use_external_reward ? r_int + r : r_int
    update_behaviour!(agent,
                      agent.last_obs,
                      obs,
                      agent.last_state,
                      agent.last_action,
                      next_state,
                      next_action,
                      is_terminal,
                      total_reward)

    agent.last_state = next_state
    agent.last_action = next_action
    agent.last_obs = obs
    return next_action
end

get_behaviour_pis(agent::Agent, state, obs) =
    get_action(agent, state, obs)[2]

function μ_π(agent::Agent, obs)
    state = proc_input(agent, obs)
    get_behaviour_pis(agent,state,obs)
end
function update_demons!(agent,obs, next_obs, state, action, next_state, next_action, is_terminal, env_reward)

    update!(agent.demon_learner,
            agent.demons,
            obs,
            next_obs,
            state,
            action,
            next_state,
            next_action,
            is_terminal,
            (state, obs) -> get_behaviour_pis(agent, state, obs),
            env_reward)

end

function update_behaviour!(agent, obs, next_obs, state, action, next_state, next_action, is_terminal, reward)

    behaviour_pis = get_behaviour_pis(agent, state, obs)
    next_behaviour_pis = get_behaviour_pis(agent, next_state, next_obs)

        #NOTE: Different call than demon updates as the reward and environment pseudotermination function
    update!(agent.behaviour_learner,
            agent.behaviour_demons,
            obs,
            next_obs,
            state,
            action,
            next_state,
            next_action,
            is_terminal,
            (state, obs) -> get_behaviour_pis(agent, state, obs),
            reward)
end

function get_demon_prediction(agent::Agent, obs, action)
    ϕ = agent.state_constructor(obs)
    agent.demon_learner(ϕ, action)
end
