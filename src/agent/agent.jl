using SparseArrays
using Distributions
using StatsBase
using GVFHordes
using MinimalRLCore

mutable struct Agent{IR<:IntrinsicReward,
                     H<:GVFHordes.AbstractHorde,
                     DL<:Learner,
                     BLU<:LearningUpdate,
                     O,
                     Φ,
                     SC} <: AbstractAgent


    behaviour_weights::Array{Float64,2}
    behaviour_lu::BLU
    behaviour_learner::Any
    behaviour_demons::Any
    behaviour_gamma::Float64

    demons::H
    demon_learner::DL

    last_obs::O
    last_state::Φ

    last_action::Int
    num_actions::Int

    intrinsic_reward::IR
    state_constructor::SC
    use_external_reward::Bool

end

function Agent(horde,
               behaviour_feature_size::Int,
               behaviour_lu,
               behaviour_learner,
               behaviour_horde,
               behaviour_gamma,
               demon_learner,
               observation_size::Int,
               num_actions::Int,
               intrinsic_reward_type,
               state_constructor,
               use_external_reward)

    behaviour_weight_dims = (num_actions, behaviour_feature_size)

    intrinsic_reward = if intrinsic_reward_type == "weight_change"
        #TODO: The intrinsic reward is defined by how the components of the agent are put together. For example, an intrinsic reward
        # could be the model error, which would then require different components that are assembled in the agent
        # Not sure how the construction of intrinisic reward could be abstracted out of this constructor
        WeightChange(get_weights(demon_learner))
    elseif intrinsic_reward_type == "no_reward"
        NoReward()
    else
        throw(ArgumentError("Not a valid intrinsic reward"))
    end


    Agent(zeros(behaviour_weight_dims),
          behaviour_lu,
          behaviour_learner,
          behaviour_horde,
          behaviour_gamma,

          horde,
          demon_learner,
          # demon_lu,

          zeros(observation_size),
          spzeros(behaviour_feature_size),

          0, # last_action
          num_actions,
          # demon_feature_size,
          # behaviour_feature_size,

          intrinsic_reward,
          state_constructor,

          use_external_reward)

end

function proc_input(agent, obs)
    return agent.state_constructor(obs)
end

function eps_greedy(qs)
    epsilon=0.2
    m = maximum(qs)
    probs = zeros(length(qs))

    maxes_ind = findall(x-> x==m, qs)
    for ind in maxes_ind
        probs[ind] += (1 - epsilon) / size(maxes_ind)[1]
    end
    probs .+= epsilon/ length(qs)
    return probs
end

function get_action(agent, state, obs)
    action_probs = if agent.behaviour_lu isa ESARSA
        get_action_probs(agent.behaviour_lu, state, obs, agent.behaviour_weights)
    else
        # qs = predict(agent.behaviour_learner, state)
        qs = agent.behaviour_learner(state)
        eps_greedy(qs)
    end
    action = sample(1:agent.num_actions, Weights(action_probs))
    return action, action_probs
end

function assign_horde!(agent, horde)
    agent.demons = horde
end

function MinimalRLCore.end!(agent, obs, reward, is_terminal)
    return step!(agent, obs, reward, is_terminal)
end

function MinimalRLCore.step!(agent::Agent, obs, r, is_terminal, args...)
    next_state = proc_input(agent, obs)
    next_action, next_action_probs = get_action(agent, next_state, obs)

    update_demons!(agent,
                   agent.last_obs,
                   obs,
                   agent.last_state,
                   agent.last_action,
                   next_state,
                   next_action,
                   is_terminal)
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

function MinimalRLCore.start!(agent::Agent, obs, args...)
    next_state = proc_input(agent, obs)
    #Always exploring starts
    next_action = sample(1:agent.num_actions, Weights(ones(agent.num_actions)))

    _, discounts, _ = get(agent.demons, obs, next_action, obs, next_action)
    agent.last_state = next_state
    agent.last_action = next_action
    agent.last_obs = obs
    zero_eligibility_traces!(agent.demon_learner)
    # zero_eligibility_traces!(agent.behaviour_learner)

    return next_action
end


# get_behaviour_pis(agent::Agent, state, obs) =
#     get_action_probs(agent.behaviour_lu, state, obs, agent.behaviour_weights)

get_behaviour_pis(agent::Agent, state, obs) =
    get_action(agent, state, obs)[2]

function update_demons!(agent,obs, next_obs, state, action, next_state, next_action, is_terminal)


    update!(# update(agent.demon_learner),
            agent.demon_learner,
            agent.demons,
            obs,
            next_obs,
            state,
            action,
            next_state,
            next_action,
            is_terminal,
            (state, obs) -> get_behaviour_pis(agent, state, obs))

end

function update_behaviour!(agent, obs, next_obs, state, action, next_state, next_action, is_terminal, reward)

    # behaviour_pis = get_action_probs(agent.behaviour_lu, state, obs, agent.behaviour_weights)
    # next_behaviour_pis = get_action_probs(agent.behaviour_lu, next_state, next_obs, agent.behaviour_weights)

    behaviour_pis = get_behaviour_pis(agent, state, obs)
    next_behaviour_pis = get_behaviour_pis(agent, next_state, next_obs)

    if agent.behaviour_lu isa ESARSA
        update!(agent.behaviour_lu,
            agent.behaviour_weights,
            [reward],
            state,
            action,
            next_state,
            next_action,
            next_behaviour_pis,
            [!is_terminal*agent.behaviour_gamma])
    elseif agent.behaviour_learner isa SARSA
        update!(agent.behaviour_lu,
                agent.behaviour_learner,
                obs,
                next_obs,
                state,
                action,
                next_state,
                next_action,
                is_terminal,
                [agent.behaviour_gamma],
                [reward])
    else
        update!(# update(agent.demon_learner),
                agent.behaviour_learner,
                agent.behaviour_demons,
                obs,
                next_obs,
                state,
                action,
                next_state,
                next_action,
                is_terminal,
                [reward],
                [agent.behaviour_gamma],
                (state, obs) -> get_behaviour_pis(agent, state, obs))
    end
end

function get_demon_prediction(agent::Agent, obs, action)
    ϕ = agent.state_constructor(obs)
    agent.demon_learner(ϕ, action)
end
