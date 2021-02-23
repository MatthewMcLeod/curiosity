using SparseArrays
using Distributions
using StatsBase
using GVFHordes
using MinimalRLCore

mutable struct Agent <: AbstractAgent
    demons::Horde
    demon_weights::Array{Float64,2}
    behaviour_weights::Array{Float64,1}
    demon_learner::Learner
    behaviour_learner::Learner
    last_state::Any
    last_action::Int
    num_actions::Int
    demon_feature_size::Int
    behaviour_feature_size::Int
    last_obs::Any
    prev_discounts::Array{Float64,1}

    function Agent(horde, demon_feature_size::Int, behaviour_feature_size::Int, num_actions::Int, demon_learner, behaviour_learner)
        new(horde,
            zeros(length(horde) * num_actions, demon_feature_size),
            zeros(behaviour_feature_size),
            demon_learner,
            behaviour_learner,
            zeros(behaviour_feature_size),
            1,
            num_actions,
            demon_feature_size,
            behaviour_feature_size,
            zeros(5),
            ones(4)*0.9
            )
    end
end

function proc_input(agent, obs)
    #TODO: How to define what is used for value estimation as in this formulation, we don't use some values of the observation space
    # This is currently only works for tabular tmaze.
    s = spzeros(agent.behaviour_feature_size)
    s[convert(Int64,obs[1])] = 1
    return s
end

function get_action(agent, state, obs)
    action_probs = get_action_probs(agent.behaviour_learner, state, obs)
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

    update_demons!(agent,agent.last_obs, obs, agent.last_state, agent.last_action, next_state, next_action, is_terminal)

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

    agent.prev_discounts = deepcopy(discounts)

    return next_action
end

function update_demons!(agent,obs, next_obs, state, action, next_state, next_action, is_terminal)
    preds = ones(length(obs))

    #TODO: Fix how to get target policy probabilities for all actions as this is needed for off-policy learning algos
    # This is very ugly and I don't like passing all the target_pis through to demons...
    target_pis = zeros(length(agent.demons), agent.num_actions)
    for (i,a) in enumerate(1:agent.num_actions)
        _, _, pi = get(agent.demons, obs, a, next_obs, next_action)
        target_pis[:,i] = pi
    end
    next_target_pis = zeros(length(agent.demons), agent.num_actions)
    for (i,a) in enumerate(1:agent.num_actions)
        _, _, pi = get(agent.demons, next_obs, a, next_obs, next_action)
        next_target_pis[:,i] = pi
    end

    # TODO: Domain is most easily understood with pseudoterm being applied to S in the S,A,S' action state... Is that a problem?
    #TODO: Because of this, NOTE the agent.prev_discounts being passed into update!
    C, discounts, _ = get(agent.demons, obs, action, next_obs, next_action)
    update!(agent.demon_learner, agent.demon_weights, C, agent.prev_discounts, state, action, target_pis, next_state, next_action, next_target_pis)
    agent.prev_discounts = deepcopy(discounts)
end

function update_behaviour!(agent,observation)
end
