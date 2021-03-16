using SparseArrays
using Distributions
using StatsBase
using GVFHordes
using MinimalRLCore

mutable struct Agent <: AbstractAgent
    demons::GVFHordes.AbstractHorde
    demon_weights::Array{Float64,2}
    behaviour_demons::Any
    behaviour_weights::Array{Float64,2}
    demon_learner::Learner
    behaviour_learner::Learner
    last_state::Any
    last_action::Int
    num_actions::Int
    demon_feature_size::Int
    behaviour_feature_size::Int
    last_obs::Any
    prev_discounts::Array{Float64,1}
    intrinsic_reward::IntrinsicReward
    state_constructor::Any
    behaviour_gamma::Float64
    use_external_reward::Bool

    function Agent(horde, behaviour_horde, demon_feature_size::Int, behaviour_feature_size::Int, observation_size::Int, num_actions::Int, demon_learner, behaviour_learner, intrinsic_reward_type, state_constructor, behaviour_gamma, use_external_reward)
        demon_weight_dims = size(demon_learner)
        @show demon_weight_dims
        behaviour_weight_dims = size(behaviour_learner)
        demon_weights = zeros(demon_weight_dims)

        intrinsic_reward = if intrinsic_reward_type == "weight_change"
            #TODO: The intrinsic reward is defined by how the components of the agent are put together. For example, an intrinsic reward
            # could be the model error, which would then require different components that are assembled in the agent
            # Not sure how the construction of intrinisic reward could be abstracted out of this constructor
            WeightChange(demon_weights)
        elseif intrinsic_reward_type == "no_reward"
            NoReward()
        else
            throw(ArgumentError("Not a valid intrinsic reward"))
        end

        # demon_weight_dims = if demon_learner isa SR
        #     size(demon_learner)
        # else
        #     (length(horde)*num_actions, demon_feature_size)
        # end
        # behaviour_weight_dims = size(behaviour_learner)

        new(horde,
            demon_weights,
            behaviour_horde,
            zeros(behaviour_weight_dims),
            demon_learner,
            behaviour_learner,
            zeros(behaviour_feature_size),
            1,
            num_actions,
            demon_feature_size,
            behaviour_feature_size,
            Array{Float64,1}(undef,observation_size),
            Array{Float64,1}(undef,length(horde)),
            intrinsic_reward,
            state_constructor,
            behaviour_gamma,
            use_external_reward,
            )
    end
end

function proc_input(agent, obs)
    return agent.state_constructor(obs)
end

function get_action(agent, state, obs)
    action_probs = get_action_probs(agent.behaviour_learner, state, obs, agent.behaviour_weights)
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
    #get intrinssic reward
    r_int = update_reward!(agent.intrinsic_reward, agent)

    total_reward = agent.use_external_reward ? r_int + r : r_int
    update_behaviour!(agent,agent.last_obs, obs, agent.last_state, agent.last_action, next_state, next_action, is_terminal, total_reward)

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

function get_demon_pis(agent::Agent, state, obs)
    target_pis = zeros(length(agent.demons), agent.num_actions)
    for (i,a) in enumerate(1:agent.num_actions)
        _, _, pi = get(agent.demons, obs, a, obs, a)
        target_pis[:,i] = pi
    end
    return target_pis
end

function get_behaviour_pis(agent::Agent, state, obs)
    _, behaviour_pis = get_action(agent, state, obs)
    return behaviour_pis
end

function update_demons!(agent,obs, next_obs, state, action, next_state, next_action, is_terminal)
    _, discounts, _ = get(agent.demons, obs, action, next_obs, next_action)

    update!(agent.demon_learner,
            agent,
            obs,
            next_obs,
            state,
            action,
            next_state,
            next_action,
            is_terminal,
            get_behaviour_pis,
            get_demon_pis)

    agent.prev_discounts = deepcopy(discounts)
end

function update_behaviour!(agent,obs, next_obs, state, action, next_state, next_action, is_terminal, reward)

    _, behaviour_pis = get_action(agent, state, obs)
    _, next_behaviour_pis = get_action(agent, next_state, next_obs)

    if agent.behaviour_learner isa GPI
        target_pis = zeros(length(agent.behaviour_demons), agent.num_actions)
        for (i,a) in enumerate(1:agent.num_actions)
            _, _, pi = get(agent.behaviour_demons, obs, a, next_obs, next_action)
            target_pis[:,i] = pi
        end
        next_target_pis = zeros(length(agent.behaviour_demons), agent.num_actions)
        for (i,a) in enumerate(1:agent.num_actions)
            _, _, pi = get(agent.behaviour_demons, next_obs, a, next_obs, next_action)
            next_target_pis[:,i] = pi
        end
        C, discounts, _ = get(agent.behaviour_demons, obs, action, next_obs, next_action)
        #NOTE: Temporary Hack for GPI as we cannot define the GVF using intrinsic rewards....
        # Also not sure how to define the gvf that learns off of weights in the agent
        C[1] = reward
        d = !is_terminal*agent.behaviour_gamma
        discounts[1] = d
        target_pis[1,:] = behaviour_pis
        next_target_pis[1,:] = next_behaviour_pis

        update!(agent.behaviour_learner,
                agent.behaviour_weights,
                C,
                state,
                action,
                target_pis,
                agent.prev_discounts,
                next_state,
                next_action,
                next_target_pis,
                discounts)

    else
        update!(agent.behaviour_learner, agent.behaviour_weights, [reward], state, action, next_state, next_action, next_behaviour_pis, [!is_terminal*agent.behaviour_gamma])
    end
end
