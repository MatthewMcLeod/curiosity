using SparseArrays
using Distributions
using StatsBase
using GVFHordes
import GVFHordes: AbstractHorde
using MinimalRLCore

ExperienceReplayCuriosity(size, obs_size, obs_type, num_act) =
    DynaExperienceReplay(size,
                         (obs_type, Int, obs_type, Int, Float64, Float64, Float64, Float64, Bool),
                         (obs_size, 1, obs_size, 1, num_act, num_act, 1, 1, 1),
                         (:s, :a, :ns, :na, :actprob, :actprob_prime, :env_r, :rint, :t))


mutable struct AgentER{O, Φ, SC, R} <: AbstractAgent

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

    random_first_action::Bool

    batch_size::Int
    replay::R

end

function AgentER(horde,
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
                 exploration_strategy,
                 random_first_action,
                 batch_size,
                 replay_size)

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

    replay = ExperienceReplayCuriosity(replay_size, observation_size, Float64, num_actions)
    

    AgentER(behaviour_learner,
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
            exploration_strategy,
            random_first_action,
            batch_size,
            replay)

end

function proc_input(agent::AgentER, obs)
    return agent.state_constructor(obs)
end

function get_action(agent::AgentER, state, obs)
    action_probs = if agent.behaviour_learner isa OneDTMazeUtils.RoundRobinPolicy || agent.behaviour_learner isa TwoDGridWorldUtils.RoundRobinPolicy
        get_action_probs(agent.behaviour_learner, state, obs)
    elseif agent.behaviour_learner.update isa TabularRoundRobin
        get_action_probs(agent.behaviour_learner.update, state, obs)
    elseif agent.behaviour_learner isa BaselineUtils.FollowDemon
        agent.behaviour_learner(obs)
    elseif agent.behaviour_learner isa BaselineUtils.RandomDemons
        qs = agent.behaviour_learner(obs)
        agent.exploration(qs)
    else
        qs = agent.behaviour_learner(state)
        agent.exploration(qs)
    end
    action = StatsBase.sample(1:agent.num_actions, Weights(action_probs))
    return action, action_probs
end

function assign_horde!(agent::AgentER, horde)
    agent.demons = horde
end

MinimalRLCore.end!(agent::AgentER, obs, reward, is_terminal) =
    MinimalRLCore.step!(agent, obs, reward, is_terminal)


function MinimalRLCore.start!(agent::AgentER, obs, args...)
    next_state = proc_input(agent, obs)
    #Always exploring starts
    step!(agent.exploration)
    next_action = if agent.random_first_action
        sample(1:agent.num_actions, Weights(ones(agent.num_actions)))
    else
        a, probs = get_action(agent,next_state, obs)
        a
    end

    # NOTE: IS THIS RIGHT, seems wierd!?
    _, discounts, _ = get(agent.demons; state_t = obs, action_t = next_action, state_tp1 = obs, action_tp1 = next_action)

    agent.last_state = next_state
    agent.last_action = next_action
    agent.last_obs = obs
    zero_eligibility_traces!(agent.demon_learner)

    return next_action
end

function MinimalRLCore.step!(agent::AgentER, obs, r, is_terminal, args...)
    next_state = proc_input(agent, obs)
    step!(agent.exploration)
    next_action, next_action_probs = get_action(agent, next_state, obs)

    #=
    Dealing with replay
    =#
    let # (:s, :a, :ns, :na, :actprob, :r, :t)
        ap = get_behaviour_pis(agent, agent.last_state, agent.last_obs)
        ap_prime = get_behaviour_pis(agent, next_state, obs)
        r_int = update_reward!(agent.intrinsic_reward, agent)
        total_reward = agent.use_external_reward ? r_int + r : r_int
        
        exp = (agent.last_obs, agent.last_action, obs, next_action, ap, ap_prime, r, total_reward, is_terminal)
        # @show typeof(exp)
        # @show typeof(agent.replay)
        push!(agent.replay, exp)
    end
    

    update_demons!(agent.demon_learner,
                   agent,
                   agent.last_obs,
                   obs,
                   agent.last_state,
                   agent.last_action,
                   next_state,
                   next_action,
                   is_terminal,
                   r)

    #get intrinssic reward

    update_behaviour!(agent)
                      # agent.last_obs,
                      # obs,
                      # agent.last_state,
                      # agent.last_action,
                      # next_state,
                      # next_action,
                      # is_terminal,
                      # total_reward)

    agent.last_state = next_state
    agent.last_action = next_action
    agent.last_obs = obs
    return next_action
end

get_behaviour_pis(agent::AgentER, state, obs) =
    get_action(agent, state, obs)[2]

function μ_π(agent::AgentER, obs)
    state = proc_input(agent, obs)
    get_behaviour_pis(agent,state,obs)
end

function update_demons!(::QLearner, agent::AgentER, args...)#, obs, next_obs, state, action, next_state, next_action, is_terminal, env_reward)

    exp = StatsBase.sample(agent.replay, agent.batch_size)
    # (:s, :a, :ns, :na, :actprob, :actprob_prime, :env_r, :rint, :t))
    for traj in exp
        state = proc_input(agent, traj.s)
        next_state = proc_input(agent, traj.ns)
        update!(agent.demon_learner,
                agent.demons,
                traj.s,
                traj.ns,
                state,
                traj.a,
                next_state,
                traj.na,
                is_terminal,
                (s, obs) -> obs === traj.s ? traj.actprob : traj.actprob_prime, #get_behaviour_pis(agent, state, obs),
                traj.env_r)
    end

end

function update_demons!(::SRLearner, agent::AgentER, obs, next_obs, state, action, next_state, next_action, is_terminal, env_reward)

    # update reward part

    
    update_rew_part!(agent.demon_learner.update,
                     agent.demon_learner,
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
    
    # update SR part
    exp = StatsBase.sample(agent.replay, agent.batch_size)
    # (:s, :a, :ns, :na, :actprob, :actprob_prime, :env_r, :rint, :t))
    for traj in exp
        state = proc_input(agent, traj.s)
        next_state = proc_input(agent, traj.ns)
        update_sr_part!(agent.demon_learner.update,
                        agent.demon_learner,
                        agent.demons,
                        traj.s,
                        traj.ns,
                        state,
                        traj.a,
                        next_state,
                        traj.na,
                        is_terminal,
                        (s, obs) -> obs === traj.s ? traj.actprob : traj.actprob_prime, 
                        traj.env_r)
    end

end

function update_behaviour!(agent::AgentER) #, obs, next_obs, state, action, next_state, next_action, is_terminal, reward)

    # behaviour_pis = get_behaviour_pis(agent, state, obs)
    # next_behaviour_pis = get_behaviour_pis(agent, next_state, next_obs)

    #NOTE: Different call than demon updates as the reward and environment pseudotermination function
    #
    exp = StatsBase.sample(agent.replay, agent.batch_size)
    
    for traj in exp
        state = proc_input(agent, traj.s)
        next_state = proc_input(agent, traj.ns)
        update!(agent.behaviour_learner,
                agent.behaviour_demons,
                traj.s,
                traj.ns,
                state,
                traj.a,
                next_state,
                traj.na,
                is_terminal,
                (s, obs) -> obs === traj.s ? traj.actprob : traj.actprob_prime, #get_behaviour_pis(agent, state, obs),
                traj.rint)

    end
end

function get_demon_prediction(agent::AgentER, obs, action)
    ϕ = agent.state_constructor(obs)
    agent.demon_learner(ϕ, action)
end
