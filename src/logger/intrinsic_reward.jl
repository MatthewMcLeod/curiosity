using JLD2
using Statistics

mutable struct IntrinsicRewardLogger <: LoggerKeyData
    intrinsic_reward::Array{Float64,1}
    log_interval::Int

    function IntrinsicRewardLogger(logger_init_info)
        num_logged_steps = logger_init_info[LoggerInitKey.TOTAL_STEPS]
        new(zeros(num_logged_steps), 1)
    end
end

function lg_step!(self::IntrinsicRewardLogger, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)
        self.intrinsic_reward[ind] = get_reward(agent.intrinsic_reward, agent)
    end
end

function lg_episode_end!(self::IntrinsicRewardLogger, cur_step_in_episode, cur_step_total)
end

function save_log(self::IntrinsicRewardLogger, save_dict::Dict)
    save_dict[:intrinsic_reward] = self.intrinsic_reward
end

function extract_demon_weights(learner_weights,gvf_i,demon_learner)
    demon_weights = if demon_learner isa SRLearner
        immediate_rewards = learner_weights[2][gvf_i,:]
        num_SFs = demon_learner.feature_size
        ψ_weights = learner_weights[1][num_SFs*(gvf_i-1)+1:num_SFs*(gvf_i)]
        flattenall([immediate_rewards, ψ_weights])
    elseif demon_learner isa QLearner
        num_actions = demon_learner.num_actions
        gvf_w = learner_weights[(gvf_i-1)*num_actions + 1: gvf_i * num_actions,:]
        flattenall(gvf_w)
    else
        @warn "Not a valid type of demon learner for monitoring weight change contribution per demon"
    end
    return demon_weights
end

mutable struct WC_Demon_Logger <: LoggerKeyData
    wc_per_demon::Array{Float64,2}
    gvf_weights::Dict
    log_interval::Int
    function WC_Demon_Logger(logger_init_info)
        new(zeros(4,logger_init_info[LoggerInitKey.TOTAL_STEPS]), Dict(),1)
    end
end

function lg_start!(self::WC_Demon_Logger, env, agent)
    learner_weights = get_weights(agent.demon_learner)
    for gvf_i in 1:4
        gvf_weights = deepcopy(extract_demon_weights(learner_weights,gvf_i, agent.demon_learner))
        self.gvf_weights[gvf_i] = gvf_weights
    end
end

function lg_step!(self::WC_Demon_Logger, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    learner_weights = deepcopy(get_weights(agent.demon_learner))
    for gvf_i in 1:4
        gvf_weights = deepcopy(extract_demon_weights(learner_weights,gvf_i, agent.demon_learner))
        wc = sum(abs.(self.gvf_weights[gvf_i] - gvf_weights))
        self.wc_per_demon[gvf_i,cur_step_total] = wc
        self.gvf_weights[gvf_i] = gvf_weights
    end
end

function lg_episode_end!(self::WC_Demon_Logger, cur_step_in_episode, cur_step_total)
end

function save_log(self::WC_Demon_Logger, save_dict::Dict)
    save_dict[:wc_per_demon] = self.wc_per_demon
end
