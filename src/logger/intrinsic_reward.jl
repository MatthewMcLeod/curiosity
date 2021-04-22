using JLD2
using Statistics

mutable struct IntrinsicRewardLogger <: LoggerKeyData
    intrinsic_reward::Array{Float64,1}
    log_interval::Int

    function IntrinsicRewardLogger(logger_init_info)
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])
        new(zeros(num_logged_steps), logger_init_info[LoggerInitKey.INTERVAL])
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
