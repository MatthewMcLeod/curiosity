using JLD2
using Statistics

mutable struct AutostepStepSize <: LoggerKeyData
    step_sizes::Array{Float64,2}
    log_interval::Int

    function AutostepStepSize(logger_init_info)
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])
        new(zeros(4,num_logged_steps), logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function step!(self::AutostepStepSize, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)

        # println(update(agent.demon_learner))

        demon_lu = update(agent.demon_learner)
        # @assert demon_lu is a Auto

        println(typeof(demon_lu))

        # err = mean((Q_est - true_values) .^ 2, dims=2)
        # self.error[:,ind] = err
    end
end

function episode_end!(self::AutostepStepSize, cur_step_in_episode, cur_step_total)
end

function save_log(self::AutostepStepSize, save_dict::Dict)
    save_dict[:autostep_stepsize] = self.step_sizes
end
