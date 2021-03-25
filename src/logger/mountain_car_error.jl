using JLD2
using Statistics

mutable struct MCError <: LoggerKeyData
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int

    function MCError(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/MCEvalSet.jld2") MCEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])

        new(zeros(2,num_logged_steps), MCEvalSet, logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_step!(self::MCError, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)

    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)
        Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
        err = mean((Q_est - self.eval_set["ests"]) .^ 2, dims=2)
        self.error[:,ind] = err
    end
end

function lg_episode_end!(self::MCError, cur_step_in_episode, cur_step_total)
end

function save_log(self::MCError, save_dict::Dict)
    save_dict[:mc_error] = self.error
end
