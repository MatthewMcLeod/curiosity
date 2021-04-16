using JLD2
using Statistics

mutable struct OneDTMazeError <: LoggerKeyData
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int

    function OneDTMazeError(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/OneDTMazeEvalSet.jld2") OneDTMazeEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1

        new(zeros(4,num_logged_steps), OneDTMazeEvalSet, logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_start!(self::OneDTMazeError, env, agent)
    Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
    true_values = OneDTMazeUtils.get_true_values(env, self.eval_set["ests"])
    err = mean((Q_est - true_values) .^ 2, dims=2)
    self.error[:,1] = sqrt.(err)
end

function lg_step!(self::OneDTMazeError, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if (rem(cur_step_total, self.log_interval) == 0)
        ind = fld(cur_step_total, self.log_interval) + 1
        Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)

        true_values = OneDTMazeUtils.get_true_values(env, self.eval_set["ests"])
        err = mean((Q_est - true_values) .^ 2, dims=2)

        self.error[:,ind] = sqrt.(err)
    end
end

function lg_episode_end!(self::OneDTMazeError, cur_step_in_episode, cur_step_total)
end

function save_log(self::OneDTMazeError, save_dict::Dict)
    save_dict[:oned_tmaze_start_error] = self.error
end
