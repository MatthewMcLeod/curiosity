using JLD2
using Statistics

mutable struct TTMazeError <: LoggerKeyData
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int

    function TTMazeError(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/TTMazeEvalSet.jld2") TTMazeEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])

        new(zeros(4,num_logged_steps), TTMazeEvalSet, logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function step!(self::TTMazeError, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if s[1] == 3
        predictions = predict(agent.demon_learner, agent, agent.demon_weights, s, a)
        # println(predictions)
    end

    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)
        Q_est = hcat([predict(agent.demon_learner, agent, agent.demon_weights, state, action) for (state,action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)

        err = mean((Q_est - self.eval_set["ests"]) .^ 2, dims=2)
        self.error[:,ind] = err
    end
end

function episode_end!(self::TTMazeError, cur_step_in_episode, cur_step_total)
end

function save_log(self::TTMazeError, save_dict::Dict)
    save_dict[:ttmaze_error] = self.error
end
