using JLD2
using Statistics

mutable struct TTMazeValueMap <: LoggerKeyData
    #(behaviour & GVF)x steps x states x actions
    error::Array{Float64,4}
    log_interval::Int

    function TTMazeValueMap(logger_init_info)
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])
        num_demons_and_behaviour = 5
        num_states = 21
        num_actions = 4
        new(zeros(num_demons_and_behaviour,num_logged_steps,num_states,num_actions), logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function step!(self::TTMazeValueMap, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)
        Q_est = hcat([predict(agent.demon_learner, agent, agent.demon_weights, state, action) for (state,action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
        err = mean((Q_est - self.eval_set["ests"]) .^ 2, dims=2)
        self.error[:,ind] = err
    end
end

function episode_end!(self::TTMazeValueMap, cur_step_in_episode, cur_step_total)
end

function save_log(self::TTMazeValueMap, save_dict::Dict)
    save_dict[:ttmaze_error] = self.error
end
