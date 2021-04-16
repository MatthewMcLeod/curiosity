using JLD2
using Statistics

mutable struct TTMazeErrorMap <: LoggerKeyData
    error_map::Array{Any}
    eval_set::Dict
    log_interval::Int
    est::Array{Any}

    function TTMazeErrorMap(logger_init_info)
        # eval_set = @load string(pwd(),"/src/data/TTMazeEvalSet.jld2") TTMazeEvalSet
        eval_set = @load string(pwd(),"/src/data/TTMazeUniformEvalSet.jld2") TTMazeUniformEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])

        new([], TTMazeUniformEvalSet, logger_init_info[LoggerInitKey.INTERVAL], [])
    end
end

function lg_step!(self::TTMazeErrorMap, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)
        Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
        true_values = TabularTMazeUtils.get_true_values(env, self.eval_set["ests"])
        err = (Q_est - true_values) .^ 2
        push!(self.error_map, err)
        push!(self.est, Q_est)
    end
end

function lg_episode_end!(self::TTMazeErrorMap, cur_step_in_episode, cur_step_total)
end

function save_log(self::TTMazeErrorMap, save_dict::Dict)
    save_dict[:ttmaze_error_map] = self.error_map
    save_dict[:ttmaze_error_map_eval_set] = self.eval_set
    save_dict[:ttmaze_error_map_est] = self.est
end
