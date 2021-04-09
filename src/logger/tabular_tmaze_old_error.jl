using JLD2
using Statistics

mutable struct TTMazeOldError <: LoggerKeyData
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int

    function TTMazeOldError(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/TTMazeOldEvalSet.jld2") TTMazeOldEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])

        new(zeros(4,num_logged_steps), TTMazeOldEvalSet, logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_step!(self::TTMazeOldError, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)
        # Q_est = hcat([predict(agent.demon_learner, agent, agent.demon_weights, state, action) for (state,action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
        num_demons = 4
        true_values = TabularTMazeUtils.get_true_values(env, self.eval_set["ests"])

        for gvf_i in 1:num_demons
            errs = []
            for (sample_i,(state,action)) in enumerate(zip(self.eval_set["states"], self.eval_set["actions"]))
                if self.eval_set["is_action"][gvf_i,sample_i]
                    pred = get_demon_prediction(agent, state, Int(action))[gvf_i]
                    err = (pred - true_values[gvf_i, sample_i]) .^2
                    push!(errs, err)
                end
            end
            self.error[gvf_i,ind] = sqrt(mean(errs))
        end


        # Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
        # true_values = TabularTMazeUtils.get_true_values(env, self.eval_set["ests"])
        # err = sqrt.(mean((Q_est - true_values) .^ 2, dims=2))
        # self.error[:,ind] = err
    end
end

function lg_episode_end!(self::TTMazeOldError, cur_step_in_episode, cur_step_total)
end

function save_log(self::TTMazeOldError, save_dict::Dict)
    save_dict[:ttmaze_old_error] = self.error
end
