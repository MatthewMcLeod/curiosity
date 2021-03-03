using JLD2
using Statistics

mutable struct MCError <: LoggerKeyData
    error::Array{Float64,2}
    eval_set::Dict

    function MCError(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/MCEvalSet.jld2") MCEvalSet
        new(zeros(2,logger_init_info[LoggerInitKey.TOTAL_STEPS]), MCEvalSet)
    end
end

function step!(self::MCError, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    Q_est = hcat([predict(agent.demon_learner, agent, agent.demon_weights, s, a) for (state,action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
    err = mean((Q_est - self.eval_set["ests"]) .^ 2, dims=2)
    self.error[:,cur_step_total] = err
end

function episode_end!(self::MCError, cur_step_in_episode, cur_step_total)
end

function save_log(self::MCError, save_dict::Dict)
    save_dict[:mc_error] = self.error
end
