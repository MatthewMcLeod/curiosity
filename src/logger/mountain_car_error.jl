using JLD2
using Statistics

mutable struct MCError <: LoggerKeyData
    error::Array{Float64,2}
    eval_set::Dict

    function MCError()
        eval_set = @load "/home/matthewmcleod/Documents/Masters/curiosity/src/data/MCEvalSet.jld2" MCEvalSet
        new(zeros(2,10000), MCEvalSet)
    end
end

function step!(self::MCError, env, agent, s, a, s_next, r, t, total_steps)
    Q_est = hcat([predict(agent.demon_learner, agent, agent.demon_weights, s, a) for (state,action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
    err = mean((Q_est - self.eval_set["ests"]) .^ 2, dims=2)
    self.error[:,total_steps] = err
end

function save_log(self::MCError, save_dict::Dict)
    save_dict[:mc_error] = self.error
end
