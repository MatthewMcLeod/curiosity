using JLD2
using Statistics

mutable struct MCErrorUniform <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int

    function MCErrorUniform(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/MCLearnedEvalSet.jld2") MCLearnedEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1

        new(zeros(2,num_logged_steps), MCLearnedEvalSet, logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_step!(self::MCErrorUniform, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)

    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval) + 1
        Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
        # @show Q_est
        # @show get_demon_prediction(agent,[0.5,0.5],1)
        err = mean((Q_est - self.eval_set["ests"]) .^ 2, dims=2)
        self.error[:,ind] = sqrt.(err)
    end
end

function lg_episode_end!(self::MCErrorUniform, cur_step_in_episode, cur_step_total)
end

function save_log(self::MCErrorUniform, save_dict::Dict)
    save_dict[:mc_uniform_error] = self.error
end
function lg_start!(self::MCErrorUniform, env, agent)
    ind = 1
    Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
    err = mean((Q_est - self.eval_set["ests"]) .^ 2, dims=2)
    self.error[:,ind] = sqrt.(err)
end

mutable struct MCErrorStartStates <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int

    function MCErrorStartStates(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/MCEvalSet.jld2") MCEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1

        new(zeros(2,num_logged_steps), MCEvalSet, logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_step!(self::MCErrorStartStates, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)

    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval) + 1
        Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
        err = mean((Q_est - self.eval_set["ests"]) .^ 2, dims=2)
        self.error[:,ind] = sqrt.(err)
    end
end

function lg_episode_end!(self::MCErrorStartStates, cur_step_in_episode, cur_step_total)
end

function save_log(self::MCErrorStartStates, save_dict::Dict)
    save_dict[:mc_starts_error] = self.error
end
function lg_start!(self::MCErrorStartStates, env, agent)
    ind = 1
    Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
    err = mean((Q_est - self.eval_set["ests"]) .^ 2, dims=2)
    self.error[:,ind] = sqrt.(err)
end
