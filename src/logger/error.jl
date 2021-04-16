using JLD2
using Statistics

function lg_start!(self::E, env, agent) where {E <: ErrorRecorder}
    Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)
    true_values = self.get_true_values(env, self.eval_set["ests"])
    err = mean((Q_est - true_values) .^ 2, dims=2)
    self.error[:,1] = sqrt.(err)
end

function lg_step!(self::E, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total) where {E <: ErrorRecorder}
    if (rem(cur_step_total, self.log_interval) == 0)
        ind = fld(cur_step_total, self.log_interval) + 1
        Q_est = hcat([get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"], self.eval_set["actions"])]...)

        true_values = self.get_true_values(env, self.eval_set["ests"])
        err = mean((Q_est - true_values) .^ 2, dims=2)

        self.error[:,ind] = sqrt.(err)
    end
end

function lg_episode_end!(self::E, cur_step_in_episode, cur_step_total) where {E<:ErrorRecorder}
end

function save_log(self::E, save_dict::Dict) where {E<:ErrorRecorder}
    save_dict[self.save_key] = self.error
end

mutable struct OneDTMazeError <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int
    save_key::Symbol
    get_true_values::Function
    function OneDTMazeError(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/OneDTMazeEvalSet.jld2") OneDTMazeEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1
        new(zeros(4,num_logged_steps), OneDTMazeEvalSet, logger_init_info[LoggerInitKey.INTERVAL], :oned_tmaze_start_error, OneDTMazeUtils.get_true_values)
    end
end

mutable struct TTMazeError <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int
    save_key::Symbol
    get_true_values::Function
    function TTMazeError(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/TTMazeEvalSet.jld2") TTMazeEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1
        new(zeros(4,num_logged_steps), TTMazeEvalSet, logger_init_info[LoggerInitKey.INTERVAL],:ttmaze_error, TabularTMazeUtils.get_true_values)
    end
end

mutable struct TTMazeUniformError <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int
    save_key::Symbol
    get_true_values::Function
    function TTMazeUniformError(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/TTMazeUniformEvalSet.jld2") TTMazeUniformEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1

        new(zeros(4,num_logged_steps), TTMazeUniformEvalSet, logger_init_info[LoggerInitKey.INTERVAL], :ttmaze_uniform_error, TabularTMazeUtils.get_true_values)
    end
end
