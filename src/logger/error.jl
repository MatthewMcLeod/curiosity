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

####
# Error Recording for 1D Tmaze
####

mutable struct OneDTMazeError <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int
    save_key::Symbol
    get_true_values::Function
    function OneDTMazeError(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/OneDTMazeEvalSet.jld2") OneDTMazeEvalSet
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1
        new(zeros(4,num_logged_steps), OneDTMazeEvalSet, logger_init_info[LoggerInitKey.INTERVAL], :oned_tmaze_old_error, OneDTMazeUtils.get_true_values)
    end
end

mutable struct OneDTMazeError_Uniform <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int
    save_key::Symbol
    get_true_values::Function
    function OneDTMazeError_Uniform(logger_init_info)
        eval_set = @load string(pwd(),"/src/data/OneDTMazeEvalSet_Uniform.jld2") OneDTMazeEvalSetUniform
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1
        new(zeros(4,num_logged_steps), OneDTMazeEvalSetUniform, logger_init_info[LoggerInitKey.INTERVAL], :oned_tmaze_uniform_error, OneDTMazeUtils.get_true_values)
    end
end

mutable struct OneDTMazeError_DPI <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int
    save_key::Symbol
    get_true_values::Function
    function OneDTMazeError_DPI(logger_init_info)
        @load string(pwd(),"/src/data/OneDTMazeEvalSet_d_pi.jld2") eval_set
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1
        new(zeros(4, num_logged_steps), eval_set, logger_init_info[LoggerInitKey.INTERVAL], :oned_tmaze_dpi_error, OneDTMazeUtils.get_true_values)
    end
end

function lg_start!(self::OneDTMazeError_DPI, env, agent)

    for gvf_idx ∈ 1:4
        Q_est = [get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"][gvf_idx], self.eval_set["actions"][gvf_idx])]
        true_values = self.get_true_values(env, self.eval_set["ests"][gvf_idx], gvf_idx)
        err = mean((getindex.(Q_est, gvf_idx) - true_values) .^ 2)
        self.error[gvf_idx, 1] = sqrt.(err)
    end

end

function lg_step!(self::OneDTMazeError_DPI, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if (rem(cur_step_total, self.log_interval) == 0)

        ind = fld(cur_step_total, self.log_interval) + 1
        for gvf_idx ∈ 1:4
            Q_est = [get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"][gvf_idx], self.eval_set["actions"][gvf_idx])]
            true_values = self.get_true_values(env, self.eval_set["ests"][gvf_idx], gvf_idx)
            err = mean((getindex.(Q_est, gvf_idx) - true_values) .^ 2)
            self.error[gvf_idx, ind] = sqrt.(err)
        end
    end
end


####
# Error Recording for Tabular Tmaze
####


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

####
# Error Recording for TwoD
####

mutable struct TwoDGridWorldError <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int
    save_key::Symbol
    get_true_values::Function
    function TwoDGridWorldError(logger_init_info)
        @load string(pwd(),"/src/data/TwoDGridWorldSet.jld2") eval_set
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1
        new(zeros(4, num_logged_steps), eval_set, logger_init_info[LoggerInitKey.INTERVAL], :twod_grid_world_error, TwoDGridWorldUtils.get_true_values)
    end
end

mutable struct TwoDGridWorldErrorCenterDPI <: ErrorRecorder
    error::Array{Float64,2}
    eval_set::Dict
    log_interval::Int
    save_key::Symbol
    get_true_values::Function
    function TwoDGridWorldErrorCenterDPI(logger_init_info)
        @load string(pwd(),"/src/data/TwoDGridWorldSet_dpi.jld2") eval_set
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1
        new(zeros(4, num_logged_steps), eval_set, logger_init_info[LoggerInitKey.INTERVAL], :twod_grid_world_error_center_dpi, TwoDGridWorldUtils.get_true_values)
    end
end

function lg_start!(self::TwoDGridWorldErrorCenterDPI, env, agent)

    for gvf_idx ∈ 1:4
        Q_est = [get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"][gvf_idx], self.eval_set["actions"][gvf_idx])]
        true_values = self.get_true_values(env, self.eval_set["ests"][gvf_idx], gvf_idx)
        err = mean((getindex.(Q_est, gvf_idx) - true_values) .^ 2)
        self.error[gvf_idx, 1] = sqrt.(err)
    end

end

function lg_step!(self::TwoDGridWorldErrorCenterDPI, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if (rem(cur_step_total, self.log_interval) == 0)

        ind = fld(cur_step_total, self.log_interval) + 1
        for gvf_idx ∈ 1:4
            Q_est = [get_demon_prediction(agent, state, Int(action)) for (state, action) in zip(self.eval_set["states"][gvf_idx], self.eval_set["actions"][gvf_idx])]
            true_values = self.get_true_values(env, self.eval_set["ests"][gvf_idx], gvf_idx)
            err = mean((getindex.(Q_est, gvf_idx) - true_values) .^ 2)
            self.error[gvf_idx, ind] = sqrt.(err)
        end
    end
end
