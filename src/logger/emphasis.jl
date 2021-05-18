using JLD2
using Statistics

mutable struct Emphasis <: LoggerKeyData
    emphasis::Array{Any}
    log_interval::Int

    function Emphasis(logger_init_info)
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])
        println("The Emphasis logger might be pretty inefficient and should probably only be used for Auto debug. If you are planning on running a sweep, it might be worth not logging step sizes.")
        new([], logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_step!(self::Emphasis, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)
        learner = agent.demon_learner
        lu = update(learner)
        @assert lu isa ETB "Update is not ETB Please set it to ETB (or add my emphasis type here) or remove LoggerKey.EMPHASIS from logger keys"

        emphasis = lu.emphasis_logging
        push!(self.emphasis, copy(emphasis))
    end
end

function lg_episode_end!(self::Emphasis, cur_step_in_episode, cur_step_total)
end

function save_log(self::Emphasis, save_dict::Dict)
    save_dict[:emphasis] = cat(self.emphasis[2:end]..., dims=2)
end