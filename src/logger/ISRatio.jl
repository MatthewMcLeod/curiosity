using JLD2
using Statistics

mutable struct ISRatio <: LoggerKeyData
    isRatio::Array{Any}
    log_interval::Int

    function ISRatio(logger_init_info)
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])
        println("The ISRatio logger might be pretty inefficient and should probably only be used for Auto debug. If you are planning on running a sweep, it might be worth not logging step sizes.")
        new([], logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_step!(self::ISRatio, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)
        learner = agent.demon_learner
        lu = update(learner)
        @assert lu isa ETB ||  lu isa PriorTB "Update is not ETB or PriorTB Please set it to ETB or PriorTB  (or add my emphasis type here) or remove LoggerKey.IS_RATIO from logger keys"

        rho = lu.rho_logging
        push!(self.isRatio, copy(rho))
    end
end

function lg_episode_end!(self::ISRatio, cur_step_in_episode, cur_step_total)
end

function save_log(self::ISRatio, save_dict::Dict)
    # Throwing away the first info.
    save_dict[:is_ratio] = cat(self.isRatio[2:end]..., dims=2)
end