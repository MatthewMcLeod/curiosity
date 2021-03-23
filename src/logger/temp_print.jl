using JLD2
using Statistics

mutable struct TempPrint <: LoggerKeyData

    function TempPrint(logger_init_info)
        new()
    end
end

function lg_step!(self::TempPrint, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if cur_step_total % 500 == 0
        println("At step: ", cur_step_total)
    end
end

function lg_episode_end!(self::TempPrint, cur_step_in_episode, cur_step_total)
    println("Finished episode: ", cur_step_in_episode)
end

function save_log(self::TempPrint, save_dict::Dict)
end
