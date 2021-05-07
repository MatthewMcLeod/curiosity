const TTMCS = TMazeCumulantSchedules

mutable struct Cumulants <: LoggerKeyData
    cs::Array{Float64,2}
    agent_pos::Array{Float64,2}
    function Cumulants(logger_init_info)
        num_logged_steps = logger_init_info[LoggerInitKey.TOTAL_STEPS]
        new(zeros(4,num_logged_steps),zeros(2,num_logged_steps))
    end
end

function lg_step!(self::Cumulants, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    true_values = TTMCS.get_cumulant_eval_values(env.cumulant_schedule)
    if s_next[3] != 0.0
        true_values[1] = s_next[3]
    end
    true_values[3] = s_next[5]
    self.cs[:,cur_step_total] = true_values
    self.agent_pos[:,cur_step_total] = s_next[1:2]
end

function lg_episode_end!(self::Cumulants, cur_step_in_episode, cur_step_total)
end

function save_log(self::Cumulants, save_dict::Dict)
    save_dict[:cumulants] = self.cs
    save_dict[:agent_pos] = self.agent_pos
end
