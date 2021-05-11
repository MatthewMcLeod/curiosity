using JLD2
using Statistics

num_bins = 12

mutable struct TTMazeStateVisitation <: LoggerKeyData
    state_visitations::Array{Int32,3}
    temp_state_visitations::Array{Int32, 2}
    log_interval::Int

    function TTMazeStateVisitation(logger_init_info)
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1
        new(zeros(7, 9, num_logged_steps), zeros(7, 9), logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_start!(self::TTMazeStateVisitation, env, agent)
end

function lg_step!(self::TTMazeStateVisitation, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)

    mask = valid_state_mask()
    env_state = convert(Integer, s[1])
    self.temp_state_visitations[mask[env_state][2], mask[env_state][1]] += 1

    if (rem(cur_step_total, self.log_interval) == 0)
        ind = fld(cur_step_total, self.log_interval) + 1
        self.state_visitations[:, :, ind] = self.temp_state_visitations

        # Reset temp state Visitation
        # println(sum(self.state_visitations))
        self.temp_state_visitations .*= 0
    end
end

function lg_episode_end!(self::TTMazeStateVisitation, cur_step_in_episode, cur_step_total)
end

function save_log(self::TTMazeStateVisitation, save_dict::Dict)
    save_dict[:ttmaze_state_visitation] = self.state_visitations
end
