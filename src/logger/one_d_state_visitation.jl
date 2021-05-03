using JLD2
using Statistics

num_bins = 12

mutable struct OneDStateVisitation <: LoggerKeyData
    state_visitations::Array{Int32,3}
    temp_state_visitations::Array{Int32, 2}
    log_interval::Int

    function OneDStateVisitation(logger_init_info)
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL]) + 1
        new(zeros(num_bins, num_bins, num_logged_steps), zeros(num_bins, num_bins), logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_start!(self::OneDStateVisitation, env, agent)
end

function lg_step!(self::OneDStateVisitation, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    x = ceil(Int32, s_next[1] * num_bins) 
    y = ceil(Int32, s_next[2] * num_bins)
    if (x == 0)
        x = 1
    end
    if (y == 0)
        y = 1
    end
    if (s_next[1] == 0.49999999999999994)
        println()
        println("found state")
        println("state: $(s[1:2])")
        println("next state: $(s_next[1:2])")
        print("================")
    end
    if (y == 3 && x != 6)
        println()
        println("BAD STATE")
        println("state: $(s[1:2])")
        println("next state: $(s_next[1:2])")
    end

    # println("state: $(s[1]), $(s[2])")
    # println("$(x), $(y)")

    # println(s)
    # print(x, y)

    self.temp_state_visitations[x, y] += 1
    if (rem(cur_step_total, self.log_interval) == 0)
        ind = fld(cur_step_total, self.log_interval) + 1
        self.state_visitations[:, :, ind] = self.temp_state_visitations

        # Reset temp state Visitation
        # println(sum(self.state_visitations))
        self.temp_state_visitations .*= 0
    end
end

function lg_episode_end!(self::OneDStateVisitation, cur_step_in_episode, cur_step_total)
end

function save_log(self::OneDStateVisitation, save_dict::Dict)
    save_dict[:oned_tmaze_state_visitation] = self.state_visitations
end
