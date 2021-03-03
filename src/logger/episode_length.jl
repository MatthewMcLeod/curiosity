
mutable struct EpisodeLength <: LoggerKeyData
    episode_length::Array{Int64}
    step_counter::Int

    function EpisodeLength(logger_init_info)
        new([],1)
    end
end

function step!(self::EpisodeLength, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if is_terminal == false
        self.step_counter += 1
    else
        push!(self.episode_length, self.step_counter)
        self.step_counter = 1
    end
end

function episode_end!(self::EpisodeLength, cur_step_in_episode, cur_step_total)
end

function save_log(self::EpisodeLength, save_dict::Dict)
    save_dict[:episode_length] = self.episode_length
end
