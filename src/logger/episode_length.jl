
mutable struct EpisodeLength <: LoggerKeyData
    episode_length::Array{Int64}
    step_counter::Int

    function EpisodeLength(logger_init_info)
        new([],0)
    end
end

function step!(self::EpisodeLength, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    self.step_counter += 1
    if is_terminal == true
        push!(self.episode_length, self.step_counter)
        self.step_counter = 0
    # else
    end

end

function episode_end!(self::EpisodeLength, cur_step_in_episode, cur_step_total)
    # Environment must have cut off trajectory for exceeding max length
    if self.step_counter > 0
        # subtract a step since the counting in step is forward
        push!(self.episode_length, self.step_counter)
        self.step_counter = 0
    end
end

function save_log(self::EpisodeLength, save_dict::Dict)
    save_dict[:episode_length] = self.episode_length
end
