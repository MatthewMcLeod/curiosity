
mutable struct EpisodeLength <: LoggerKeyData
    episode_length::Array{Int64}
    step_counter::Int

    function EpisodeLength()
        new([],1)
    end
end

function step!(self::EpisodeLength, env, agent, s, a, s_next, r, t)
    if t == false
        self.step_counter += 1
    else
        push!(self.episode_length, self.step_counter)
        self.step_counter = 1
    end

end

function save_log(self::EpisodeLength, save_dict::Dict)
    save_dict[:episode_length] = self.episode_length
end
