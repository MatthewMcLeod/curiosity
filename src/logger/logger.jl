import Reproduce

abstract type LoggerKeyData end

include("goal_visitation.jl")
include("episode_length.jl")
include("mountain_car_error.jl")

# Module for scoping key names
module LoggerKey
    const GOAL_VISITATION = "GOAL_VISITATION"
    const EPISODE_LENGTH = "EPISODE_LENGTH"
    const MC_ERROR = "MC_ERROR"
end

const LOGGER_KEY_MAP = Dict(
    LoggerKey.GOAL_VISITATION => GoalVisitation(),
    LoggerKey.EPISODE_LENGTH => EpisodeLength(),
    LoggerKey.MC_ERROR => MCError()
)

# Common logger for all experiments. It has multiple functionalities so pass in what you need to get started
mutable struct Logger
    save_file:: String
    # Tracks what needs to be logged
    logger_keys::Array{String}
    # Array of structs for logger logic
    logger_key_data::Array{LoggerKeyData}

    function Logger(parsed, save_file)
        logger_keys = parsed["logger_keys"]
        logger_key_data = LoggerKeyData[]
        save_results = Dict()


        # Set up for each key
        for key in logger_keys
            if haskey(LOGGER_KEY_MAP, key)
                push!(logger_key_data, LOGGER_KEY_MAP[key])
            else
                throw(ArgumentError("Logger Key $(key) is not in the LOGGER_KEY_MAP"))
            end
        end
        new(save_file, logger_keys, logger_key_data)
    end
end

function logger_step!(self::Logger, env, agent, s, a, s_next, r, is_terminal, total_steps)
    for data in self.logger_key_data
        step!(data, env, agent, s, a, s_next, r, is_terminal, total_steps)
    end
end

function logger_save(self::Logger)
    save_dict = Dict()
    for data in self.logger_key_data
        save_log(data, save_dict)
    end
    save_results(self.save_file, save_dict)
end
