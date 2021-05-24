import Reproduce

abstract type LoggerKeyData end

function lg_step!(self::LoggerKeyData, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    error("lg_step! for this logger is unimplemented")
end

function lg_episode_end!(self::LoggerKeyData, cur_step_in_episode, cur_step_total)
    error("lg_episode_end! for this logger is unimplemented")
end

function save_log(self::LoggerKeyData, save_dict::Dict)
    error("save_log for this logger is unimplemented")
end

abstract type ErrorRecorder <: LoggerKeyData end

include("error.jl")
include("goal_visitation.jl")
include("episode_length.jl")
include("mountain_car_error.jl")
include("cumulants.jl")
include("temp_print.jl")
include("value_map.jl")
include("autostep_stepsize.jl")
include("tabular_tmaze_old_error.jl")
include("tabular_tmaze_error_map.jl")
include("intrinsic_reward.jl")
include("behaviour_action_values.jl")
include("one_d_state_visitation.jl")
include("ttmaze_state_visitation.jl")
include("emphasis.jl")
include("ISRatio.jl")

# Module for scoping key names
module LoggerKey
    const GOAL_VISITATION = "GOAL_VISITATION"
    const ONED_GOAL_VISITATION = "ONED_GOAL_VISITATION"
    const EPISODE_LENGTH = "EPISODE_LENGTH"
    const MC_ERROR = "MC_ERROR"
    const TEMP_PRINT = "TEMP_PRINT"
    const TTMAZE_ERROR = "TTMAZE_ERROR"
    const VALUE_MAP = "VALUE_MAP"
    const TWODGRIDWORLDERROR = "TWODGRIDWORLDERROR"
    const TWODGRIDWORLDERRORDPI = "TWODGRIDWORLDERRORDPI"
    const AUTOSTEP_STEPSIZE = "AUTOSTEP_STEPSIZE"
    const TTMAZE_UNIFORM_ERROR = "TTMAZE_UNIFORM_ERROR"
    const TTMAZE_OLD_ERROR = "TTMAZE_OLD_ERROR"
    const TTMAZE_ERROR_MAP = "TTMAZE_ERROR_MAP"
    const INTRINSIC_REWARD = "INTRINSIC_REWARD"
    const TTMAZE_DIRECT_ERROR = "TTMAZE_DIRECT_ERROR"
    const TTMAZE_ROUNDROBIN_ERROR = "TTMAZE_ROUNDROBIN_ERROR"
    const BEHAVIOUR_ACTION_VALUES = "BEHAVIOUR_ACTION_VALUES"
    const ONED_STATE_VISITATION = "ONED_STATE_VISITATION"
    const CUMULANTS = "CUMULANTS"
    const WC_PER_DEMON = "WC_PER_DEMON"
    const ONEDTMAZEERROR = "ONEDTMAZEERROR" # this old error metric gives true uniform distribution, but this is not reachable with no action noise
    const ONEDTMAZEERROR_DPI = "ONEDTMAZEERROR_DPI" # error based on following d_pi for each GVF in the onedtmaze
    const ONEDTMAZEERROR_UNIFORM = "ONEDTMAZEERROR_UNIFORM" # error with states following RR but using uniform aciton weightings
    const ONEDTMAZEERROR_DMU = "ONEDTMAZEERROR_DMU" # State weighting according to round robin with exploring starts
    const TTMAZE_STATE_VISITATION = "TTMAZE_STATE_VISITATION"
    const EMPHASIS = "EMPHASIS"
    const IS_RATIO = "IS_RATIO"
end

module LoggerInitKey
    const TOTAL_STEPS = "TOTAL_STEPS"
    const INTERVAL = "INTERVAL"
    const ENV = "ENV"
end

const LOGGER_KEY_MAP = Dict(
    LoggerKey.GOAL_VISITATION => GoalVisitation,
    LoggerKey.EPISODE_LENGTH => EpisodeLength,
    LoggerKey.MC_ERROR => MCError,
    LoggerKey.TTMAZE_ERROR => TTMazeError,
    LoggerKey.TEMP_PRINT => TempPrint,
    LoggerKey.VALUE_MAP => ValueMap,
    LoggerKey.ONEDTMAZEERROR => OneDTMazeError,
    LoggerKey.TWODGRIDWORLDERROR => TwoDGridWorldError,
    LoggerKey.TWODGRIDWORLDERRORDPI => TwoDGridWorldErrorCenterDPI,
    LoggerKey.ONED_GOAL_VISITATION => OneDGoalVisitation,
    LoggerKey.AUTOSTEP_STEPSIZE => AutostepStepSize,
    LoggerKey.TTMAZE_UNIFORM_ERROR => TTMazeUniformError,
    LoggerKey.TTMAZE_OLD_ERROR => TTMazeOldError,
    LoggerKey.TTMAZE_ERROR_MAP => TTMazeErrorMap,
    LoggerKey.TTMAZE_ROUNDROBIN_ERROR => TTMazeRoundRobinError,
    LoggerKey.INTRINSIC_REWARD => IntrinsicRewardLogger,
    LoggerKey.TTMAZE_DIRECT_ERROR => TTMazeDirectError,
    LoggerKey.BEHAVIOUR_ACTION_VALUES => BehaviourActionValues,
    LoggerKey.ONED_STATE_VISITATION => OneDStateVisitation,
    LoggerKey.CUMULANTS => Cumulants,
    LoggerKey.WC_PER_DEMON => WC_Demon_Logger,
    LoggerKey.ONEDTMAZEERROR_DPI => OneDTMazeError_DPI,
    LoggerKey.ONEDTMAZEERROR_UNIFORM => OneDTMazeError_Uniform,
    LoggerKey.TTMAZE_STATE_VISITATION => TTMazeStateVisitation,
    LoggerKey.EMPHASIS => Emphasis,
    LoggerKey.IS_RATIO => ISRatio,
    LoggerKey.ONEDTMAZEERROR_DMU => OneDTMazeError_dmu,
)

# Common logger for all experiments. It has multiple functionalities so pass in what you need to get started
mutable struct Logger
    save_file:: String
    # Tracks what needs to be logged
    logger_keys::Array{String}
    # Array of structs for logger logic
    logger_key_data::Array{LoggerKeyData}

    # This should only track info that makes sense to track across ALL loggers.
    cur_step_in_episode::Int64
    cur_step_total::Int64

    function Logger(parsed, logger_init_info, save_file)
        logger_keys = parsed["logger_keys"]
        logger_key_data = LoggerKeyData[]
        save_results = Dict()

        cur_step_in_episode = 0;
        cur_step_total = 0;


        # Set up for each key
        for key in logger_keys
            if haskey(LOGGER_KEY_MAP, key)
                push!(logger_key_data, LOGGER_KEY_MAP[key](logger_init_info))
            else
                throw(ArgumentError("Logger Key $(key) is not in the LOGGER_KEY_MAP"))
            end
        end
        new(save_file, logger_keys, logger_key_data, cur_step_in_episode, cur_step_total)
    end
end

function logger_start!(self::Logger, env, agent)
    for data in self.logger_key_data
        if applicable(lg_start!, data, env, agent)
            lg_start!(data, env, agent)
        # else
        #     @warn "No lg_start! defined for $(typeof(data))"
        end
    end
end

function logger_step!(self::Logger, env, agent, s, a, s_next, r, is_terminal)
    self.cur_step_in_episode += 1;
    self.cur_step_total += 1;

    for data in self.logger_key_data
        lg_step!(data, env, agent, s, a, s_next, r, is_terminal, self.cur_step_in_episode, self.cur_step_total)
    end
end

function logger_episode_end!(self::Logger)
    for data in self.logger_key_data
        lg_episode_end!(data, self.cur_step_in_episode, self.cur_step_total)
    end

    self.cur_step_in_episode = 0
end

function logger_save(self::Logger)
    save_dict = Dict()
    for data in self.logger_key_data
        save_log(data, save_dict)
    end
    save_results(self.save_file, save_dict)
end
