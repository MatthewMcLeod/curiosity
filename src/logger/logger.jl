import Reproduce

# Common logger for all experiments. It has multiple functionalities so pass in what you need to get started
mutable struct Logger
    save_file:: String
    save_results::Dict

    # Add any temp things here. I'm prioritizing flexibility over fixed structs right now...
    temp_dict::Dict

    function Logger(parsed, save_file)
        # Initialize Me
        new(save_file, Dict(), Dict())
    end
end

function logger_step!(self::Logger, env, agent, s, a, s_next, r)
    # println(typeof(env), typeof(agent), typeof(s), typeof(a), typeof(s_next), typeof(r))
    # logger_step!(logger, env, agent, s, a, s_next, r)
    # Look in https://github.com/mkschleg/ActionRNNs.jl/blob/master/src/utils/experiment.jl
end

function logger_save(self::Logger)
    # Add any post-processing needed here on the save_results

    save_results(self.save_file, self.save_results)
end