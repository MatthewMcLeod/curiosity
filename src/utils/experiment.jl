using JLD2
using FileIO
import Reproduce


function save_setup(parsed;
    save_dir_key="save_dir",
    def_save_file="results.jld2",
    filter_keys=["verbose",
                 "working",
                 "exp_loc",
                 "visualize",
                 "progress",
                 "synopsis"])

    savefile = def_save_file
    Reproduce.create_info!(
        parsed,
        parsed[save_dir_key];
        filter_keys=filter_keys)
    savepath = Reproduce.get_save_dir(parsed)
    joinpath(savepath, def_save_file)
end

function save_results(savefile, results)
    JLD2.@save savefile results
end

function check_save_file_loadable(savefile)
    try
        JLD2.@load savefile results
    catch
        return false
    end
    return true
end

function experiment_wrapper(exp_func::Function, parsed, working)
    save_file = save_setup(parsed)
    if isfile(save_file) && Curiosity.check_save_file_loadable(save_file)
        return
    end
    logger = Logger(parsed, save_file)

    println(typeof(exp_func))
    ret = exp_func(parsed, logger)
    
    if working
        ret
    else
        logger_save(logger)
    end
end
