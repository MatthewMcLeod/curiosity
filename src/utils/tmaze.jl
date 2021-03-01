
module TabularTMazeUtils
import ..TabularTMazeCumulantSchedules
const TTMCS = TabularTMazeCumulantSchedules

DrifterDistractor(parsed) = begin
    if "drifter" ∈ keys(parsed)
        TTMCS.DrifterDistractor(
            parsed["constant_target"],
            parsed["drifter"][1],
            parsed["drifter"][2],
            parsed["distractor"][1],
            parsed["distractor"][2])
    else
        TTMCS.DrifterDistractor(
            parsed["constant_target"],
            parsed["drifter_init"],
            parsed["drifter_std"],
            parsed["distractor_mean"],
            parsed["distractor_std"])
    end
end

function get_cumulant_schedule(parsed)
    sched = parsed["cumulant_schedule"]
    if parsed["cumulant_schedule"] == "DrifterDistractor"
        DrifterDistractor(parsed)
    elseif parsed["cumulant_schedule"] == "Constant"
        if parsed["cumulant"] isa Number
            TTMCS.Constant(parsed["cumulant"])
        else
            TTMCS.Constant(parsed["cumulant"]...)
        end
            
    else
        throw("$(sched) Not Implemented")
    end
end


end
