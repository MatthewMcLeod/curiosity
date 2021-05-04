using Plots
using Reproduce
using JLD2
using FileIO
using Statistics
using ProgressMeter

include("plot_utils.jl")

previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

save_file = "plotting/chunlok/generated_plots/state_vistation_heatmap_reproduce.gif"


num_steps = 40000
log_interval = 100

p = plot()

ic = ItemCollection("./OneDTMaze_NEW_DYNAMICS_NEW_EVAL/")
best_ic = ic

data = load_results(best_ic, :oned_tmaze_state_visitation; return_type="array")

state_visitation_data = cat(data..., dims=4)

num_log = size(state_visitation_data)[3]
num_frames = 50
frames_between = floor(Int32, num_log / num_frames)

start_frame = 1
prog = Progress(num_frames, 1)
anim = @animate for i ∈ start_frame:num_frames
    next!(prog)
    init_frame = i * frames_between
    last_frame = min((i+1) * frames_between, num_log)
    frame_info = state_visitation_data[:, :, init_frame:last_frame, 12:12]
    # println(size(frame_info))

    # println(size(sum(frame_info, dims=(3, 4))))

    frame_info = dropdims(sum(frame_info, dims=(3,4)), dims=(3, 4))

    # println(sum(frame_info))
    frame_info = frame_info ./ sum(frame_info)

    # println(size(frame_info))

    frame_info = permutedims(frame_info, [2, 1])
    cgradient = cgrad(:rainbow, scale=:exp)
    heatmap!(frame_info, title="Step: $(i * frames_between * log_interval)", c=cgradient, clim=(0, 1), zscale=:log10)
end


gif(anim, save_file, fps = 2)
ProgressMeter.finish!(prog)


# gif(anim, save_file, fps=15)    
# savefig(save_file)




ENV["GKSwstype"] = previous_GKSwstype 