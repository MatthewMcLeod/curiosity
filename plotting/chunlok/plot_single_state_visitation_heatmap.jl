using Plots
using Reproduce
using JLD2
using FileIO
using ProgressMeter

previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"


p = plot()
folder = "OneDTMazeExperiment/RP_0_0xa59186950e76228a/"
save_file = "plotting/chunlok/generated_plots/state_vistation_heatmap.gif"


results_file = folder * "results.jld2"
@load results_file results
# println(results)

settings_file = folder * "settings.jld2"
settings = FileIO.load(settings_file)["parsed_args"]

state_visitation_data = results[:oned_tmaze_state_visitation]

num_log = size(state_visitation_data)[3]
num_frames = 50
frames_between = floor(Int32, num_log / num_frames)

prog = Progress(num_frames, 1)
anim = @animate for i ∈ 1:num_frames
    next!(prog)
    init_frame = i * frames_between
    last_frame = min((i+1) * frames_between, num_log)
    frame_info = state_visitation_data[:, :, init_frame:last_frame]

    frame_info = dropdims(sum(frame_info, dims=3), dims=3)
    # println(frame_info)
    display("text/plain", frame_info)
    asdasd

    # println(sum(frame_info))
    frame_info = frame_info ./ sum(frame_info)

    # println(size(frame_info))

    frame_info = permutedims(frame_info, [2, 1])
    cgradient = cgrad(:rainbow, scale=:exp)
    heatmap!(frame_info, title="$(i)", c=cgradient, clim=(0, 1), zscale=:log10)
end


gif(anim, save_file, fps = 2)
ProgressMeter.finish!(prog)


# gif(anim, save_file, fps=15)    
# savefig(save_file)




ENV["GKSwstype"] = previous_GKSwstype 