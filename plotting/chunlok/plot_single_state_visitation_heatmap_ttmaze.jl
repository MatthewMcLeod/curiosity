using Plots
using Reproduce
using JLD2
using FileIO
using ProgressMeter

function plot_single_state_visitation_heatmap(results, frames, log_interval; fps=2)
    previous_GKSwstype = get(ENV, "GKSwstype", "")
    ENV["GKSwstype"] = "100"
    
    save_file = "plotting/chunlok/generated_plots/ttmaze_state_vistation_heatmap.gif"
    p = plot()

    state_visitation_data = results[:ttmaze_state_visitation]
    
    num_log = size(state_visitation_data)[3]
    frames_between = floor(Int32, num_log / frames)
    
    prog = Progress(frames, 1)
    anim = @animate for i ∈ 1:frames
        next!(prog)
        init_frame = i * frames_between
        last_frame = min((i+1) * frames_between, num_log)
        frame_info = state_visitation_data[:, :, init_frame:last_frame]
    
        frame_info = dropdims(sum(frame_info, dims=3), dims=3)
    
        frame_info = frame_info ./ sum(frame_info)
    
        frame_info = permutedims(frame_info, [2, 1])
        cgradient = cgrad(:rainbow, scale=:exp)
        heatmap!(frame_info, title="Step: $(i * frames_between * log_interval)", c=cgradient, clim=(0, 1), zscale=:log10, yflip=true)
    end
    
    
    gif(anim, save_file, fps = fps)
    ProgressMeter.finish!(prog)
    println("State visitation heatmap animation saved to $(save_file)")
    
    
    
    
    ENV["GKSwstype"] = previous_GKSwstype 

end