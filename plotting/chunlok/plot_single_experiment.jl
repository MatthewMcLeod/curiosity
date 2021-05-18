include("plot_single_reward.jl")
include("plot_single_intrinsic_reward.jl")
include("plot_single_state_visitation_heatmap_ttmaze.jl")
include("plot_single_goal_visitation.jl")
include("plot_single_autostep.jl")
include("plot_single_emphasis.jl")
include("plot_single_is_ratio.jl")

# folder = "TabularTMazeExperiment/RP_0_0x6c12151ed3862fec/"
folder = "OneDTMazeExperiment/RP_0_0xe5590dac76ee51e8/"

log_interval = 100
num_steps = 60000

# Loading results file
results_file = folder * "results.jld2"
@load results_file results

# Loading settings file
settings_file = folder * "settings.jld2"
settings = FileIO.load(settings_file)["parsed_args"]

# Various plotting scripts
plot_single_reward(results, log_interval, :oned_tmaze_dpi_error)
# plot_single_reward(results,log_interval, :ttmaze_direct_error)
# plot_single_intrinsic_reward(results)
plot_single_goal_visitation(results; step_size=20)
# plot_single_state_visitation_heatmap(results, 50, log_interval; fps=10)
plot_single_autostep(results, log_interval)
plot_single_emphasis(results, log_interval)