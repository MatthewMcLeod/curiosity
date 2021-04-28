using Revise
using ProgressMeter


using Curiosity, Plots, PlutoUI, Statistics, Random

include("./experiments/1d-tmaze.jl")

ODTMU = Curiosity.OneDTMazeUtils

default_args(ϵ) =
Dict(
	"logger_interval" => 100,

	# Behaviour Items
	# "behaviour_eta" => 0.1/8,
	"behaviour_gamma" => 0.9,
	"behaviour_learner" => "RoundRobin",
	"behaviour_update" => "ESARSA",
	"behaviour_reward_projector" => "base",
	"behaviour_rp_tilings" => 1,
	"behaviour_rp_tiles" => 16,
	"behaviour_trace" => "AccumulatingTraces",
	"behaviour_opt" => "Descent",
	"behaviour_lambda" => 0.9,
	"exploration_param" => ϵ,
	"exploration_strategy" => "epsilon_greedy",
	"ϵ_range" => (0.4,0.1),
	"decay_period" => 5000,
	"warmup_steps" => 1000,
	"behaviour_w_init" => 10,

	# Demon Attributes
	"demon_alpha_init" => 0.1,
	# "demon_eta" => 0.1/8,
	"demon_discounts" => 0.9,
	"demon_learner" => "Q",
	"demon_update" => "TB",
	"demon_policy_type" => "greedy_to_cumulant",
	"demon_opt" => "Auto",
	"demon_lambda" => 0.9,
	"demon_trace"=> "AccumulatingTraces",
	"demon_beta_m" => 0.99,
	"demon_beta_v" => 0.99,

	#shared
	"num_tiles" => 2,
	"num_tilings" =>8,
	"demon_rep" => "ideal_martha",
	# "demon_rep" => "tilecoding",
	"demon_num_tiles" => 6,
	"demon_num_tilings" => 1,
	"eta" => 0.05,

	# Environment Config
	"constant_target"=> [-10.0,10.0],
	"cumulant_schedule" => "DrifterDistractor",
	"distractor" => (1.0, 5.0),
	"drifter" => (1.0, sqrt(0.01)),
	"exploring_starts"=>"beg",
	"env_step_penalty" => -0.005,


	# Agent and Logger
	"horde_type" => "regular",
	"intrinsic_reward" => "weight_change",
	# "logger_keys" => [LoggerKey.TTMAZE_ERROR],
	"save_dir" => "OneDTMazeExperiment",
	"seed" => 1,
	"steps" => 10000,
	"use_external_reward" => true,

	"logger_keys" => [LoggerKey.ONEDTMAZEERROR, LoggerKey.ONED_GOAL_VISITATION, LoggerKey.EPISODE_LENGTH, LoggerKey.INTRINSIC_REWARD]
)

env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"], parsed["env_step_penalty"])

begin
	start!(env)
	plot(env)
end