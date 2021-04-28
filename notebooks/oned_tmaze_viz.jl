### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ caa2cdba-a76b-11eb-3461-bb8695eb2e3b
using Revise, Curiosity, Plots, PlutoUI, Statistics, Random

# ╔═╡ cfb65788-a6be-4d5e-a404-3ebd38a1c539
include("../experiments/1d-tmaze.jl")

# ╔═╡ dfd01378-0c1d-4385-b84c-d06fb049364e
ODTMU = Curiosity.OneDTMazeUtils

# ╔═╡ 578231af-8b90-4f2f-9745-989eae047a38
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
	"steps" => 100,
	"use_external_reward" => true,

	"logger_keys" => [LoggerKey.ONEDTMAZEERROR, LoggerKey.ONED_GOAL_VISITATION, LoggerKey.EPISODE_LENGTH, LoggerKey.INTRINSIC_REWARD]
)

# ╔═╡ 6d51db0c-72b1-47ed-8ce2-8822f15473cc
env = OneDTMaze(Curiosity.TMazeCumulantSchedules.Constant(1), "beg")

# ╔═╡ dd541710-f2c0-4723-8f5c-03dd37240eb5
begin 
	# @userplot OneDTMaze
	# function Plots.plot(env::OneDTMaze)
	# 	p = scatter(env.pos[1:1],env.pos[2:2])
	# 	plot!([0.5,0.5],[0.0,0.8], label="", color=:black)
	# 	plot!([0.0,0.0],[0.6,1.0], label="", color=:black)
	# 	plot!([1.0,1.0],[0.6,1.0], label="", color=:black)
	# 	plot!([0.0,1.0],[0.8,0.8], label="", color=:black)
	# 	return p
	# end
end

# ╔═╡ db493356-572c-4475-93ac-f9c785d8cefd
let
	@show MinimalRLCore.step!(env, 1)
	plot(env)
end

# ╔═╡ 99ad1698-d4f7-449d-8865-18a5917bb47c
savefig("1dtmaze_example.pdf")

# ╔═╡ Cell order:
# ╠═caa2cdba-a76b-11eb-3461-bb8695eb2e3b
# ╠═cfb65788-a6be-4d5e-a404-3ebd38a1c539
# ╠═dfd01378-0c1d-4385-b84c-d06fb049364e
# ╠═578231af-8b90-4f2f-9745-989eae047a38
# ╠═6d51db0c-72b1-47ed-8ce2-8822f15473cc
# ╠═dd541710-f2c0-4723-8f5c-03dd37240eb5
# ╠═db493356-572c-4475-93ac-f9c785d8cefd
# ╠═99ad1698-d4f7-449d-8865-18a5917bb47c
