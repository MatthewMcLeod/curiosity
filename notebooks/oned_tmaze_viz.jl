### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ caa2cdba-a76b-11eb-3461-bb8695eb2e3b
using Revise, Curiosity, Plots, PlutoUI, Statistics, Random

# ╔═╡ 0af438b0-ffe6-4f8e-9077-2f7d37b3bf3f
using ProgressMeter

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
	"behaviour_gamma" => 0.95,
	"behaviour_learner" => "Q",
	"behaviour_update" => "ESARSA",
	"behaviour_reward_projector" => "base",
	"behaviour_rp_tilings" => 1,
	"behaviour_rp_tiles" => 16,
	"behaviour_trace" => "ReplacingTraces",
	"behaviour_opt" => "Auto",
	"behaviour_lambda" => 0.9,
	"behaviour_alpha_init" => 0.1,
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
	"demon_learner" => "SR",
	"demon_update" => "TB",
	"demon_policy_type" => "greedy_to_cumulant",
	"demon_opt" => "Auto",
	"demon_lambda" => 0.9,
	"demon_trace"=> "AccumulatingTraces",
	"demon_beta_m" => 0.99,
	"demon_beta_v" => 0.99,

	#shared
	"num_tiles" => 8,
	"num_tilings" =>2,
	"demon_rep" => "ideal_martha",
	# "demon_rep" => "tilecoding",
	"demon_num_tiles" => 6,
	"demon_num_tilings" => 1,
	"eta" => 0.2,

	# Environment Config
	"constant_target"=> [-10.0,10.0],
	"cumulant_schedule" => "DrifterDistractor",
	"distractor" => (1.0, 5.0),
	"drifter" => (1.0, sqrt(0.01)),
	"exploring_starts"=>"beg",
	"env_step_penalty" => -0.01,


	# Agent and Logger
	"horde_type" => "regular",
	"random_first_action" => false,
	"intrinsic_reward" => "weight_change",
	# "logger_keys" => [LoggerKey.TTMAZE_ERROR],
	"save_dir" => "OneDTMazeExperiment",
	"seed" => 1,
	"steps" => 10000,
	"use_external_reward" => true,

	"logger_keys" => [LoggerKey.ONEDTMAZEERROR, LoggerKey.ONED_GOAL_VISITATION, LoggerKey.EPISODE_LENGTH, LoggerKey.INTRINSIC_REWARD]
)

# ╔═╡ 6d51db0c-72b1-47ed-8ce2-8822f15473cc
env = OneDTMaze(Curiosity.TMazeCumulantSchedules.Constant(1), "beg")

# ╔═╡ db493356-572c-4475-93ac-f9c785d8cefd
let
	@show MinimalRLCore.step!(env, 2)
	plot(env)
end

# ╔═╡ 99ad1698-d4f7-449d-8865-18a5917bb47c
function main_experiment(parsed=default_args(); progress=false, working=false)

    num_steps = parsed["steps"]
    Random.seed!(parsed["seed"])

    cumulant_schedule = ODTMU.get_cumulant_schedule(parsed)

    # exploring_starts = parsed["exploring_starts"]
    env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"], parsed["env_step_penalty"])

    agent = OneDTmazeExperiment.construct_agent(parsed)
	@show parsed["behaviour_learner"],parsed["demon_learner"]

    goal_visitations = zeros(4)

	anim = Animation()

    eps = 1
    max_num_steps = num_steps
    steps = Int[]

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    while sum(steps) < max_num_steps
        cur_step = 0
        max_episode_steps = max_num_steps - sum(steps)
        tr, stp =
            run_episode!(env, agent, max_episode_steps) do (s, a, s_next, r, t)
                #This is a callback for every timestep where logger can go
                # agent is accesible in this scope
				plot(env)
                if t == true && working==true
                    goals = s_next[3:end]
                    f = findfirst(!iszero, goals)
                    goal_visitations[f] += 1
                end

                if progress
                    next!(prg_bar)
                end
                cur_step+=1
				frame(anim)

            end
        push!(steps, stp)
        eps += 1
    end
    if working == true
        println("Goal Visitation: ", goal_visitations)
    end
	anim
end

# ╔═╡ 15a72300-bd1a-4476-a3b6-d108035d0f25
anim = main_experiment(default_args(0.1),progress=true)

# ╔═╡ bcea5162-66e6-4e1f-8bcc-b4d738277213
mp4(anim,"../plotting/plots/tst2.mp4")

# ╔═╡ Cell order:
# ╠═caa2cdba-a76b-11eb-3461-bb8695eb2e3b
# ╠═cfb65788-a6be-4d5e-a404-3ebd38a1c539
# ╠═dfd01378-0c1d-4385-b84c-d06fb049364e
# ╠═578231af-8b90-4f2f-9745-989eae047a38
# ╠═6d51db0c-72b1-47ed-8ce2-8822f15473cc
# ╠═db493356-572c-4475-93ac-f9c785d8cefd
# ╠═99ad1698-d4f7-449d-8865-18a5917bb47c
# ╠═0af438b0-ffe6-4f8e-9077-2f7d37b3bf3f
# ╠═15a72300-bd1a-4476-a3b6-d108035d0f25
# ╠═bcea5162-66e6-4e1f-8bcc-b4d738277213
