using ProgressMeter

include("./plot_ic_reward.jl")

folder = "M:/globus/InterestTB/Experiment1/"
ic = ItemCollection(folder)

# Experiment parameters
log_interval = 50
num_steps = 2000

p = plot()


configs = [
    Dict(
        :search => Dict("demon_learner" => "Q", "demon_update" => "ETB"),
        :label => "Q ETB"
    )
    Dict(
        :search => Dict("demon_learner" => "Q", "demon_update" => "EmphESARSA"),
        :label => "Q EmphESARSA"
    )
    Dict(
        :search => Dict("demon_learner" => "Q", "demon_update" => "InterestTB"),
        :label => "Q InterestTB"
    )
    Dict(
        :search => Dict("demon_learner" => "SR", "demon_update" => "ETB"),
        :label => "SR ETB"
    )
    Dict(
        :search => Dict("demon_learner" => "SR", "demon_update" => "EmphESARSA"),
        :label => "SR EmphESARSA"
    )
    Dict(
        :search => Dict("demon_learner" => "SR", "demon_update" => "InterestTB"),
        :label => "SR InterestTB"
    )
]

@showprogress for config in configs
    local search_ic = search(ic, config[:search])
    local best_ic = get_best(search_ic, ["demon_alpha_init", "demon_eta"], :ttmaze_direct_error)
    plot_ic_reward!(p, best_ic, :ttmaze_direct_error, log_interval; smooth_step=1, label=config[:label])
end

savefig("plotting/chunlok/generated_plots/Experiment1_rewards.svg")
