using ProgressMeter

include("./plot_ic_reward.jl")
include("./plot_ic_goal_visitation.jl")

# Experiment parameters
log_interval = 100
num_steps = 60000


# Getting baseline TB
folder = "M:/globus/InterestTB/Experiment2_dpi/"
ic = ItemCollection(folder)

# println(diff(ic))
# # asdasd
search_ic = search(ic, Dict("demon_learner" => "SR", "demon_update" => "TB", "behaviour_learner" => "GPI"))
gpi_tb_ic = get_best(search_ic, ["eta",  "alpha_init"], :ttmaze_direct_error)

search_ic = search(ic, Dict("demon_learner" => "SR", "demon_update" => "TB", "behaviour_learner" => "Q"))
esarsa_tb_ic = get_best(search_ic, ["eta",  "alpha_init"], :ttmaze_direct_error)



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


folder = "M:/globus/InterestTB/Experiment2_ESARSA/"
ic = ItemCollection(folder)

p = plot()

@showprogress for config in configs
    local search_ic = search(ic, config[:search])
    local best_ic = get_best(search_ic, ["demon_alpha_init", "eta", "behaviour_alpha_init"], :ttmaze_direct_error)
    plot_ic_reward!(p, best_ic, :ttmaze_direct_error, log_interval; smooth_step=1, label=config[:label])
end

plot_ic_reward!(p, esarsa_tb_ic, :ttmaze_direct_error, log_interval; smooth_step=1, label="SR TB")
plot!(p, title="Q")

savefig("plotting/chunlok/generated_plots/Experiment2_rewards_esarsa.svg")




##########################################################################################
# Plotting configs with GPI behaviour
##########################################################################################

folder = "M:/globus/InterestTB/Experiment2_GPI/"
ic = ItemCollection(folder)

# # println(diff(ic))

# Plotting the RMSE plots for all configs with GPI behaviour
p = plot()
@showprogress for config in configs
    local search_ic = search(ic, config[:search])
    local best_ic = get_best(search_ic, ["demon_alpha_init", "eta", "behaviour_alpha_init"], :ttmaze_direct_error)
    plot_ic_reward!(p, best_ic, :ttmaze_direct_error, log_interval; smooth_step=1, label=config[:label])
end

plot_ic_reward!(p, gpi_tb_ic, :ttmaze_direct_error, log_interval; smooth_step=1, label="SR TB")
plot!(p, title="GPI")
savefig("plotting/chunlok/generated_plots/Experiment2_rewards_gpi.svg")

# Plotting goal visitations for each config for GPI.
# for config in configs
#     local p = plot()
#     local search_ic = search(ic, config[:search])
#     local best_ic = get_best(search_ic, ["demon_alpha_init", "eta", "behaviour_alpha_init"], :ttmaze_direct_error)
#     plot_ic_goal_visitation!(p, best_ic)
#     plot!(title=config[:label])
#     savefig("plotting/chunlok/generated_plots/Experiment2_GPI_goal_visitation_$(config[:label])")
# end


# Plotting reward per GVF
# best_ics = []
# for config in configs
#     local search_ic = search(ic, config[:search])
#     local best_ic = get_best(search_ic, ["demon_alpha_init", "eta", "behaviour_alpha_init"], :ttmaze_direct_error)
#     push!(best_ics, best_ic)
# end


# for gvf_i in 1:4
#     p = plot()
#     for (config_i, config) in enumerate(configs)
#         plot_ic_reward_per_gvf!(p, best_ics[config_i], :ttmaze_direct_error, log_interval, num_steps, gvf_i; smooth_step=1, label=config[:label])
#     end
#     plot!(title="GVF $(gvf_i)")
#     savefig("plotting/chunlok/generated_plots/Experiment2_GPI_rewards_gvf_$(gvf_i)")
# end







sr_configs = [
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

# SR reward plots
# p = plot()
# folder = "M:/globus/InterestTB/Experiment2_GPI/"
# ic = ItemCollection(folder)
# @showprogress for config in sr_configs
#     local search_ic = search(ic, config[:search])
#     local best_ic = get_best(search_ic, ["demon_alpha_init", "eta", "behaviour_alpha_init"], :ttmaze_direct_error)
#     plot_ic_reward!(p, best_ic, :ttmaze_direct_error, log_interval, num_steps; smooth_step=1, label="GPI " * config[:label])
# end

# folder = "M:/globus/InterestTB/Experiment2_ESARSA/"
# ic = ItemCollection(folder)

# @showprogress for config in sr_configs
#     local search_ic = search(ic, config[:search])
#     local best_ic = get_best(search_ic, ["demon_alpha_init", "eta", "behaviour_alpha_init"], :ttmaze_direct_error)
#     plot_ic_reward!(p, best_ic, :ttmaze_direct_error, log_interval, num_steps; smooth_step=1, label="ESARSA " * config[:label])
# end

# savefig("plotting/chunlok/generated_plots/Experiment2_rewards_SR.svg")