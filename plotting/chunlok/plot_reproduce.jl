using Plots
using Reproduce
using JLD2
using FileIO
using Statistics

include("plot_utils.jl")

ic = ItemCollection("./TMazeDrifterDIstractor")

ic = search(ic, Dict(
    "demon_learner" => "Q",
    "demon_update" => "TBAuto"
))

println(diff(ic))

sweep_params = ["demon_alpha"]


best_ic = get_best(ic, sweep_params, :ttmaze_error)

print_params(best_ic, sweep_params, ["distractor", "demon_alpha_init"])

# print(best_ic)


data = load_results(best_ic, :ttmaze_error)

println(size(data))

p = plot()
for i in 1:4
    # if i == 3
    #     continue
    # end
    line = mean(data[i, :, :], dims=2)
    # print(line[1:5])
    line = line .^0.5 
    # print(line[1:5])
    # print(size(line))
    plot!(p, 1:400, line)
end

# println("done")
savefig("plotting/chunlok_plots/generated_plots/test_plot_reproduce.svg")
© 2021 GitHub, Inc.