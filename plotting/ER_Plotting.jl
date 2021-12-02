using Markdown
using InteractiveUtils

# ╔═╡ f9fb3324-bf08-11eb-3893-85f5df3d567d
using Revise

# ╔═╡ 216ccd98-4953-4d9d-b770-69caee1c3fcc
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 50c823af-0573-4be3-98b8-b40cc1c8e501
color_scheme = [
    colorant"#44AA99",
    colorant"#332288",
    colorant"#DDCC77",
    colorant"#999933",
    colorant"#CC6677",
    colorant"#AA4499",
    colorant"#DDDDDD",
	colorant"#117733",
	colorant"#882255",
	colorant"#88CCEE",
]

# ╔═╡ 44b42144-1ca4-4814-993f-6984687e3c7b
const RPU = ReproducePlotUtils

# ╔═╡ ded0bd85-55b5-4d22-8ad2-b510381aadae
ic_sr, dd_sr = RPU.load_data("../data/ER/OneDTMaze_RR_ER_cor_err/")

# ╔═╡ 303bfc74-2abc-492f-89ef-c5f15322cfec
data_sr_avg = RPU.get_line_data_for(
		ic_sr,
        ["batch_size"],
        ["alpha_init", "eta"];
        comp=:min,
		get_comp_data=(x)->sum(x["results"][:oned_tmaze_dmu_error][:, end-30]),
        get_data=(x)->mean(x["results"][:oned_tmaze_dmu_error], dims=1)[1, :])

# ╔═╡ e9527191-dcac-43b6-aafc-d8a7f9750917
data_sr = [RPU.get_line_data_for(
		ic_sr,
        ["batch_size"],
        ["alpha_init", "eta"];
        comp=:min,
		get_comp_data=(x)->sum(x["results"][:oned_tmaze_dmu_error][:, end-30]),
        get_data=(x)->x["results"][:oned_tmaze_dmu_error][i,:]) for i in 1:4]

# ╔═╡ d508ab1b-8016-459e-ba70-7553ca26ae04
let
	tls = ["Distractor", "Constant-1", "Drifter", "Constant-2"]
	plts = [plot(data_sr[i], label_idx="batch_size",
		 legendtitle=nothing, palette=color_scheme, title=tls[i], size=(800,800)) for i in 1:4]
	plot(plts...)
end

# ╔═╡ db7ffcb4-e6f9-4591-a35a-9fb4b7d5925f
FileIO.load(joinpath(ic_sr[1].folder_str, "results.jld2"))

# ╔═╡ c427e9d6-dbe7-49ae-bfc9-164612233a5d
ic_q, dd_q = RPU.load_data("../local_data/OneDTMaze_RR_ER_2/")

# ╔═╡ 151330f1-b8b5-4a62-be6d-37fe0870e263
data_q_avg = RPU.get_line_data_for(
		ic_q,
        ["batch_size"],
        ["alpha_init", "eta"];
        comp=:min,
		get_comp_data=(x)->sum(x["results"][:oned_tmaze_dmu_error][:, end-30]),
        get_data=(x)->mean(x["results"][:oned_tmaze_dmu_error], dims=1)[1, :])

# ╔═╡ d06a43c2-b4e7-48c7-b844-4a12ce7b8d98
let
	bs = 1
	plot(data_sr_avg, Dict("batch_size"=>bs), label="SR $(bs)", label_idx="batch_size",
		 legendtitle=nothing, palette=color_scheme)
	plot!(data_q_avg, Dict("batch_size"=>bs), label="Q $(bs)", label_idx="batch_size",
		 legendtitle=nothing, palette=color_scheme)
end

# ╔═╡ 0744dfcf-d695-4257-a2d1-a2433cc96428
data_q = [RPU.get_line_data_for(
		ic_q,
        ["batch_size"],
        ["alpha_init", "eta"];
        comp=:min,
		get_comp_data=(x)->sum(x["results"][:oned_tmaze_dmu_error][:, end-30]),
        get_data=(x)->x["results"][:oned_tmaze_dmu_error][i,:]) for i in 1:4]

# ╔═╡ 4e168ab6-4cc0-48c5-a5f4-ade5e4b87746
let
	tls = ["Distractor", "Constant-1", "Drifter", "Constant-2"]
	plts = [plot(data_q[i], label_idx="batch_size",
		 legendtitle=nothing, palette=color_scheme, title=tls[i], size=(800,800)) for i in 1:4]
	plot(plts...)
end
