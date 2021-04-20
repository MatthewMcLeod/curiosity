### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 194aeca0-9c99-11eb-35c7-891390734658
using Reproduce, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, NaNMath

# ╔═╡ f2964d90-9257-4131-924e-1be16e38b162
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

# ╔═╡ 51cea0d1-bdd3-49e5-8eae-e89c9edbe522
md"""

# Data Functions

"""

# ╔═╡ 2c3a4b8f-0f40-4445-9462-2644b1157c15
function mean_uneven(d::Vector{Array{F, 1}}) where {F}
    ret = zeros(F, maximum(length.(d)))
    n = zeros(Int, maximum(length.(d)))
    for v ∈ d
        ret[1:length(v)] .+= v
        n[1:length(v)] .+= 1
    end
    ret ./ n
end

# ╔═╡ 93549b83-fffc-421c-a23e-376ce5787a9e
function std_uneven(d::Vector{Array{F, 1}}) where {F}
    
    m = mean_uneven(d)
    
    ret = zeros(F, maximum(length.(d)))
    n = zeros(Int, maximum(length.(d)))
    for v ∈ d
        ret[1:length(v)] .+= (v .- m[1:length(v)]).^2
        n[1:length(v)] .+= 1
    end
    sqrt.(ret ./ n)
end

# ╔═╡ abb330d0-8144-4395-b22b-7262055cba3f
function Reproduce.search(f, ic::ItemCollection)

    found_items = Vector{Reproduce.Item}()
    for (item_idx, item) in enumerate(ic.items)
        if f(item)
            push!(found_items, item)
        end
    end
    return ItemCollection(found_items)

end

# ╔═╡ f41e07cd-e7e7-4fce-bed2-d310d1998ec9
load_data(item) = FileIO.load(joinpath(item.folder_str, "results.jld2"))
# 	catch
# 		@show "$(f) not loadable"
# 	end
# end

# ╔═╡ 4ed2764d-cf06-4dd8-8458-cac9a8f4d0a5
function get_runs(ic, get_data)
    tmp = get_data(FileIO.load(joinpath(ic[1].folder_str, "results.jld2")))
    d = typeof(tmp)[]
    for (idx, item) ∈ enumerate(ic)
        push!(d, get_data(load_data(item)))
    end
    d
end

# ╔═╡ 5d660aa4-8e6b-464a-bd90-af616a61fc43
function internal_func(
		ic, 
		param_keys;
		comp=findmax,
		get_comp_data, 
		get_data=get_comp_data)
	
	# @show length(ic)
    ic_diff = diff(ic)
	# @show ic_diff
    params = if param_keys isa String
        ic_diff[param_keys]
    else
        collect(Iterators.product([ic_diff[k] for k ∈ param_keys]...))
    end
    
    s = zeros(length(params))
    for (p_idx, p) ∈ enumerate(params)
        sub_ic = if param_keys isa String
            search(ic, Dict(param_keys=>p))
        else
            search(ic, Dict(param_keys[i]=>p[i] for i ∈ 1:length(p)))
        end
        s[p_idx] = mean(get_runs(sub_ic, get_comp_data))
    end
	
	v, idx = comp(s)
	
	# get data of best setting
	best_ic = search(
		ic, 
		Dict(param_keys[i]=>params[idx][i] for i ∈ 1:length(params[idx])))
	data = get_runs(best_ic, get_data)

    data, v, params[idx]
end

# ╔═╡ f5885dc3-c6e8-4af9-a78e-4f240bf88807
function get_line_data_for(
		ic::ItemCollection, 
		line_keys, 
		param_keys; 
		comp=findmin,
	    get_comp_data,
	    get_data)
	ic_diff = diff(ic)
	params = if line_keys isa String
        ic_diff[line_keys]
    else
        collect(Iterators.product([ic_diff[k] for k ∈ line_keys]...))
    end

	strg = []
	
	Threads.@threads for p_idx ∈ 1:length(params)
		p = params[p_idx]
		sub_ic = if line_keys isa String
            search(ic, Dict(line_keys=>p))
        else
            search(ic, Dict(line_keys[i]=>p[i] for i ∈ 1:length(p)))
        end
		if length(sub_ic) != 0
			d, c, ps = internal_func(
				sub_ic, 
				param_keys;
				get_comp_data=get_comp_data, 
				get_data=get_data)
			push!(strg, (params[p_idx], ps, d, c))
		end
	end
	strg
end

# ╔═╡ 883d2bf4-5fa7-452b-8cd2-8ca33886689f
function new_diff(items::Array{Reproduce.Item, 1};
                 exclude_keys::Union{Array{String,1}, Array{Symbol,1}} = Array{String, 1}(),
                 exclude_parse_values::Bool=true)

    kt = keytype(items[1].parsed_args)
    exclude_keys = kt.(exclude_keys)
    if exclude_parse_values == true
        exclude_keys = [exclude_keys; kt.([Reproduce.HASH_KEY, Reproduce.SAVE_NAME_KEY, Reproduce.GIT_INFO_KEY])]
    end
    diff_parsed = Dict{kt, Array}()
    for item in items
        tmp_dict = items[1] - item
        for key in filter((k)->k ∉ exclude_keys, keys(tmp_dict))
            if key ∉ keys(diff_parsed)
				diff_parsed[key] = Array{Any, 1}()
            end
			
            if tmp_dict[key][1] ∉ diff_parsed[key]
                push!(diff_parsed[key], tmp_dict[key][1])
				new_type = typeof(tmp_dict[key])
            end
            if tmp_dict[key][2] ∉ diff_parsed[key]
                push!(diff_parsed[key], tmp_dict[key][2])
            end
        end
    end
    for key in keys(diff_parsed)
		diff_parsed[key] = collect(promote(diff_parsed[key]...))
        sort!(diff_parsed[key])
    end
    return diff_parsed
end

# ╔═╡ c394afe9-7780-4e07-8e40-4f52265f8afa
new_diff(ic::Reproduce.ItemCollection; kwargs...) = new_diff(ic.items; kwargs...)

# ╔═╡ b749f892-15b8-4512-963a-ebc3e63b3f55
function get_oned_error(d; agg=(x)->mean(x;dims=1)[1, :])
	ret = agg(d["results"][:oned_tmaze_start_error])
	ret[isnan.(ret)] .= Inf
	ret
end

# ╔═╡ d6a4d95b-4953-48b7-a31a-e66a69a1c90c
begin
	get_oned_error_AUC(d) = sum(get_oned_error(d))
	get_oned_error_AUE(d; final_perc = 0.1) = begin	
		x = get_oned_error(d)
		len_inc = Int(clamp(round(length(x)*final_perc), 1, Inf))
		sum(x[end-len_inc:end])
	end
	get_oned_error_gvf(d; idx) = get_oned_error(d; agg=(x)->x[idx,:])
end

# ╔═╡ bf5417e3-965b-4bb4-948c-ed954ac18cdb
function get_gvf_data(sub_ic; get_comp_data=get_oned_error_AUC)
	# descent_tb_ic = search(ic, Dict(
	# 		"cumulant_schedule"=>"DrifterDistractor", 
	# 		"demon_opt"=>"Auto", 
	# 		"demon_update"=>"ESARSA"))
	data, v, ps = internal_func(
		sub_ic, 
		["num_tilings", "num_tiles", "demon_eta"]; 
		comp=findmin,
		get_comp_data = get_comp_data, 
		get_data=(x)->[get_oned_error_gvf(x; idx=i) for i ∈ 1:4])
	[(mean(getindex.(data, i)), std(getindex.(data, i))./sqrt(length(data))) for i in 1:4], ps
end

# ╔═╡ 7f05f324-2600-46d1-9468-4ee6b60fef57
md"""
# Data Analysis

Get Item Collections for OneDTMazeQ and OneDTMazeSR and combine.

Sarsa also run w/ Q learners, but failed for some optimizers.

"""

# ╔═╡ ea84781a-cc17-48f7-8cc6-cedb3bc06686
begin
	ic_q = search(ItemCollection("../local_data/OneDTMazeQ/")) do item
		item.parsed_args["demon_update"] ∈ ["ESARSA", "TB"]
	end
	ic_sr = ItemCollection("../local_data/OneDTMazeSR_ideal/")
	ic = ItemCollection([ic_q.items; ic_sr.items])
	diff_dict = new_diff(ic.items)
end

# ╔═╡ eeed36c6-326f-4cc9-93f4-143bb8cae54e
md"""
#### Get data with parameters chosen over AUC for all gvfs.

"""

# ╔═╡ 3c167a75-f726-4579-abb1-7e3308d5b704
begin
	gvf_best_data = get_line_data_for(
		ic, 
		["demon_learner", "demon_update", "demon_opt", "num_tiles", "num_tilings", "cumulant_schedule"], 
		["demon_eta"]; 
		comp=findmin,
	    get_comp_data = get_oned_error_AUC,
	    get_data = (x)->[get_oned_error_gvf(x; idx=i) for i ∈ 1:4])
end

# ╔═╡ d92544b9-b571-4bbd-af4b-db83d360dc68
function plot_gvf_data(ic, gvf_idx;	cumulant_schedule = "DrifterDistractor", demon_learner="Q", demon_update="TB")

	sub_ic = search(ic, Dict(
			"cumulant_schedule"=>cumulant_schedule, 
			"demon_learner"=>demon_learner, 
			"demon_update"=>demon_update))
	plt = nothing
	d = new_diff(sub_ic)
	prms = if "demon_opt" ∈ keys(d)
		d["demon_opt"]
	else
		[sub_ic.items[1].parsed_args["demon_opt"]]
	end
	@show prms
	for (demon_opt) ∈ prms
		sub_ic_2 = search(sub_ic,
			   Dict("demon_opt"=>demon_opt))
		data, params = get_gvf_data(sub_ic_2)
		if plt isa Nothing
			plt = plot(data[gvf_idx][1], ribbon=data[gvf_idx][2], label="$(demon_opt), $(demon_update), $(params)", title="GVF $(gvf_idx)", palette=color_scheme)
		else
			plot!(plt, data[gvf_idx][1], ribbon=data[gvf_idx][2], label="$(demon_opt), $(demon_update), $(params)", palette=color_scheme)
		end
	end
	plt
end

# ╔═╡ 76b499f8-950e-4fb5-9e5e-be7b42cc18e0
# plot([plot_gvf_data(ic, i; demon_learner="Q", demon_update="TB", cumulant_schedule="DrifterDistractor") for i ∈ 1:4]..., size=(800, 800))

# ╔═╡ d5e946b4-b534-46fe-b4f6-e6aaadf18bd4
# plot([plot_gvf_data(ic, i; demon_learner="SR", demon_update="TB") for i ∈ 1:4]..., size=(800, 800))

# ╔═╡ bc2f558d-f4e6-48df-9c1d-06c1f5f4cf69
function get_gvf_plot_data(ic; line_args)
	d = new_diff(ic)
	data_col = []
	for args ∈ Iterators.product([d[la] for la ∈ line_args]...)
		sub_ic_2 = search(ic,
			   Dict(line_args[i]=>args[i] for i ∈ 1:length(line_args)))
		if length(sub_ic_2) != 0
			data, params = get_gvf_data(sub_ic_2)
			push!(data_col, (data, params, args))
		end
	end
	data_col
end

# ╔═╡ b7d30b13-9e5f-4f0e-bb03-c58cf667328a
function plot_all_gvf_data(data_col, gvf_idx; plt_kwargs...)

	plt = nothing
	for d ∈ data_col
			data, params, args = d
			if plt isa Nothing
				plt = plot(data[gvf_idx][1], ribbon=data[gvf_idx][2], label="$(args), $(params)", title="GVF $(gvf_idx)"; plt_kwargs...)
			else
				plot!(plt, data[gvf_idx][1], ribbon=data[gvf_idx][2], label="$(args), $(params)"; plt_kwargs...)
			end
	end
	plt
end

# ╔═╡ 0ee7ee2c-8998-4062-a43d-9e0a5f34f1c6
# gvf_data_dd = get_gvf_plot_data(search(ic, Dict("cumulant_schedule"=>"DrifterDistractor")); 
# 	line_args=["demon_opt", "demon_update", "demon_learner"])

# ╔═╡ effda83c-8c65-4e75-b695-22f4644604ea
# begin
# 	for i ∈ 1:4
# 		plot_all_gvf_data(gvf_data_dd, i; palette=color_scheme, size=(600, 600), lw=2, xlims=(0,200), ylims=(0.0,1.5))
# 		savefig("oned_tmaze_dd_round_robin_gvf_$(i).pdf")
# 	end
# 	for i ∈ 1:4
# 		plot_all_gvf_data(gvf_data_dd, i; palette=color_scheme, size=(600, 600), lw=2, xlims=(0,200))
# 		savefig("oned_tmaze_dd_round_robin_gvf_$(i)_nl.pdf")
# 	end
# end

# ╔═╡ 2a50b3e8-d06b-4a1f-b274-4386ef8a4081
# gvf_data_c = get_gvf_plot_data(search(ic, Dict("cumulant_schedule"=>"Constant")); 
# 	line_args=["demon_opt", "demon_update", "demon_learner"])

# ╔═╡ b44f8824-72c6-4500-971a-4adf90dafed8
# begin
# 	for i ∈ 1:4
# 		plot_all_gvf_data(gvf_data_c, i; palette=color_scheme, size=(600, 600), lw=2, xlims=(0,200), ylims=(0.0,1.5))
# 		savefig("oned_tmaze_c_round_robin_gvf_$(i).pdf")
# 	end
# 	for i ∈ 1:4
# 		plot_all_gvf_data(gvf_data_c, i; palette=color_scheme, size=(600, 600), lw=2, xlims=(0,200))
# 		savefig("oned_tmaze_c_round_robin_gvf_$(i)_nl.pdf")
# 	end
# end

# ╔═╡ Cell order:
# ╠═194aeca0-9c99-11eb-35c7-891390734658
# ╟─f2964d90-9257-4131-924e-1be16e38b162
# ╟─51cea0d1-bdd3-49e5-8eae-e89c9edbe522
# ╟─2c3a4b8f-0f40-4445-9462-2644b1157c15
# ╟─93549b83-fffc-421c-a23e-376ce5787a9e
# ╟─abb330d0-8144-4395-b22b-7262055cba3f
# ╟─5d660aa4-8e6b-464a-bd90-af616a61fc43
# ╟─f41e07cd-e7e7-4fce-bed2-d310d1998ec9
# ╟─4ed2764d-cf06-4dd8-8458-cac9a8f4d0a5
# ╟─f5885dc3-c6e8-4af9-a78e-4f240bf88807
# ╟─883d2bf4-5fa7-452b-8cd2-8ca33886689f
# ╟─c394afe9-7780-4e07-8e40-4f52265f8afa
# ╟─b749f892-15b8-4512-963a-ebc3e63b3f55
# ╟─d6a4d95b-4953-48b7-a31a-e66a69a1c90c
# ╟─bf5417e3-965b-4bb4-948c-ed954ac18cdb
# ╟─7f05f324-2600-46d1-9468-4ee6b60fef57
# ╟─ea84781a-cc17-48f7-8cc6-cedb3bc06686
# ╟─eeed36c6-326f-4cc9-93f4-143bb8cae54e
# ╠═3c167a75-f726-4579-abb1-7e3308d5b704
# ╟─d92544b9-b571-4bbd-af4b-db83d360dc68
# ╠═76b499f8-950e-4fb5-9e5e-be7b42cc18e0
# ╠═d5e946b4-b534-46fe-b4f6-e6aaadf18bd4
# ╠═bc2f558d-f4e6-48df-9c1d-06c1f5f4cf69
# ╠═b7d30b13-9e5f-4f0e-bb03-c58cf667328a
# ╠═0ee7ee2c-8998-4062-a43d-9e0a5f34f1c6
# ╠═effda83c-8c65-4e75-b695-22f4644604ea
# ╠═2a50b3e8-d06b-4a1f-b274-4386ef8a4081
# ╠═b44f8824-72c6-4500-971a-4adf90dafed8
