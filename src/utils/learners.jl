function row_order_reshape(A, reshape_dims)
    #I dont think Julia natively supports row based reshape
    # https://github.com/JuliaLang/julia/issues/20311
    return transpose(reshape(transpose(A), reshape_dims...))
end

function get_active_action_state_vector(state::SparseVector, action, feature_size, num_actions)
    vec_length = feature_size * num_actions
    new_ind = (state.nzind .- 1) * num_actions .+ action
    active_state_action = sparsevec(new_ind, state.nzval, vec_length)
    return active_state_action
end
