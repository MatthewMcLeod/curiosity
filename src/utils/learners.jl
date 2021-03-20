function row_order_reshape(A, reshape_dims)
    #I dont think Julia natively supports row based reshape
    # https://github.com/JuliaLang/julia/issues/20311
    return transpose(reshape(transpose(A), reshape_dims...))
end




