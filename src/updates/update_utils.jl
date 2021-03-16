function get_demon_pis(horde, num_actions, state, obs)
    target_pis = zeros(length(horde), num_actions)
    for (i,a) in enumerate(1:num_actions)
        target_pis[:,i] = get(demons[i], obs, a)
    end
    return target_pis
end
