
function get_demon_pis(horde::GVFSRHordes.SRHorde, num_actions, state, obs)
    target_pis = zeros(length(horde), num_actions)
    for a in 1:num_actions
        _, _, pi = get(horde, obs, a, obs, a)
        target_pis[:,a] = pi
    end
    return target_pis
end

function get_demon_pis(horde, num_actions, state, obs)
    target_pis = zeros(length(horde), num_actions)
    for i in 1:length(horde)
        for a in 1:num_actions
            target_pis[i, a] = get(GVFHordes.policy(horde.gvfs[i]), obs, a)
        end
    end
    return target_pis
end
