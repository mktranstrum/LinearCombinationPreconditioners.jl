# Assume dot(u,u) = 1
@inline proj(v, u) = dot(u, v)*u

function gramschmidt(V)
    n, k = size(V)
    U = zero(V)
    U[:,1] = V[:,1] ./ norm(V[:,1])
    for i = 2:k
        U[:,i] = V[:,i]
        for j = 1:i-1
            U[:,i] .-= proj(U[:,i], U[:,j])
        end
        U[:,i] ./= norm(U[:,i])
    end
    return U
end
