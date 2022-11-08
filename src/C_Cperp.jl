"""
Returns basis for column space and its orthogonal complement using QR decomposition
"""
C_Cperp(A) = C_Cperp(qr(A))

function C_Cperp(qrfact::QRCompactWY)
    n = size(qrfact.Q, 1)
    absdiagR = abs.(diag(qrfact.R))
    indices = sortperm(absdiagR)
    rnk = count(absdiagR .> 1e-12) # This could be done better
    Cperpindices = indices[1:n-rnk]
    Cindices = indices[n-rnk+1:end]
    return qrfact.Q[:,Cindices], qrfact.Q[:,Cperpindices]    
end

function C_Cperp(qrfact::QRSparse)
    n = size(qrfact.Q, 1)
    q = inv_permutation(qrfact.prow)
    absdiagR = abs.(diag(qrfact.R))
    rnk = count(absdiagR .> 1e-12) # Same as above
    Cindices = 1:rnk
    Cperpindices = rnk+1:n
    return qrfact.Q[q,Cindices], qrfact.Q[q,Cperpindices]
end

function inv_permutation(p::Vector{Int})
    q = Vector{Int}(undef, length(p))
    for i = 1:length(p)
        q[p[i]] = i
    end
    return q
end
