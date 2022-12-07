include("GramSchmidt.jl")

"""
Returns basis for column space and its orthogonal complement using QR decomposition
"""
C_Cperp(A::AbstractMatrix) = C_Cperp(qr(A))

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

C_Cperp(A::SparseRankOneMatrix) = C_Cperp(A.u)

function C_Cperp(v::SparseVector)
    N = length(v)
    nn = length(v.nzval)
    U = hcat(v.nzval, diagm(nn, nn-1, ones(nn-1))) |> gramschmidt

    C = spzeros(N, 1)
    C[v.nzind, 1] .= U[:,1]

    Cperp = spzeros(N, N - 1)
    for icol = 1:nn - 1
        Cperp[v.nzind, icol] .= U[:,icol+1]
    end
    irows = trues(N)
    irows[v.nzind] .= false
    for (i,j) = zip((1:N)[irows], nn:N-1)
        Cperp[i,j] = 1
    end

    return C, Cperp
end
