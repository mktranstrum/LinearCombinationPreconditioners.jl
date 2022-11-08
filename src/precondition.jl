"""
For linear problem with A(θ) = A0 + ∑_k θ_k A_k, finds a preconditioner that is well behaved as θ_i → ∞.

Returns:
T: Preconditioner
TA: Preconditioner times A
"""
function precondition(A0::M, θs, Aks) where M<:AbstractMatrix
    n = size(A0, 1)
    TA = copy(A0)
    Tmat = zero(A0)
    for i = 1:size(A0, 1)
        Tmat[i,i] = 1
    end
    for (θ, Ak) = zip(θs, Aks)
        TAk = Tmat * Ak
        C, Cperp = C_Cperp(TAk)
        Tnew = T(θ, C, Cperp)
        TA .= Tnew*TA + finite_xform(θ) * vcat( C'*TAk, Cperp'*TAk) # 2nd term should be zero
        Tmat .= Tnew*Tmat
    end
    return TA, Tmat
end
