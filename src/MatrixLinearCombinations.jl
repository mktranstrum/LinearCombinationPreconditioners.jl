"""
Structure for storing linear problem with A(θ) = A0 + ∑_k θ_k A_k, finds a preconditioner that is well behaved as θ_i → ∞.
"""
struct MatrixLinearCombination{M, K<:AbstractMatrix}
    A0::M
    Aks::Vector{K}
end

A0(mlc::MatrixLinearCombination) = mlc.A0

Aks(mlc::MatrixLinearCombination) = mlc.Aks
