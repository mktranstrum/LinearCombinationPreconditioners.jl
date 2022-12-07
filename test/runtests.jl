using LinearCombinationPreconditioners
using LinearAlgebra
using SparseArrays
using Test

mlc = MatrixLinearCombination(randn(5,5), [ randn(5) * randn(5)' for _ = 1:3])
θ = ones(3)
TA, T = precondition(mlc, θ)
A = mlc.A0 + sum( θ .* mlc.Aks)
@assert isapprox(T*A, TA)
θ[1] = Inf
TA, T = precondition(mlc, θ)
@assert any(.~isinf.(TA))
@assert any(.~isinf.(T))

A0 = spdiagm(ones(ComplexF64, 5))
A1 = spzeros(ComplexF64, 5, 5)
A1[1,1] = A1[2,2] = -1
A1[1,2] = A1[2,1] = 1
A2 = spzeros(ComplexF64, 5, 5)
A2[2,2] = A2[3,3] = -1
A2[2,3] = A2[3,2] = 1
A3 = spzeros(ComplexF64, 5, 5)
A3[3,3] = A3[4,4] = -1
A3[3,4] = A3[4,3] = 1
A4 = spzeros(ComplexF64, 5, 5)
A4[4,4] = A3[5,5] = -1
A4[4,5] = A3[5,4] = 1
mlc = MatrixLinearCombination(A0, [A1, A2, A3, A4])
θ = ones(4)
TA, T = precondition(mlc, θ)
A = A0 + sum( θ .* mlc.Aks)
@assert isapprox(T*A, TA)
@assert any(.~isinf.(TA))
@assert any(.~isinf.(T))

################################################################################
# Sparse Rank One Matrices
################################################################################

m = SparseRankOneMatrix(SparseVector(Float64[1, 2, 3]),
                        SparseVector(Float64[1, 0, 0, 0]))
@assert size(m) == (3,4)
m_dense = m.u * m.v' |> collect
@assert all(m .== m_dense)

T1 = sparse(randn(5,3))
@assert all( T1*m .== T1*m_dense)
T2 = sparse(randn(4, 5))
@assert all( m*T2 .== m_dense*T2)
m2 = SparseRankOneMatrix(SparseVector(Float64[1,0,0,0]),
                        SparseVector(Float64[1, 1]))
m2_dense = m2.u * m2.v' |> collect
@assert all(m*m2 .== m_dense * m2_dense)

import LinearCombinationPreconditioners: C_Cperp

C, Cperp = C_Cperp(m)
@assert all(isapprox.(C' * Cperp, 0; atol = eps()))
@assert all(isapprox.(Cperp' * m, 0; atol = eps()))

A0 = spdiagm(ones(ComplexF64, 5))
A1 = SparseRankOneMatrix( SparseVector(ComplexF64[1, -1, 0, 0, 0]),
                          SparseVector(ComplexF64[1, 1, 0, 0, 0]))
A2 = SparseRankOneMatrix( SparseVector(ComplexF64[0, 1, -1, 0, 0]),
                          SparseVector(ComplexF64[0, 1, 1, 0, 0]))
A3 = SparseRankOneMatrix( SparseVector(ComplexF64[0, 0, 1, -1, 0]),
                          SparseVector(ComplexF64[0, 0, 1, 1, 0]))
A4 = SparseRankOneMatrix( SparseVector(ComplexF64[0, 0, 0, 1, -1]),
                          SparseVector(ComplexF64[0, 0, 0, 1, 1]))
mlc = MatrixLinearCombination(A0, [A1, A2, A3, A4])
θ = ones(4)
TA, T = precondition(mlc, θ)
A = A0 + sum( θ .* mlc.Aks)
@assert isapprox(T*A, TA)
@assert any(.~isinf.(TA))
@assert any(.~isinf.(T))
