using LinearCombinationPreconditioners
using LinearAlgebra
using SparseArrays
using Test

A0 = randn(5,5)
Aks = [ randn(5) * randn(5)' for _ = 1:3]
θ = ones(3)
TA, T = precondition(A0, θ, Aks)
A = A0 + sum( θ .* Aks)
@assert isapprox(T*A, TA)
θ[1] = Inf
TA, T = precondition(A0, θ, Aks)
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
Aks = [A1, A2, A3, A4]
θ = ones(4)
TA, T = precondition(A0, θ, Aks)
A = A0 + sum( θ .* Aks)
@assert isapprox(T*A, TA)
@assert any(.~isinf.(TA))
@assert any(.~isinf.(T))
