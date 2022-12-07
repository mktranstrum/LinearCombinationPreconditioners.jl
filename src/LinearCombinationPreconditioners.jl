module LinearCombinationPreconditioners

export SparseRankOneMatrix, MatrixLinearCombination, precondition

using LinearAlgebra: qr, diag, I, QRCompactWY, diagm, norm, dot
using SparseArrays: spdiagm, spzeros, SparseVector
using SuiteSparse.SPQR: QRSparse

include("SparseRankOneMatrices.jl")
include("MatrixLinearCombinations.jl")
include("precondition.jl")
include("finite_xform.jl")
include("C_Cperp.jl")
include("T.jl")

end # module
