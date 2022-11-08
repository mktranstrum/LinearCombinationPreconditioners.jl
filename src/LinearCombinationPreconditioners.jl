module LinearCombinationPreconditioners

export precondition

using LinearAlgebra: qr, diag, I, QRCompactWY
using SparseArrays: spdiagm
using SuiteSparse.SPQR: QRSparse

include("precondition.jl")
include("finite_xform.jl")
include("C_Cperp.jl")
include("T.jl")

end # module
