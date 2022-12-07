
struct SparseRankOneMatrix{T} <: AbstractMatrix{T}
    u::SparseVector{T}
    v::SparseVector{T}
end

import Base: size, getindex, *

Base.size(m::SparseRankOneMatrix) = (length(m.u), length(m.v))

Base.getindex(m::SparseRankOneMatrix, I::Vararg{Int, 2}) = m.u[I[1]] * m.v[I[2]]

function Base.getindex(m::SparseRankOneMatrix, i::Int)
    M, N = size(m)
    q = div(i-1, M) + 1
    p = ((i-1) % M) + 1
    return m.u[p] * m.v[q]
end

Base.:*(T::M, m::SparseRankOneMatrix) where M<:AbstractMatrix = SparseRankOneMatrix( T*m.u, m.v)

Base.:*(m::SparseRankOneMatrix, T::M) where M<:AbstractMatrix = SparseRankOneMatrix( m.u, vec(m.v'*T))

Base.:*(m::SparseRankOneMatrix, n::SparseRankOneMatrix) where M<:AbstractMatrix = SparseRankOneMatrix( m.u, (m.v' * n.u) * n.v)


