using Zygote
using Random
using Flux

include("operators.jl")
include("diffrules.jl")
Random.seed!(1)

struct EquivLayer
    name::Symbol
    paths::AbstractVector{NamedTuple}
    pathsmap::AbstractVector
    dims::Int
    dx::Real
    rmax::Real
    σ::Any
    w::AbstractVector
    nonlinearity_params::AbstractMatrix
end

function Flux.trainable(m::EquivLayer)
    if m.name == :conv
        return vcat(
            [x.op.radfunc for x in m.paths],
            [m.nonlinearity_params],
            [m.w],
        )
    elseif m.name == :prod
        return m.w
    end
end

"""
    function EquivLayer(
        name,
        inranks,
        outranks;
        dims = 3,
        dx = 1.0,
        rmax = 1.0,
        rank_max = max(1, inranks..., outranks...),
        σ = identity
    )
"""
function EquivLayer(
    name,
    inranks,
    outranks;
    dims = 3,
    dx = 1.0,
    rmax = 1.0,
    rank_max = max(1, inranks..., outranks...),
    σ = identity,
)
    Zygote.ignore() do
        grid = Grid(dx, rmax; rank_max)

        nin = length(inranks)
        nout = length(outranks)
        inparities = (-1) .^ inranks
        outparities = (-1) .^ outranks

        paths = []
        pathsmap = [[] for i = 1:nout]
        # iterate all tensor product paths
        # if isempty(paths)

        if name == :conv
            for lf in (rmax == 0 ? [0] : 0:rank_max)
                for (i, li, si) in zip(1:nin, inranks, inparities)
                    loutrange = abs(li - lf):min(li + lf, rank_max)
                    parity = si * (-1)^lf
                    for (o, lo) in enumerate(outranks)
                        if lo in loutrange && outparities[o] == parity
                            ranks = (li, lf, lo)
                            io = (i, o)
                            op = LinearOperator(:neural; grid, ranks, rmax)
                            path = (; i, o, ranks, op)
                            push!(paths, path)
                        end
                        # end
                    end
                end
            end
        elseif name == :prod
            for (i1, li1, si1) in zip(1:nin, inranks, inparities)
                for (i2, li2, si2) in
                    zip(0:i1, [0, inranks[1:i1]...], [1, inparities[1:i1]...])
                    loutrange = abs(li1 - li2):min(li1 + li2, rank_max)
                    parity = si1 * si2
                    for (o, lo) in enumerate(outranks)
                        if lo in loutrange && outparities[o] == parity
                            prod = FieldProd(li1, li2, lo)
                            path = (; i1, i2, o, prod)
                            push!(paths, path)
                        end
                    end
                end
            end
        end

        for (i, path) in enumerate(paths)
            push!(pathsmap[path.o], i)
        end

        w = ones(length(paths))
        nonlinearity_params = ones(2, nout)
        return EquivLayer(
            name,
            paths,
            pathsmap,
            dims,
            dx,
            rmax,
            σ,
            w,
            nonlinearity_params,
        )
    end
end

function rescale(x, params, σ)
    a, b = params
    if length(x) == 1
        return [σ.(a * x[1] .+ b) .* x[1]]
    end
    # norm=field_norm(x)
    # rescales = [[σ.(norms[i] .+ b[i]) ./ (norms[i] .+ tol)] for i in eachindex(res)]
    # [Prod(0, field_rank(res[i]), field_rank(res[i]))(rescales[i], res[i]) for i in eachindex(res)]
end

"""
"""
function (f::EquivLayer)(X::AbstractVector, grid::Grid; remake = true)
    name, paths, pathsmap, dx, rmax, σ, w, nonlinearity_params = f.name,
    f.paths,
    f.pathsmap,
    f.dx,
    f.rmax,
    f.σ,
    f.w,
    f.nonlinearity_params

    if name == :conv
        pathfunc = i -> w[i] * paths[i].op(X[paths[i].i], grid)
        if remake
            for x in paths
                remake!(x.op)
            end
        end
    elseif name == :prod
        pathfunc =
            i ->
                w[i] * (
                    paths[i].i2 === 0 ? X[paths[i].i1] :
                    paths[i].prod(X[paths[i].i1], X[paths[i].i2])
                )
    end

    res = [
        sum([pathfunc(i) for i in pathsmap[o]]) for o in eachindex(pathsmap)
    ]

    if σ === identity
        return res
    end

    # rescales = [[σ.(norms[i] .+ b[i]) ./ (norms[i] .+ tol)] for i in eachindex(res)]
    # rescales = [[σ.(x .+ bi) ./ (x .+ tol)] for (x, bi) in zip(norms, b)]
    # [Prod(0, field_rank(x), field_rank(x))(n, x) for (n, x) in zip(norms, res)]
    # rescale.(res, eachcol(nonlinearity_params), σ)
end
# struct NormDense
#     f::Flux.Dense
#     # W::AbstractMatrix
#     # b::AbstractVector
# end
# # Flux.trainable(m::EquivAttn) = [x.params for x in m.paths]
#
#
# """
# initialize params
# """
# function NormDense(n)
#     f = Dense(n, n, swish)
#     return NormDense(f)
# end
#
# """
# layer function
# """
# function (f::NormDense)(X::AbstractVector;) where {T<:AbstractFloat}
#     W, b, σ = f.f.W, f.f.b, f.f.σ
#     sz = Base.size(X[1][1])
#     norms = tfnorm.(X)
#     norms = [σ.(x) for x in (W * norms + b .* ones(sz...))]
# end
