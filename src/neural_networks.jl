using Zygote
using Random
using Flux
using UnPack


include("operators.jl")
include("diffrules.jl")
Random.seed!(1)

struct EquivConv
    in::Props
    out::Props
    paths::AbstractVector{NamedTuple}
    pathsmap::AbstractVector
    dx::Real
    rmax::Real
end

function Flux.trainable(m::EquivConv)
    [x.op.radfunc for x in m.paths]
end
function sep(x, nranks)
    [body]
end
"""
    function EquivConv(
        name,
        in.ranks,
        out.ranks;
        dims = 3,
        dx = 1.0,
        rmax = 1.0,
        rank_max = max(1, in.ranks..., out.ranks...),
        σ = identity
    )
"""
function EquivConv(
    in_::Props,
    out::Props,
    rmax;
    dims = 3,
    rank_max = max(1, length(in_.nranks), length(out.nranks))-1,
)
    Zygote.ignore() do
        grid = Grid(dx, rmax; rank_max)
        nin = length(in_.ranks)
        nout = length(out.ranks)

        paths = []
        pathsmap = [[] for i = 1:nout]
        # iterate all tensor product paths
        # if isempty(paths)

        for lf in (rmax == 0 ? [0] : 0:rank_max)
            for (i, li, si) in zip(1:nin, in_.ranks, in_.parities)
                loutrange = abs(li - lf):min(li + lf, rank_max)
                parity = si * (-1)^lf
                for (o, lo) in enumerate(out.ranks)
                    if (lo in loutrange) && out.parities[o] == parity
                        ranks = (li, lf, lo)
                        op = LinearOperator(:neural; grid, ranks, rmax)
                        path = (; i, o, ranks, op)
                        push!(paths, path)
                    end
                    # end
                end
            end
        end

        for (i, path) in enumerate(paths)
            push!(pathsmap[path.o], i)
        end

        return EquivConv(in_, out, paths, pathsmap, dx, rmax)
    end
end


"""
"""
function (f::EquivConv)(X::AbstractArray; remake = true)
    @unpack in, out, paths, pathsmap, dx, rmax = f
    pathfunc = i -> paths[i].op(get(X, in, paths[i].i), in.grid)
    if remake
        for x in paths
            remake!(x.op)
        end
    end

    res = cat(
        [
            sum([pathfunc(i) for i in pathsmap[o]]) for
            o in eachindex(pathsmap)
        ]...,
        dims = 4,
    )

end


struct EquivProd
    in::Props
    out::Props
    paths::AbstractVector{NamedTuple}
    spectra::Bool
end
function EquivProd(
    in::Props;
    rank_max = max(1, length(in.ranks)),
    spectra = false,
)
    Zygote.ignore() do
        paths = []
        nranks_out = zeros(Int, rank_max + 1)
        for (i1, li1, si1) in zip(1:length(in.ranks), in.ranks, in.parities)
            for (i2, li2, si2) in
                zip(0:i1, [0, in.ranks[1:i1]...], [1, in.parities[1:i1]...])
                loutrange = abs(li1 - li2):min(li1 + li2, rank_max)
                so = si1 * si2
                for lo in loutrange
                    prod = FieldProd(li1, li2, lo)
                    if prod !== nothing && (i1!=i2 || prod!=prod111)
                        nranks_out[lo+1] += 1
                        path = (; i1, i2, lo, so, prod)
                        push!(paths, path)
                    end
                end
            end
        end
        if spectra
            nranks_out = (sum(nranks_out),)
        end
        out = Props(nranks_out, in.grid)
        sort!(paths, by = x -> x.lo)
        EquivProd(in, out, paths, spectra)
    end
end
function (f::EquivProd)(X::AbstractArray)
    in, out, paths, spectra = f.in, f.out, f.paths, f.spectra
    res = [
        path.i2 === 0 ? get(X, in, path.i1) :
        path.prod(get(X, in, path.i1), get(X, in, path.i2)) for
        path in paths
    ]
    if spectra
        res = [fieldnorm(x) for x in res]
        # res = fieldnorm.(res)
    end
    cat(res..., dims = 4)
end
struct LocalDense
m::Dense
function LocalDense(args...)
    Random.seed!(1)
    new(Dense(args...))
end
end
function Flux.trainable(m::LocalDense)
    return [m.m]
    # return vcat(m.W, [m.b])
end

function (m::LocalDense)(x::AbstractArray)
    @unpack W, b, σ = m.m
    out,in=size(W)
    res = σ.(cat([
            b[i].+sum([
                W[i,j] * x[:, :, :, j:j] for
                j in 1:in
            ]) for i in 1:out
    ]...,dims=4))
end
struct EquivDense
    in::Props
    out::Props
    W::AbstractVector
    b::AbstractVector
    σ::Any
end
function Flux.trainable(m::EquivDense)
    return [m.W, m.b]
    # return vcat(m.W, [m.b])
end

function EquivDense(in, out; σ = identity)
    Zygote.ignore() do
        W = [ones(b, a) / a for (a, b) in zip(in.nranks, out.nranks)]
        b = zeros(out.nranks[1])
        EquivDense(in, out, W, b, σ)
    end
end
function (f::EquivDense)(X::AbstractArray)
    @unpack in, out, W, b, σ = f
    res = [
        [
            sum([
                wi[j] * X[:, :, :, in.grouped_slices[a][j]] for
                j in eachindex(in.grouped_slices[a])
            ]) for wi in eachrow(W[a])
        ] for a in eachindex(in.grouped_slices)
    ]
    # res = [w * [X[:,:,:,s] for s in S] for (w, S) in zip(W, in.grouped_slices)]
    res1 = [σ.(res[1][i] .+ b[i]) for i in eachindex(b)]
    # res1 = [σ.(x.+bi) for (x,bi) in zip(res[1], b)]
    # res[1] = res1
    res = [res1]
    cat(vcat(res...)..., dims = 4)
end

function nae(yhat, y;sumy=sum(abs.(y)))
    if sumy == 0
        error()
    end
    sum(abs.(yhat .- y)) / sumy
end
