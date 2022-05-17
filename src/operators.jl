using Random
using Functors
using Zygote

include("grid.jl")
include("radfuncs.jl")
include("diffrules.jl")
Random.seed!(1)

mutable struct Op
    l
    kernel::AbstractArray
    grid::Grid
    radfunc
    rmin
    rmax
    boundary
end
@functor Op
Flux.trainable(m::Op) = [m.radfunc]

function makekernel(radfunc, rmin, rmax, l, grid)
    @unpack cell, coords, dv = grid
    n = size(cell, 1)
rnorms=0
Zygote.ignore() do
    rnorms = fieldnorm(coords)
end
    f = r-> rmin <= r <= rmax ? radfunc(r) : 0.0
    rscalars = f.(rnorms)
    if l == 0
        kernel = cat(rscalars, dims = n + 1)
    elseif l == 1
        kernel = broadcast(
            (a, r, s) -> (r == 0 || r > rmax) ? 0.0 : a * s / r,
            coords,
            repeat(rnorms, ones(Int, n)..., size(coords)[end]),
            repeat(rscalars, ones(Int, n)..., size(coords)[end]),
        )
    end
    dv * kernel
end
function Op(
    radfunc,
    rmin,
    rmax,
    cell::Matrix;
    l = 0,
    boundary =:zero,
)
    grid = Grid(cell, rmax)
    kernel = makekernel(radfunc, rmin, rmax, l, grid)

    Op(l, kernel, grid, radfunc,rmin, rmax, boundary)
end

"""
    Op(
        name::Union{Symbol,String},
        cell;
        boundary =:zero,
        rmin = 0,
        rmax = Inf,
        l = 0,
        σ = 1.0,
    )
    Op(
        radfunc,
        rmin::AbstractFloat,
        rmax::AbstractFloat,
        cell;
        l = 0,
        boundary =:zero,
    )

`Op` constructs finite difference operators. Prebuilt operators like differential operators (`▽`) & common Green's functions can be specified by name. Custom equivariant operators can be made by specifying radial function.
"""
function Op(
    name::Union{Symbol,String},
    cell;
    rmin = 0.,
    rmax = Inf,
    boundary =:zero,
    l = 0,
    σ = 1.0,
)
    name = Symbol(name)
    if name == :▽
        name = :grad
    end

    n = size(cell, 1)
    radfunc=nothing
    if name == :neural
    elseif name == :Gaussian
        radfunc = r -> exp(-r^2 / (2 * σ^2)) / sqrt(2π * σ^(2n))
        return Op(radfunc, rmin,3σ, cell)
    elseif name == :grad
        boundary = :smooth
        l = 1
        grid = Grid(cell, fill(3, n))

        if n == 1
            kernel = [1 / (2 * cell[1]), 0, -1 / (2 * cell[1])]
        elseif n == 2
            kernel = [
                sum(abs.([x, y])) > 1 ? zeros(2) : -cell' \ [x, y] / 2
                for x = -1:1, y = -1:1
            ]
            kernel = cat([getindex.(kernel, i) for i = 1:n]..., dims = n + 1)
        elseif n == 3
            kernel = [
                sum(abs.([x, y, z])) > 1 ? zeros(3) :
                    -cell' \ [x, y, z] / 2 for x = -1:1, y = -1:1, z = -1:1
            ]
            kernel = cat([getindex.(kernel, i) for i = 1:n]..., dims = n + 1)
        end
    end
    Op(l, kernel, grid, radfunc,rmin, rmax, boundary)
end

"""
    function (m::Op)(x::AbstractArray, )

"""
function (m::Op)(x::AbstractArray)
    li = 0
    lo = m.l
    m(x, li, lo)
end

function (m::Op)(x::AbstractArray, li, lo)
    @unpack grid, kernel, l, boundary = m
    @unpack origin, cell = grid
    ix = 0
    Zygote.ignore() do
        if boundary == :smooth
            ix = [
                Int.([a + 1, ((a+1):(b+a-2))..., b + a - 2])
                for (a, b) in zip(origin, size(x))
            ]
        else
            ix = [Int(a):Int(b + a - 1) for (a, b) in zip(origin, size(x))]
        end
    end

    FieldConv(li, l, lo)(x, kernel)[ix..., :]
end

function LinearAlgebra.:⋅(m::Op, x)
    m(x, m.l, 0)
end
function ⨉(m::Op, x)
    m(x, 1, 1)
end
function remake!(m)
    @unpack radfunc,rmin, rmax, l, grid = m
    m.kernel = makekernel(radfunc,rmin, rmax, l, grid)
end
