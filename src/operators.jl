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
    convfunc
    kwargs
    radfunc
    rmin
    rmax
end
@functor Op
Flux.trainable(m::Op) = [m.radfunc]

function makekernel(radfunc, rmin, rmax, l, grid)
    @unpack cell, rhat,R, dv = grid
    n = size(cell, 1)
    f = r-> rmin <= r <= rmax ? radfunc(r) : 0.0
    rscalars = f.(R)
    if l == 0
        kernel = rscalars
    elseif l == 1
        kernel = rscalars.*rhat
    end
    dv * kernel
end
function Op(
    radfunc,
    rmin,
    rmax,
    cell::Matrix;
    l = 0,
    convfunc=conv,
    boundary =:same
)
    grid = Grid(cell, rmax)
    kernel = makekernel(radfunc, rmin, rmax, l, grid)
    kwargs=(;boundary)

    Op(l, kernel, grid,convfunc,kwargs, radfunc,rmin, rmax)
end

"""
    Op(
        name::Union{Symbol,String},
        cell;
        boundary =:same,
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
        boundary =:same,
    )

`Op` constructs finite difference operators. Prebuilt operators like differential operators (`▽`) & common Green's functions can be specified by name. Custom equivariant operators can be made by specifying radial function.
"""
function Op(
    name::Union{Symbol,String},
    cell;
    rmin = 0.,
    rmax = Inf,
    convfunc=conv,
    boundary =:same,
    l = 0,
    σ = 1.0,
)
    name = Symbol(name)
    if name == :▽
        name = :grad
    end

    dims = size(cell, 1)
    radfunc=nothing
if name == :Gaussian
    radfunc = r -> exp(-r^2 / (2 * σ^2)) / sqrt(2π * σ^(2n))
    rmax=2σ
    return Op(radfunc,rmin,rmax,cell)
end

    if name == :neural
    elseif name == :grad
        boundary = :smooth
        l = 1
        grid = Grid(cell, fill(3, dims))


            kernel = [
                sum(abs.(v)) > 1 ? zeros(dims) :
                    -cell' \ collect(v) / 2 for v in Iterators.product(fill(-1:1,dims)...)
            ]
    end
    kwargs=(;boundary)
    Op(l, kernel, grid, convfunc,kwargs,radfunc,rmin, rmax)
end

"""
    function (m::Op)(x::AbstractArray, )

"""
function (m::Op)(x::AbstractArray,product=*)
    @unpack grid, kernel, convfunc,kwargs = m
    convfunc(x,kernel;product,kwargs...)
end

function LinearAlgebra.:⋅(m::Op, x)
    m(x,⋅)
end
function ⨉(m::Op, x)
    m(x,⨉)
end
function remake!(m)
    @unpack radfunc,rmin, rmax,  grid,l = m
    m.kernel = makekernel(radfunc,rmin, rmax, l, grid)
end
