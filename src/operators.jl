using Random
include("field_grid.jl")
include("radfuncs.jl")
Random.seed!(1)

mutable struct LinearOperator
    ranks::Any
    rmin::AbstractFloat
    rmax::AbstractFloat

    radfunc::Any
    conv::Function

    filter::AbstractArray
    grid::Grid
end

"""
    function LinearOperator(
        name;
        dx = nothing,
        rmax = nothing,
        ranks = nothing,
        grid = nothing,
        dims = 3,
        radfunc = nothing,
        rmin = 0.0,
        σ = 1.0
    )
"""
function LinearOperator(
    name;
    dx = nothing,
    rmax = nothing,
    ranks = nothing,
    grid = nothing,
    dims = 3,
    radfunc = nothing,
    rmin = 0.0,
    σ = 1.0,
)
    if name == :neural
        radfunc = Radfunc(; rmin, rmax)
    elseif name == :potential
        li = lo = lf = 0
        rmin = dx
        radfunc = r-> 1 / (4π * r)
    elseif name == :inverse_squared_field
        li = 0
        lf = lo = 1
        rmin = dx
        radfunc = r-> 1 / (4π * r^2)
    elseif name == :Gaussian
        li = lf = lo = 1
        radfunc = r-> exp(-r^2 / (2σ^2)) / (2π)^(dims / 2)
    elseif name == :grad
        dV = dx^dims
        li = 0
        lf = lo = 1
        rmax = dx
        n = 1
        radfunc = r-> δn(r, dx, n, dims)
    elseif name == :div
        dV = dx^dims
        lf = li = 1
        lo = 0
        rmax = dx
        n = 1
        radfunc = r-> δn(r, dx, n, dims)
    elseif name == :curl
        dV = dx^dims
        lf = lo = li = 1
        rmax = dx
        n = 1
        radfunc = r-> -δn(r, dx, n, dims)
    elseif name == :Laplacian
        dV = dx^dims
        lf = li = lo = 0
        rmax = dx
        n = 2
        radfunc = r-> δn(r, dx, n, dims)
    end

    if ranks === nothing
        ranks = (li, lf, lo)
    end
    if grid === nothing
        grid = Grid(dx, rmax; dims, rank_max = ranks[2])
    end
    conv = FieldConv(ranks...)

if typeof(radfunc)===Radfunc
    radfunc1=radfunc
else
    radfunc1(r) = (rmin - tol) < r < (rmax + tol) ? radfunc(r) : 0.0
end

    filter = Field(; radfunc=radfunc1, grid, rank = ranks[2])
     LinearOperator(ranks, rmin, rmax, radfunc1, conv, filter, grid)
end

"""
    function (m::LinearOperator)(x::AbstractArray, grid::Grid)

"""
function (m::LinearOperator)(x::AbstractArray, grid::Grid)
    if m.rmax == 0
        return m.radfunc(0) * x
    end

    ix = 0
    x_ = 0
    dV = 0.0
    Zygote.ignore() do
        ix = [
            Int(a):Int(b + a - 1) for (a, b) in zip(m.grid.origin, grid.sz)
            # Int(a):Int(b + a - 1) for (a, b) in zip(y.grid.origin, x.grid.sz)
        ]
        x_ = x
        # x_ = x.components
        dV = m.grid.dV
    end
    y_ = m.filter

    res = m.conv(x_, y_)
    res[ix...,:] .* dV
end
# makefilter(radfunc,grid) = Field(;radfunc, grid)
function remake!(m)
    rank, radfunc, grid = field_rank(m.filter), m.radfunc, m.grid
    m.filter = Field(; rank, radfunc, grid)
end
