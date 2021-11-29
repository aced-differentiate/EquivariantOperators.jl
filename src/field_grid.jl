include("field_operations.jl")
include("spherical_harmonics.jl")

struct Grid
    dx::AbstractFloat
    sz::Any
    origin::Any
    spherical_harmonics::Any
    r::Any
    dΩ::AbstractFloat
end

"""
    Grid(dx::AbstractFloat, rmax::AbstractFloat; dims = 3, rank_max = 1)
"""
function Grid(dx::AbstractFloat, rmax::AbstractFloat; dims = 3, rank_max = 1)
    dΩ = dx^dims

    nrmax = round(Int, rmax / dx)
    origin = (nrmax + 1) * ones(dims)
    sz = fill(2nrmax + 1, dims)

    r, spherical_harmonics =
        make_spherical_harmonics(dx, sz, origin, rank_max; dims)
    Grid(dx, sz, origin, spherical_harmonics, r, dΩ)
end

"""
    Grid(
    dx::AbstractFloat,
    sz::Union{AbstractVector,Tuple};
    origin = nothing,
    rank_max = 1)
"""
function Grid(
    dx::AbstractFloat,
    sz::Union{AbstractVector,Tuple};
    origin = nothing,
    rank_max = 1,
)
    dims = length(sz)
    dΩ = dx^dims
    if origin === nothing
        origin = (sz .+ 1.0) / 2
    end
    r, spherical_harmonics =
        make_spherical_harmonics(dx, sz, origin, rank_max; dims)
    Grid(dx, sz, origin, spherical_harmonics, r, dΩ)
end

GRID_ALIASES = (
    x = x -> x.spherical_harmonics[1][1],
    y = x -> x.spherical_harmonics[1][2],
    z = x -> x.spherical_harmonics[1][3],
)

function Base.getproperty(grid::Grid, v::Symbol)
    v in GRID_ALIASES ? GRID_ALIASES[v](grid) : getfield(grid, v)
end

"""
    Field(; grid = nothing, rank = nothing, radfunc = nothing)
"""
function Field(; grid = nothing, rank = nothing, radfunc = nothing)
    if radfunc !== nothing
        r = radfunc.(grid.r)
        r = concat([r])
        return rank == 0 ? r :
               FieldProd(0, rank, rank)(r, grid.spherical_harmonics[rank])
    end

    zeros(grid.sz..., 2rank + 1)
end

"""
    (field::AbstractArray)(rvec, grid::Grid)

Makes field objects callable. Returns interpolated tensor value of the tensor field at position rvec.
"""
function (field::AbstractArray)(rvec, grid::Grid)
    sum([
        w * getindex.(eachslice(field, dims = length(size(field))), ix...) for
        (ix, w) in nearest(grid.dx, grid.origin, rvec)
    ])
end

"""
    (field::AbstractArray)(i::Int)

Makes field objects callable. Returns ith component array (2d/3d) of tensor field.
"""
function (field::AbstractArray)(i::Int)
    selectdim(field, dim(field), i)
end

function nearest(dx, origin, rvec)
    ix = origin .+ rvec / dx
    ixfloor = floor.(Int, ix)
    er = ix - ixfloor
    [
        (ixfloor + [x, y, z], prod(ones(3) - abs.([x, y, z] - er))) for x = 0:1,
        y = 0:1, z = 0:1
    ]
end

"""
    put_point_source!(
    field::AbstractArray,
    grid::Grid,
    rvec::AbstractVector,
    val::AbstractVector,
)


"""
function put_point_source!(
    field::AbstractArray,
    grid::Grid,
    rvec::AbstractVector,
    val::AbstractVector,
)
    for (ix, w) in nearest(grid.dx, grid.origin, rvec)
        for (c, vi) in zip(1:size(field)[end], val)
            field[ix..., c] += w / grid.dΩ * vi
        end
    end
end

function put_point_source!(f, grid, rvecs::AbstractMatrix, vals::AbstractMatrix)
    for (val, rvec) in zip(eachcol(vals), eachcol(rvecs))
        put_point_source!(f, grid, rvec, val)
    end
end

"""
field_rank(x::AbstractArray)
"""
field_rank(x::AbstractArray) = (size(x)[end] - 1) ÷ 2
