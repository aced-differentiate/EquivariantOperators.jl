include("field_operations.jl")
include("spherical_harmonics.jl")

struct Grid
    dx::AbstractFloat
    sz::Any
    origin::Any
    spherical_harmonics::Any
    r::Any
    dV::AbstractFloat
end

"""
    Grid(dx::AbstractFloat, rmax::AbstractFloat; dims = 3, rank_max = 1)
"""
function Grid(dx::AbstractFloat, rmax::AbstractFloat; dims = 3, rank_max = 1)
    dV = dx^dims

    nrmax = round(Int, rmax / dx)
    origin = (nrmax + 1) * ones(dims)
    sz = fill(2nrmax + 1, dims)

    r, spherical_harmonics =
        make_spherical_harmonics(dx, sz, origin, rank_max; dims)
    Grid(dx, sz, origin, spherical_harmonics, r, dV)
end

"""
    Grid(
        dx::AbstractFloat,
        sz::Union{AbstractVector,Tuple};
        origin = nothing,
        rank_max = 1
        )
"""
function Grid(
    dx::AbstractFloat,
    sz::Union{AbstractVector,Tuple};
    origin = nothing,
    rank_max = 1,
)
    dims = length(sz)
    dV = dx^dims
    if origin === nothing
        origin = (sz .+ 1.0) / 2
    end
    r, spherical_harmonics =
        make_spherical_harmonics(dx, sz, origin, rank_max; dims)
    Grid(dx, sz, origin, spherical_harmonics, r, dV)
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
        r = join([r])
        return rank == 0 ? r : field_prod(r, grid.spherical_harmonics[rank])
    end

    zeros(grid.sz..., 2rank + 1)
end


struct Props

    grid::Grid
    nranks::Any
    ranks::Any
    parities::Any
    slices::Any
    grouped_slices::Any
end

function Props(nranks, grid::Grid)
    ranks = vcat([fill(i - 1, n) for (i, n) in enumerate(nranks)]...)
ends =cumsum(2ranks .+ 1)
    slices =[(i==1 ? 1 : ends[i-1]+1):ends[i] for i in eachindex(ends)]

    starts = [1, (1 .+ cumsum(nranks))...]
    grouped_slices = [slices[starts[i]:starts[i+1]-1] for i = 1:length(nranks)]
    parities = (-1) .^ ranks
    Props(grid,nranks, ranks, parities, slices, grouped_slices)
end

"""
    (field::AbstractArray)(rvec, grid::Grid)

Makes field objects callable. Returns interpolated tensor value of the tensor field at position rvec.
"""
function Base.get(fields::AbstractArray, props::Props, i::Int)
    fields[:, :, :, props.slices[i]]
end

function Base.get(field::AbstractArray, grid::Grid, rvec::AbstractVector)
    sum([
        w * getindex.(eachslice(field, dims = 4), ix...) for
        (ix, w) in nearest(grid.dx,grid.origin, rvec)
    ])
end
function Base.get(fields::AbstractArray, props::Props, i::Int, rvec::AbstractVector)
    field = get(fields, props, i)
    get(field,props.grid,rvec)
end
function Base.get(fields::AbstractArray, props::Props, rvec::AbstractVector)
    get(fields, props, 1, rvec)
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
        val::AbstractVector)


"""
function put_point_source!(
    field::AbstractArray,
    grid::Grid,
    rvec::AbstractVector,
    val::AbstractVector,
)
    for (ix, w) in nearest(grid.dx, grid.origin, rvec)
        for (c, vi) in zip(1:size(field)[end], val)
            field[ix..., c] += w / grid.dV * vi
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
