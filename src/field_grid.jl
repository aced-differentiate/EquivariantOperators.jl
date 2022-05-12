include("$DIR/field_operations.jl")
using Functors
using UnPack


struct Grid
    cell::AbstractMatrix
    origin::Any
    coords::AbstractArray
    dv::AbstractFloat
end
@functor Grid

Base.size(g::Grid)=size(g.coords)[1:end-1]
# function Base.getproperty(g::Grid,f)
#     @unpack coords=g
#     if f===:x


"""
    Grid(
        )
"""
function Grid(
    cell::AbstractMatrix,
    sz::Union{AbstractVector,Tuple};
    origin = (sz .+ 1) ./ 2,
)
    n = size(cell, 1)
    if n == 1
        rvecs = [cell * ([x] .- origin) for x = 1:sz[1]]
    elseif n == 2

        rvecs = [cell * ([x, y] .- origin) for x = 1:sz[1], y = 1:sz[2]]
    elseif n == 3

        rvecs = [
            cell * ([x, y, z] .- origin)
            for x = 1:sz[1], y = 1:sz[2], z = 1:sz[3]
        ]
    end
    coords = cat([getindex.(rvecs, i) for i = 1:n]..., dims = n + 1)
    dv=det(cell)
    Grid(cell, origin, coords,dv)
end

function Grid(cell, rmax::AbstractFloat)
    n = size(cell, 1)
    Grid(cell, 1 .+ 2 * ceil.(rmax * (cell \ ones(n))))
end


struct Props

    nranks::Any
    ranks::Any
    parities::Any
    slices::Any
end

function Props(ranks)
    Props(; ranks)
end
function Props(; nranks = nothing, ranks = nothing)
    if ranks === nothing
        ranks = vcat([fill(i - 1, n) for (i, n) in enumerate(nranks)]...)
    elseif nranks === nothing
        nranks = zeros(Int, maximum(ranks) + 1)
        for x in ranks
            nranks[x+1] += 1
        end
    end
    ends = Int.(cumsum(2 .* ranks .+ 1))
    slices = [(i == 1 ? 1 : ends[i-1] + 1):ends[i] for i in eachindex(ends)]

    # starts = [1, (1 .+ cumsum(nranks))...]
    # grouped_slices = [slices[starts[i]:starts[i+1]-1] for i = 1:length(nranks)]
    parities = (-1) .^ ranks
    Props(nranks, ranks, parities, slices)
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
        w * getindex.(eachslice(field, dims = 1+length(size(grid))), ix...)
        for (ix, w) in nearest(grid, rvec)
    ])
end
function Base.get(
    fields::AbstractArray,
    props::Props,
    i::Int,
    rvec::AbstractVector,
)
    field = get(fields, props, i)
    get(field, props.grid, rvec)
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

function nearest(grid, rvec)
    @unpack cell,origin=grid
    n=length(size(grid))
    ix = origin .+ (cell\rvec)
    ixfloor = floor.(Int, ix)
    er = ix - ixfloor

    if n==2
res=        [
        (ixfloor + [x, y], prod(ones(n) - abs.([x, y] - er)))
        for x = 0:1, y = 0:1
            if [0,0] < ixfloor + [x, y] < collect(size(grid))
                ]
    elseif         n==3
    res=[
        (ixfloor + [x, y, z], prod(ones(3) - abs.([x, y, z] - er)))
        for x = 0:1, y = 0:1, z = 0:1
            if [0,0,0] < ixfloor + [x, y, z] < collect(size(grid))
    ]
end
res
end

"""
    put_point_source!(
        field::AbstractArray,
        grid::Grid,
        rvec::AbstractVector,
        val::AbstractVector)


"""
function Base.put!(
    field::AbstractArray,
    grid::Grid,
    rvec::AbstractVector,
    val::AbstractVector,
)
    for (ix, w) in nearest(grid, rvec)
        for (c, vi) in zip(1:size(field)[end], val)
            field[ix..., c] += w / grid.dv * vi
        end
    end
end

function Base.put!(f, grid, rvecs::AbstractMatrix, vals::AbstractMatrix)
    for (val, rvec) in zip(eachcol(vals), eachcol(rvecs))
        put!(f, grid, rvec, val)
    end
end

"""
    field_rank(x::AbstractArray)
"""
field_rank(x::AbstractArray) = (size(x)[end] - 1) ÷ 2
