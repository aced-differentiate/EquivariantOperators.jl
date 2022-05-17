# using Flux: conv
using DSP
using LinearAlgebra

prod0__(x, y) = cat(
    [x .* y for y in eachslice(y, dims = length(size(y)))]...,
    dims = length(size(x)),
)
prod_0_(x, y) = cat(
    [x .* y for x in eachslice(x, dims = length(size(x)))]...,
    dims = length(size(x)),
)
prod__0(x, y) = cat(
    sum([
        x .* y
        for
        (x, y) in zip(
            eachslice(x, dims = length(size(x))),
            eachslice(y, dims = length(size(y))),
        )
    ]),
    dims = length(size(x)),
)
# prod111(x, y) = join([
#     x[:, :, :, 2] .* y[:, :, :, 3] - x[:, :, :, 3] .* y[:, :, :, 2],
#     x[:, :, :, 3] .* y[:, :, :, 1] - x[:, :, :, 1] .* y[:, :, :, 3],
#     x[:, :, :, 1] .* y[:, :, :, 2] - x[:, :, :, 2] .* y[:, :, :, 1],
# ])

function myconv(x, y)
    DSP.conv(x, y)
    # Flux.conv(cat(x,dims=5),cat(y,dims=5);pad=(size(y).-1).÷2)[:,:,:,1,1]
end

function conv0__(x, y)
    dims = length(size(y))
    cat([myconv(dropdims(x; dims), y) for y in eachslice(y; dims)]...; dims)
end

function conv_0_(x, y)
    dims = length(size(x))
    cat([myconv(x, dropdims(y; dims)) for x in eachslice(x; dims)]; dims)
end

conv__0(x, y) = cat(
    sum([
        myconv(x, y)
        for
        (x, y) in zip(
            eachslice(x, dims = length(size(x))),
            eachslice(y, dims = length(size(y))),
        )
    ]),
    dims = length(size(x)),
)
# conv111 =
#     (x, y) -> join([
#         myconv(x[:, :, :, 2], y[:, :, :, 3]) -
#         myconv(x[:, :, :, 3], y[:, :, :, 2]),
#         myconv(x[:, :, :, 3], y[:, :, :, 1]) -
#         myconv(x[:, :, :, 1], y[:, :, :, 3]),
#         myconv(x[:, :, :, 1], y[:, :, :, 2]) -
#         myconv(x[:, :, :, 2], y[:, :, :, 1]),
#     ])
# conv121 =
#     (x, y) -> join([
#         myconv(x[:, :, :, 1], -y[:, :, :, 3] / sqrt(3) .- y[:, :, :, 5]) +
#         myconv(x[:, :, :, 2], y[:, :, :, 2]) +
#         myconv(x[:, :, :, 3], y[:, :, :, 1]),
#         myconv(x[:, :, :, 1], y[:, :, :, 2]) +
#         myconv(x[:, :, :, 2], 2 / sqrt(3) * y[:, :, :, 3]) +
#         myconv(x[:, :, :, 3], y[:, :, :, 4]),
#         myconv(x[:, :, :, 1], y[:, :, :, 1]) +
#         myconv(x[:, :, :, 2], y[:, :, :, 4]) +
#         myconv(x[:, :, :, 3], -y[:, :, :, 3] / sqrt(3) .+ y[:, :, :, 5]),
#     ])

function FieldProd(l1, l2, l)
    # l1, l2, l = ranks
    if l1 == 0
        return prod0__
    elseif l2 == 0
        return prod_0_
    elseif l1 == l2 && l == 0
        return prod__0
    elseif l1 == l2 == l == 1
        return prod111
    end

    # (x, y) -> yield(f(x, y), x.grid)
    # (x,y)->yield
end


function FieldConv(l1, l2, l)
    if l1 == 0
        f = conv0__
    elseif l2 == 0
        f = conv_0_
    elseif l1 == l2 && l == 0
        f = conv__0
    elseif l1 == l2 == l == 1
        f = conv111
    elseif (l1, l2, l) == (1, 2, 1)
        f = conv121
    end
    f
    # (x, y) -> tfconv(f,x,y)
end

"""
    ⊗(x,y)
    fieldprod(x::AbstractArray, y::AbstractArray)
    fieldprod(x::AbstractArray, y::AbstractArray,l)

    Alias of fieldprod(u, v), `u ⊗ v` computes the appropriate pointwise product between 2 scalar or vector fields (inferred as scalar-scalar, scalar-vector, dot, cross). For greater clarity one may also write `u ⋅ v` for dot and `u ⨉ v` for cross
"""
function fieldprod(x::AbstractArray, y::AbstractArray, l)
    lx, ly = getl.([x, y])
    FieldProd(lx, ly, l)(x, y)
end
function fieldprod(x::AbstractArray, y::AbstractArray)
    lx, ly = getl.([x, y])
    l = abs(lx - ly)
    FieldProd(lx, ly, l)(x, y)
end
function getl(x)
    n = size(x)[end]
    if n == 1
        return 0
    end

    dims = ndims(x) - 1
    if dims == 2
        return 1
    elseif dims == 3
        return (n - 1) ÷ 2
    end

end
function ⊗(x,y)
    fieldprod(x,y)
end
#
# function LinearAlgebra.:⋅(x::Array{T,3}, y::Array{T,3}) where T<:AbstractFloat
#     fieldprod(x,y,0)
# end
# function LinearAlgebra.:⋅(x::Array{T,4}, y::Array{T,4}) where T<:AbstractFloat
#     fieldprod(x,y,0)
# end
# function ⨉(x::Array{T,4}, y::Array{T,4}) where T<:AbstractFloat
#     fieldprod(x,y,1)
# end
#
# function LinearAlgebra.adjoint(x::AbstractArray)
#     x
# end
"""
"""
function field_conv(x::AbstractArray, y::AbstractArray; rank = nothing)
    lx, ly = l.([x, y])
    FieldConv(lx, ly, rank === nothing ? lx + ly : rank)(x, y)
end

"""
    function field_norm(field::AbstractArray)
"""
function fieldnorm(x::AbstractArray)
    # n=length(size(x))-1
    if size(x)[end] == 1
        return abs.(x)
    end
    # r=sum([x[:,:,:,i:i] .^ 2 for i in 1:size(x,4)])
    # sqrt.(r.+.01).-.1
    # sqrt.(sum([x .^ 2 for x in eachslice(x, dims = length(size(x)))]))
    cat(
        sqrt.(sum([x .^ 2 for x in eachslice(x, dims = ndims(x))])),
        dims = ndims(x),
    )
end

function Δ(x, y)
    sum(abs.(x .- y)) / sum(abs.(y))
end


function nae(yhat, y; sumy = sum(abs.(y)))
    if sumy == 0
        error()
    end
    sum(abs.(yhat .- y)) / sumy
end
