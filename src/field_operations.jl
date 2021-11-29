using DSP: conv

dim(x) = length(size(x))
join(x) = cat(x...; dims = dim(x[1]) + 1)

prod0__(x, y) = join([x(1) .* y for y in eachslice(y, dims = dim(y))])
prod_0_(x, y) = join([x .* y[1] for x in x])
prod__0(x, y) = join([sum([x[i] .* y[i] for i = 1:length(x)])])

conv0__ =
    (x, y; kwargs...) ->
        join([conv(x(1), y; kwargs...) for y in eachslice(y, dims = dim(y))])
conv111 =
    (x, y; kwargs...) -> join([
        conv(x[2], y[3]; kwargs...) - conv(x[3], y[2]; kwargs...),
        conv(x[3], y[1]; kwargs...) - conv(x[1], y[3]; kwargs...),
        conv(x[1], y[2]; kwargs...) - conv(x[2], y[1]; kwargs...),
    ])
conv__0 = (x, y; kwargs...) -> join(sum(conv.(x, y; kwargs...)))
conv121 =
    (x, y; kwargs...) -> join([
        conv(x[1], -y[3] / sqrt(3) .- y[5]; kwargs...) +
        conv(x[2], y[2]; kwargs...) +
        conv(x[3], y[1]; kwargs...),
        conv(x[1], y[2]; kwargs...) +
        conv(x[2], 2 / sqrt(3) * y[3]; kwargs...) +
        conv(x[3], y[4]; kwargs...),
        conv(x[1], y[1]; kwargs...) +
        conv(x[2], y[4]; kwargs...) +
        conv(x[3], -y[3] / sqrt(3) .+ y[5]; kwargs...),
    ])

function FieldProd(l1, l2, l)
    # l1, l2, l = ranks
    if l1 == 0
        f = prod0__
    elseif l2 == 0
        f = prod_0_
    elseif l1 == l2 && l == 0
        f = prod__0
    end
    f
    # (x, y) -> yield(f(x, y), x.grid)
    # (x,y)->yield
end


function FieldConv(l1, l2, l; kwargs...)

    # l1,l2,l=ranks
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
    field_prod(x::AbstractArray, y::AbstractArray; rank = nothing)
"""
function field_prod(x::AbstractArray, y::AbstractArray; rank = nothing)
    lx, ly = field_rank.([x, y])
    FieldProd(lx, ly, rank === nothing ? lx + ly : rank)(x, y)
end

"""
"""
function field_conv(x::AbstractArray, y::AbstractArray; rank = nothing)
    lx, ly = field_rank.([x, y])
    FieldConv(lx, ly, rank === nothing ? lx + ly : rank)(x, y)
end

"""
    function field_norm(field::AbstractArray)
"""
function field_norm(field::AbstractArray)
    join(sqrt.(sum([x .^ 2 for x in field])))
end
