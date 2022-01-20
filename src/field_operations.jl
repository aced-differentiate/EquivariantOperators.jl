using DSP: conv

# 4 = length(size(x))
join(x) = cat(x...; dims = 4)

prod0__(x, y) = join([x[:, :, :, 1] .* y for y in eachslice(y, dims = 4)])
prod_0_(x, y) = join([x .* y[:, :, :, 1] for x in eachslice(x, dims = 4)])
prod__0(x, y) =
    join([sum([x[:, :, :, i] .* y[:, :, :, i] for i = 1:size(x, 4)])])

conv0__ =
    (x, y) -> join([conv(x[:, :, :, 1], y) for y in eachslice(y, dims = 4)])
conv111 =
    (x, y) -> join([
        conv(x[:, :, :, 2], y[:, :, :, 3]) - conv(x[:, :, :, 3], y[:, :, :, 2]),
        conv(x[:, :, :, 3], y[:, :, :, 1]) - conv(x[:, :, :, 1], y[:, :, :, 3]),
        conv(x[:, :, :, 1], y[:, :, :, 2]) - conv(x[:, :, :, 2], y[:, :, :, 1]),
    ])
conv__0 =
    (x, y) -> join(sum(conv.(eachslice(x, dims = 4), eachslice(y, dims = 4))))
conv121 =
    (x, y) -> join([
        conv(x[:, :, :, 1], -y[:, :, :, 3] / sqrt(3) .- y[:, :, :, 5]) +
        conv(x[:, :, :, 2], y[:, :, :, 2]) +
        conv(x[:, :, :, 3], y[:, :, :, 1]),
        conv(x[:, :, :, 1], y[:, :, :, 2]) +
        conv(x[:, :, :, 2], 2 / sqrt(3) * y[:, :, :, 3]) +
        conv(x[:, :, :, 3], y[:, :, :, 4]),
        conv(x[:, :, :, 1], y[:, :, :, 1]) +
        conv(x[:, :, :, 2], y[:, :, :, 4]) +
        conv(x[:, :, :, 3], -y[:, :, :, 3] / sqrt(3) .+ y[:, :, :, 5]),
    ])

function FieldProd(l1, l2, l)
    # l1, l2, l = ranks
    if l1 == 0
        return  prod0__
    elseif l2 == 0
        return  prod_0_
    elseif l1 == l2 && l == 0
        return  prod__0
    end

    # (x, y) -> yield(f(x, y), x.grid)
    # (x,y)->yield
end


function FieldConv(l1, l2, l)

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
function fieldnorm(field::AbstractArray)
sqrt.(sum([x .^ 2 for x in eachslice(field,dims=4)]))
# cat(sqrt.(sum([x .^ 2 for x in eachslice(field,dims=4)])),dims=4)
end
