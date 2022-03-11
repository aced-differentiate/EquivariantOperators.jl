using Flux: conv
using DSP: conv

# 4 = length(size(x))
join(x) = cat(x...; dims = 4)

prod0__(x, y) = join([x .* y[:,:,:,i:i] for i in 1:size(y,4)])
prod_0_(x, y) = join([x[:,:,:,i:i].*y for i in 1:size(x,4)])
prod__0(x, y) =
    join([sum([x[:, :, :, i] .* y[:, :, :, i] for i = 1:size(x, 4)])])
prod111(x,y) =
     join([
    x[:, :, :, 2].*y[:, :, :, 3] - x[:, :, :, 3].* y[:, :, :, 2],
    x[:, :, :, 3].* y[:, :, :, 1] - x[:, :, :, 1].* y[:, :, :, 3],
    x[:, :, :, 1].* y[:, :, :, 2] - x[:, :, :, 2].* y[:, :, :, 1],
    ])

function myconv(x,y)
    DSP.conv(x,y)
    # Flux.conv(cat(x,dims=5),cat(y,dims=5);pad=(size(y).-1).÷2)[:,:,:,1,1]
end
conv0__ =
    (x, y) -> join([myconv(x[:, :, :, 1], y) for y in eachslice(y, dims = 4)])
conv_0_ =
    (x, y) -> join([myconv(x, y[:, :, :, 1]) for y in eachslice(x, dims = 4)])
conv111 =
    (x, y) -> join([
        myconv(x[:, :, :, 2], y[:, :, :, 3]) - myconv(x[:, :, :, 3], y[:, :, :, 2]),
        myconv(x[:, :, :, 3], y[:, :, :, 1]) - myconv(x[:, :, :, 1], y[:, :, :, 3]),
        myconv(x[:, :, :, 1], y[:, :, :, 2]) - myconv(x[:, :, :, 2], y[:, :, :, 1]),
    ])
conv__0 =
    (x, y) -> join(sum(myconv.(eachslice(x, dims = 4), eachslice(y, dims = 4))))
conv121 =
    (x, y) -> join([
        myconv(x[:, :, :, 1], -y[:, :, :, 3] / sqrt(3) .- y[:, :, :, 5]) +
        myconv(x[:, :, :, 2], y[:, :, :, 2]) +
        myconv(x[:, :, :, 3], y[:, :, :, 1]),
        myconv(x[:, :, :, 1], y[:, :, :, 2]) +
        myconv(x[:, :, :, 2], 2 / sqrt(3) * y[:, :, :, 3]) +
        myconv(x[:, :, :, 3], y[:, :, :, 4]),
        myconv(x[:, :, :, 1], y[:, :, :, 1]) +
        myconv(x[:, :, :, 2], y[:, :, :, 4]) +
        myconv(x[:, :, :, 3], -y[:, :, :, 3] / sqrt(3) .+ y[:, :, :, 5]),
    ])

function FieldProd(l1, l2, l)
    # l1, l2, l = ranks
    if l1 == 0
        return  prod0__
    elseif l2 == 0
        return  prod_0_
    elseif l1 == l2 && l == 0
        return  prod__0
    elseif l1 == l2 ==l == 1
        return  prod111
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
function fieldnorm(x::AbstractArray;squared=false)
# sqrt.(sum([x[:,:,:,i:i] .^ 2 for i in 1:size(x,4)]))
if  size(x,4)==1
    return x
end
r=sum([x[:,:,:,i:i] .^ 2 for i in 1:size(x,4)])
sqrt.(r.+.01).-.1
# sqrt.(sum([x .^ 2 for x in eachslice(field,dims=4)]))
# cat(sqrt.(sum([x .^ 2 for x in eachslice(field,dims=4)])),dims=4)
end
