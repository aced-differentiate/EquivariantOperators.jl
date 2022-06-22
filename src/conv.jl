# using Flux: conv
using LinearAlgebra
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
function tp(x, f, l, p)
    r = l .+ size(f).-1
    xl = max.(l, 1)
    xr = min.(r, size(x))
    fl = 1 .+ xl .- l
    fr = size(f) .- r .+ xr
    sum(p.(
        x[[a:b for (a, b) in zip(xl, xr)]...],
        f[[a:b for (a, b) in zip(fl, fr)]...],
    ))
end
"""
    xcor(x, f; product = *, stride = 1, pad = 0)

Cross correlation, or "convolution" in computer vision.
"""
function xcor(x, f; product = *, stride = 1, pad = 0)
    l = Iterators.product([
        a:stride:b for (a, b) in zip(ones(Int,ndims(x)) .- pad, size(x) .- size(f) .+ 1 .+ pad)
    ]...)
    [tp(x, f, l, product) for l in l]
end

"""
    conv(x, f; product = *, boundary = :extend)

Convolution in signal processing. For "convolution" in computer vision, use xcor instead.
"""
function conv(x, f; product = *, boundary = :extend)
    if boundary == :smooth
        r = xcor(x, reverse(f); product)
        ix = [Int.([1, (1:a)..., a]) for a in size(r)]
        return r[ix...]
    end

    if boundary == :extend
        pad = size(f) .- 1
    elseif boundary == :same
        pad = (size(f) .- 1) .÷ 2
    end
    xcor(x, reverse(f); product, pad)

end
function Δ(x, y)
    sum(abs.(x .- y)) / sum(abs.(y))
end
