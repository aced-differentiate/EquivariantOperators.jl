using Flux, Functors

"""
radial function
"""
struct Radfunc
    p::AbstractVector
    f
    rmin
    rmax
end
@functor Radfunc
Flux.trainable(m::Radfunc) = [m.f]


function Radfunc(; rmin = 0, rmax = 1e16, n = 32)
    p = ones(4)
    f = Chain(Dense(1, n, leakyrelu), Dense(n, 1))
    Radfunc(p, f, rmin, rmax)
end

function (m::Radfunc)(r)
    @unpack f, p, rmin, rmax = m
    c, k = p[1:2]
    # rmin < r < rmax ? c * exp(-abs(k) * r) : 0.0
     f([r])[1]
end
