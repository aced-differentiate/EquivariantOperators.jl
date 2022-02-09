using Flux
const tol = 1e-18
invr = (r; p = 1, kwargs...) -> r == 0 ? 0.0 : 1 / r^p

DIFF_COEFFS = ((0.0, 0.5), (-2.0, 1.0))
function δn(x, dx, n, dim)
    i = round(Int, x / dx)
    if abs(x - i) < tol
        return i < length(DIFF_COEFFS[n]) ? DIFF_COEFFS[n][i+1] / dx^dim : 0.0
    end
    return 0.0
end

"""
radial function
"""
struct Radfunc
    # c::AbstractVector
    # μ::AbstractVector
    # σ::AbstractVector
    p::AbstractVector
    f
    rmin::AbstractFloat
    rmax::AbstractFloat
end

# Flux.trainable(m::Radfunc) = [m.p]
Flux.trainable(m::Radfunc) = [m.f,m.p]
# Flux.trainable(m::Radfunc) = [m.c, m.μ, m.σ]

function Radfunc(; rmin = 0.0, rmax = nothing, n = 8)
    # r = Array(LinRange(rmin, rmax, n))
    # c = exp.(-r / rmax)
    # μ = 1.0r
    # σ = 1.0r.+rmax/n
    # Radfunc(c, μ, σ, rmin, rmax)
    p=ones(4)
    f=Chain(Dense(1,n,leakyrelu),Dense(n,1))
    Radfunc(p,f, rmin, rmax)
end

function (m::Radfunc)(r)
    @unpack f, rmin, rmax ,p= m
    # @unpack f, rmin, rmax = m
    # @unpack c, μ, σ, rmin, rmax = m
    if r < m.rmin - tol || r > m.rmax + tol
        return 0.0
    end
    # dot(c, exp.(-((r .- μ) ./ σ) .^ 2))
    # (f([r])-f([rmax]))[1]
    c,k=p[1:2]
    # a=p[3:6]
    # c*exp(-abs(k)*r)*dot(a,r.^(0:3))
    c*exp(-abs(k)*r)*f([r])[1]
end
