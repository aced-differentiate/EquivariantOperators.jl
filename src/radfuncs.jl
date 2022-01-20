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
    # f1::Any
    # f2::Any
    params::AbstractVector
    r::AbstractVector
    dr::AbstractFloat
    rmin::AbstractFloat
    rmax::AbstractFloat
end
Flux.trainable(m::Radfunc) = m.params
# Flux.trainable(m::Radfunc) = [m.f1, m.f2, m.params]
function Radfunc(; rmin = 0.0, rmax = 1 / tol, n = rmax == 0 ? 1 : 16)

    # f1 = Dense(1, n, swish)
    # f2 = Dense(n, 1)
    r = LinRange(rmin, rmax, n)
    dr = (rmax - rmin) / n
    params = ones(n + 2)
    Radfunc(params, r, dr, rmin, rmax)
end

function (m::Radfunc)(r)

    # if r ⪉m.rmin  || r ⪊ m.rmax
    if r < m.rmin - tol || r > m.rmax + tol
        return 0.0
    end
    c, k, C = m.params[1], abs(m.params[2]), m.params[3:end]
    # (m.f2 ∘ m.f1)([r])[1] * (c1 * exp(-abs(k) * r) + c2 * exp(-(r / σ)^2))
    m.rmax == 0 ? c :
    c * exp(-k * r / m.rmax) * dot(C, exp.(-((r .- m.r) / m.dr) .^ 2))
end
