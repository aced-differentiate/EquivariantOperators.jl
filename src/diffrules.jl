using Zygote: @adjoint
using Zygote
using ForwardDiff
using DSP

val(x::ForwardDiff.Dual) = ForwardDiff.value(x)
val(x::Float64) = x
val(x::Float32) = x

function dconv(x, a, b)
    size=Base.size
    n = length(size(a))
    x = reshape(vec(x), (size(a) .+ size(b) .- ones(Int, n))...)
    r = (
        DSP.conv(x, reverse(b))[(
            # DSP.xcorr(x, b, padmode = :none)[(
            i:j for (i, j) in zip(size(b), size(x))
        )...],
        DSP.conv(x, reverse(a))[(
            # DSP.xcorr(x, a, padmode = :none)[(
            i:j for (i, j) in zip(size(a), size(x))
        )...],
    )
    return r
end
# @adjoint DSP.conv(a, b) = DSP.conv(a, b), x -> dconv(x, a, b)
@adjoint DSP.conv(a, b) = DSP.conv(val.(a), val.(b)),
x -> dconv(val.(x), val.(a), val.(b))
