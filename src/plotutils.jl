# using GLMakie
# using CairoMakie
using Plots

function vfplot(v,g;kwargs...)
x=g.x
y=g.y
u =getindex.(v,1)
v =getindex.(v,2)
x, y, u, v = vec.([x, y, u, v])
quiver(x, y; quiver = (u, v), kwargs...)
end
# function vis(x::EquivConv)
#     r=LinRange(0.,x.rmax,64)
#     plots = [Plots.plot(r,p.op.radfunc.(r)) for p in x.paths]
#     Plots.plot(plots..., layout = length(plots))
# end
