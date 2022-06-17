"""
Finite difference calculation & machine learning of electric potential & electric field from charge
"""

using LinearAlgebra
using Plots
using Random
using Flux
Random.seed!(1)
include("../src/operators.jl")
include("../src/plotutils.jl")
# using Main.EquivariantOperators # end users should omit Main.

# make grid
dims = 2
dx = 0.1
cell = dx * Matrix(I, dims, dims)
rmax = 1.0
grid = Grid(cell, rmax)

# make operators
rmin = 1e-9
ϕ = Op(r -> 1 / (4π * r), rmin, rmax, cell)
E = Op(r -> 1 / (4π * r^2), rmin, rmax, cell; l = 1)
▽ = Op(:▽, cell)

# put dipole charges
ρf = zeros(size(grid))
put!(ρf, grid, [0.5, 0.0],1)
put!(ρf, grid, [-0.5, 0.0],-1)

# calculate fields
Ef = E(ρf)
ϕf = ϕ(ρf)

# test
rvec = [0, 0]
@show get(ϕf, grid, rvec), [0.0]
@show get(Ef, grid, rvec), get(-▽(ϕf), grid, rvec), [-2 / (4π * 0.5^2), 0]

p = []
push!(p, heatmap(ρf',title = "dipole charge"), )
push!(p, heatmap(ϕf', title = "dipole potential"),)
plot(p..., layout = length(p))
vfplot(Ef,grid; title = "dipole electric field")

##
# make neural operators
ϕ_ = Op(Radfunc(), rmin, rmax, cell)
E_ = Op(Radfunc(), rmin, rmax, cell; l = 1)

ps = Flux.params(ϕ_, E_)
function loss()
    remake!(E_)
    remake!(ϕ_)
    global E_f = E_(ρf)
    global ϕ_f = ϕ_(ρf)
    @show l = (nae(E_f, Ef) + nae(ϕ_f, ϕf)) / 2
    l
end

data = [()]
loss()
opt = ADAM(0.1)
Flux.@epochs 10 Flux.train!(loss, ps, data, opt)

## plot
p = []
push!(p, heatmap(ϕ_.kernel, title = "learned potential kernel"))
r = 0:0.01:1
push!(
    p,
    plot(r, ϕ_.radfunc.(r), title = "learned potential kernel radial function"),
)
plot(p..., layout = length(p))

vfplot(E_.kernel,grid; title = "learned E field kernel ")
