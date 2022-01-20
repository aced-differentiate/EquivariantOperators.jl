"""
We demonstrate core capabilities of EquivariantOperators.jl: particle mesh, finite differences, and machine learning. In a particle mesh context, we place 2 +1 point charges in a grid creating a scalar field. We then do a finite difference calculation of the electric field and potential. Switching gears to machine learning, we train our equivariant neural network to learn this transformation from the charge distribution (scalar field) to the electric potential (scalar field) and field (vector field), essentially learning the Green's function solution to Poisson's Equation. Finally, we treat our point charges as proton nuclei and train another neural network to predict the ground state electronic density of H2.
"""

repo = ".."
include("$repo/src/neural_networks.jl")
# include("../src/plotutils.jl")
# include("$repo/src/EquivariantOperators.jl")
# using .EquivariantOperators

using FileIO
using Plots
using Random
Random.seed!(1)

name = "h2"
case = load("..\\density-prediction\\data\\$name.jld2", "cases")[1]
@show positions = case.positions'
@show charges = case.charges
natoms = length(charges)
@show r = abs(positions[1, 1] - positions[1, 2]) / 2
@show center_charge = vec(sum(positions, dims = 2) / 2)

# grid params
@show dx = case.resolution
@show origin = case.origin
electronic_density = case.density
@show sz = size(electronic_density)
grid = Grid(dx, sz; origin)

chargesum = sum(electronic_density) * grid.dV
@assert chargesum ≈ sum(charges)


# place dipole point charges 2r apart in grid as input
rank = 0
dims = 3
proton_density = Field(; grid, rank)
put_point_source!(
    proton_density,
    grid,
    positions,
    reshape(charges, (1, length(charges))),
)
# plot(proton_density(1))

# tensor convolution params

# generate data with Green's functions for Poisson's Equation, Gauss's law
rmax=1.
rank_max = 1
in_ =Props((1,),grid) # input scalar field
out =Props((1,),grid) # input scalar field
X = cat(proton_density,dims=4)
Y =cat( electronic_density,dims=4)
# define tensor field convolution layer
L = EquivConv(Props((1,),grid),Props((2,1),grid), rmax)
Q = EquivProd(L.out;spectra=true)
D1 = EquivDense(Q.out,Props((1,),grid))

function normalize_density(density, total, dV)

    density / (dV*sum(density) )* total
end

f0(X)=normalize_density(abs.((D1∘Q∘L)(X)),chargesum,grid.dV)
f(X)=f0(X)-f0(zeros(grid.sz...,1))
function loss()
    Yhat= f(X)
    l = Flux.mae(Y, Yhat)
    println(l)
    l
end
loss()

ps = Flux.params(L,Q,D1)
data = [()]
opt = ADAM(0.1)

for i = 1:5
    Flux.train!(loss, ps, data, opt)
end

plot(Y[:,:,:,1])
