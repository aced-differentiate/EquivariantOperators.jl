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
electronic_density = [case.density]
@show sz = size(electronic_density[1])
grid = Grid(dx, sz; origin)

chargesum = sum(electronic_density[1]) * grid.dΩ
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
rmax = 2.0
name = :inverse_squared_field
field_op = LinearOperator(name; dx, dims, rmax)
name = :potential
potential_op = LinearOperator(name; dx, dims, rmax)

# output data
potential = potential_op(proton_density, grid)
efield = field_op(proton_density, grid)
# field_plot(potential)
# plot(Array(potential(1)))
# field_plot(field)

# check
rvec = center_charge
# automatic interpolation at rvec
@show potential(rvec, grid), [2 / (4π * (r))]
# @show potential(rvec,grid) ,[1 / (4π * (.1+r)) + 1 / (4π * (r-.1))]
@show efield(rvec, grid), zeros(3)
# @show efield(rvec,grid) ,[1 / (4π * (.1+r)^2) + 1 / (4π * (r-.1)^2), 0, 0]

rank_max = 1
inranks = [0] # input scalar field
outranks = [0] # output scalar field, vector field
X = [proton_density]
Y = [potential]
# define tensor field convolution layer
L = EquivLayer(:conv, inranks, outranks; dims, dx, rmax)
y1hat = L(X, grid)[1]

function loss()
    y1hat = L(X, grid)[1]
    l = Flux.mae(Y[1], y1hat)
    # l = mean(sum(abs.(Y[1] - y1hat)))
    # l +=Flux.mae(Y[2], y2hat)
    println(l)
    l
end
loss()
ps = Flux.params(L)
data = [()]
opt = ADAM(0.1)

for i = 1:3
    Flux.train!(loss, ps, data, opt)
end

##
# define tensor field convolution layer
inranks = outranks = [0]
midranks = [0, 1, 2]

# @load "../data/density.jld", ρ
dims = 3
rmax = 3.0
L = EquivLayer(:conv, inranks, midranks; dims, dx, rmax)
Q = EquivLayer(:prod, midranks, outranks)

function normalize_density(density, total, grid.dΩ)
    density = abs.(density[1])
    [density / sum(density) * total / grid.dΩ]
end

electronic_density_pred = 0
function loss()
    global electronic_density_pred = normalize_density(
        Q(L([proton_density], grid; remake = true), grid)[1],
        chargesum,
        grid.dΩ,
    )
    l =
        grid.dΩ *
        sum(abs.(electronic_density_pred[1] .- electronic_density[1])) /
        chargesum
    println(l)
    l
end
loss()

ps = Flux.params(L, Q)
data = [()]
opt = ADAM(0.1)

for i = 1:30
    Flux.train!(loss, ps, data, opt)
end

plot(electronic_density_pred[1])
