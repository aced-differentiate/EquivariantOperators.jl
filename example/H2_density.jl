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
sz = size(electronic_density)
grid = Grid(dx, sz; origin)
@unpack sz, dV = grid

chargesum = sum(electronic_density) * dV
@assert chargesum ≈ sum(charges)


# place dipole point charges 2r apart in grid as input
rank = 0
dims = 3
ρp = Field(; grid, rank)
put_point_source!(
    ρp,
    grid,
    positions,
    reshape(charges, (1, length(charges))),
)
# plot(ρp(1))

# tensor convolution params

# generate data with Green's functions for Poisson's Equation, Gauss's law
rmax = 3.0
rank_max = 1
in_ = Props((1,), grid) # input scalar field
x = cat(ρp, dims = 4)
y = cat(electronic_density, dims = 4)
# define tensor field convolution layer
L = EquivConv(Props((1,), grid), Props((1,), grid), rmax)
Q = EquivProd(L.out; spectra = true)
D1 = EquivDense(Q.out, Props((1,), grid))

function normalize_density(density, total)
    # return density
    s = sum(density)
    if s == 0
        return density
    end
    density * total / s
end

function calcforces(rvecs,charges,ρ,grid;rmax=grid.dx*norm(grid.sz))
    name = :inverse_squared_field
    rmin=2grid.dx
    efield_op = LinearOperator(name; dx,rmin, rmax)
    efield =efield_op(ρ)
    vcat([charge*get(efield,grid,rvec) for (charge,rvec) in zip(charges,eachcol(rvecs))]...)
end
a=rand(length(Q.out.ranks))
ps = Flux.params(L, a)
# ps = Flux.params(L, D1)
f0(x)=abs.(sum([a[i]*Q(L(x))[:,:,:,i:i] for i=eachindex(a)]))
# f0(x) =sum( abs.((Q ∘ L)(x)))
# f0(x) = abs.((D1 ∘ Q ∘ L)(x))
# f0(x)=abs.(L(x))
# f0(x)=normalize_density(abs.((D1∘Q∘L)(x)),chargesum,dV)
# f(x) = normalize_density(f0(x) - f0(zeros(sz..., 1)),chargesum,dV)
f(x) = normalize_density(f0(x) - f0(zeros(sz..., 1)),chargesum/dV)

sumy=sum(y)
function loss()
    yhat = f(x)
    @show l = nae(yhat,y;sumy)
    l
end
loss()

data = [()]
opt = ADAM(0.1)

for i = 1:20
    Flux.train!(loss, ps, data, opt)
end
ρ=electronic_density+ρp
@show forces=calcforces(positions,charges,ρ,grid)
electronic_density_hat=yhat = f(x)
ρhat=electronic_density_hat+ρp
@show forces_hat=calcforces(positions,charges,ρ_hat,grid)

plot(x[:, :, :, 1])
plot(yhat[:, :, :, 1])
plot(y[:, :, :, 1])
