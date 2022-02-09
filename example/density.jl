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

# name = "AAACQNRQYWKRAR-SCLBCKFNSA-N"
# name = "benzene"
# name = "h2"
# case = load("..\\density-prediction\\data\\$name.jld2", "cases")[1]

folder = "..\\density-prediction\\data"
name="dsgdb9nsd_000001"
file = "$folder\\$name.jld2"
case = load(file,"case")

@unpack resolution, density, charges, positions = case
# positions=positions'
# case = load("..\\density-prediction\\data\\$name.jld2", "case")
# @unpack resolution, density, core_density,charges,core_charges, positions = case
# grid params
origin=ones(3)
@show dx =resolution
sz = size(density)
grid = Grid(dx, sz; origin)
@unpack sz, dV = grid

electronic_density = cat(density,dims=4)
total = sum(charges)
# electronic_density *=total/(sum(electronic_density*dV))


proton_density = zeros(sz...,1)
put_point_source!(
    proton_density,
    grid,
    positions,
    reshape(charges, (1, length(charges))),
)
@assert total ≈ sum(electronic_density)*dV≈ sum(proton_density)*dV
# plot(proton_density(1))

# tensor convolution params

# generate data with Green's functions for Poisson's Equation, Gauss's law
x = proton_density
y = electronic_density

rmax = 4.0
rank_max = 2
in_ = Props((1,), grid) # input scalar field
# define tensor field convolution layer
L = EquivConv(Props((1,), grid), Props((2,), grid), rmax)
Q = EquivProd(L.out; spectra = true)

n=length(Q.out.ranks)
D = LocalDense(4n,1)
D1 = LocalDense(n,4n,leakyrelu)
D2 = LocalDense(2n,4n,leakyrelu)
D3 = LocalDense(4n,1)

function normalize_density(density, total)
    # return density
    s = sum(density)
    if s == 0
        return density
    end
    density * total / s
end

# a=rand(length(Q.out.ranks))
# ps = Flux.params(L, a)
# f0(x)=abs.(sum([a[i]*Q(L(x))[:,:,:,i:i] for i=eachindex(a)]))
ps = Flux.params(L, D1,D2,D3)
f0(x) =abs.((D3∘D2∘D1∘Q ∘ L)(x))
ps = Flux.params(L, D1,D)
f0(x) =(D∘D1∘Q ∘ L)(x)
f(x) = normalize_density(abs.(f0(x) - f0(zeros(sz..., 1))),total/dV)

sumy=sum(y)
function loss()
    yhat = f(x)
    @show l = nae(yhat,y;sumy)
    l
end

@show sum(abs.(y-yhat))
@show sum(yhat)*dV,dV*sumy
@show minimum.([yhat,y])
@show maximum.([yhat,y])
yhat = f(x)
loss()

data = [()]
opt = ADAM(0.1)

for i = 1:100
    Flux.train!(loss, ps, data, opt)
end

z =(Q ∘ L)(x)
##
using GLMakie
include("$repo/src/plotutils.jl")

# volume(x[:, :, :, 1])
# volume(y[:, :, :, 1])
GLMakie.inline!(false)
# fig=GLMakie.contour(10yhat[:, :, :, 1])
# fig=volume(10yhat[:, :, :, 1],algorithm = :iso,)
# fig=volume(10y[:, :, :, 1],algorithm = :iso,)
# fig=volume(10yhat[:, :, :, 1],algorithm = :absorption,)
fig=volume(yhat[:, :, :, 1])
display(fig)
# volume(10z[:, :, :, 3])

# vis(L)
