"""
We map a charge distribution (scalar field) to electric potential (scalar field) and electric field (vector field).
In fluid mechanics the equivalent mapping is from source/sink distribution to flow potential and velocity field.
"""
dir = ".."
include("$dir/src/neural_networks.jl")
include("$dir/src/plotutils.jl")
Random.seed!(1)

# grid params
dx = 0.1
dim = 3
sz = fill(11, dim) #
grid = Grid(dx, sz)
@show grid.dΩ, grid.origin

rank = 0
charges = Field(; grid, rank)
# @show charges.components


# place dipole point charges 2r apart in grid as input
r = 0.5
pos = [r 0 0; -r 0 0]'
vals = [1.0; 1.0]'
put_point_source!(charges,grid, pos, vals)
field_plot(charges)

# tensor convolution params

# generate data with Green's functions for Poisson's Equation, Gauss's law
rmax = 1.0
name = :inverse_squared_field
field_op = LinearOperator(name; dx, dim, rmax)
name = :potential
potential_op = LinearOperator(name; dx, dim, rmax)

# output data
potential = potential_op(charges,grid)
efield = field_op(charges,grid)
field_plot(potential)
# field_plot(field)

# check
rvec = [0.1, 0, 0]
# automatic interpolation at rvec
# @assert get_tensor(rvec,potential,grid) ≈ [1 / (4π * r) - 1 / (4π * 3r)]
# @assert get_tensor(rvec,efield,grid) ≈ [1 / (4π * r^2) - 1 / (4π * (3r)^2), 0, 0]
@assert potential(rvec,grid) ≈ [1 / (4π * .6) + 1 / (4π * .4)]
@assert efield(rvec,grid) ≈ [1 / (4π * .6^2) + 1 / (4π * (.4)^2), 0, 0]

rank_max = 1
inranks = [0] # input scalar field
outranks = [0, 1] # output scalar field, vector field
X = [charges]
Y = [potential, field]
# define tensor field convolution layer
L = EquivLayer(:conv,inranks, outranks, dim, dx, rmax)

function loss()
    y1hat, y2hat = L(X,grid)
    l1 = sum(Flux.mae.(Y[1], y1hat))
    l2 = sum(Flux.mae.(Y[2], y2hat))
    l = l1 + l2
    println(l)
    l
end
loss()

ps = Flux.params(L)
data = [()]
opt = ADAM(0.1)

for i = 1:5
    Flux.train!(loss, ps, data, opt)
end
