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
sz = fill(11,3) #
grid = Grid(dx, sz)
@show grid.dV, grid.origin

rank = 0
charges = zeros(sz...,1)
# @show charges.components


# place dipole point charges 2r apart in grid as input
pos = zeros(3)
vals = [1.0]
put_point_source!(charges,grid, pos, vals)
# field_plot(charges)

# tensor convolution params

# generate data with Green's functions for Poisson's Equation, Gauss's law
rmax = 1.0
name = :inverse_squared_field
field_op = LinearOperator(name; dx, rmax)
name = :potential
potential_op = LinearOperator(name; dx, rmax)

# output data
potential = potential_op(charges,grid)
efield = field_op(charges,grid)
# field_plot(potential)
# field_plot(field)

# check
r=.2
rvec = [r, 0, 0]
# automatic interpolation at rvec
# @assert get_tensor(rvec,potential,grid) ≈ [1 / (4π * r) - 1 / (4π * 3r)]
# @assert get_tensor(rvec,efield,grid) ≈ [1 / (4π * r^2) - 1 / (4π * (3r)^2), 0, 0]
@assert get(potential,grid,rvec) ≈ [1 / (4π * r)]
@assert get(efield,grid,rvec) ≈ [1 / (4π * r^2), 0, 0]

rank_max = 1
in_ =Props((1,),grid) # input scalar field
out =Props((1,1),grid) # input scalar field
X = charges
Y = cat(potential, efield,dims=4)
# define tensor field convolution layer
L = EquivConv(in_, out, rmax)

function loss()
    Yhat= L(X)
    l = Flux.mae(Y, Yhat)
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
