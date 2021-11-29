include("../src/nn.jl")
using Test
Random.seed!(1)



"""
charge distribution -> potential (Poisson's eqn), electric field (Gauss's law)
"""

# input Scalar field
inranks = [0]
# output scalar field, vector field
outranks = [0, 1]
sz=(8,8,8)
dx = 0.1
dV=dx^3
rmax = 2dx
lmax = 1

# charge distribution
x = [zeros(sz...)]
ix=[5, 5, 5]
x[1][ix...] = 1.0
X=[x]

# generate data
# Green's fn for Poisson, Gauss
f1 = LinearOperator(:potential,dx;rmax=rmax)
f2 = LinearOperator(:field,dx;rmax=rmax)

y1 = f1(X[1])
y2 = f2(X[1])

# check
v=[1,1,1]
ix = ix.+v
r=norm(v)*dx
@test y1[1][ix...]≈dV/r
@test [y2[i][ix...] for i = 1:3]≈dV/r^2*ones(3)/sqrt(3)

##
# train
# linear layer: tensor field convolution
L = EquivConv(inranks, outranks, dx; rmax = rmax)

function nn(X)
    L(X)
end


function loss()
    y1hat, y2hat = nn(X)
    l1 = Flux.mae(toArray(y1), toArray(y1hat))
    l2 = Flux.mae(toArray(y2), toArray(y2hat))
    l = l1 + l2
    println(l)
    l
end
loss()
##
ps = Flux.params(L)
data = [()]
opt = ADAM(0.1)

println("===\nTraining")
for i = 1:5
    # global doplot = i % 50 == 0
    Flux.train!(loss, ps, data, opt)
end

##
Random.seed!(1)
n=4
inranks=[0,0]
outranks=[0]
X=[[rand(n,n,n)],[rand(n,n,n)]]
y=[[X[1][1].*X[2][1]]]

# train
# linear layer: tensor field convolution

A = EquivAttn(inranks, outranks)

function nn(X)
    A(X)
end


function loss()
    yhat = nn(X)
    l = Flux.mae(toArray(y[1]), toArray(yhat[1]))
    println(l)
    l
end
loss()

ps = Flux.params(A)
data = [()]
opt = ADAM(0.1)

println("===\nTraining")
for i = 1:10
    # global doplot = i % 50 == 0
    Flux.train!(loss, ps, data, opt)
end
