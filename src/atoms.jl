include("neural_networks.jl")

using IterTools
using Functors

using Random
using Optim
using Flux

Random.seed!(1)
function normalize_density(density, total)
    # return density
    s = sum(density)
    if s == 0
        return density
    end
    density * total / s
end
struct DensityPredictor
    # dx
    # rmax
    L::EquivConv
    Q::EquivProd
    D1::LocalDense
    D::LocalDense
    function DensityPredictor(dx, rmax)
        L = EquivConv(Props((0,)), Props((0, 2),), dx, rmax) |> mygpu
        Q = EquivProd(Props((0, 2, 0, 1)); spectra = true)
        n = length(Q.out.ranks)
        D = LocalDense(4n, 1) |> mygpu
        D1 = LocalDense(n, 4n, leakyrelu) |> mygpu
        # D2 = LocalDense(2n, 4n, leakyrelu) |> mygpu
        # D3 = LocalDense(4n, 1) |> mygpu
        new(L, Q, D1, D)
    end
end
@functor DensityPredictor

function Flux.params(m::DensityPredictor)
    @unpack L, Q, D1, D = m
    [L, D1, D]
end

function f_(x, ψ, E,L, Q, D1, D)
    (D ∘ D1 ∘ Q)(cat(L(x), ψ, E, dims = 4))
end
function (m::DensityPredictor)(x, ψ, E)
    @unpack L, Q, D1, D = m

    sz = size(x)
    x0 = ψ0 = zeros(sz)
    E0 = zeros(sz[1:3]..., 3)
    normalize_density(abs.(f_(x, ψ, E,L, Q, D1, D) - f_(x0, ψ0, E0,L, Q, D1, D)), sum(x))
end

mutable struct Frame
    atoms
    positions
    grid
    ops
    ρp
    ρe
    cache
    more
    function Frame(atoms, positions, grid;ρe=nothing)
        @unpack dV, dx, sz = grid
        total = sum(atoms)
        ρp = zeros(sz..., 1)
        put_point_source!(
            ρp,
            grid,
            positions,
            reshape(atoms, (1, length(atoms))),
        )
        @assert total ≈ sum(ρp) * dV

        rmax = dx * norm(sz)
        rmin = 2dx
        name = :inverse_squared_field
        E = LinearOperator(name; dx, rmin, rmax)
        name = :potential
        ψ = LinearOperator(name; dx, rmin, rmax)

        ops = (; ψ, E)
        cache = ()
        more = ()
        # x = ρp |> mygpu
        # y = electronic_density |> mygpu
        new(
            atoms,
            positions,
            grid,
            ops,
            ρp,
            ρe,
            cache,
            more,
        )
    end
end

function (m::DensityPredictor)(x::Frame)
    @unpack ρp, ops,cache = x
    if isempty(cache)
    ψp = ops.ψ(ρp)
    Ep = ops.E(ρp)
    x.cache=(;ψp,Ep)
else
    @unpack ψp,Ep=cache
end
m(ρp, ψp, Ep)
    # m.ρe=m(ρp, ψp, Ep)
end

function calc!(x, p)
    @unpack ρe,ρp, grid, atoms, positions, ops = x
    if p == :forces
        # ψ=ops.ψ(ρp)
        E = ops.E(ρe)
        F = hcat([
            atom * get(E, grid, position)
            for (atom, position) in zip(atoms, eachcol(positions))
        ]...)
        more = (; E, F)
        x.more = merge(x.more, more)
        return F
    end
    # more
end
##
# function fg!(F, G, x)
#     electronic_density_hat, ρp = predict_density(x, atoms, grid)
#     ρhat = -electronic_density_hat + ρp
#     @show forces_hat = calcforces(x, atoms, ρhat, grid)
#     if G != nothing
#         G .= -forces_hat
#     end
#     norm(forces_hat) / sum(atoms)
# end;
#
# x0 = positions
#
# xres = optimize(
#     Optim.only_fg!(fg!),
#     x0,
#     LBFGS(),
#     Optim.Options(show_trace = true, f_tol = tol),
# )
# xmin = Optim.minimizer(xres)
# dmin = norm(xmin[:, 1] - xmin[:, 2])
# @printf "\nOptimal bond length for Ecut=%.2f: %.3f Bohr\n" Ecut dmin
