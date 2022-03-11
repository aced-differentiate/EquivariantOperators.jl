include("operators.jl")



function calcforces(rvecs,charges,ρ,grid;rmax=grid.dx*norm(grid.sz))
    name = :inverse_squared_field
    rmin=2grid.dx
    efield_op = LinearOperator(name; dx,rmin, rmax)
    efield =efield_op(ρ)
hcat([charge*get(efield,grid,rvec) for (charge,rvec) in zip(charges,eachcol(rvecs))]...)
end
function calcfield(name,ρ,grid;rmax=grid.dx*norm(grid.sz))
@unpack dx=grid
    op = LinearOperator(name; dx, rmax)
    op(ρ)
end

function prep(positions, charges, grid)
    ρp = zeros(grid.sz..., 1)
    put_point_source!(
        ρp,
        grid,
        positions,
        reshape(charges, (1, length(charges))),
    )
    ψ=calcfield(:potential,ρp,grid)
    E=calcfield("inverse_squared_field",ρp,grid)
ρp,ψ,E
end
function predict_density(positions, charges, grid)
    ρp,ψ,E
    electronic_density = f(ρp)
    @assert sum(charges) ≈
            sum(electronic_density) * dV ≈
            sum(ρp) * dV
    electronic_density, ρp
end
