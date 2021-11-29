using DSP: conv
using LinearAlgebra


Funcs3D = (
    (x, y, z) -> cat(x, y, z,dims=4),
    (x, y, z) -> cat(
        sqrt(3) * x .* z,
        sqrt(3) * x .* y,
        y .^ 2 - (x .^ 2 + z .^ 2) / 2,
        sqrt(3) * y .* z,
        (z .^ 2 - x .^ 2) * sqrt(3) / 2,
    dims=4),
)

 Funcs2D = (
    (x, y) -> [x, y],
    #     # (x,y)->        [
    #     #             sqrt(3) * x*z,
    #     #             sqrt(3) * x*y,
    #     #             y^ 2 - (x^ 2 + z^ 2) / 2,
    #     #             sqrt(3) *y*z,
    #     #             (z^ 2 - x^ 2) * sqrt(3) / 2,
    #     #         ],
)

function make_spherical_harmonics(dx, size, origin, rank_max; dims = 3)
    if rank_max < 0
        return [], []
    end
    if dims == 3
        rvecs =
            [
                [x, y, z] .- origin for x = 1:size[1], y = 1:size[2],
                z = 1:size[3]
            ] * dx
        Funcs = Funcs3D
    elseif dims == 2
        rvecs = [[x, y] .- origin for x = 1:size[1], y = 1:size[2]] * dx
        Funcs = Funcs2D
    end
    r = norm.(rvecs)
    if rank_max > 0
        rhat = rvecs ./ broadcast(x -> x ≈ 0.0 ? 1.0 : x, r)
        rhat = [getindex.(rhat, i) for i = 1:dims]
    end
    sh = [Funcs[i](rhat...) for i = 1:rank_max]
    r, sh
end
