# using GLMakie
# using CairoMakie
using Plots

function field_plot(x; title = "", kwargs...)
    l = size(x,4)
    plot(
        [
            plot(
        x,
                title = "$title\ncomponent $i",
                c = cgrad([:blue, :red], [0.0, 0.5, 1.0]),
                kwargs...,
            ) for x in eachslice(x,dims=4)
        ]...,
        layout = l,
    )
    # fig = Figure()
    # fig = Figure(
    #     backgroundcolor = RGBf0(0.98, 0.98, 0.98),
    #     resolution = (1000, 700),
    # )
    # for i = 1:length(X)
    #     volume(fig[1, i], X[i])
    # end
    # plots = [volume(X[i]) for i = 1:length(X)]
    # plot(plots..., layout = length(X))
end
