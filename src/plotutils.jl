# using GLMakie
# using CairoMakie
using Plots

function field_plot(x; title = "", kwargs...)
    l = length(x)
    high=maximum(maximum.(x))
    low=minimum(minimum.(x))
    plot(
        [
            plot(
                # x[i],
                (x[i] .- low) ./ (high-low),
                title = "$title\ncomponent $i",
                # st = :heatmap,
                # clims=[-maxval,maxval],
                c = cgrad([:blue, :red], [0.0, 0.5, 1.0]),
                kwargs...,
            ) for i = 1:length(x)
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
