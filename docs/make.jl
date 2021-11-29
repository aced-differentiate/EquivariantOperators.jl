using Documenter

# include("../src/EquivariantOperators.jl")
# using .EquivariantOperators

include("../src/neural_networks.jl")

##
makedocs(
    sitename = "EquivariantOperators.jl",
    pages = ["index.md", "architecture.md", "publications.md", "tutorials.md"],
)
