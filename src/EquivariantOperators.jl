module EquivariantOperators

include("./neural_networks.jl")
export Grid, put_point_source!, field, field_rank, field_norm
export FieldProd, field_prod, FieldConv, field_conv
export EquivLayer, LinearOperator
#
end # module
