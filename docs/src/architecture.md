# Architecture

## Scalar & vector fields

Scalar & vector fields in 2d/3d are represented as 3d/4d arrays with the last dimension for the field component. For example, a 3d vector field would be sized XxYxZx3 and while a 2d scalar field would be XxYx1.

## Customizable grid

Grid is specified by its discrete cell vectors (column-wise matrix), overall size and origin. For a uniform Cartesian 5x5x5 grid discretized at 0.1 with a centered origin, we get `cell = [0.1 0; 0 0.1]` & `origin = [3, 3, 3]`. Grid cell can in general be nonuniform & noncartesian.

## Pointwise products

`u ⊗ v` computes the appropriate pointwise product between 2 scalar or vector fields (inferred as scalar-scalar, scalar-vector, dot, cross). For greater clarity one may also write `u ⋅ v` for dot and `u ⨉ v` for cross

## Particle mesh placement and interpolation

With grid info we can interpolate a scalar or vector field at any location. We can also place a scalar or vector point source anywhere with automatic normalization wrt discretization. Both work via a proximity weighted average of the closest grid points (in general up to 4 in 2d and 8 in 3d).

## Finite difference operators

`Op` constructs finite difference operators. Prebuilt operators like differential operators (`▽`) & common Green's functions can be specified by name. Custom equivariant operators can be made by specifying radial function.
