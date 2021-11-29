# Architecture

## Tensor fields over grid

Tensor fields are represented as 3d/4d arrays for 2d/3d uniform Cartesian grids with the last dimension for the field component. For example, a 3d vector field would be sized XxYxZx3 and while a 2d scalar field would be XxYx1. We provide

We store grid information including resolution, size and origin in `Grid`. By default, the origin indices are centered in the grid. Julia is 1-indexed, so the origin of a 3x3x3 grid defaults to [2, 2, 2].

```@docs
Grid
Field
```

## Tensor field products and operations

All `Arrays` operations (eg +, -, abs) apply to fields. Local pointwise products however depend on the ranks of input fields and output field. `field_prod` infers the appropriate pointwise product (eg scalar, dot, cross, matrix vector) from the ranks. For example, locally multiplying two vector fields into a scalar field (ranks 1, 1 -> 0) involves the dot product. We also provide convenience functions for retrieving or computing the pointwise norm and rank.

```@docs
field_prod
field_norm
field_rank
```

## Particle mesh placement and interpolation

With grid info we can interpolate a tensor field at any location. We can also place a point tensor source (eg scalar particle) anywhere. This particle mesh placement applies integral normalization, so the array value is scaled by 1/dV (or 1/dA). Both work via a proximity weighted average of the closest lattice points (in general up to 4 in 2d and 8 in 3d).

```@docs
put_point_source!
```

## Linear operators

`LinearOperator` constructs functions for common differential operators and Green's functions. operators diverging at 0 are zeroed out within `rmin`. Any custom equivariant operator can also be made by specifying its radial function and ranks of the input output fields.

Under the hood, we implement all linear operators as tensor field convolutions between the input field and the impulse response field of the operator. We compute the kernel field as a product of the radial function and the appropriate spherical harmonic tensor. The operator's kernel field has a `Grid` with `origin` centered on a lattice point. The output field's components are truncated to have same size and `Grid` as those of input.

Long ranged convolutions are automatically computed in Fourier space by dependency `DSP.jl`

```@docs
LinearOperator
```

## Machine learning

We can learn rotation equivariant mappings between sets of scalar, vector and tensor fields. `EquivLayer` constructs neural network layers compatible with Julia's machine learning library `Flux.jl`.

...

```@docs
EquivLayer
```
