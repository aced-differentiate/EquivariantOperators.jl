# Guide

## Scalar & vector fields

Scalar & vector fields in 2d/3d are represented as 3d/4d arrays with the last dimension for the field component. For example, a 3d vector field would be sized XxYxZx3 and while a 2d scalar field would be XxYx1.

## Customizable grid

```@docs
Grid
```

## Pointwise products

```@docs
fieldprod
```

## Particle mesh placement and interpolation

```@docs
get(::AbstractArray, ::Grid, ::AbstractVector)
```

## Finite difference operators

```@docs
Op
```
