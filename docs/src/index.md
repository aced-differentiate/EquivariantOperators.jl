# Home

!!! note
    Documentation website under construction. Expected release of code base in early December 2021.

## Synopsis

EquivariantOperators.jl is a Julia package implementing **equivariant machine learning**, **finite difference** operators and **particle mesh methods** on scalar, vector and **tensor fields** over uniform grids in 2d/3d. It's a fully **differentiable finite differences engine** that can run forwards for simulation or backwards for machine learning and inverse problems. Emphasis is on **rotation equivariant** operators which consequently preserve symmetry. This includes common differential operators (eg div, curl, Laplacian), Green's functions (eg inverse-square fields, Gaussians, Stokeslet), and parametrized equivariant neural operators.

Tensor fields are represented as multidim arrays supporting particle mesh methods of interpolation and point source placement. Operators are implemented as tensor field convolutions in real or Fourier space using rank appropriate products (eg scalar, dot, cross). For machine learning, we provide tensor convolution, product and nonlinear scaling layers for inferring equivariant mappings between arbitrary sets of tensor fields.

## Use cases
- Machine learning rotation equivariant and symmetry preserving behavior of dynamical systems and solutions to PDEs
- Solving inverse problems via adjoint methods
- Applying finite difference differential operators (eg grad, div) and Green's functions (eg inverse-square fields, Gaussians, Stokeslet) on images and vector fields
- Particle mesh point source placement, interpolation, and Fourier space field calculations

Check out our tutorials on Google Colab and our Arxiv preprint!

## Contributors

[Paul Shen](xingpins@andrew.cmu.edu), [Michael Herbst][herbst@acom.rwth-aachen.de], PI: [Venkat Viswanathan](venkatv@andrew.cmu.edu)

In consultation with Rachel Kurchin, Dhairya Gandhi, Chris Rackauckas

In collaboration with Julia Computing
