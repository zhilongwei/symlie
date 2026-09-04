# symlie: Symbolic Lie Symmetry Analysis on Jet Spaces with SymPy

[![CI](https://github.com/zhilongwei/symlie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/zhilongwei/symlie/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SymPy](https://img.shields.io/badge/SymPy-1.14+-green.svg)](https://www.sympy.org/)

**`symlie`** is a modern, exact symbolic framework for **Lie symmetry analysis**, **jet space prolongations**, **differential invariants**, and **variational mechanics** built on top of SymPy.

---

## Key Features

- **Jet Space & Multi-Index Infrastructure**: Exact parsing and differential substitution for multi-dimensional scalar and coupled PDE/ODE systems on regular, acyclic solved branches.
- **Fréchet Linearization & Formal Adjoints**: Computation of forward Fréchet derivative operator matrices $\mathbf{D}_F(Q)$ and formal $L^2$ adjoint operators $\mathbf{D}_F^*(v)$ via integration by parts on jet spaces.
- **Variational Calculus & Euler–Lagrange Operators**: Automatic derivation of equations of motion $\mathbf{E}_u(\mathcal{L}) = 0$ for higher-order Lagrangians.
- **Hamiltonian Dynamics & Poisson Brackets**: Canonical Poisson brackets $\{F, G\}$, equations of motion $\dot{z} = \{z, H\}$, and direct Hamilton's equations.
- **Lie Prolongation & Determining Systems**: Exact construction of linear overdetermined PDE determining systems for unknown infinitesimals $\xi^i(x, u), \phi^\alpha(x, u)$ on a regular equation branch by splitting on unconstrained jet coordinates.
- **Polynomial Point Symmetry Solver**: Exact symmetry calculation within an explicit finite polynomial ansatz, with basis extraction using nullspace decomposition.
- **Lie Bracket Commutator Algebra**: Base-space vector field Lie brackets $[X_1, X_2]$ to determine Lie algebra structure constants.
- **Invariance Verification**: Automatic symbolic check $\left. \mathrm{pr}^{(n)} X(\Delta) \right|_{\Delta = 0} \equiv 0$ for any vector field candidate.

---

## Mathematical Foundations

### 1. Jet Spaces and Prolongation Formula
For independent variables $x = (x^1, \dots, x^p)$ and dependent fields $u = (u^1, \dots, u^q)$, a point symmetry generator has the form:

$$
X = \sum_{i=1}^p \xi^i(x, u) \frac{\partial}{\partial x^i} + \sum_{\alpha=1}^q \phi^\alpha(x, u) \frac{\partial}{\partial u^\alpha}
$$

Its evolutionary characteristic is $Q^\alpha = \phi^\alpha - \sum_{i=1}^p \xi^i u^\alpha_{x^i}$.

The $n$-th prolongation $\mathrm{pr}^{(n)} X$ applied to a differential equation $\Delta(x, u^{(n)}) = 0$ is computed via the Fréchet linearization:

$$
\mathrm{pr}^{(n)} X(\Delta) = \mathbf{D}_\Delta(Q) + \sum_{i=1}^p \xi^i D_i \Delta
$$

restricted to the differential equation manifold $\Delta = 0$ and its differential consequences.

### 2. Fréchet Derivative & Formal Adjoint
For a differential expression $F(x, u^{(n)})$:

$$
\mathbf{D}_F(Q) = \sum_{J} \frac{\partial F}{\partial u_J} D_J Q
$$

$$
\mathbf{D}_F^*(v) = \sum_{J} (-1)^{|J|} D_J \left( \frac{\partial F}{\partial u_J} v \right)
$$

### 3. Euler–Lagrange Operator
For a Lagrangian density $\mathcal{L}(x, u^{(n)})$:

$$
\mathbf{E}_{u^\alpha}(\mathcal{L}) = \sum_J (-1)^{|J|} D_J \left( \frac{\partial \mathcal{L}}{\partial u^\alpha_J} \right) = 0
$$

---

## Installation

### Directly from GitHub

Using `pip`:
```bash
pip install git+https://github.com/zhilongwei/symlie.git
```

Using `uv`:
```bash
uv add git+https://github.com/zhilongwei/symlie.git
```

### From Source (Development)

```bash
git clone https://github.com/zhilongwei/symlie.git
cd symlie
uv sync
```

---

## Quickstart Guide

### 1. Lie Symmetries of the Heat Equation
```python
import sympy as sp
from symlie import infinitesimals, verify_generator, lie_bracket

x, t = sp.symbols("x t")
u = sp.Function("u")(x, t)

# Heat equation: u_t - u_xx = 0
heat_eq = u.diff(t) - u.diff(x, 2)

# Compute point symmetries within a total-degree-1 polynomial ansatz
sol = infinitesimals(heat_eq, u, (x, t), ansatz_degree=1)
print(f"Dimension within ansatz: {sol.ansatz_dimension}")  # 6

for i, gen in enumerate(sol.basis, 1):
    print(f"X_{i}: xi = {gen.xi}, phi = {gen.phi}")
    assert verify_generator(heat_eq, u, (x, t), gen)
```

### 2. Variational Mechanics (Wave Equation)
```python
import sympy as sp
from symlie import euler_lagrange

x, t = sp.symbols("x t")
c = sp.symbols("c", positive=True)
u = sp.Function("u")(x, t)

# Lagrangian density: L = 1/2 u_t^2 - 1/2 c^2 u_x^2
L = sp.Rational(1, 2) * u.diff(t) ** 2 - sp.Rational(1, 2) * c**2 * u.diff(x) ** 2

# Euler-Lagrange operator
el = euler_lagrange(L, u, (x, t))
# Returns: -u_tt + c^2 u_xx
```

### 3. Canonical Hamiltonian Dynamics
```python
import sympy as sp
from symlie import poisson_bracket, hamilton_equations

t = sp.symbols("t")
q = sp.Function("q")(t)
p = sp.Function("p")(t)
m, k = sp.symbols("m k", positive=True)

# Harmonic Oscillator Hamiltonian
H = p**2 / (2 * m) + sp.Rational(1, 2) * k * q**2

# Poisson brackets
print("{q, p} =", poisson_bracket(q, p, q, p))  # 1
print("{q, H} =", poisson_bracket(q, H, q, p))  # p(t)/m
print("{p, H} =", poisson_bracket(p, H, q, p))  # -k*q(t)

# Hamilton's equations of motion
eqs = hamilton_equations(H, q, p, time=t)
# Returns: (Eq(Derivative(q(t), t), p(t)/m), Eq(Derivative(p(t), t), -k*q(t)))
```

---

## Examples & Tutorials

Check out the interactive tutorials in the [`examples/`](examples) directory:

| Notebook | Description | Key Physical Concepts |
| :--- | :--- | :--- |
| [`heat_equation.ipynb`](examples/heat_equation.ipynb) | 1D Heat Conduction | Parabolic scaling, Gaussian heat kernel self-similar reduction |
| [`burgers_equation.ipynb`](examples/burgers_equation.ipynb) | Viscous Burgers' Equation | Galilean boosts, projective symmetry, Cole–Hopf linearization |
| [`kdv.ipynb`](examples/kdv.ipynb) | Korteweg–de Vries (KdV) | Dispersion, Galilean boost, exact 1-soliton traveling wave |
| [`wave_equation.ipynb`](examples/wave_equation.ipynb) | Hyperbolic Wave Equation | Lorentz boost invariance, D'Alembert wave reduction |
| [`klein_gordon.ipynb`](examples/klein_gordon.ipynb) | Cubic Klein–Gordon Field | Relativistic Lorentz invariance, singular traveling profile |
| [`porous_medium.ipynb`](examples/porous_medium.ipynb) | Porous Medium Equation | Degenerate nonlinear diffusion, Barenblatt–Pattle solution |
| [`coupled_wave_system.ipynb`](examples/coupled_wave_system.ipynb) | Coupled $2 \times 2$ Wave System | Multi-field $2 \times 2$ Fréchet matrix, degree-1 polynomial symmetries |
| [`boundary_layer_instability.ipynb`](examples/boundary_layer_instability.ipynb) | Boundary Layer & Stability | Blasius similarity reduction, Orr–Sommerfeld & Rayleigh operators |
| [`ode_and_variational_mechanics.ipynb`](examples/ode_and_variational_mechanics.ipynb) | ODEs & Variational Calculus | Harmonic oscillator, scale-invariant ODE, Poisson brackets |
| [`laplace_and_liouville.ipynb`](examples/laplace_and_liouville.ipynb) | Laplace & Liouville Equations | Conformal $\mathfrak{so}(3,1)$ generators, formal self-adjointness, elliptic/hyperbolic Liouville solutions |
| [`drift_diffusion_and_kolmogorov.ipynb`](examples/drift_diffusion_and_kolmogorov.ipynb) | Drift-Diffusion & Kolmogorov | Variable-coefficient diffusion group classification, 3D Kolmogorov PDE symmetries |
| [`ermakov_pinney_and_linearization.ipynb`](examples/ermakov_pinney_and_linearization.ipynb) | Ermakov–Pinney & Linearization | $\mathfrak{sl}(2, \mathbb{R})$ symmetry, Lewis–Riesenfeld invariant, Lie's linearization theorem & quadratic Liénard ODEs |
| [`differential_invariants_and_reductions.ipynb`](examples/differential_invariants_and_reductions.ipynb) | Differential Invariants & Reductions | Euclidean $E(2)$ curvature $\kappa$ and order reduction of a fiber-preserving ODE |
| [`darboux_and_euler_poisson_darboux.ipynb`](examples/darboux_and_euler_poisson_darboux.ipynb) | Darboux & Euler–Poisson–Darboux | Singular hyperbolic wave equations, pseudo-conformal Killing vector fields, (2+1)-D Darboux scale invariance |

---

## Scope and Regularity

`infinitesimals(..., ansatz_degree=d)` returns the solution space inside a
finite total-degree-$d$ polynomial ansatz. Its dimension is not, in general,
the dimension of the complete Lie point-symmetry algebra. Linear differential
equations also possess the infinite solution-superposition ideal
$h(x)\partial_u$, where $h$ is any solution of the homogeneous equation.

Automatic differential substitution solves the equations simultaneously for
distinct leading jets and reports any nonzero regularity conditions.
Use `infer_substitution_rules` with `leading_derivatives=...` and
`return_conditions=True` when the automatic derivative ranking is unsuitable,
or pass explicit `substitution_rules` to the symmetry routines. Exceptional
parameter values should be substituted before analysis because they can define
different equation branches and symmetry algebras. If no unique regular branch
can be inferred, the symmetry routines raise `ValueError` instead of evaluating
the invariance condition off the equation manifold. Explicit substitution rules
are checked against the equations and must define a regular solved branch.

The polynomial solver constrains the infinitesimal coefficients to a finite
polynomial ansatz; the differential equation itself may contain common
transcendental factors such as exponentials and trigonometric functions. After
symbolic simplification, distinct non-polynomial factors are split as additional
algebraically independent generators. Simplify or rewrite identities between
those factors before solving. Returned regularity conditions include both
equation reduction assumptions and parameter conditions needed by the generic
nullspace basis. Substitute exceptional parameter values and solve again to
obtain their separate symmetry branches.

---

## Running Tests

Run the complete test suite with `pytest`:
```bash
uv run pytest
```

---

## References

The theoretical foundation and design of `symlie` draw upon the following classic and modern literature:

1. **Peter J. Olver**, [*Applications of Lie Groups to Differential Equations*](https://doi.org/10.1007/978-1-4612-4350-2), 2nd ed., Graduate Texts in Mathematics, Springer, 1993.
2. **Gerd Baumann**, [*Symmetry Analysis of Differential Equations with Mathematica*](https://doi.org/10.1007/978-1-4612-1360-4), Springer, 2000 (and the `MathLie` package).
3. **Brian J. Cantwell**, [*Introduction to Symmetry Analysis*](https://doi.org/10.1017/CBO9780511613999), Cambridge University Press, 2002 (and the accompanying `IntroToSymmetry.m` package).
4. **F. Güngör**, [*Lie symmetry group methods for differential equations*](https://arxiv.org/abs/1901.01543), arXiv:1901.01543, 2019.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
