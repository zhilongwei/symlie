"""symlie: Symbolic Lie Symmetry Analysis on Jet Spaces with SymPy."""

from symlie.core import (
    DeterminingSystem,
    InfinitesimalGenerator,
    InfinitesimalSolution,
    adjoint_frechet_derivative,
    determining_equations,
    differential_substitute,
    euler_lagrange,
    frechet_derivative,
    hamilton_equations,
    infer_substitution_rules,
    infinitesimals,
    lie_bracket,
    max_derivative_order,
    poisson_bracket,
    prolongation,
    total_derivative,
    verify_generator,
)

__version__ = "0.1.0"

__all__ = [
    "DeterminingSystem",
    "InfinitesimalGenerator",
    "InfinitesimalSolution",
    "__version__",
    "adjoint_frechet_derivative",
    "determining_equations",
    "differential_substitute",
    "euler_lagrange",
    "frechet_derivative",
    "hamilton_equations",
    "infer_substitution_rules",
    "infinitesimals",
    "lie_bracket",
    "max_derivative_order",
    "poisson_bracket",
    "prolongation",
    "total_derivative",
    "verify_generator",
]
