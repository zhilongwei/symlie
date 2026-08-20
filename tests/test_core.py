"""Test suite for symlie.core using common ODEs and PDEs."""

import pytest
import sympy as sp

from symlie.core import (
    InfinitesimalGenerator,
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
    total_derivative,
    verify_generator,
)

# ==============================================================================
# Fixtures for Standard Coordinates, ODEs, and PDEs
# ==============================================================================


@pytest.fixture
def independent_symbols():
    """Standard space-time independent variables (x, t)."""
    x, t = sp.symbols("x t")
    return x, t


@pytest.fixture
def dependent_functions(independent_symbols):
    """Standard dependent field variables u(x, t) and v(x, t)."""
    x, t = independent_symbols
    u = sp.Function("u")(x, t)
    v = sp.Function("v")(x, t)
    return u, v


@pytest.fixture
def test_functions(independent_symbols):
    """Test / characteristic functions Q_u(x, t) and Q_v(x, t)."""
    x, t = independent_symbols
    Q_u = sp.Function("Q_u")(x, t)
    Q_v = sp.Function("Q_v")(x, t)
    return Q_u, Q_v


# ==============================================================================
# 1. Tests for max_derivative_order
# ==============================================================================


def test_max_derivative_order_pdes(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, v = dependent_functions

    # Heat equation: u_t = u_xx (order 2)
    heat_eq = u.diff(t) - u.diff(x, 2)
    assert max_derivative_order(heat_eq, u, (x, t)) == 2

    # KdV equation: u_t + u*u_x + u_xxx = 0 (order 3)
    kdv_eq = u.diff(t) + u * u.diff(x) + u.diff(x, 3)
    assert max_derivative_order(kdv_eq, u, (x, t)) == 3

    # Coupled system: u_t - v_x = 0, v_tt - u_xxxx = 0 (max order 4)
    sys = (u.diff(t) - v.diff(x), v.diff(t, 2) - u.diff(x, 4))
    assert max_derivative_order(sys, (u, v), (x, t)) == 4


def test_core_inputs_require_dependents_and_scalar_equations(independent_symbols):
    x, t = independent_symbols

    with pytest.raises(ValueError, match="Dependent variables cannot be empty"):
        max_derivative_order(0, (), (x, t))
    with pytest.raises(TypeError, match="scalar Sympy expressions"):
        max_derivative_order(sp.Matrix([x, t]), sp.Function("u"), (x, t))


# ==============================================================================
# 2. Tests for total_derivative
# ==============================================================================


def test_total_derivative():
    x = sp.symbols("x")
    u = sp.Function("u")(x)

    expr = u**2 + sp.sin(x)
    d1 = total_derivative(expr, x, order=1)
    assert d1 == 2 * u * u.diff(x) + sp.cos(x)

    d2 = total_derivative(expr, x, order=2)
    expected_d2 = 2 * (u.diff(x) ** 2 + u * u.diff(x, 2)) - sp.sin(x)
    assert sp.simplify(d2 - expected_d2) == 0


# ==============================================================================
# 3. Tests for frechet_derivative
# ==============================================================================


def test_frechet_derivative_heat(
    independent_symbols, dependent_functions, test_functions
):
    x, t = independent_symbols
    u, _ = dependent_functions
    Q_u, _ = test_functions

    heat_eq = u.diff(t) - u.diff(x, 2)
    res = frechet_derivative(heat_eq, u, (x, t), Q_u)

    expected = sp.ImmutableDenseMatrix([[Q_u.diff(t) - Q_u.diff(x, 2)]])
    assert res == expected


def test_frechet_derivative_kdv(
    independent_symbols, dependent_functions, test_functions
):
    x, t = independent_symbols
    u, _ = dependent_functions
    Q_u, _ = test_functions

    kdv_eq = u.diff(t) + u * u.diff(x) + u.diff(x, 3)
    res = frechet_derivative(kdv_eq, u, (x, t), Q_u)

    expected = sp.ImmutableDenseMatrix(
        [[Q_u.diff(t) + u * Q_u.diff(x) + u.diff(x) * Q_u + Q_u.diff(x, 3)]]
    )
    assert res == expected


def test_frechet_derivative_coupled_system(
    independent_symbols, dependent_functions, test_functions
):
    x, t = independent_symbols
    u, v = dependent_functions
    Q_u, Q_v = test_functions

    eq1 = u.diff(t) - v.diff(x)
    eq2 = v.diff(t) - u.diff(x)

    res = frechet_derivative([eq1, eq2], [u, v], [x, t], [Q_u, Q_v])
    expected = sp.ImmutableDenseMatrix(
        [
            [Q_u.diff(t), -Q_v.diff(x)],
            [-Q_u.diff(x), Q_v.diff(t)],
        ]
    )
    assert res == expected


# ==============================================================================
# 4. Tests for adjoint_frechet_derivative
# ==============================================================================


def test_adjoint_frechet_derivative_heat(
    independent_symbols, dependent_functions, test_functions
):
    x, t = independent_symbols
    u, _ = dependent_functions
    v, _ = test_functions  # dual test function

    heat_eq = u.diff(t) - u.diff(x, 2)
    # L = D_t - D_x^2 => L* = -D_t - D_x^2
    res = adjoint_frechet_derivative(heat_eq, u, (x, t), v)
    expected = (-v.diff(t) - v.diff(x, 2),)
    assert res == expected


def test_adjoint_frechet_derivative_kdv(
    independent_symbols, dependent_functions, test_functions
):
    x, t = independent_symbols
    u, _ = dependent_functions
    v, _ = test_functions

    kdv_eq = u.diff(t) + u * u.diff(x) + u.diff(x, 3)
    # D_KdV = D_t + u*D_x + u_x + D_xxx
    # D_KdV* = -D_t - D_x(u * .) + u_x - D_xxx = -D_t - u*D_x - D_xxx
    res = adjoint_frechet_derivative(kdv_eq, u, (x, t), v)
    expected = (-v.diff(t) - u * v.diff(x) - v.diff(x, 3),)
    assert res == expected


# ==============================================================================
# 5. Tests for euler_lagrange
# ==============================================================================


def test_euler_lagrange_harmonic_oscillator():
    t = sp.symbols("t")
    x = sp.Function("x")(t)
    m, k = sp.symbols("m k")

    # L = 1/2 m \dot{x}^2 - 1/2 k x^2
    L = sp.Rational(1, 2) * m * x.diff(t) ** 2 - sp.Rational(1, 2) * k * x**2
    el = euler_lagrange(L, x, t)

    # E_x(L) = \partial L / \partial x - d/dt(\partial L / \partial \dot{x}) = -k*x - m*\ddot{x}
    expected = (-k * x - m * x.diff(t, 2),)
    assert el == expected


def test_euler_lagrange_wave_equation(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, _ = dependent_functions
    c = sp.symbols("c")

    # Lagrangian density: L = 1/2 u_t^2 - c^2/2 u_x^2
    density = (
        sp.Rational(1, 2) * u.diff(t) ** 2 - sp.Rational(1, 2) * c**2 * u.diff(x) ** 2
    )
    el = euler_lagrange(density, u, (x, t))

    # E_u(L) = -D_t(u_t) - D_x(-c^2 u_x) = -u_tt + c^2 u_xx
    expected = (-u.diff(t, 2) + c**2 * u.diff(x, 2),)
    assert el == expected


# ==============================================================================
# 6. Tests for Hamiltonian Mechanics & Poisson Bracket
# ==============================================================================


def test_poisson_bracket_and_hamilton_equations():
    t = sp.symbols("t")
    m, k = sp.symbols("m k")
    q_fn = sp.Function("q")(t)
    p_fn = sp.Function("p")(t)

    # Harmonic oscillator Hamiltonian: H = p(t)^2/(2m) + 1/2 k q(t)^2
    H = p_fn**2 / (2 * m) + sp.Rational(1, 2) * k * q_fn**2

    # Canonical Poisson bracket {q, p} = 1
    assert poisson_bracket(q_fn, p_fn, q_fn, p_fn) == 1
    assert poisson_bracket(p_fn, q_fn, q_fn, p_fn) == -1
    assert poisson_bracket(q_fn, q_fn, q_fn, p_fn) == 0

    # Equations of motion via Poisson bracket: \dot{q} = {q, H} = p/m, \dot{p} = {p, H} = -kq
    assert poisson_bracket(q_fn, H, q_fn, p_fn) == p_fn / m
    assert poisson_bracket(p_fn, H, q_fn, p_fn) == -k * q_fn

    # Hamilton equations with time=None (returns RHS rate expressions)
    rates = hamilton_equations(H, q_fn, p_fn, time=None)
    assert rates == (p_fn / m, -k * q_fn)

    # Hamilton equations directly with time=t
    eqs = hamilton_equations(H, q_fn, p_fn, time=t)
    assert eqs == (
        sp.Eq(q_fn.diff(t), p_fn / m),
        sp.Eq(p_fn.diff(t), -k * q_fn),
    )

    # Validate that passing plain symbols when time is given raises TypeError
    q_sym, p_sym = sp.symbols("q p")
    H_sym = p_sym**2 / (2 * m) + sp.Rational(1, 2) * k * q_sym**2
    with pytest.raises(TypeError, match="plain Symbol"):
        hamilton_equations(H_sym, q_sym, p_sym, time=t)

    # Validate non-symbol time raises TypeError
    with pytest.raises(TypeError, match="Time variable must be a Sympy Symbol"):
        hamilton_equations(H, q_fn, p_fn, time="t")

    # Validate mismatched lengths raises ValueError
    with pytest.raises(ValueError, match="same length"):
        hamilton_equations(H, (q_fn,), (p_fn, p_fn))

    # Reject composite and repeated canonical variables before asking SymPy to
    # differentiate with respect to them.
    with pytest.raises(TypeError, match="not a valid Sympy differentiation"):
        hamilton_equations(H, q_fn + 1, p_fn, time=t)
    with pytest.raises(ValueError, match="unique"):
        poisson_bracket(q_fn, H, (q_fn, q_fn), (p_fn, sp.Function("r")(t)))
    with pytest.raises(TypeError, match="Hamiltonian must be a scalar"):
        hamilton_equations(sp.Matrix([H]), q_fn, p_fn)
    with pytest.raises(TypeError, match="operands must be scalar"):
        poisson_bracket(sp.Matrix([q_fn]), H, q_fn, p_fn)


# ==============================================================================
# 7. Tests for infer_substitution_rules & differential_substitute
# ==============================================================================


def test_differential_substitution_heat(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, _ = dependent_functions

    # Heat equation: u_t - u_xx = 0 => u_t = u_xx
    heat_eq = u.diff(t) - u.diff(x, 2)
    rules = infer_substitution_rules(heat_eq, u, (x, t))
    assert rules == {u.diff(t): u.diff(x, 2)}

    # Substitute u_tt => u_xxxx
    reduced = differential_substitute(u.diff(t, 2), rules, u, (x, t))
    assert reduced == u.diff(x, 4)


def test_differential_substitution_burgers(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, _ = dependent_functions

    # Inviscid Burgers equation: u_t + u*u_x = 0 => u_t = -u*u_x
    burgers_eq = u.diff(t) + u * u.diff(x)
    rules = infer_substitution_rules(burgers_eq, u, (x, t))
    assert rules == {u.diff(t): -u * u.diff(x)}

    # Substitute u_tt = D_t(-u*u_x) = -u_t*u_x - u*u_xt = 2*u*u_x^2 + u^2*u_xx
    reduced = differential_substitute(u.diff(t, 2), rules, u, (x, t))
    expected = 2 * u * u.diff(x) ** 2 + u**2 * u.diff(x, 2)
    assert sp.simplify(reduced - expected) == 0


def test_differential_substitution_iteration_boundary(
    independent_symbols, dependent_functions
):
    x, t = independent_symbols
    u, _ = dependent_functions
    rules = {u.diff(t): u.diff(x, 2)}

    # Sixteen updates exactly exhaust the default budget and must still return
    # the converged normal form.
    assert differential_substitute(u.diff(t, 16), rules, u, (x, t)) == u.diff(x, 32)

    with pytest.raises(ValueError, match="positive integer"):
        differential_substitute(u.diff(t), rules, u, (x, t), max_iterations=0)


def test_substitution_inference_reports_regular_parameter_branch(
    independent_symbols, dependent_functions
):
    x, t = independent_symbols
    u, _ = dependent_functions
    a = sp.symbols("a")

    rules, conditions = infer_substitution_rules(
        a * u.diff(t) - u.diff(x),
        u,
        (x, t),
        return_conditions=True,
    )

    assert rules == {u.diff(t): u.diff(x) / a}
    assert conditions == (sp.Ne(a, 0),)
    determining = determining_equations(a * u.diff(t) - u.diff(x), u, (x, t))
    assert determining.regularity_conditions == conditions


# ==============================================================================
# 8. Tests for Lie Symmetries: Prolongation, Determining Equations, Infinitesimals
# ==============================================================================


def test_lie_symmetries_heat_equation(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, _ = dependent_functions
    heat_eq = u.diff(t) - u.diff(x, 2)

    # 1. Determining equations
    det = determining_equations(heat_eq, u, (x, t))
    assert len(det.equations) > 0

    # 2. Polynomial infinitesimals (degree 1)
    sol = infinitesimals(heat_eq, u, (x, t), ansatz_degree=1)
    # The degree-1 polynomial ansatz contains six independent solutions; this is
    # not the dimension of the complete heat-equation point-symmetry algebra.
    assert sol.dimension == 6

    # 3. Verify that all computed basis generators satisfy the invariance condition
    for gen in sol.basis:
        assert verify_generator(heat_eq, u, (x, t), gen)


def test_lie_bracket_algebra(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, _ = dependent_functions

    # Space translation: X_1 = \partial_x
    gen_x = InfinitesimalGenerator((1, 0), (0,))
    # Scaling: X_3 = x \partial_x + 2t \partial_t
    gen_scale = InfinitesimalGenerator((x, 2 * t), (0,))

    # Commutator [X_1, X_3] = [d/dx, x d/dx + 2t d/dt] = d/dx = X_1
    bracket = lie_bracket(gen_x, gen_scale, u, (x, t))
    assert bracket.xi == (1, 0)
    assert bracket.phi == (0,)


def test_lie_symmetries_wave_equation(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, _ = dependent_functions
    wave_eq = u.diff(t, 2) - u.diff(x, 2)

    sol = infinitesimals(wave_eq, u, (x, t), ansatz_degree=1)
    assert sol.dimension == 8
    for gen in sol.basis:
        assert verify_generator(wave_eq, u, (x, t), gen)


def test_lie_symmetries_burgers_equation(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, _ = dependent_functions
    burgers_eq = u.diff(t) + u * u.diff(x) - u.diff(x, 2)

    sol = infinitesimals(burgers_eq, u, (x, t), ansatz_degree=1)
    assert sol.dimension == 4
    for gen in sol.basis:
        assert verify_generator(burgers_eq, u, (x, t), gen)


def test_lie_symmetries_porous_medium(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, _ = dependent_functions
    pme = u.diff(t) - u.diff(x) ** 2 - u * u.diff(x, 2)

    sol = infinitesimals(pme, u, (x, t), ansatz_degree=1)
    assert sol.dimension == 4
    for gen in sol.basis:
        assert verify_generator(pme, u, (x, t), gen)


def test_lie_symmetries_cubic_klein_gordon(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, _ = dependent_functions
    ckg = u.diff(t, 2) - u.diff(x, 2) + u**3

    sol = infinitesimals(ckg, u, (x, t), ansatz_degree=1)
    assert sol.dimension == 4
    for gen in sol.basis:
        assert verify_generator(ckg, u, (x, t), gen)


def test_lie_symmetries_coupled_system(independent_symbols, dependent_functions):
    x, t = independent_symbols
    u, v = dependent_functions
    eq1 = u.diff(t) - v.diff(x)
    eq2 = v.diff(t) - u.diff(x)

    sol = infinitesimals([eq1, eq2], [u, v], [x, t], ansatz_degree=1)
    assert sol.dimension == 12
    for gen in sol.basis:
        assert verify_generator([eq1, eq2], [u, v], [x, t], gen)


def test_coupled_system_reduction_is_invariant_under_equation_recombination(
    independent_symbols, dependent_functions
):
    x, t = independent_symbols
    u, v = dependent_functions
    eq1 = u.diff(t) - v.diff(x)
    eq2 = v.diff(t) - u.diff(x)
    recombined = (sp.expand(eq1 + eq2), sp.expand(eq1 - eq2))
    expected_rules = {
        u.diff(t): v.diff(x),
        v.diff(t): u.diff(x),
    }

    assert infer_substitution_rules(recombined, (u, v), (x, t)) == expected_rules
    det = determining_equations(recombined, (u, v), (x, t))
    sol = infinitesimals(recombined, (u, v), (x, t), ansatz_degree=1)

    assert det.substitution_rules == expected_rules
    assert sol.ansatz_dimension == 12
    for generator in sol.basis:
        assert verify_generator(
            recombined,
            (u, v),
            (x, t),
            generator,
            substitution_rules=expected_rules,
        )


def test_lie_symmetries_odes():
    t = sp.symbols("t")
    y = sp.Function("y")(t)

    # 1. Harmonic oscillator ODE: y'' + y = 0
    osc_ode = y.diff(t, 2) + y
    sol_osc = infinitesimals(osc_ode, y, t, ansatz_degree=1)
    assert sol_osc.dimension >= 2
    for gen in sol_osc.basis:
        assert verify_generator(osc_ode, y, t, gen)

    # 2. Scale-invariant ODE: y' - y/t = 0
    scale_ode = y.diff(t) - y / t
    sol_scale = infinitesimals(scale_ode, y, t, ansatz_degree=1)
    assert sol_scale.dimension == 4
    for gen in sol_scale.basis:
        assert verify_generator(scale_ode, y, t, gen)


# ==============================================================================
# 9. Tests from Güngör (arXiv:1901.01543)
# ==============================================================================


def test_laplace_and_conformal_symmetries():
    x, y = sp.symbols("x y")
    u = sp.Function("u")(x, y)
    laplace_eq = u.diff(x, 2) + u.diff(y, 2)

    # 1. Self-adjointness
    Q = sp.Function("Q")(x, y)
    D_lap = frechet_derivative(laplace_eq, u, (x, y), Q)
    D_star_lap = adjoint_frechet_derivative(laplace_eq, u, (x, y), Q)
    assert sp.simplify(D_lap[0, 0] - D_star_lap[0]) == 0

    # 2. Polynomial point symmetries (degree 1)
    sol_lap = infinitesimals(laplace_eq, u, (x, y), ansatz_degree=1)
    assert sol_lap.dimension == 8
    for gen in sol_lap.basis:
        assert verify_generator(laplace_eq, u, (x, y), gen)

    # 3. Special conformal generator c_x = (x^2 - y^2) d/dx + 2xy d/dy
    c_x = InfinitesimalGenerator(xi=(x**2 - y**2, 2 * x * y), phi=(0,))
    assert verify_generator(laplace_eq, u, (x, y), c_x)

    # 4. Elliptic Liouville equation: u_xx + u_yy = K*e^u
    K = sp.symbols("K")
    ell_liouv = u.diff(x, 2) + u.diff(y, 2) - K * sp.exp(u)
    v_conf_liouv = InfinitesimalGenerator(
        xi=(x**2 - y**2, 2 * x * y),
        phi=(-4 * x,),
    )
    assert verify_generator(ell_liouv, u, (x, y), v_conf_liouv)


def test_drift_diffusion_and_kolmogorov():
    x, t = sp.symbols("x t", positive=True)
    u = sp.Function("u")(x, t)

    # Drift-diffusion: u_t - x*u_xx - 1/2*u_x = 0
    drift_eq = u.diff(t) - x * u.diff(x, 2) - sp.Rational(1, 2) * u.diff(x)
    v1 = InfinitesimalGenerator(xi=(0, 1), phi=(0,))
    v4 = InfinitesimalGenerator(xi=(sp.sqrt(x), 0), phi=(0,))
    assert verify_generator(drift_eq, u, (x, t), v1)
    assert verify_generator(drift_eq, u, (x, t), v4)

    # Kolmogorov equation in 3 variables (t, x, y): u_t - u_xx + x*u_y = 0
    y_sym = sp.symbols("y")
    u_3d = sp.Function("u")(t, x, y_sym)
    kolm = u_3d.diff(t) - u_3d.diff(x, 2) + x * u_3d.diff(y_sym)
    assert max_derivative_order(kolm, u_3d, (t, x, y_sym)) == 2

    det_kolm = determining_equations(kolm, u_3d, (t, x, y_sym))
    assert len(det_kolm.equations) > 0


def test_ermakov_pinney_and_cubic_ode():
    t = sp.symbols("t")
    y = sp.Function("y")(t)
    K = sp.symbols("K", positive=True)

    # 1. Ermakov-Pinney equation: y'' = K / y^3
    ep_eq = y.diff(t, 2) - K / y**3
    v1 = InfinitesimalGenerator(xi=(1,), phi=(0,))
    v2 = InfinitesimalGenerator(xi=(t,), phi=(y / 2,))
    v3 = InfinitesimalGenerator(xi=(t**2,), phi=(t * y,))

    assert verify_generator(ep_eq, y, t, v1)
    assert verify_generator(ep_eq, y, t, v2)
    assert verify_generator(ep_eq, y, t, v3)

    # Commutators of sl(2, R)
    b12 = lie_bracket(v1, v2, y, t)
    b13 = lie_bracket(v1, v3, y, t)
    b23 = lie_bracket(v2, v3, y, t)
    assert b12.xi == (1,)
    assert b13.xi == (2 * t,) and b13.phi == (y,)
    assert b23.xi == (t**2,) and b23.phi == (t * y,)

    # 2. Cubic nonlinear ODE: y'' + 3yy' + y^3 = 0
    cubic_ode = y.diff(t, 2) + 3 * y * y.diff(t) + y**3
    sol_cubic = infinitesimals(cubic_ode, y, t, ansatz_degree=1)
    assert sol_cubic.dimension == 2
    for gen in sol_cubic.basis:
        assert verify_generator(cubic_ode, y, t, gen)


def test_epd_and_fiber_preserving_ode():
    # 1. EPD equation: u_tt + b/t*u_t - u_xx = 0
    t, x = sp.symbols("t x", positive=True)
    b = sp.symbols("b")
    u = sp.Function("u")(x, t)
    epd_eq = u.diff(t, 2) + (b / t) * u.diff(t) - u.diff(x, 2)

    v1 = InfinitesimalGenerator(xi=(1, 0), phi=(0,))
    v2 = InfinitesimalGenerator(xi=(x, t), phi=(0,))
    v3 = InfinitesimalGenerator(xi=(0, 0), phi=(u,))
    assert verify_generator(epd_eq, u, (x, t), v1)
    assert verify_generator(epd_eq, u, (x, t), v2)
    assert verify_generator(epd_eq, u, (x, t), v3)

    # 2. Fiber-preserving ODE: y'' = y'^2/y - y'
    t_sym = sp.symbols("t")
    y_fn = sp.Function("y")(t_sym)
    fiber_ode = y_fn.diff(t_sym, 2) - y_fn.diff(t_sym) ** 2 / y_fn + y_fn.diff(t_sym)
    sol_fiber = infinitesimals(fiber_ode, y_fn, t_sym, ansatz_degree=1)
    assert sol_fiber.dimension == 2
    for gen in sol_fiber.basis:
        assert verify_generator(fiber_ode, y_fn, t_sym, gen)


def test_public_api_excludes_typo_alias():
    import symlie

    assert "poisson_bracket" in symlie.__all__
    assert "possion_bracket" not in symlie.__all__
    assert not hasattr(symlie, "possion_bracket")
