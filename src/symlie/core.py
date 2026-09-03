from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import overload

import sympy as sp
from sympy.core.function import AppliedUndef, FunctionClass


def _as_tuple(value):
    if isinstance(value, (list, tuple, sp.Tuple)):
        return tuple(value)
    return (value,)


def _is_unique(items) -> bool:
    return len(items) == len(set(items))


def _ensure_unique(items, label: str) -> None:
    if not _is_unique(items):
        raise ValueError(f"{label} must be unique.")


def _normalize_independents(independent_variables: Sequence[sp.Symbol]):
    independents = _as_tuple(independent_variables)
    if not independents:
        raise ValueError("Independent variables cannot be empty.")
    if not all(isinstance(var, sp.Symbol) for var in independents):
        raise TypeError("Independent variables must be Sympy symbols.")
    _ensure_unique(independents, "Independent variables")
    return independents


def _normalize_dependents(
    dependent_variables: Sequence[sp.Basic], independent_variables: Sequence[sp.Symbol]
):
    independents = _normalize_independents(independent_variables)
    dependents = []

    raw_dependents = _as_tuple(dependent_variables)
    if not raw_dependents:
        raise ValueError("Dependent variables cannot be empty.")

    for dependent in raw_dependents:
        if isinstance(dependent, AppliedUndef):
            applied = dependent
        elif isinstance(dependent, FunctionClass) or callable(dependent):
            applied = dependent(*independents)
        else:
            raise TypeError(
                "Dependent variable must contain undefined Sympy functions or applied undefined functions."
            )

        if not isinstance(applied, AppliedUndef):
            raise TypeError("Dependent variable must be undefined Sympy functions.")
        dependents.append(applied)

    _ensure_unique(dependents, "Dependent variables")

    return tuple(dependents)


def _normalize_equations(equations) -> tuple[sp.Expr, ...]:
    expressions = []

    for equation in _as_tuple(equations):
        if isinstance(equation, sp.Equality):
            expression = equation.lhs - equation.rhs
        else:
            expression = sp.sympify(equation)
        if isinstance(expression, sp.MatrixBase) or not isinstance(expression, sp.Expr):
            raise TypeError("Equations must be scalar Sympy expressions or equalities.")
        expressions.append(sp.simplify(expression))

    if not expressions:
        raise ValueError("Equations cannot be empty.")

    return tuple(expressions)


def _normalize_expressions(values, expected_length: int, label: str):
    expressions = tuple(sp.simplify(value) for value in _as_tuple(values))

    if len(expressions) != expected_length:
        raise ValueError(f"{label} must contain {expected_length} expressions.")

    return expressions


def _multi_index(jet, dependent, independents) -> tuple[int, ...] | None:
    if jet == dependent:
        return (0,) * len(independents)

    if not isinstance(jet, sp.Derivative) or jet.expr != dependent:
        return None

    counts = dict(jet.variable_count)
    if any(variable not in independents for variable in counts):
        return None

    return tuple(int(counts.get(variable, 0)) for variable in independents)


def _derivative_from_multi_index(expression, multi_index, independents):
    diff_specs = [
        (variable, count)
        for variable, count in zip(independents, multi_index)
        if count > 0
    ]

    if not diff_specs:
        return expression

    return sp.diff(expression, *diff_specs)


def _dependent_jets(expression, dependents, independents, include_zeros=True):
    jets: dict[tuple[int, tuple[int, ...]], sp.Expr] = {}

    if include_zeros:
        for index, dependent in enumerate(dependents):
            if expression.has(dependent):
                jets[(index, (0,) * len(independents))] = dependent

    for derivative in expression.atoms(sp.Derivative):
        for index, dependent in enumerate(dependents):
            multi_index = _multi_index(derivative, dependent, independents)
            if multi_index is not None:
                jets[(index, multi_index)] = derivative
                break

    return jets


def max_derivative_order(equations, dependent_variables, independent_variables) -> int:
    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    expressions = _normalize_equations(equations)

    max_order = 0
    for expr in expressions:
        for _, multi_index in _dependent_jets(expr, dependents, independents):
            max_order = max(max_order, sum(multi_index))

    return max_order


def total_derivative(
    expression: sp.Basic, independent_variable: sp.Symbol, order: int = 1
) -> sp.Expr:
    """Compute the total derivative of ``expression`` on the jet space.

    Differentiates ``expression`` with respect to the independent variable
    ``independent_variable``, applying the chain rule through applied (unknown)
    functions while holding other independent variables fixed.  For example,
    with ``u = sp.Function("u")(x)``::

        total_derivative(u**2, x)    == 2*u*u.diff(x)
        total_derivative(u**2, x, 2) == 2*u.diff(x)**2 + 2*u*u.diff(x, 2)

    This matches the jet-space total derivative operator ``D_i``: because
    SymPy's ``diff`` already differentiates through applied functions of the
    variable, differentiating with respect to an independent coordinate while
    holding the others fixed yields exactly ``D_i`` for expressions built from
    functions of the independent coordinates.

    Parameters
    ----------
    expression : sp.Basic
        Expression to differentiate.
    independent_variable : sp.Symbol
        Independent variable with respect to which the derivative is taken.
    order : int, optional
        Number of times the derivative is applied (default 1).

    Returns
    -------
    sp.Expr
        The ``order``-th total derivative of ``expression``.

    Raises
    ------
    TypeError
        If ``independent_variable`` is not a SymPy symbol.
    ValueError
        If ``order`` is not a non-negative integer.
    """
    if not isinstance(independent_variable, sp.Symbol):
        raise TypeError("Independent variable must be a Sympy symbol.")

    if not isinstance(order, int) or order < 0:
        raise ValueError("Order of derivative must be a non-negative integer.")

    return sp.diff(sp.simplify(expression), independent_variable, order)


def _jets_for_dependent(jets, dependent_index):
    """Yield ``(multi_index, jet)`` pairs belonging to ``dependent_index``."""
    return (
        (multi_index, jet)
        for (index, multi_index), jet in jets.items()
        if index == dependent_index
    )


def _frechet_derivative_rows(expressions, dependents, independents, tests):
    rows = []
    for expr in expressions:
        jets = _dependent_jets(expr, dependents, independents)
        row = []
        for dependent_index in range(len(dependents)):
            component = sum(
                sp.diff(expr, jet)
                * _derivative_from_multi_index(
                    tests[dependent_index], multi_index, independents
                )
                for multi_index, jet in _jets_for_dependent(jets, dependent_index)
            )
            row.append(sp.expand(component))
        rows.append(row)
    return rows


def frechet_derivative(
    equations, dependent_variables, independent_variables, test_functions
):
    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    expressions = _normalize_equations(equations)
    tests = _normalize_expressions(test_functions, len(dependents), "Test functions")

    rows = _frechet_derivative_rows(expressions, dependents, independents, tests)
    return sp.ImmutableDenseMatrix(rows)


def adjoint_frechet_derivative(
    equations, dependent_variables, independent_variables, test_functions
) -> tuple[sp.Expr, ...]:
    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    expressions = _normalize_equations(equations)
    tests = _normalize_expressions(test_functions, len(expressions), "Test functions")
    jets_by_expression = tuple(
        _dependent_jets(expression, dependents, independents)
        for expression in expressions
    )

    results = []
    for dependent_index in range(len(dependents)):
        component = sum(
            (-1) ** sum(multi_index)
            * _derivative_from_multi_index(
                sp.diff(expr, jet) * test, multi_index, independents
            )
            for expr, test, jets in zip(expressions, tests, jets_by_expression)
            for multi_index, jet in _jets_for_dependent(jets, dependent_index)
        )
        results.append(sp.expand(component))

    return tuple(results)


def euler_lagrange(density, dependent_variables, independent_variables):
    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    density = sp.simplify(density)
    jets = _dependent_jets(density, dependents, independents)

    results = []
    for dependent_index in range(len(dependents)):
        component = sum(
            (-1) ** sum(multi_index)
            * _derivative_from_multi_index(
                sp.diff(density, jet), multi_index, independents
            )
            for multi_index, jet in _jets_for_dependent(jets, dependent_index)
        )
        results.append(sp.expand(component))

    return tuple(results)


def poisson_bracket(first, second, coordinates, momenta):
    coordinates, momenta = _normalize_canonical_variables(coordinates, momenta)
    first = sp.sympify(first)
    second = sp.sympify(second)
    if any(
        isinstance(expression, sp.MatrixBase) or not isinstance(expression, sp.Expr)
        for expression in (first, second)
    ):
        raise TypeError("Poisson-bracket operands must be scalar Sympy expressions.")

    return sp.expand(
        sum(
            sp.diff(first, coordinate) * sp.diff(second, momentum)
            - sp.diff(first, momentum) * sp.diff(second, coordinate)
            for coordinate, momentum in zip(coordinates, momenta)
        )
    )


@overload
def hamilton_equations(
    hamiltonian: sp.Expr,
    coordinates: sp.Basic | Sequence[sp.Basic],
    momenta: sp.Basic | Sequence[sp.Basic],
    time: None = None,
) -> tuple[sp.Expr, ...]: ...


@overload
def hamilton_equations(
    hamiltonian: sp.Expr,
    coordinates: sp.Basic | Sequence[sp.Basic],
    momenta: sp.Basic | Sequence[sp.Basic],
    time: sp.Symbol,
) -> tuple[sp.Equality, ...]: ...


def hamilton_equations(
    hamiltonian: sp.Expr,
    coordinates: sp.Basic | Sequence[sp.Basic],
    momenta: sp.Basic | Sequence[sp.Basic],
    time: sp.Symbol | None = None,
) -> tuple[sp.Expr, ...] | tuple[sp.Equality, ...]:
    """Compute Hamilton's equations of motion from a Hamiltonian.

    Parameters
    ----------
    hamiltonian : sp.Expr
        Hamiltonian scalar function H(q, p, t).
    coordinates : sp.Basic or Sequence[sp.Basic]
        Generalized coordinate(s) q.
    momenta : sp.Basic or Sequence[sp.Basic]
        Canonical conjugate momenta p.
    time : sp.Symbol, optional
        Time variable t. When provided, coordinates and momenta must be functions
        of time (e.g. AppliedUndef instances like q(t), p(t)) so that explicit
        derivatives dq/dt and dp/dt are well-defined. In this case, returns a tuple
        of equality objects (sp.Eq(Derivative(q, t), dH/dp), ...).
        When None, returns a tuple of velocity and momentum rate expressions (dH/dp, -dH/dq).

    Returns
    -------
    tuple[sp.Expr, ...] | tuple[sp.Equality, ...]
        Equations of motion as RHS expressions (when time is None) or Eq relations (when time is provided).

    Raises
    ------
    ValueError
        If coordinates and momenta have different lengths.
    TypeError
        If time is specified but is not a SymPy Symbol, or if coordinates/momenta
        are plain Symbols rather than functions of time.
    """
    coordinates, momenta = _normalize_canonical_variables(
        coordinates, momenta, time=time
    )
    hamiltonian = sp.sympify(hamiltonian)
    if isinstance(hamiltonian, sp.MatrixBase) or not isinstance(hamiltonian, sp.Expr):
        raise TypeError("Hamiltonian must be a scalar Sympy expression.")
    q_rhs = tuple(sp.diff(hamiltonian, momentum) for momentum in momenta)
    p_rhs = tuple(-sp.diff(hamiltonian, coordinate) for coordinate in coordinates)

    if time is None:
        return q_rhs + p_rhs

    return tuple(
        sp.Eq(sp.diff(variable, time), rhs)
        for variable, rhs in zip(coordinates + momenta, q_rhs + p_rhs)
    )


def _normalize_canonical_variables(coordinates, momenta, *, time=None):
    coordinates = _as_tuple(coordinates)
    momenta = _as_tuple(momenta)

    if not coordinates or not momenta:
        raise ValueError("Coordinates and momenta cannot be empty.")
    if len(coordinates) != len(momenta):
        raise ValueError("Coordinates and momenta must have the same length.")
    if not _is_unique(coordinates) or not _is_unique(momenta):
        raise ValueError("Coordinates and momenta must each be unique.")
    if set(coordinates) & set(momenta):
        raise ValueError("Coordinates and momenta must be distinct.")

    if time is not None and not isinstance(time, sp.Symbol):
        raise TypeError("Time variable must be a Sympy Symbol.")

    for variable in coordinates + momenta:
        if not isinstance(variable, sp.Basic) or not variable._diff_wrt:
            raise TypeError(
                f"Canonical variable '{variable}' is not a valid Sympy differentiation variable."
            )
        if time is None:
            continue
        if isinstance(variable, sp.Symbol):
            raise TypeError(
                f"Variable '{variable}' is a plain Symbol. When 'time' is specified, "
                f"coordinates and momenta must be undefined functions of time '{time}'."
            )
        if not isinstance(variable, AppliedUndef) or time not in variable.args:
            raise TypeError(
                f"Variable '{variable}' must be an undefined function of time '{time}'."
            )

    return coordinates, momenta


def _ordered_derivative_candidates(expressions, dependents, independents):
    candidates = {}
    for expression in expressions:
        jets = _dependent_jets(expression, dependents, independents, include_zeros=False)
        for (dependent_index, multi_index), derivative in jets.items():
            candidates[derivative] = (dependent_index, multi_index)

    def candidate_key(item):
        derivative, (dependent_index, multi_index) = item
        last_order = multi_index[-1]
        pure_last_derivative = last_order > 0 and sum(multi_index) == last_order
        return (
            -int(pure_last_derivative),
            -last_order,
            -sum(multi_index),
            dependent_index,
            multi_index,
            sp.default_sort_key(derivative),
        )

    return tuple(
        derivative for derivative, _ in sorted(candidates.items(), key=candidate_key)
    )


def _append_condition(conditions, expression):
    expression = sp.factor(sp.simplify(expression))
    if expression.is_nonzero is True:
        return
    condition = sp.Ne(expression, 0, evaluate=False)
    if condition not in conditions:
        conditions.append(condition)


def _solve_substitution_system(expressions, leaders):
    try:
        solutions = sp.solve(
            expressions,
            leaders,
            dict=True,
            simplify=True,
        )
    except (NotImplementedError, ValueError, TypeError):
        return None

    if len(solutions) != 1 or any(leader not in solutions[0] for leader in leaders):
        return None

    # solve(..., simplify=True) already simplifies each solution; re-simplifying
    # here would just repeat that work.
    rules = {leader: solutions[0][leader] for leader in leaders}
    if any(right_hand_side.has(*leaders) for right_hand_side in rules.values()):
        return None
    if any(sp.simplify(expression.xreplace(rules)) != 0 for expression in expressions):
        return None

    conditions = []
    for right_hand_side in rules.values():
        denominator = sp.denom(sp.together(right_hand_side))
        if denominator != 1:
            _append_condition(conditions, denominator)

    # A regular, acyclic reduction requires the leaders to be locally isolated,
    # regular roots of the (possibly overdetermined) system: the Jacobian
    # restricted to the leader columns must have full column rank at the
    # solution. Rank is witnessed by at least one non-vanishing square minor;
    # search all len(leaders)-row combinations for one that is nonzero there.
    jacobian = sp.Matrix(expressions).jacobian(leaders)
    regular = False
    for row_indices in combinations(range(len(expressions)), len(leaders)):
        minor = jacobian[list(row_indices), :]
        regularity_factor = sp.simplify(minor.det().xreplace(rules))
        if regularity_factor != 0:
            _append_condition(conditions, regularity_factor)
            regular = True
            break
    if not regular:
        return None

    return rules, tuple(conditions)


def _infer_substitution_rules_core(
    expressions, dependents, independents, *, leading_derivatives=None
):
    """Core of :func:`infer_substitution_rules`, assuming already-normalized inputs.

    Always returns ``(rules, conditions)``.
    """
    candidates = _ordered_derivative_candidates(expressions, dependents, independents)

    if leading_derivatives is not None:
        leaders = _as_tuple(leading_derivatives)
        if not leaders or not _is_unique(leaders):
            raise ValueError("Leading derivatives must be non-empty and unique.")
        if any(leader not in candidates for leader in leaders):
            raise ValueError(
                "Every leading derivative must occur in the equations as a derivative "
                "of a dependent variable."
            )
        solved = _solve_substitution_system(expressions, leaders)
        if solved is None:
            raise ValueError(
                "The equations do not define a unique regular, acyclic reduction "
                "for the requested leading derivatives."
            )
        return solved

    solved = None
    maximum_leaders = min(len(expressions), len(candidates))
    trial_count = 0
    for leader_count in range(maximum_leaders, 0, -1):
        for leaders in combinations(candidates, leader_count):
            solved = _solve_substitution_system(expressions, leaders)
            trial_count += 1
            if solved is not None or trial_count >= 64:
                break
        if solved is not None or trial_count >= 64:
            break
    if solved is None and trial_count >= 64:
        raise ValueError(
            "Automatic leader selection exceeded its search limit; provide "
            "leading_derivatives explicitly."
        )
    if solved is None:
        solved = ({}, ())
    return solved


def infer_substitution_rules(
    equations,
    dependent_variables,
    independent_variables,
    *,
    leading_derivatives=None,
    return_conditions: bool = False,
) -> (
    dict[sp.Derivative, sp.Expr]
    | tuple[dict[sp.Derivative, sp.Expr], tuple[sp.Basic, ...]]
):
    """Infer an orthonomic differential reduction on a regular equation branch.

    Equations are solved simultaneously for distinct leading derivatives.  When
    ``leading_derivatives`` is omitted, a deterministic ranking prefers pure
    derivatives in the last independent variable, which conventionally serves
    as the evolution variable.  The inferred identities are valid only where
    the returned regularity conditions are nonzero.

    Set ``return_conditions=True`` to receive ``(rules, conditions)``.  If the
    automatic ranking is unsuitable, provide explicit leading derivatives or
    pass explicit substitution rules to the downstream symmetry functions.
    """
    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    expressions = _normalize_equations(equations)
    solved = _infer_substitution_rules_core(
        expressions, dependents, independents, leading_derivatives=leading_derivatives
    )
    return solved if return_conditions else solved[0]


def _rule_multi_index(leading_derivative, dependents, independents):
    for index, dependent in enumerate(dependents):
        multi_index = _multi_index(leading_derivative, dependent, independents)
        if multi_index is not None:
            return index, multi_index
    return None


def differential_substitute(
    expression,
    substitution_rules: Mapping[sp.Derivative, sp.Expr],
    dependent_variables,
    independent_variables,
    *,
    max_iterations: int = 16,
):
    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer.")
    parsed_rules = []

    for leading, right_hand_side in substitution_rules.items():
        parsed = _rule_multi_index(leading, dependents, independents)
        if parsed is None:
            raise ValueError(
                f"Leading derivative {leading} is not a valid dependent variable derivative."
            )
        parsed_rules.append(
            (parsed[0], parsed[1], leading, sp.simplify(right_hand_side))
        )
    # Break ties between equal-order rules by rule content (dependent index,
    # multi-index, then a canonical Sympy sort key) rather than by the
    # incidental iteration order of the input mapping, so the same rule set
    # always resolves the same way regardless of dict insertion order.
    parsed_rules.sort(
        key=lambda item: (
            -sum(item[1]),
            item[0],
            item[1],
            sp.default_sort_key(item[2]),
        )
    )

    def find_replacements(current):
        replacements = {}
        candidates = set(current.atoms(sp.Derivative))
        candidates.update(dependent for dependent in dependents if current.has(dependent))
        derivatives = sorted(
            candidates,
            key=lambda item: (
                -sum(count for _, count in item.variable_count)
                if isinstance(item, sp.Derivative)
                else 0,
                sp.default_sort_key(item),
            ),
        )

        for derivative in derivatives:
            for dependent_index, dependent in enumerate(dependents):
                target_index = _multi_index(derivative, dependent, independents)
                if target_index is None:
                    continue
                for rule_dependent, rule_index, _, right_hand_side in parsed_rules:
                    if rule_dependent != dependent_index:
                        continue
                    if all(
                        target >= lead for target, lead in zip(target_index, rule_index)
                    ):
                        remainder = tuple(
                            target - lead
                            for target, lead in zip(target_index, rule_index)
                        )
                        replacements[derivative] = _derivative_from_multi_index(
                            right_hand_side, remainder, independents
                        )
                        break
                if derivative in replacements:
                    break
        return replacements

    result = sp.expand(sp.simplify(expression))
    seen = set()
    for _ in range(max_iterations):
        signature = sp.srepr(result)
        if signature in seen:
            raise RuntimeError("Differential substitution entered a cycle.")

        seen.add(signature)
        replacements = find_replacements(result)

        if not replacements:
            return result
        updated = sp.expand(result.xreplace(replacements))
        if updated == result:
            return result
        result = updated

    if not find_replacements(result):
        return result
    raise RuntimeError(
        "Differential substitution did not converge within the maximum number of iterations."
    )


def prolongation(
    equations,
    dependent_variables,
    independent_variables,
    xi,
    phi,
    *,
    substitution_rules: Mapping[sp.Derivative, sp.Expr] | None = None,
) -> tuple[sp.Expr, ...]:
    """Prolong a point vector field and apply it to differential equations.

    ``xi`` contains the infinitesimals of the independent variables and
    ``phi`` those of the dependent variables.  Returned expressions are the
    invariance residuals ``pr X(F)`` restricted to the equation manifold.
    """

    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    expressions = _normalize_equations(equations)
    xi = _normalize_expressions(xi, len(independents), "xi")
    phi = _normalize_expressions(phi, len(dependents), "phi")
    characteristics = tuple(
        phi_component
        - sum(
            xi_component * sp.diff(dependent, variable)
            for xi_component, variable in zip(xi, independents)
        )
        for dependent, phi_component in zip(dependents, phi)
    )
    # expressions/dependents/independents are already normalized above, so the
    # *_core helpers are used directly here to avoid re-running _normalize_equations
    # (and its sp.simplify pass) a second time inside frechet_derivative /
    # infer_substitution_rules.
    frechet_rows = _frechet_derivative_rows(
        expressions, dependents, independents, characteristics
    )
    residuals = []
    for row_index, expression in enumerate(expressions):
        residual = sum(frechet_rows[row_index])
        residual += sum(
            xi_component * sp.diff(expression, variable)
            for xi_component, variable in zip(xi, independents)
        )
        residuals.append(sp.expand(residual))

    if substitution_rules is None:
        substitution_rules = _infer_substitution_rules_core(
            expressions, dependents, independents
        )[0]
    if substitution_rules:
        residuals = [
            differential_substitute(
                residual,
                substitution_rules,
                dependents,
                independents,
            )
            for residual in residuals
        ]
    return tuple(sp.expand(residual) for residual in residuals)


def _normalize_zero_expression(expression, conditions=None):
    expression = sp.factor_terms(sp.cancel(sp.sympify(expression).doit()))
    numerator, denominator = sp.fraction(expression)
    if conditions is not None and denominator != 1:
        _append_condition(conditions, denominator)
    _, primitive = sp.sympify(numerator).as_content_primitive()
    primitive = sp.factor_terms(primitive)
    if primitive.could_extract_minus_sign():
        primitive = -primitive
    return primitive


def _append_unique(expressions, candidate, conditions=None):
    candidate = _normalize_zero_expression(candidate, conditions)
    if candidate == 0:
        return
    for existing in expressions:
        if candidate == existing or sp.simplify(candidate - existing) == 0:
            return
    expressions.append(candidate)


def _jet_replacement_map(expression, dependents, independents):
    jets = _dependent_jets(expression, dependents, independents, include_zeros=False)
    ordered_jets = sorted(jets.values(), key=sp.default_sort_key)
    jet_symbols = sp.symbols(f"J0:{len(ordered_jets)}")
    dependent_symbols = sp.symbols(f"U0:{len(dependents)}")
    mapping = dict(zip(ordered_jets, jet_symbols))
    mapping.update(zip(dependents, dependent_symbols))
    return mapping, tuple(jet_symbols), tuple(dependent_symbols)


def _split_on_jets(expression, dependents, independents, conditions=None):
    mapping, jet_symbols, dependent_symbols = _jet_replacement_map(
        expression, dependents, independents
    )
    algebraic = sp.expand(expression.xreplace(mapping))
    numerator, denominator = sp.together(algebraic).as_numer_denom()
    if conditions is not None and denominator != 1:
        _append_condition(conditions, denominator)
    if not jet_symbols:
        return [_normalize_zero_expression(numerator, conditions)], dependent_symbols
    try:
        polynomial = sp.Poly(sp.expand(numerator), *jet_symbols)
    except sp.PolynomialError as error:
        raise ValueError(
            "the invariance condition is not polynomial in the unconstrained jets; "
            "provide different substitution_rules"
        ) from error
    return [coefficient for _, coefficient in polynomial.terms()], dependent_symbols


@dataclass(frozen=True)
class InfinitesimalGenerator:
    """Coefficients of a point-symmetry vector field."""

    xi: tuple[sp.Expr, ...]
    phi: tuple[sp.Expr, ...]

    def __post_init__(self):
        object.__setattr__(self, "xi", tuple(sp.sympify(value) for value in self.xi))
        object.__setattr__(self, "phi", tuple(sp.sympify(value) for value in self.phi))

    def scaled(self, factor):
        return InfinitesimalGenerator(
            tuple(sp.expand(factor * value) for value in self.xi),
            tuple(sp.expand(factor * value) for value in self.phi),
        )

    @property
    def components(self):
        return self.xi + self.phi


@dataclass(frozen=True)
class DeterminingSystem:
    """A system of determining PDEs for unknown infinitesimals."""

    equations: tuple[sp.Equality, ...]
    residuals: tuple[sp.Expr, ...]
    xi: tuple[sp.Expr, ...]
    phi: tuple[sp.Expr, ...]
    substitution_rules: Mapping[sp.Derivative, sp.Expr]
    dependent_symbols: tuple[sp.Symbol, ...]
    regularity_conditions: tuple[sp.Basic, ...] = ()


@dataclass(frozen=True)
class InfinitesimalSolution:
    """Finite-dimensional polynomial point symmetries of an equation system."""

    basis: tuple[InfinitesimalGenerator, ...]
    general: InfinitesimalGenerator
    constants: tuple[sp.Symbol, ...]
    ansatz_degree: int
    determining_equations: tuple[sp.Expr, ...]
    regularity_conditions: tuple[sp.Basic, ...] = ()

    @property
    def dimension(self):
        """Compatibility alias for :attr:`ansatz_dimension`."""
        return len(self.basis)

    @property
    def ansatz_dimension(self):
        """Dimension found within the configured polynomial ansatz."""
        return len(self.basis)


def determining_equations(
    equations,
    dependent_variables,
    independent_variables,
    *,
    substitution_rules: Mapping[sp.Derivative, sp.Expr] | None = None,
) -> DeterminingSystem:
    """Construct the point-symmetry determining equations.

    This is the direct counterpart of MathLie's ``DeterminingEquations``.
    The returned equations are linear PDEs for functions named ``xi1``, ...,
    ``phi1``, ... .
    """

    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    expressions = _normalize_equations(equations)
    arguments = independents + dependents
    xi = tuple(
        sp.Function(f"xi{index + 1}")(*arguments) for index in range(len(independents))
    )
    phi = tuple(
        sp.Function(f"phi{index + 1}")(*arguments) for index in range(len(dependents))
    )
    conditions = []
    if substitution_rules is None:
        substitution_rules, inferred_conditions = _infer_substitution_rules_core(
            expressions, dependents, independents
        )
        conditions.extend(inferred_conditions)
    residuals = prolongation(
        expressions,
        dependents,
        independents,
        xi,
        phi,
        substitution_rules=substitution_rules,
    )
    coefficients = []
    dependent_symbols = tuple(sp.symbols(f"U0:{len(dependents)}"))
    for residual in residuals:
        split, current_dependent_symbols = _split_on_jets(
            residual, dependents, independents, conditions
        )
        dependent_symbols = current_dependent_symbols
        for coefficient in split:
            _append_unique(coefficients, coefficient, conditions)
    return DeterminingSystem(
        equations=tuple(sp.Eq(coefficient, 0) for coefficient in coefficients),
        residuals=residuals,
        xi=xi,
        phi=phi,
        substitution_rules=dict(substitution_rules),
        dependent_symbols=dependent_symbols,
        regularity_conditions=tuple(conditions),
    )


def _monomials(variables, degree):
    if degree < 0:
        raise ValueError("ansatz_degree must be non-negative")
    monomials = sp.polys.monomials.itermonomials(variables, degree)
    return tuple(
        sorted(
            monomials,
            key=lambda item: (
                sp.Poly(item, *variables).total_degree(),
                sp.default_sort_key(item),
            ),
        )
    )


def _normalize_null_vector(vector):
    vector = tuple(sp.simplify(value) for value in vector)
    first = next((value for value in vector if value != 0), None)
    if first is None:
        return vector
    return tuple(sp.simplify(value / first) for value in vector)


def infinitesimals(
    equations,
    dependent_variables,
    independent_variables,
    *,
    ansatz_degree: int = 1,
    substitution_rules: Mapping[sp.Derivative, sp.Expr] | None = None,
    constant_prefix: str = "k",
) -> InfinitesimalSolution:
    """Find polynomial point infinitesimals by exact linear algebra.

    The legacy package used a specialised overdetermined PDE solver.  This
    Python API makes the finite ansatz explicit: all infinitesimals are
    polynomials of total degree at most ``ansatz_degree`` in the independent
    and dependent variables.  The resulting determining system is solved
    exactly, and a basis plus a general linear combination are returned.

    This is the solution space inside the selected ansatz, not generally the
    complete point-symmetry algebra.  In particular, linear equations have an
    infinite solution-superposition ideal.  For degrees above one, the finite
    ansatz space is also not automatically closed under Lie brackets.
    """

    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    expressions = _normalize_equations(equations)
    conditions = []
    if substitution_rules is None:
        substitution_rules, inferred_conditions = _infer_substitution_rules_core(
            expressions, dependents, independents
        )
        conditions.extend(inferred_conditions)

    dependent_symbols = tuple(sp.symbols(f"U0:{len(dependents)}"))
    algebraic_variables = independents + dependent_symbols
    monomials = _monomials(algebraic_variables, ansatz_degree)
    coefficient_groups = []
    ansatz_components = []
    component_count = len(independents) + len(dependents)
    for component_index in range(component_count):
        coefficients = sp.symbols(f"a{component_index + 1}_0:{len(monomials)}")
        coefficient_groups.append(tuple(coefficients))
        polynomial = sum(
            coefficient * monomial
            for coefficient, monomial in zip(coefficients, monomials)
        )
        ansatz_components.append(
            polynomial.xreplace(dict(zip(dependent_symbols, dependents)))
        )
    xi = tuple(ansatz_components[: len(independents)])
    phi = tuple(ansatz_components[len(independents) :])
    residuals = prolongation(
        expressions,
        dependents,
        independents,
        xi,
        phi,
        substitution_rules=substitution_rules,
    )

    determining = []
    for residual in residuals:
        mapping, jet_symbols, current_dependent_symbols = _jet_replacement_map(
            residual, dependents, independents
        )
        algebraic = sp.expand(residual.xreplace(mapping))
        numerator, denominator = sp.together(algebraic).as_numer_denom()
        if denominator != 1:
            _append_condition(conditions, denominator)
        generators = independents + current_dependent_symbols + jet_symbols
        try:
            polynomial = sp.Poly(sp.expand(numerator), *generators)
        except sp.PolynomialError as error:
            raise ValueError(
                "the polynomial infinitesimal ansatz produced a non-polynomial "
                "invariance condition"
            ) from error
        for _, coefficient in polynomial.terms():
            _append_unique(determining, coefficient, conditions)

    all_coefficients = tuple(
        coefficient for group in coefficient_groups for coefficient in group
    )
    if determining:
        matrix, right_hand_side = sp.linear_eq_to_matrix(determining, all_coefficients)
        if any(value != 0 for value in right_hand_side):
            raise ValueError("the determining system unexpectedly became inhomogeneous")
        vectors = tuple(_normalize_null_vector(vector) for vector in matrix.nullspace())
    else:
        vectors = tuple(
            tuple(
                sp.S.One if row == column else sp.S.Zero
                for row in range(len(all_coefficients))
            )
            for column in range(len(all_coefficients))
        )

    basis = []
    for vector in vectors:
        substitutions = dict(zip(all_coefficients, vector))
        basis.append(
            InfinitesimalGenerator(
                tuple(sp.simplify(value.subs(substitutions)) for value in xi),
                tuple(sp.simplify(value.subs(substitutions)) for value in phi),
            )
        )
    constants = tuple(sp.symbols(f"{constant_prefix}1:{len(basis) + 1}"))
    general = InfinitesimalGenerator(
        tuple(
            sp.expand(
                sum(
                    constant * generator.xi[index]
                    for constant, generator in zip(constants, basis)
                )
            )
            for index in range(len(independents))
        ),
        tuple(
            sp.expand(
                sum(
                    constant * generator.phi[index]
                    for constant, generator in zip(constants, basis)
                )
            )
            for index in range(len(dependents))
        ),
    )
    return InfinitesimalSolution(
        basis=tuple(basis),
        general=general,
        constants=constants,
        ansatz_degree=ansatz_degree,
        determining_equations=tuple(determining),
        regularity_conditions=tuple(conditions),
    )


def verify_generator(
    equations,
    dependent_variables,
    independent_variables,
    generator: InfinitesimalGenerator,
    *,
    substitution_rules: Mapping[sp.Derivative, sp.Expr] | None = None,
) -> bool:
    """Return whether a generator is invariant on the selected regular branch.

    Automatically inferred substitution rules can require nonzero regularity
    conditions.  Use :func:`infer_substitution_rules` with
    ``return_conditions=True`` when those conditions need to be inspected.
    """

    residuals = prolongation(
        equations,
        dependent_variables,
        independent_variables,
        generator.xi,
        generator.phi,
        substitution_rules=substitution_rules,
    )
    return all(sp.simplify(residual) == 0 for residual in residuals)


def lie_bracket(
    first: InfinitesimalGenerator,
    second: InfinitesimalGenerator,
    dependent_variables,
    independent_variables,
) -> InfinitesimalGenerator:
    """Calculate the commutator of two point-symmetry generators."""

    independents = _normalize_independents(independent_variables)
    dependents = _normalize_dependents(dependent_variables, independent_variables)
    if len(first.xi) != len(independents) or len(second.xi) != len(independents):
        raise ValueError("generator xi dimensions do not match independent variables")
    if len(first.phi) != len(dependents) or len(second.phi) != len(dependents):
        raise ValueError("generator phi dimensions do not match dependent variables")
    # The commutator lives on the base space (x, u), so derivatives with
    # respect to x must hold u fixed.  Replacing applied functions by ordinary
    # symbols prevents SymPy's ``diff`` from taking a total derivative here.
    dependent_symbols = tuple(sp.Dummy(f"u{index}") for index in range(len(dependents)))
    to_algebraic = dict(zip(dependents, dependent_symbols))
    from_algebraic = dict(zip(dependent_symbols, dependents))
    variables = independents + dependent_symbols
    first_components = tuple(value.xreplace(to_algebraic) for value in first.components)
    second_components = tuple(
        value.xreplace(to_algebraic) for value in second.components
    )
    bracket = tuple(
        sp.simplify(
            sum(
                left * sp.diff(target_right, variable)
                - right * sp.diff(target_left, variable)
                for left, right, variable in zip(
                    first_components, second_components, variables
                )
            )
        ).xreplace(from_algebraic)
        for target_left, target_right in zip(first_components, second_components)
    )
    return InfinitesimalGenerator(
        bracket[: len(independents)], bracket[len(independents) :]
    )
