"""Exact sign-change brackets for polynomial contact guards on a unit interval."""

from fractions import Fraction
from math import comb


def _trim(values):
    values = list(values)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return values


def _divide(a, b):
    remainder = _trim(a)
    b = _trim(b)
    if b == [0]:
        raise ZeroDivisionError("Zero polynomial divisor")
    quotient = [Fraction(0)] * max(1, len(a) - len(b) + 1)
    while remainder != [0] and len(remainder) >= len(b):
        shift = len(remainder) - len(b)
        coefficient = remainder[-1] / b[-1]
        quotient[shift] += coefficient
        for i, value in enumerate(b):
            remainder[i + shift] -= coefficient * value
        remainder = _trim(remainder)
    return _trim(quotient), remainder


def _gcd(a, b):
    while b != [0]:
        a, b = b, _divide(a, b)[1]
    return [value / a[-1] for value in a]


def _multiply(a, b):
    result = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] += x * y
    return _trim(result)


def evaluate(polynomial, x):
    result = Fraction(0)
    for value in reversed(polynomial):
        result = result * x + value
    return result


def _odd_part(polynomial):
    derivative = [i * value for i, value in enumerate(polynomial)][1:]
    repeated = _gcd(polynomial, derivative)
    remaining = _divide(polynomial, repeated)[0]
    result = [Fraction(1)]
    multiplicity = 1
    while len(remaining) > 1:
        common = _gcd(remaining, repeated)
        factor = _divide(remaining, common)[0]
        if multiplicity % 2:
            result = _multiply(result, factor)
        remaining = common
        repeated = _divide(repeated, common)[0]
        multiplicity += 1
    return result


def _bernstein(polynomial, lo=Fraction(0), hi=Fraction(1)):
    degree = len(polynomial) - 1
    polynomial = [
        (hi - lo) ** k
        * sum(polynomial[i] * comb(i, k) * lo ** (i - k) for i in range(k, degree + 1))
        for k in range(degree + 1)
    ]
    return [
        sum(polynomial[k] * Fraction(comb(i, k), comb(degree, k)) for k in range(i + 1))
        for i in range(degree + 1)
    ]


def _split(coefficients):
    rows = [coefficients]
    while len(rows[-1]) > 1:
        row = rows[-1]
        rows.append([(a + b) / 2 for a, b in zip(row[:-1], row[1:])])
    return [row[0] for row in rows], [row[-1] for row in reversed(rows)]


def _sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def _direction(polynomial, root):
    value = evaluate(polynomial, root)
    while value == 0 and len(polynomial) > 1:
        polynomial = [i * coefficient for i, coefficient in enumerate(polynomial)][1:]
        value = evaluate(polynomial, root)
    return _sign(value)


def crossing_brackets(coefficients):
    """Return (lo, hi, direction) for every sign change in (0, 1].

    Coefficients are ordered by ascending power and converted exactly to
    rationals. Even-multiplicity tangencies keep the same contact mode.
    Brackets use Bernstein sign variations and de Casteljau subdivision;
    rational roots discovered during subdivision are returned as lo == hi.
    """
    original = _trim([Fraction(value) for value in coefficients])
    if len(original) == 1:
        return []
    polynomial = _odd_part(original)
    exact = []
    for endpoint in [Fraction(0), Fraction(1)]:
        if len(polynomial) > 1 and evaluate(polynomial, endpoint) == 0:
            polynomial, remainder = _divide(polynomial, [-endpoint, Fraction(1)])
            assert remainder == [0]
            if endpoint:
                exact.append(endpoint)
    while len(polynomial) > 1:
        brackets = []
        bounds = sorted(set([Fraction(0), *exact, Fraction(1)]))
        pending = [(lo, hi, _bernstein(polynomial, lo, hi)) for lo, hi in zip(bounds[:-1], bounds[1:])]
        found = None
        while pending:
            lo, hi, values = pending.pop()
            signs = [_sign(value) for value in values if value]
            variations = sum(a != b for a, b in zip(signs[:-1], signs[1:]))
            if variations == 0:
                continue
            if variations == 1:
                brackets.append((lo, hi))
                continue
            left, right = _split(values)
            midpoint = (lo + hi) / 2
            if left[-1] == 0:
                found = midpoint
                break
            pending.append((midpoint, hi, right))
            pending.append((lo, midpoint, left))
        if found is None:
            break
        exact.append(found)
        polynomial, remainder = _divide(polynomial, [-found, Fraction(1)])
        assert remainder == [0]
    else:
        brackets = []
    result = [(root, root, _direction(original, root)) for root in exact]
    for lo, hi in brackets:
        # Excluded tangencies/endpoints may still be zeros of the original guard.
        for lower in [True, False]:
            endpoint = lo if lower else hi
            if evaluate(original, endpoint) != 0:
                continue
            fraction = Fraction(1, 2)
            while True:
                candidate = endpoint + ((hi if lower else lo) - endpoint) * fraction
                if (
                    _sign(evaluate(polynomial, candidate)) == _sign(evaluate(polynomial, endpoint))
                    and evaluate(original, candidate) != 0
                ):
                    if lower:
                        lo = candidate
                    else:
                        hi = candidate
                    break
                fraction /= 2
        result.append((lo, hi, _sign(evaluate(original, hi))))
    return sorted(result)
