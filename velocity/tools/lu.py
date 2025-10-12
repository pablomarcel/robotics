# velocity/tools/lu.py
# Minimal LU utilities. Exposes the "old" (L,U) API expected by apis.py/tests,
# and optional SciPy-like (LU, piv) helpers under *_piv names for future use.

from __future__ import annotations
import numpy as np

__all__ = [
    # Old, non-pivoting API (matches tests and current apis.py)
    "lu_factor",
    "lu_solve",
    "lu_inverse",
    "LU",
    # Optional pivoting helpers (SciPy-like, names end with _piv)
    "lu_factor_piv",
    "lu_solve_piv",
    "lu_inverse_piv",
]

# ----------------------------- Old interface (no pivoting) -----------------------------

def lu_factor(A: np.ndarray):
    """
    Doolittle LU without pivoting.
    Returns (L, U) with L unit-diagonal lower-triangular and U upper-triangular.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("lu_factor: A must be square")
    n = A.shape[0]
    L = np.eye(n, dtype=float)
    U = A.copy()

    for i in range(n - 1):
        piv = U[i, i]
        if piv == 0.0:
            # Non-pivoting version; behave like the tests' toy impl (let it blow up later)
            continue
        for j in range(i + 1, n):
            m = U[j, i] / piv
            L[j, i] = m
            U[j, i:] -= m * U[i, i:]
    return L, U


def _forward_sub(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def _back_sub(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        rhs = y[i] - U[i, i + 1:] @ x[i + 1:]
        denom = U[i, i]
        if denom == 0.0:
            raise np.linalg.LinAlgError("singular matrix: zero on U diagonal")
        x[i] = rhs / denom
    return x


def lu_solve(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b using (L, U) from lu_factor (no pivoting).
    Accepts b shape (n,) and returns (n,).
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    y = _forward_sub(L, b)
    x = _back_sub(U, y)
    return x


def lu_inverse(L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Invert A given (L, U) by solving A X = I column-wise.
    """
    n = L.shape[0]
    I = np.eye(n, dtype=float)
    cols = [lu_solve(L, U, I[:, i]) for i in range(n)]
    return np.column_stack(cols)


# ---------------------- Optional pivoting helpers (SciPy-like) ----------------------

def lu_factor_piv(A: np.ndarray):
    """
    SciPy-like LU with partial pivoting using combined storage.
    Returns (LU, piv) where piv records the row index swapped with k at step k.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("lu_factor_piv: A must be square")
    n = A.shape[0]
    LU = A.copy()
    piv = np.arange(n)

    for k in range(n - 1):
        i_max = k + np.argmax(np.abs(LU[k:, k]))
        if LU[i_max, k] == 0.0:
            continue
        if i_max != k:
            LU[[k, i_max], :] = LU[[i_max, k], :]
            piv[[k, i_max]] = piv[[i_max, k]]
        if LU[k, k] != 0.0:
            LU[k + 1:, k] /= LU[k, k]
            LU[k + 1:, k + 1:] -= np.outer(LU[k + 1:, k], LU[k, k + 1:])
    return LU, piv


def _apply_pivots(b: np.ndarray, piv: np.ndarray) -> np.ndarray:
    bp = b.copy()
    for k, i_max in enumerate(piv):
        if k != i_max:
            bp[[k, i_max], ...] = bp[[i_max, k], ...]
    return bp


def _forward_sub_unit_lower(LU: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = LU.shape[0]
    y = b.copy()
    for i in range(n):
        y[i, ...] -= LU[i, :i] @ y[:i, ...]
    return y


def _back_sub_upper(LU: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = LU.shape[0]
    x = y.copy()
    for i in range(n - 1, -1, -1):
        x[i, ...] -= LU[i, i + 1:] @ x[i + 1:, ...]
        denom = LU[i, i]
        if denom == 0.0:
            raise np.linalg.LinAlgLinAlgError("singular matrix: zero on U diagonal")
        x[i, ...] /= denom
    return x


def lu_solve_piv(fact: tuple[np.ndarray, np.ndarray], b: np.ndarray) -> np.ndarray:
    """
    SciPy-like solve using (LU, piv). b may be (n,) or (n, k).
    """
    LU, piv = fact
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b2 = b.reshape(-1, 1)
        squeeze = True
    elif b.ndim == 2:
        b2 = b
        squeeze = False
    else:
        raise ValueError("lu_solve_piv: b must be 1D or 2D")

    bp = _apply_pivots(b2, piv)
    y = _forward_sub_unit_lower(LU, bp)
    x = _back_sub_upper(LU, y)
    return x.ravel() if squeeze else x


def lu_inverse_piv(A: np.ndarray) -> np.ndarray:
    LU, piv = lu_factor_piv(A)
    n = LU.shape[0]
    I = np.eye(n, dtype=float)
    return lu_solve_piv((LU, piv), I)


# ------------------------------ Optional OO façade ------------------------------

class LU:
    # Old style helpers
    @staticmethod
    def factor(A): return lu_factor(A)
    @staticmethod
    def solve(A, b):
        L, U = lu_factor(A)
        return lu_solve(L, U, b)
    @staticmethod
    def inverse(A):
        L, U = lu_factor(A)
        return lu_inverse(L, U)

    # Pivoting style helpers under different names
    @staticmethod
    def factor_piv(A): return lu_factor_piv(A)
    @staticmethod
    def solve_piv(A, b):
        LUfac = lu_factor_piv(A)
        return lu_solve_piv(LUfac, b)
    @staticmethod
    def inverse_piv(A): return lu_inverse_piv(A)
