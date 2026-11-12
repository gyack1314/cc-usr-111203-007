"""纯标准库线性代数：Householder QR、加权最小二乘、Jacobi对称特征分解。

刻意不引入第三方依赖。所有秩判定走显式容差，秩不足时返回最小残差解并报告秩，
绝不以随意ridge把欠定方程组伪装成唯一确定值。
"""
import math


def zeros(m, n):
    return [[0.0 for _ in range(n)] for _ in range(m)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A):
    return [list(row) for row in zip(*A)]


def matmul(A, B):
    p, q, s = len(A), len(B), len(B[0])
    C = zeros(p, s)
    for i in range(p):
        Ai = A[i]
        Ci = C[i]
        for k in range(q):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(s):
                Ci[j] += a * Bk[j]
    return C


def vdot(a, b):
    return sum(x * y for x, y in zip(a, b))


def qr(A):
    """Householder QR：A(m×n)=Q R，Q为m×m正交阵，R为m×n上梯形。"""
    m, n = len(A), len(A[0])
    R = [row[:] for row in A]
    Q = identity(m)
    for k in range(min(m, n)):
        x = [R[i][k] for i in range(k, m)]
        norm = math.sqrt(sum(v * v for v in x))
        if norm == 0.0:
            continue
        v = x[:]
        v[0] += math.copysign(norm, x[0] if x[0] != 0.0 else 1.0)
        vn = math.sqrt(sum(t * t for t in v))
        v = [t / vn for t in v]
        for j in range(k, n):
            dotvr = sum(v[i - k] * R[i][j] for i in range(k, m))
            for i in range(k, m):
                R[i][j] -= 2.0 * v[i - k] * dotvr
        for i in range(m):
            qdot = sum(Q[i][t] * v[t - k] for t in range(k, m))
            for t in range(k, m):
                Q[i][t] -= 2.0 * qdot * v[t - k]
    return Q, R


def weighted_lstsq(A, y, w=None, rank_tolerance=1e-10):
    """解 min ||W^(1/2)(A c - y)||^2，返回(系数, 数值秩, 奇异判定阈值)。

    秩不足时自由分量置零（QR给出的最小残差解），不添加任何ridge项；
    调用方依据返回秩决定是否标记不可识别。
    """
    m, n = len(A), len(A[0])
    sw = [math.sqrt(x) for x in w] if w is not None else [1.0] * m
    Aw = [[A[i][j] * sw[i] for j in range(n)] for i in range(m)]
    yw = [y[i] * sw[i] for i in range(m)]
    Q, R = qr(Aw)
    qty = [sum(Q[i][k] * yw[i] for i in range(m)) for k in range(min(m, n))]
    diag = [abs(R[k][k]) for k in range(min(m, n))]
    scale = max(diag, default=0.0)
    threshold = rank_tolerance * max(1.0, scale)
    rank = sum(1 for d in diag if d > threshold)
    coef = [0.0] * n
    for j in range(n - 1, -1, -1):
        if j >= m or abs(R[j][j]) <= threshold:
            coef[j] = 0.0
            continue
        s = qty[j] - sum(R[j][k] * coef[k] for k in range(j + 1, n))
        coef[j] = s / R[j][j]
    return coef, rank, threshold


def eigh(M, sweep_tolerance=1e-14, max_sweeps=100):
    """实对称阵Jacobi特征分解，返回(升序特征值, 列特征向量)。"""
    n = len(M)
    A = [row[:] for row in M]
    V = identity(n)
    for _ in range(max_sweeps):
        off = math.sqrt(sum(A[i][j] ** 2 for i in range(n) for j in range(i + 1, n)))
        fro = math.sqrt(sum(A[i][j] ** 2 for i in range(n) for j in range(n)))
        if off <= sweep_tolerance * max(1.0, fro):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                if apq == 0.0:
                    continue
                tau = (A[q][q] - A[p][p]) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = c * akp - s * akq
                    A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k] = c * apk - s * aqk
                    A[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: A[i][i])
    values = [A[i][i] for i in order]
    vectors = [[V[i][j] for j in order] for i in range(n)]
    return values, vectors


def area_orthonormalize(V, area, rank_tolerance=1e-10):
    """把V(n×r)变换为 V B，使 (V B)^T diag(area) (V B)=I。

    M=V^T A V=Q Λ Q^T，取 B=Q Λ^{-1/2}；同步返回 U 的变换阵 B^{-T}=Q Λ^{1/2}，
    使 (U B^{-T})(V B)^T=U V^T，重建不变。最小特征值过小时报告秩不足。
    """
    n, r = len(V), len(V[0])
    M = zeros(r, r)
    for i in range(n):
        ai = area[i]
        Vi = V[i]
        for p in range(r):
            for q in range(r):
                M[p][q] += ai * Vi[p] * Vi[q]
    values, Q = eigh(M)
    top = max((abs(x) for x in values), default=0.0)
    threshold = rank_tolerance * max(1.0, top)
    full_rank = all(abs(x) > threshold for x in values)
    B = zeros(r, r)
    Binvt = zeros(r, r)
    for k in range(r):
        lam = values[k]
        scale = 1.0 / math.sqrt(lam) if abs(lam) > threshold else 0.0
        grow = math.sqrt(lam) if abs(lam) > threshold else 0.0
        for i in range(r):
            B[i][k] = Q[i][k] * scale
            Binvt[i][k] = Q[i][k] * grow
    Vnew = matmul(V, B)
    return Vnew, Binvt, values, full_rank


class SplitMix:
    """确定性伪随机源（SplitMix64 + Box-Muller），多起点可复现。"""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF
        self.spare = None

    def _next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self):
        return (self._next_u64() >> 11) * (2.0 ** -53)

    def gaussian(self):
        if self.spare is not None:
            x, self.spare = self.spare, None
            return x
        u = 0.0
        while u == 0.0:
            u = self.uniform()
        v = self.uniform()
        mag = math.sqrt(-2.0 * math.log(u))
        z0 = mag * math.cos(2.0 * math.pi * v)
        self.spare = mag * math.sin(2.0 * math.pi * v)
        return z0
