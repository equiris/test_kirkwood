"""
Simple iterations Q_{k+1} = A(Q_k), Q_0 = 0 for equilibrium operator A.
"""
import numpy as np


def solve(b, dp, sigma_m, sigma_w, d, L, n = None, tol = 1e-13, maxit = 200000):

    def _calculate_grid(sigma, L):
        """
        Choose n so that dx = L / n <= sigma_min / 8 (heuristic). Then choose the next power of two for FFT.
        """
        return int(2 ** np.ceil(np.log2(8 * L / sigma)))
    sw = min(sigma_m, sigma_w)
    n = _calculate_grid(sw, L)
    dx = L / n
    x = np.arange(n) * dx
    x = np.where(x > L / 2, x - L, x)

    def _kernel(s):
        g = np.exp(-0.5 * (x / s) ** 2)
        return g / (g.sum() * dx)

    m, w = b * _kernel(sigma_m), dp * _kernel(sigma_w)
    m_fft, w_fft = np.fft.rfft(m), np.fft.rfft(w)
    conv = lambda f, g: dx * np.fft.irfft(f * g, n)

    Q = np.zeros(n)
    for it in range(1, maxit + 1):
        Y = dp + dx * np.sum(w * Q)
        Q_fft = np.fft.rfft(Q)
        t = conv(np.fft.rfft(w * Q), Q_fft) + conv(Q_fft, w_fft)
        AQ = (Y * m / (b - d) - w + conv(Q_fft, m_fft)
              - (b - d) / Y * (1 + Q) * t) / (w + b)
        step = np.max(np.abs(AQ - Q))
        Q = AQ
        if not np.isfinite(step) or step < tol:
            break
    ok = np.isfinite(step) and step < tol
    Y = dp + dx * np.sum(w * Q)
    r = np.fft.fftshift(np.where(np.arange(n) * dx > L / 2, np.arange(n) * dx - L, np.arange(n) * dx))
    return dict(r=r, Q=np.fft.fftshift(Q), N=(b - d) / Y, it=it, step=step, converged=ok)