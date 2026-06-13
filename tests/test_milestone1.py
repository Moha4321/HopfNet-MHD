"""
Milestone 1 unit tests:
  - 3D real FFT round-trip (forward -> inverse recovers the original field)
  - Single-mode spectrum: transforming a known sinusoid concentrates energy
    at the expected wavevector
  - Wavevector array correctness against hand-computed values (N=8)
  - Dealiasing mask shape and corner-case correctness (2/3 rule)

Run with: pytest tests/test_milestone1.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng

N = 8
L = 2 * np.pi
DK = 2 * np.pi / L


def test_fft_roundtrip_random_field():
    fft = eng.FFT3D(N)
    rng = np.random.default_rng(42)
    field = rng.standard_normal((N, N, N))

    spectrum = fft.forward(field)
    recovered = fft.inverse(spectrum)

    assert recovered.shape == field.shape
    np.testing.assert_allclose(recovered, field, atol=1e-12)


def test_fft_roundtrip_constant_field():
    # The k=0 mode only; trivial but catches normalization bugs.
    fft = eng.FFT3D(N)
    field = np.full((N, N, N), 3.7)

    spectrum = fft.forward(field)
    recovered = fft.inverse(spectrum)

    np.testing.assert_allclose(recovered, field, atol=1e-12)
    # All energy should sit in the k=0 mode.
    assert np.isclose(spectrum[0, 0, 0].real, 3.7 * N**3)
    mag = np.abs(spectrum)
    assert mag[0, 0, 0] / mag.sum() > 0.999


def test_single_mode_spectrum():
    fft = eng.FFT3D(N)
    x = np.arange(N) * (L / N)
    X, _, _ = np.meshgrid(x, x, x, indexing='ij')

    # cos(1 * x) = single Fourier mode with integer wavenumber n = +-1 along x
    field = np.cos(1.0 * X)
    spectrum = fft.forward(field)

    mag = np.abs(spectrum)
    total_energy = mag.sum()

    # In the rFFT layout, nx=+1 lives at index 1; nx=-1 is folded into index N-1.
    peak_energy = mag[1, 0, 0] + mag[N - 1, 0, 0]
    assert peak_energy / total_energy > 0.99


def test_wavevector_values():
    grid = eng.SpectralGrid(N, L)

    kx = grid.kx()
    # Standard FFT ordering: 0,1,2,3,4,-3,-2,-1 (for N=8)
    expected_kx = np.array([0, 1, 2, 3, 4, -3, -2, -1]) * DK
    np.testing.assert_allclose(kx, expected_kx)
    np.testing.assert_allclose(grid.ky(), expected_kx)

    kz = grid.kz()
    assert kz.shape[0] == N // 2 + 1
    np.testing.assert_allclose(kz, np.arange(N // 2 + 1) * DK)


def test_k2_consistency():
    grid = eng.SpectralGrid(N, L)
    kx, ky, kz = grid.kx(), grid.ky(), grid.kz()
    k2 = grid.k2()

    assert k2.shape == (N, N, N // 2 + 1)
    # Spot-check a handful of points by brute-force reconstruction.
    for i, j, k in [(0, 0, 0), (1, 0, 0), (4, 4, 4), (5, 2, 3)]:
        expected = kx[i]**2 + ky[j]**2 + kz[k]**2
        assert np.isclose(k2[i, j, k], expected)


def test_dealias_mask_shape_and_values():
    grid = eng.SpectralGrid(N, L)
    mask = grid.dealias_mask()

    assert mask.shape == (N, N, N // 2 + 1)

    # k=0 mode (mean field) is always retained.
    assert mask[0, 0, 0] == 1.0

    # For N=8, cutoff = floor((2/3)*4) = 2.
    # nx index 1,2 -> |n|<=2, kept; index 3 -> |n|=3, masked; Nyquist index 4
    # (nx=-4 under the FFT convention) -> |n|=4, masked.
    assert mask[1, 0, 0] == 1.0
    assert mask[2, 0, 0] == 1.0
    assert mask[3, 0, 0] == 0.0
    assert mask[4, 0, 0] == 0.0

    # Mask must be symmetric for +/- wavenumbers on the x and y axes.
    np.testing.assert_array_equal(mask[1, :, :], mask[N - 1, :, :])
    np.testing.assert_array_equal(mask[:, 1, :], mask[:, N - 1, :])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
