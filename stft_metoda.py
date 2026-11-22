# ============================================================
# STFT METODA - UMETANJE, EKSTRAKCIJA I SLIKE
# Hann prozor 512 uzoraka, preklapanje 256, binovi 3,9-15,7 Hz.
# Umetanje: Z_wm(k, n) = Z(k, n) + alpha * (-1)^k * w(n)
# Fazni faktor (-1)^k postavlja umetnutu komponentu u središte
# prozora, gdje Hann prozor ima vrijednost 1.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

from zajednicko import to_bits, save_figure

FIGURE_NUMBER = 9   # broj slike razlike koeficijenata u radu
NAME = "STFT"
WINDOW = "hann"
NPERSEG = 512
NOVERLAP = 256
F_MIN = 3.9
F_MAX = 15.7
ALPHA = 0.0614   # kalibrisano na PSNR = 60 dB


def stft_forward(signal, fs):
    return stft(signal, fs=fs, window=WINDOW,
                nperseg=NPERSEG, noverlap=NOVERLAP)


def stft_inverse(Zxx, fs, length):
    _, signal = istft(Zxx, fs=fs, window=WINDOW,
                      nperseg=NPERSEG, noverlap=NOVERLAP)
    return signal[:length]


def embed(ecg_signal, watermark, alpha=ALPHA, fs=1000):

    f, t, Zxx_original = stft_forward(ecg_signal, fs)

    # Frekvencijski binovi za umetanje (3,9-15,7 Hz)
    selected_bins = np.where((f >= F_MIN) & (f <= F_MAX))[0]

    # Fazni faktor (-1)^k za svaki odabrani bin
    phase = (-1.0) ** selected_bins

    # Jedan element vodenog žiga po vremenskom segmentu
    reference = np.resize(watermark, Zxx_original.shape[1])

    Zxx_watermarked = Zxx_original.copy()
    Zxx_watermarked[selected_bins, :] += (
        alpha * phase[:, np.newaxis] * reference[np.newaxis, :]
    )

    watermarked_signal = stft_inverse(Zxx_watermarked, fs, len(ecg_signal))

    state = dict(
        name=NAME, binary=False, alpha=alpha, fs=fs,
        param=f"α = {alpha}",
        f=f, t=t,
        selected_bins=selected_bins,
        phase=phase,
        Zxx_original=Zxx_original,
        Zxx_watermarked=Zxx_watermarked,
        reference=reference,
        reference_bits=to_bits(reference),
        capacity=len(reference),
        element_label="Indeks vremenskog segmenta",
    )
    return watermarked_signal, state


def extract(signal, state):
    """
    Neslijepa ekstrakcija: razlika STFT koeficijenata napadnutog i
    originalnog signala u odabranim binovima, pomnožena faznim
    faktorom, usrednjena po binovima i podijeljena sa alpha.
    """
    _, _, Zxx_attacked = stft_forward(signal, state["fs"])
    sel = state["selected_bins"]

    difference = Zxx_attacked[sel, :] - state["Zxx_original"][sel, :]
    difference = difference * state["phase"][:, np.newaxis]

    return np.real(np.mean(difference, axis=0)) / state["alpha"]


def amplitude_data(watermarked_signal, state):
    return (state["reference"], extract(watermarked_signal, state),
            "Vrijednost vodenog žiga")


def figures(ecg_signal, watermarked_signal, state, fs):
    """
    Slika 9: zadata promjena STFT koeficijenata (upisana u matricu)
    i ostvarena promjena (STFT signala sa vodenim žigom).
    """
    f, t, sel = state["f"], state["t"], state["selected_bins"]
    Zxx_original = state["Zxx_original"]

    _, _, Zxx_reconstructed = stft_forward(watermarked_signal, fs)

    expected = state["Zxx_watermarked"] - Zxx_original
    actual = Zxx_reconstructed - Zxx_original

    expected_sel, actual_sel = expected[sel, :], actual[sel, :]
    gain = (np.real(np.sum(actual_sel * np.conj(expected_sel)))
            / np.sum(np.abs(expected_sel) ** 2))
    outside = np.setdiff1d(np.arange(Zxx_original.shape[0]), sel)

    print("\nSTFT - zadata i ostvarena promjena koeficijenata")
    print("Odabrani binovi:", sel, "| frekvencije (Hz):", np.round(f[sel], 4))
    print(f"Srednja |zadata promjena|:    {np.mean(np.abs(expected_sel)):.6f}")
    print(f"Srednja |ostvarena promjena|: {np.mean(np.abs(actual_sel)):.6f}")
    print(f"Pojačanje (projekcija):       {gain:.4f}")
    print(f"Maks. zadata promjena:        {np.max(np.abs(expected_sel)):.6f}")
    print(f"Maks. ostvarena promjena:     {np.max(np.abs(actual_sel)):.6f}")
    print(f"Maks. zadata promjena izvan binova:    "
          f"{np.max(np.abs(expected[outside, :])):.6f}")
    print(f"Maks. ostvarena promjena izvan binova: "
          f"{np.max(np.abs(actual[outside, :])):.6f}")

    vmax = max(np.max(np.abs(expected)), np.max(np.abs(actual)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True,
                             constrained_layout=True)
    for ax, D, title in [
        (axes[0], expected, "Zadata promjena (STFT matrica)"),
        (axes[1], actual, "Ostvarena promjena (STFT signala sa žigom)"),
    ]:
        mesh = ax.pcolormesh(t, f, np.abs(D), shading="gouraud",
                             vmin=0, vmax=vmax)
        ax.set_ylim(0, 20)
        ax.set_title(title)
        ax.set_xlabel("Vrijeme (s)")
    axes[0].set_ylabel("Frekvencija (Hz)")
    fig.colorbar(mesh, ax=axes, label="Apsolutna razlika koeficijenata")
    save_figure(f"Slika_{FIGURE_NUMBER:02d}_razlika_STFT_koeficijenata.png", fig)
