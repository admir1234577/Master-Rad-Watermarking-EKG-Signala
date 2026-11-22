# ============================================================
# STOCKWELL (ST) METODA - UMETANJE, EKSTRAKCIJA I SLIKE
# Aditivno umetanje u frekvencijske redove 3,9-15,7 Hz; jedna
# vrijednost vodenog žiga po redu, konstantna duž vremenske ose.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

from zajednicko import to_bits, calculate_nc, save_figure

FIGURE_NUMBER = 10   # broj slike razlike koeficijenata u radu
NAME = "ST"
F_MIN = 3.9
F_MAX = 15.7
ALPHA = 0.0114   # kalibrisano na PSNR = 60 dB
F_PLOT = 50      # gornja granica frekvencije čuvanih redova (Hz)
F_DIFF_PLOT = 30 # gornja granica frekvencije na Slici 10 (Hz)


def stockwell_transform(signal, gaussian_width=1.0, rows=None):
    """
    Stockwell transformacija sa podesivom širinom Gaussovog prozora.
    rows=None računa sve frekvencijske redove (0 ... N/2); zadavanjem
    'rows' računaju se samo navedeni redovi (ista formula, manje memorije).
    """
    if gaussian_width <= 0:
        raise ValueError("gaussian_width mora biti veći od nule.")

    N = len(signal)
    X = fft(signal)
    X_extended = np.concatenate((X, X))
    n_freq = N // 2 + 1
    if rows is None:
        rows = np.arange(n_freq)

    k = np.arange(N)
    k_shifted = np.where(k <= N // 2, k, k - N)

    S = np.zeros((len(rows), N), dtype=np.complex128)
    for i, m in enumerate(rows):
        if m == 0:
            S[i, :] = np.mean(signal)
            continue
        sigma = gaussian_width * m
        gaussian = np.exp(-2 * (np.pi ** 2) * (k_shifted ** 2) / (sigma ** 2))
        S[i, :] = ifft(X_extended[m:m + N] * gaussian)
    return S


def inverse_stockwell_transform(S, signal_length):
    """Inverzna ST: zbir svakog reda po vremenu daje DFT koeficijent."""
    spectrum_positive = np.sum(S, axis=1)
    if signal_length % 2 == 0:
        spectrum_full = np.concatenate(
            (spectrum_positive, np.conj(spectrum_positive[-2:0:-1])))
    else:
        spectrum_full = np.concatenate(
            (spectrum_positive, np.conj(spectrum_positive[-1:0:-1])))
    return np.real(ifft(spectrum_full))[:signal_length]


def embed(ecg_signal, watermark, alpha=ALPHA, fs=1000,
          gaussian_width=1.0, verbose=True):

    N = len(ecg_signal)

    # Kompletna ST matrica originalnog EKG-a (N/2 + 1) x N
    S_original = stockwell_transform(ecg_signal, gaussian_width)

    if verbose:
        reconstructed = inverse_stockwell_transform(S_original, N)
        error = ecg_signal - reconstructed
        print("\nST - kontrola rekonstrukcije")
        print("Oblik Stockwell matrice:", S_original.shape)
        print(f"MSE rekonstrukcije: {np.mean(error ** 2):.3e}")
        print(f"Maksimalna apsolutna greška: {np.max(np.abs(error)):.3e}")

    frequencies = np.fft.rfftfreq(N, d=1 / fs)
    selected_bins = np.where(
        (frequencies >= F_MIN) & (frequencies <= F_MAX))[0]

    # Jedna vrijednost vodenog žiga po odabranom frekvencijskom redu
    reference = np.resize(watermark, len(selected_bins))

    S_watermarked = S_original.copy()
    S_watermarked[selected_bins, :] += alpha * reference[:, np.newaxis]

    watermarked_signal = inverse_stockwell_transform(S_watermarked, N)

    # Čuvaju se samo redovi potrebni za ekstrakciju i za slike
    plot_rows = np.where(frequencies <= F_PLOT)[0]
    state = dict(
        name=NAME, binary=False, alpha=alpha, fs=fs,
        gaussian_width=gaussian_width,
        param=f"α = {alpha}",
        frequencies=frequencies,
        selected_bins=selected_bins,
        S_original_selected=S_original[selected_bins, :].copy(),
        plot_rows=plot_rows,
        S_original_plot=S_original[plot_rows, :].copy(),
        S_watermarked_plot=S_watermarked[plot_rows, :].copy(),
        reference=reference,
        reference_bits=to_bits(reference),
        capacity=len(reference),
        element_label="Indeks frekvencijskog reda",
    )
    del S_original, S_watermarked
    return watermarked_signal, state


def extract(signal, state):
    """
    Neslijepa ekstrakcija: srednja realna razlika ST koeficijenata
    napadnutog i originalnog signala po vremenskoj osi, podijeljena
    sa alpha. Računaju se samo odabrani frekvencijski redovi.
    """
    S_attacked = stockwell_transform(
        signal, state["gaussian_width"], rows=state["selected_bins"])
    difference = S_attacked - state["S_original_selected"]
    return np.real(np.mean(difference, axis=1)) / state["alpha"]


def amplitude_data(watermarked_signal, state):
    return (state["reference"], extract(watermarked_signal, state),
            "Vrijednost vodenog žiga")


def figures(ecg_signal, watermarked_signal, state, fs):
    N = len(ecg_signal)
    time_axis = np.arange(N) / fs

    print("\nST - provjera amplitude")
    print(f"Originalni EKG: min = {np.min(ecg_signal):.6f}, "
          f"max = {np.max(ecg_signal):.6f}")
    print(f"EKG sa žigom:   min = {np.min(watermarked_signal):.6f}, "
          f"max = {np.max(watermarked_signal):.6f}")

    # Slika 10: zadata i ostvarena promjena Stockwell koeficijenata (0-30 Hz)
    #   zadata:    S_wm - S_original (vrijednosti upisane u ST matricu, izraz 66)
    #   ostvarena: ST{x_wm} - ST{x}  (ST signala sa vodenim žigom nakon inverzne ST)
    rows_mask = state["frequencies"][state["plot_rows"]] <= F_DIFF_PLOT
    rows = state["plot_rows"][rows_mask]
    freqs = state["frequencies"][rows]
    S_original = state["S_original_plot"][rows_mask, :]

    expected = state["S_watermarked_plot"][rows_mask, :] - S_original
    actual = stockwell_transform(watermarked_signal, state["gaussian_width"],
                                 rows=rows) - S_original

    inside = np.isin(rows, state["selected_bins"])
    gain = (np.real(np.sum(actual[inside] * np.conj(expected[inside])))
            / np.sum(np.abs(expected[inside]) ** 2))
    mean_error = np.max(np.abs(np.real(np.mean(actual[inside], axis=1))
                               - np.real(np.mean(expected[inside], axis=1))))
    energy_outside = (np.sum(np.abs(actual[~inside]) ** 2)
                      / np.sum(np.abs(actual) ** 2))

    print(f"\nST - zadata i ostvarena promjena koeficijenata (0-{F_DIFF_PLOT} Hz)")
    print(f"Maks. zadata promjena:                    {np.max(np.abs(expected)):.6f}")
    print(f"Maks. zadata promjena izvan opsega:       {np.max(np.abs(expected[~inside])):.6f}")
    print(f"Maks. ostvarena promjena unutar opsega:   {np.max(np.abs(actual[inside])):.6f}")
    print(f"Maks. ostvarena promjena izvan opsega:    {np.max(np.abs(actual[~inside])):.6f}")
    print(f"Srednja |zadata| promjena unutar opsega:  {np.mean(np.abs(expected[inside])):.6f}")
    print(f"Srednja |ostvarena| promjena unutar:      {np.mean(np.abs(actual[inside])):.6f}")
    print(f"Udio energije ostvarene promjene izvan opsega: {100 * energy_outside:.2f} %")
    print(f"Pojačanje (projekcija):                   {gain:.4f}")
    print(f"Najveća razlika srednjih vrijednosti redova (zadata - ostvarena): {mean_error:.2e}")

    vmax = max(np.max(np.abs(expected)), np.max(np.abs(actual)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True,
                             constrained_layout=True)
    for ax, D, title in [
        (axes[0], expected, "Zadata promjena (ST matrica)"),
        (axes[1], actual, "Ostvarena promjena (ST signala sa žigom)"),
    ]:
        mesh = ax.pcolormesh(time_axis, freqs, np.abs(D), shading="auto",
                             vmin=0, vmax=vmax, rasterized=True)
        for f_edge in (F_MIN, F_MAX):
            ax.axhline(f_edge, color="white", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, F_DIFF_PLOT)
        ax.set_title(title)
        ax.set_xlabel("Vrijeme (s)")
    axes[0].set_ylabel("Frekvencija (Hz)")
    fig.colorbar(mesh, ax=axes, label="Apsolutna razlika koeficijenata")
    save_figure(f"Slika_{FIGURE_NUMBER:02d}_razlika_ST_koeficijenata.png", fig)


def additional_experiments(ecg_signal, watermark, alpha=ALPHA, fs=1000,
                           widths=(0.5, 1.0, 2.0, 4.0)):
    """
    Dodatni eksperimenti (poglavlje 7.5):
    1) Stockwell ekstrakcija naspram direktne DFT razlike na istim binovima
    2) uticaj širine Gaussovog prozora
    """
    N = len(ecg_signal)
    X_original = np.fft.rfft(ecg_signal)
    rows = []

    for width in widths:
        watermarked, state = embed(ecg_signal, watermark, alpha, fs,
                                   gaussian_width=width, verbose=False)
        extracted_st = extract(watermarked, state)

        sel = state["selected_bins"]
        extracted_dft = np.real(np.fft.rfft(watermarked)[sel] - X_original[sel])

        ref, ref_bits = state["reference"], state["reference_bits"]
        mse = np.mean((ecg_signal - watermarked) ** 2)
        rows.append({
            "Širina prozora": width,
            "NC (ST)": calculate_nc(ref, extracted_st),
            "BER (ST)": np.mean(ref_bits != to_bits(extracted_st)),
            "NC (DFT)": calculate_nc(ref, extracted_dft),
            "BER (DFT)": np.mean(ref_bits != to_bits(extracted_dft)),
            "MSE": mse,
            "PSNR (dB)": 10 * np.log10(np.max(np.abs(ecg_signal)) ** 2 / mse),
        })

    table = pd.DataFrame(rows)

    return table
