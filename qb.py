# ============================================================
# QB (QIM) METODA - UMETANJE, EKSTRAKCIJA I SLIKE
# Kvantizacija A4 koeficijenata (db4, 4 nivoa) sa dither
# vrijednostima 0 i delta/2; slijepa ekstrakcija.
# ============================================================

import numpy as np
import pywt

from zajednicko import to_bits, plot_assigned_vs_realized

FIGURE_NUMBER = 8   # broj slike razlike koeficijenata u radu
NAME = "QB"
WAVELET = "db4"
LEVEL = 4
DELTA = 2.2497   # korak kvantizacije, kalibrisan na PSNR = 60 dB


def embed(ecg_signal, watermark, delta=DELTA, fs=1000):
    """QIM umetanje: A4_wm = round((A4 - d) / delta) * delta + d."""

    coeffs = pywt.wavedec(ecg_signal, WAVELET, level=LEVEL)
    original_A4 = coeffs[0].copy()

    # Binarni vodeni žig prilagođen broju A4 koeficijenata
    reference = np.resize(watermark, original_A4.shape)
    reference_bits = to_bits(reference)

    # Dither: 0 za bit 0, delta/2 za bit 1
    dither = np.where(reference_bits == 0, 0.0, delta / 2)

    watermarked_A4 = np.round((original_A4 - dither) / delta) * delta + dither
    coeffs[0] = watermarked_A4

    watermarked_signal = pywt.waverec(coeffs, WAVELET)[:len(ecg_signal)]

    state = dict(
        name=NAME, binary=True, delta=delta,
        param=f"Δ = {delta}",
        original_A4=original_A4,
        watermarked_A4=watermarked_A4,
        reference=reference,
        reference_bits=reference_bits,
        capacity=len(reference_bits),
        element_label="Indeks A4 koeficijenta",
    )
    return watermarked_signal, state


TIE_TOLERANCE = 1e-9   # relativna tolerancija (u jedinicama Δ) za granicu odluke
TIE_SEED = 12345       # seed za ponovljivo rješavanje neodlučivih bitova


def extract(signal, state):
    """
    Slijepa ekstrakcija: bit određuje bliža kvantizaciona mreža.

    Koeficijent koji je jednako udaljen od obje mreže (do greške
    zaokruživanja, TIE_TOLERANCE·Δ) leži tačno na granici odluke i
    njegov bit se ne može odrediti. Bez posebnog pravila takav bit bi
    zavisio od greške zaokruživanja računara (reda 1e-13), pa rezultat
    ne bi bio ponovljiv na različitim računarima. Zato se ovi bitovi
    određuju slučajno, ali ponovljivo (seed TIE_SEED), a njihov broj
    se čuva u state["last_ties"].
    """
    delta = state["delta"]
    attacked_A4 = pywt.wavedec(signal, WAVELET, level=LEVEL)[0]

    q0 = delta * np.round(attacked_A4 / delta)                       # mreža za bit 0
    q1 = delta * (np.round((attacked_A4 - delta / 2) / delta) + 0.5)  # mreža za bit 1
    d0 = np.abs(attacked_A4 - q0)
    d1 = np.abs(attacked_A4 - q1)

    bits = (d1 < d0).astype(int)

    ties = np.abs(d0 - d1) < TIE_TOLERANCE * delta
    if np.any(ties):
        rng = np.random.default_rng(TIE_SEED)
        bits[ties] = rng.integers(0, 2, size=int(np.sum(ties)))
    state["last_ties"] = int(np.sum(ties))

    return bits


def amplitude_data(watermarked_signal, state):
    """
    Žig je binarni, pa se porede promjena A4 koeficijenata nastala
    kvantizacijom i promjena izmjerena nakon rekonstrukcije signala.
    """
    received_A4 = pywt.wavedec(watermarked_signal, WAVELET, level=LEVEL)[0]
    return (state["watermarked_A4"] - state["original_A4"],
            received_A4 - state["original_A4"],
            "Promjena A4 koeficijenta")


def figures(ecg_signal, watermarked_signal, state, fs):
    """
    Slika 6 (DWT), 7 (BWT) ili 8 (QB): zadata promjena A4 koeficijenata
    (upisana pri umetanju) i ostvarena promjena (A4 koeficijenti signala sa
    vodenim žigom nakon ponovne dekompozicije minus originalni A4).
    """
    assigned = state["watermarked_A4"] - state["original_A4"]
    realized = pywt.wavedec(watermarked_signal, WAVELET, level=LEVEL)[0] \
        - state["original_A4"]
    plot_assigned_vs_realized(assigned, realized, NAME, FIGURE_NUMBER)
