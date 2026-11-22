# ============================================================
# BWT METODA - UMETANJE, EKSTRAKCIJA I SLIKE
# Biortogonalni talasić bior4.4, 4 nivoa, aditivno umetanje
# u aproksimacione koeficijente A4.
# ============================================================

import numpy as np
import pywt

from zajednicko import to_bits, plot_assigned_vs_realized

FIGURE_NUMBER = 7   # broj slike razlike koeficijenata u radu
NAME = "BWT"
WAVELET = "bior4.4"
LEVEL = 4
ALPHA = 0.6410   # kalibrisano na PSNR = 60 dB


def embed(ecg_signal, watermark, alpha=ALPHA, fs=1000):
    """Umetanje: A4_wm = A4 + alpha * w."""

    coeffs = pywt.wavedec(ecg_signal, WAVELET, level=LEVEL)

    # Originalni A4 koeficijenti - potrebni za neslijepu ekstrakciju
    original_A4 = coeffs[0].copy()

    # Vodeni žig prilagođen broju A4 koeficijenata
    reference = np.resize(watermark, original_A4.shape)

    watermarked_A4 = original_A4 + alpha * reference
    coeffs[0] = watermarked_A4

    watermarked_signal = pywt.waverec(coeffs, WAVELET)[:len(ecg_signal)]

    state = dict(
        name=NAME, binary=False, alpha=alpha,
        param=f"α = {alpha}",
        original_A4=original_A4,
        watermarked_A4=watermarked_A4,
        reference=reference,
        reference_bits=to_bits(reference),
        capacity=len(reference),
        element_label="Indeks A4 koeficijenta",
    )
    return watermarked_signal, state


def extract(signal, state):
    """Neslijepa ekstrakcija: w = (A4_napadnuti - A4_original) / alpha."""
    attacked_A4 = pywt.wavedec(signal, WAVELET, level=LEVEL)[0]
    return (attacked_A4 - state["original_A4"]) / state["alpha"]


def amplitude_data(watermarked_signal, state):
    """Umetnuti i ekstrahovani žig bez napada (za poređenje amplituda)."""
    return (state["reference"], extract(watermarked_signal, state),
            "Vrijednost vodenog žiga")


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
