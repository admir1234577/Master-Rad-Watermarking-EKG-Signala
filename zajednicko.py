# ============================================================
# ZAJEDNIČKE FUNKCIJE ZA SVIH PET METODA
# - učitavanje EKG signala i generisanje vodenog žiga
# - metrike kvaliteta (MSE, PSNR, SNR, PRD) i robusnosti (NC, BER)
# - šest napada, identičnih za svih pet metoda
# - snimanje tabela i slika
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from scipy.signal import resample, medfilt
from scipy.fft import dct, idct

SHOW_FIGURES = False   # True = slike se prikazuju i na ekranu
if not SHOW_FIGURES:
    matplotlib.use("Agg")   # slike se samo snimaju u fajlove
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Putanje (relativne u odnosu na folder sa kodom)
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "r01.csv"
OUT_DIR = BASE_DIR / "rezultati"
FIG_DIR = OUT_DIR / "slike"
TAB_DIR = OUT_DIR / "tabele"

FIG_DPI = 600

# ------------------------------------------------------------
# Opšti parametri eksperimenta
# ------------------------------------------------------------

FS = 1000                 # frekvencija uzorkovanja (Hz)
ECG_CHANNEL = "Direct_1"  # kanal direktnog fetalnog EKG-a
WATERMARK_SEED = 42       # seed pseudo-slučajnog vodenog žiga
NOISE_SEED = 12345        # seed šuma za AWGN napad
CROP_SEED = 12345         # seed položaja nuliranih segmenata
TARGET_PSNR = 60.0        # ciljni PSNR pri kalibraciji (dB)


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Učitavanje EKG signala i vodeni žig
# ------------------------------------------------------------

def load_ecg(path=DATA_PATH):
    """Učitava kanal Direct_1 iz r01.csv (druga linija su jedinice)."""
    data = pd.read_csv(path, skiprows=[1], quotechar="'")
    return data[ECG_CHANNEL].values.astype(float)


def generate_watermark(signal_length):
    """Pseudo-slučajni vodeni žig, dužine N/8, seed 42."""
    np.random.seed(WATERMARK_SEED)
    return np.random.randn(signal_length // 8)


# ------------------------------------------------------------
# Metrike kvaliteta
# ------------------------------------------------------------

def evaluate_performance(original_signal, watermarked_signal):
    """MSE, PSNR (dB), SNR (dB) i PRD (%)."""
    error = original_signal - watermarked_signal
    mse = np.mean(error ** 2)
    psnr = 10 * np.log10(np.max(np.abs(original_signal)) ** 2 / mse)
    snr = 10 * np.log10(np.mean(original_signal ** 2) / mse)
    prd = 100 * np.sqrt(np.sum(error ** 2) / np.sum(original_signal ** 2))
    return mse, psnr, snr, prd


# ------------------------------------------------------------
# Metrike robusnosti
# ------------------------------------------------------------

def calculate_nc(reference, extracted):
    """Normalizovana korelacija (NC), sa uklonjenom srednjom vrijednošću."""
    reference = np.asarray(reference, dtype=float)
    extracted = np.asarray(extracted, dtype=float)
    reference = reference - np.mean(reference)
    extracted = extracted - np.mean(extracted)
    denominator = np.linalg.norm(reference) * np.linalg.norm(extracted)
    if denominator == 0:
        return np.nan
    return np.dot(reference, extracted) / denominator


def calculate_ber(reference_bits, extracted_bits):
    """Stopa greške bitova (BER)."""
    return np.mean(np.asarray(reference_bits) != np.asarray(extracted_bits))


def to_bits(values):
    """Realni vodeni žig -> bitovi (1 za vrijednosti >= 0)."""
    return (np.asarray(values) >= 0).astype(int)


def evaluate_extraction(state, extracted):
    """
    NC i BER za ekstrahovani vodeni žig.
    Kod QB metode ekstrakcija daje bitove, pa se NC računa nad bitovima;
    kod ostalih metoda NC se računa nad realnim vrijednostima.
    """
    if state["binary"]:
        extracted_bits = np.asarray(extracted).astype(int)
        nc = calculate_nc(state["reference_bits"], extracted_bits)
    else:
        extracted_bits = to_bits(extracted)
        nc = calculate_nc(state["reference"], extracted)
    ber = calculate_ber(state["reference_bits"], extracted_bits)
    errors = np.where(state["reference_bits"] != extracted_bits)[0] + 1
    return nc, ber, errors


# ------------------------------------------------------------
# Napadi (identični za svih pet metoda)
# ------------------------------------------------------------

def add_awgn(signal, snr_db, standard_noise):
    """Aditivni bijeli Gaussov šum zadatog SNR-a."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return signal + np.sqrt(noise_power) * standard_noise


def crop_signal(signal, start, crop_length):
    """Nuliranje segmenta signala."""
    cropped = np.copy(signal)
    cropped[start:start + crop_length] = 0
    return cropped


def resampling_attack(signal, factor):
    """Ponovno odabiranje sa faktorom i vraćanje na prvobitnu dužinu."""
    n = len(signal)
    return resample(resample(signal, int(n * factor)), n)


def median_filter_attack(signal, kernel_size):
    """Medijansko filtriranje."""
    return medfilt(signal, kernel_size=kernel_size)


def amplitude_scaling_attack(signal, scale_factor):
    """Skaliranje amplitude."""
    return signal * scale_factor


def compression_attack(signal, keep_ratio):
    """Kompresija sa gubicima: zadržava se dio DCT koeficijenata najveće magnitude."""
    coefficients = dct(signal, norm="ortho")
    keep_count = max(1, int(len(coefficients) * keep_ratio))
    keep_indices = np.argsort(np.abs(coefficients))[-keep_count:]
    compressed = np.zeros_like(coefficients)
    compressed[keep_indices] = coefficients[keep_indices]
    return idct(compressed, norm="ortho")


def build_attacks(signal_length):
    """
    Definicije šest napada. Nivoi su poredani kao kolone u tabelama 5-10.
    """
    # Isti standardni šum za sve metode i sve SNR nivoe
    rng_noise = np.random.default_rng(NOISE_SEED)
    standard_noise = rng_noise.standard_normal(signal_length)

    # Iste pozicije nuliranih segmenata za sve metode (zajednički centar)
    crop_levels = [0.30, 0.20, 0.10, 0.05]
    rng_crop = np.random.default_rng(CROP_SEED)
    max_crop_length = int(signal_length * max(crop_levels))
    max_start = int(rng_crop.integers(0, signal_length - max_crop_length + 1))
    crop_center = max_start + max_crop_length // 2
    crop_positions = {
        c: crop_center - int(signal_length * c) // 2 for c in crop_levels
    }

    return [
        dict(key="awgn", table=5, xlabel="SNR (dB)",
             title="Dodavanje šuma (AWGN)",
             levels=[30, 25, 20, 15, 10],
             label=lambda v: f"SNR = {v} dB",
             apply=lambda s, v: add_awgn(s, v, standard_noise)),

        dict(key="nuliranje", table=6, xlabel="Nulirani segmenti (%)", x_scale=100,
             title="Nuliranje segmenata",
             levels=crop_levels,
             label=lambda v: f"{int(round(v * 100))}%",
             apply=lambda s, v: crop_signal(
                 s, crop_positions[v], int(len(s) * v)),
             positions=crop_positions),

        dict(key="resampling", table=7, xlabel="Faktor ponovnog odabiranja",
             title="Promjena intervala odabiranja",
             levels=[1.5, 1.25, 0.75, 0.5],
             label=lambda v: f"faktor = {v}",
             apply=resampling_attack),

        dict(key="medijan", table=8, xlabel="Veličina prozora filtera",
             title="Medijansko filtriranje",
             levels=[9, 7, 5, 3],
             label=lambda v: f"kernel = {v}",
             apply=median_filter_attack),

        dict(key="skaliranje", table=9, xlabel="Faktor skaliranja amplitude",
             title="Skaliranje amplitude",
             levels=[1.5, 1.25, 0.75, 0.5],
             label=lambda v: f"faktor = {v}",
             apply=amplitude_scaling_attack),

        dict(key="kompresija", table=10, xlabel="Zadržani DCT koeficijenti (%)", x_scale=100, x_reverse=True,
             title="DCT kompresija",
             levels=[0.90, 0.75, 0.50, 0.25],
             label=lambda v: f"{int(round(v * 100))}%",
             apply=compression_attack),
    ]


# ------------------------------------------------------------
# Snimanje slika i tabela
# ------------------------------------------------------------

def save_figure(file_name, fig=None):
    fig = fig or plt.gcf()
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(FIG_DIR / file_name, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def save_table(df, file_name, title=None):
    df.to_csv(TAB_DIR / file_name, index=False, encoding="utf-8-sig")
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    print(df.to_string(index=False))


def plot_signal(time_axis, signal, title, label, file_name):
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, signal, color="blue", label=label)
    plt.title(title)
    plt.xlabel("Vrijeme (s)")
    plt.ylabel("Amplituda (µV)")
    plt.legend()
    save_figure(file_name)


def plot_assigned_vs_realized(assigned, realized, method_name, figure_number,
                              n_edge=10):
    """
    Slike 6-8: zadata promjena A4 koeficijenata (upisana pri umetanju) i
    ostvarena promjena (A4 koeficijenti signala sa vodenim žigom nakon
    ponovne dekompozicije, minus originalni A4). Indeksi počinju od 1.
    """
    assigned = np.asarray(assigned, dtype=float)
    realized = np.asarray(realized, dtype=float)
    difference = realized - assigned
    idx = np.arange(1, len(assigned) + 1)

    middle = slice(n_edge, len(assigned) - n_edge)
    largest = np.argsort(np.abs(difference))[::-1][:6] + 1
    gain = np.dot(assigned, realized) / np.dot(assigned, assigned)

    print(f"\n{method_name} - zadata i ostvarena promjena A4 koeficijenata")
    print(f"Minimalna zadata promjena A4:             {np.min(assigned):.6f}")
    print(f"Maksimalna zadata promjena A4:            {np.max(assigned):.6f}")
    print(f"Maks. |zadata promjena|:                  {np.max(np.abs(assigned)):.6f}")
    print(f"Maks. |ostvarena promjena|:               {np.max(np.abs(realized)):.6f}")
    print(f"Maks. |ostvarena - zadata| (cijeli niz):  {np.max(np.abs(difference)):.6f}")
    print(f"Maks. |ostvarena - zadata| (indeksi {n_edge + 1}-{len(assigned) - n_edge}): "
          f"{np.max(np.abs(difference[middle])):.2e}")
    print(f"Pozicije najvećih razlika:                {', '.join(map(str, sorted(largest)))}")
    print(f"Pojačanje (projekcija):                   {gain:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             constrained_layout=True)
    axes[0].plot(idx, assigned, color="blue", linewidth=1.2, label="Zadata promjena")
    axes[0].plot(idx, realized, "--", color="red", linewidth=1.0,
                 label="Ostvarena promjena")
    axes[0].set_title(f"(a) Zadata i ostvarena promjena A4 koeficijenata - {method_name}")
    axes[0].set_ylabel("Promjena koeficijenta")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(idx, difference, color="black", linewidth=1.0)
    axes[1].set_title("(b) Razlika ostvarene i zadate promjene")
    axes[1].set_xlabel("Indeks A4 koeficijenta")
    axes[1].set_ylabel("Razlika")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1, len(assigned))
    # Oznake na x-osi počinju od 1 (kao pozicije grešaka u Tabeli 3)
    ticks = [1] + list(range(100, len(assigned) - 50, 100)) + [len(assigned)]
    axes[1].set_xticks(ticks)

    save_figure(f"Slika_{figure_number:02d}_razlika_A4_koeficijenata_{method_name}.png", fig)

