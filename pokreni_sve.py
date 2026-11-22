# ============================================================
# GLAVNI PROGRAM
# Od fajla r01.csv generiše sve tabele i slike iz rada:
#   1. Učitavanje signala i preliminarna analiza (Slika 4, Slika 5a i 5b)
#   2. Umetanje vodenog žiga, metrike kvaliteta (Tabele 1 i 2, Slike 6-10)
#   3. Provjera kalibracije parametara na PSNR = 60 dB
#   4. Ekstrakcija bez napada (Tabela 3)
#   5. Amplituda umetnutog i ekstrahovanog žiga (Tabela 4, Slika 11)
#   6. Napadi, NC i BER (Tabele 5-10, Slika 12)
#   7. Dodatni eksperimenti za Stockwell metodu
#
# Pokretanje:  python pokreni_sve.py
# Rezultati:   rezultati/tabele (CSV) i rezultati/slike (PNG)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft

import zajednicko as z
import dwt
import bwt
import qb
import stft_metoda
import stockwell

METHODS = [dwt, bwt, qb, stft_metoda, stockwell]
PARAMETER_NAME = {"DWT": "α", "BWT": "α", "QB": "Δ", "STFT": "α", "ST": "α"}

z.ensure_dirs()


# ============================================================
# 1. UČITAVANJE SIGNALA I PRELIMINARNA ANALIZA
# ============================================================

ecg_signal = z.load_ecg()
watermark = z.generate_watermark(len(ecg_signal))
fs = z.FS
N = len(ecg_signal)
time_axis = np.arange(N) / fs

print("Broj uzoraka EKG signala:", N)
print("Frekvencija uzorkovanja:", fs, "Hz")
print("Trajanje signala:", N / fs, "s")
print("Dužina vodenog žiga:", len(watermark))
print(f"Minimum / maksimum: {ecg_signal.min():.3f} / {ecg_signal.max():.3f} µV")

# Slika 4: originalni EKG signal
z.plot_signal(time_axis, ecg_signal, "Originalni EKG signal",
              "Originalni EKG", "Slika_04_originalni_EKG.png")

# --- Raspodjela spektralne energije (poglavlje 6.3.7) ---
spectrum = np.fft.rfft(ecg_signal)
freqs = np.fft.rfftfreq(N, d=1 / fs)
power = np.abs(spectrum) ** 2
energy_rows = [
    {"Frekvencija do (Hz)": limit,
     "Udio energije (%)": round(100 * np.sum(power[freqs <= limit]) / np.sum(power), 2)}
    for limit in [5, 10, 20, 30, 40, 50, 100]
]
z.save_table(pd.DataFrame(energy_rows), "dodatno_spektralna_energija.csv",
             "RASPODJELA SPEKTRALNE ENERGIJE")

# --- Slika 5: vremensko-frekvencijski prikaz originalnog EKG-a (dB) ---
# (A) STFT: Hann prozor 512 uzoraka, preklapanje 256
# (B) Stockwell transformacija: redovi 0-50 Hz (izraz 60, stockwell.py)
F_MAX_PLOT = 50

f0, t0, Z0 = stft(ecg_signal, fs=fs, window="hann", nperseg=512, noverlap=256)
stft_db = 20 * np.log10(np.abs(Z0) + 1e-12)

st_freqs = np.fft.rfftfreq(N, d=1 / fs)
st_rows = np.where(st_freqs <= F_MAX_PLOT)[0]
S0 = stockwell.stockwell_transform(ecg_signal, rows=st_rows)
st_db = 20 * np.log10(np.abs(S0) + 1e-12)

fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

# Isti dinamički opseg (100 dB ispod maksimuma) za oba prikaza
panels = [
    (axes[0], t0, f0, stft_db, "gouraud",
     "STFT (Hann prozor, 512 uzoraka, preklapanje 50%)", "a"),
    (axes[1], time_axis, st_freqs[st_rows], st_db, "auto",
     "Stockwell transformacija", "b"),
]
for ax, x, y, values, shading, title, letter in panels:
    vmax = np.max(values)
    mesh = ax.pcolormesh(x, y, values, shading=shading, cmap="viridis",
                         vmin=vmax - 100, vmax=vmax, rasterized=True)
    ax.set_ylim(0, F_MAX_PLOT)
    ax.set_xlim(0, N / fs)
    ax.set_xlabel("Vrijeme (s)")
    ax.set_ylabel("Frekvencija (Hz)")
    ax.set_title(f"({letter}) {title}", fontsize=12)
    fig.colorbar(mesh, ax=ax, label="Amplituda (dB)")

z.save_figure("Slika_05_vf_prikaz_originalnog_EKG_STFT_ST.png", fig)
del S0, st_db

# --- Poređenje Hann i Hamming prozora (poglavlja 6.3.7 i 7.4), samo tabela ---
window_rows = []
for window_type in ["hann", "hamming"]:
    for nperseg in [256, 512, 1024]:
        f_w, t_w, Z_w = stft(ecg_signal, fs=fs, window=window_type,
                             nperseg=nperseg, noverlap=nperseg // 2)
        window_rows.append({
            "Prozor": window_type, "Dužina (uzoraka)": nperseg,
            "Trajanje (s)": nperseg / fs, "Hop (uzoraka)": nperseg // 2,
            "Razmak binova (Hz)": round(fs / nperseg, 6),
            "Frekvencijskih binova": Z_w.shape[0],
            "Vremenskih segmenata": Z_w.shape[1],
        })
z.save_table(pd.DataFrame(window_rows), "dodatno_poredjenje_prozora.csv",
             "POREĐENJE STFT PROZORA")


# ============================================================
# 2. UMETANJE VODENOG ŽIGA I METRIKE KVALITETA
# ============================================================

watermarked = {}
states = {}
quality_rows = []
parameter_rows = []

for method in METHODS:
    name = method.NAME
    print(f"\n==================== {name} ====================")

    signal_wm, state = method.embed(ecg_signal, watermark, fs=fs)
    watermarked[name] = signal_wm
    states[name] = state

    # Razlika transformacionih koeficijenata (Slike 6-10)
    method.figures(ecg_signal, signal_wm, state, fs)

    # Metrike kvaliteta (Tabela 2)
    mse, psnr, snr, prd = z.evaluate_performance(ecg_signal, signal_wm)
    quality_rows.append({
        "Metoda": name, "MSE": round(mse, 4), "PSNR (dB)": round(psnr, 2),
        "SNR (dB)": round(snr, 2), "PRD (%)": round(prd, 4),
    })

    # Parametri metode (Tabela 1)
    parameter_rows.append({
        "Metoda": name,
        "Broj umetnutih elemenata": state["capacity"],
        "Tip vodenog žiga": "Binarni" if state["binary"] else "Pseudo-slučajni",
        "Parametar umetanja": state["param"],
        "PSNR (dB)": round(psnr, 2),
        "Način umetanja": "QIM" if state["binary"] else "Aditivni",
        "Način ekstrakcije": "Slijepa" if state["binary"] else "Neslijepa",
        "Kapacitet (bit/s)": round(state["capacity"] / (N / fs), 1),
    })

z.save_table(pd.DataFrame(parameter_rows), "Tabela_01_parametri.csv",
             "TABELA 1 - PARAMETRI UMETANJA I EKSTRAKCIJE")

# Tabela 2: kvalitet EKG signala za svih pet metoda
z.save_table(pd.DataFrame(quality_rows)[["Metoda", "PSNR (dB)", "SNR (dB)", "MSE", "PRD (%)"]],
             "Tabela_02_kvalitet.csv",
             "TABELA 2 - KVALITET EKG SIGNALA NAKON UMETANJA")


# ============================================================
# 3. PROVJERA KALIBRACIJE (PSNR = 60 dB)
# Aditivne metode su linearne, pa MSE raste sa alpha^2:
#   alpha_60 = alpha * sqrt(MSE_cilj / MSE(alpha)).
# QB: greška QIM kvantizacije je približno ravnomjerna u [-Δ/2, Δ/2]
# (srednji kvadrat Δ^2/12), a A4 koeficijenti čine N_A4/N uzoraka, pa je
#   MSE ≈ (N_A4 / N) * Δ^2 / 12  =>  Δ_60 ≈ sqrt(12 * MSE_cilj * N / N_A4).
# ============================================================

mse_target = np.max(np.abs(ecg_signal)) ** 2 / 10 ** (z.TARGET_PSNR / 10)
calibration_rows = []

for method in METHODS:
    name = method.NAME
    mse_used = np.mean((watermarked[name] - ecg_signal) ** 2)
    psnr_used = 10 * np.log10(np.max(np.abs(ecg_signal)) ** 2 / mse_used)
    if name == "QB":
        used = states[name]["delta"]
        estimate = np.sqrt(12 * mse_target * N / states[name]["capacity"])
        note = "teorijska procjena Δ iz MSE ≈ (N_A4/N)·Δ²/12"
    else:
        used = states[name]["alpha"]
        estimate = used * np.sqrt(mse_target / mse_used)
        note = "α za tačno 60 dB (MSE ~ α²)"
    calibration_rows.append({
        "Metoda": name, "Parametar": PARAMETER_NAME[name],
        "Korišćena vrijednost": used,
        "PSNR pri korišćenoj vrijednosti (dB)": round(psnr_used, 4),
        "Vrijednost za PSNR = 60 dB": round(estimate, 4),
        "Napomena": note,
    })

print(f"\nCiljni MSE za PSNR = 60 dB: {mse_target:.6f}")
z.save_table(pd.DataFrame(calibration_rows), "dodatno_kalibracija.csv",
             "PROVJERA KALIBRACIJE PARAMETARA")


# ============================================================
# 4. EKSTRAKCIJA BEZ NAPADA (Tabela 3)
# ============================================================

no_attack_rows = []
for method in METHODS:
    name = method.NAME
    extracted = method.extract(watermarked[name], states[name])
    nc, ber, errors = z.evaluate_extraction(states[name], extracted)
    no_attack_rows.append({
        "Metoda": name, "NC": round(nc, 4), "BER": round(ber, 4),
        "Broj grešaka": len(errors),
        "Broj bitova": states[name]["capacity"],
        "Pozicije grešaka": ", ".join(map(str, errors)) if len(errors) else "-",
    })

z.save_table(pd.DataFrame(no_attack_rows), "Tabela_03_bez_napada.csv",
             "TABELA 3 - EKSTRAKCIJA BEZ NAPADA")


# ============================================================
# 5. AMPLITUDA UMETNUTOG I EKSTRAHOVANOG ŽIGA BEZ NAPADA
#    (Tabela 4, Slika 11)
# Pojačanje g = <w, w_ext> / <w, w>; 1 = sačuvana cijela amplituda.
# Kod QB se porede promjene A4 koeficijenata.
# ============================================================

amplitude_rows = []
N_SHOW = 60   # broj prikazanih elemenata po metodi

fig, axes = plt.subplots(len(METHODS), 1, figsize=(12, 16),
                         constrained_layout=True)

for ax, method in zip(axes, METHODS):
    name = method.NAME
    state = states[name]
    embedded, extracted, ylabel = method.amplitude_data(watermarked[name], state)
    embedded = np.asarray(embedded, float)
    extracted = np.asarray(extracted, float)
    gain = np.dot(embedded, extracted) / np.dot(embedded, embedded)

    amplitude_rows.append({
        "Metoda": name, "Parametar": state["param"],
        "Broj elemenata": len(embedded),
        "Pojačanje g": round(gain, 4),
        "Odnos RMS": round(np.sqrt(np.mean(extracted ** 2) / np.mean(embedded ** 2)), 4),
        "max|umetnuti|": round(np.max(np.abs(embedded)), 4),
        "max|ekstrahovani|": round(np.max(np.abs(extracted)), 4),
    })

    idx = np.arange(1, len(embedded) + 1)[:N_SHOW]
    e, x = embedded[:N_SHOW], extracted[:N_SHOW]
    ax.plot(idx, e, "o-", color="tab:blue", linewidth=1.8, markersize=4,
            label="Umetnuti vodeni žig")
    ax.plot(idx, x, "s--", color="tab:red", linewidth=1.2, markersize=3,
            label="Ekstrahovani vodeni žig (bez napada)")
    lim = 1.15 * max(np.max(np.abs(e)), np.max(np.abs(x)))
    ax.set_ylim(-lim, lim)
    ax.set_xlim(idx[0] - 0.5, idx[-1] + 0.5)
    ax.set_title(f"{name}  ({state['param']})", loc="left",
                 fontsize=11, fontweight="bold")
    ax.set_title(f"pojačanje g = {gain:.3f}", loc="right", fontsize=10)
    ax.set_xlabel(state["element_label"], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=2,
           fontsize=11, frameon=False)
z.save_figure("Slika_11_amplituda_sve_metode.png", fig)

z.save_table(pd.DataFrame(amplitude_rows), "Tabela_04_amplituda.csv",
             "TABELA 4 - AMPLITUDA UMETNUTOG I EKSTRAHOVANOG ŽIGA")


# ============================================================
# 6. NAPADI: EKSTRAKCIJA, NC I BER (Tabele 5-10, Slika 12a-f)
# ============================================================

attacks = z.build_attacks(N)
all_results = []
ber_by_attack = {}   # za Sliku 12

for attack in attacks:
    print(f"\n==================== NAPAD: {attack['title']} ====================")
    if "positions" in attack:
        for level, start in attack["positions"].items():
            print(f"Nuliranje {int(round(level * 100))}%: početni uzorak {start}, "
                  f"dužina {int(N * level)}")

    table_rows = []
    ber_matrix = {}

    for method in METHODS:
        name = method.NAME
        state = states[name]
        nc_row = {"Metoda": name, "Mjera": "NC"}
        ber_row = {"Metoda": "", "Mjera": "BER"}
        ber_matrix[name] = []

        for level in attack["levels"]:
            # Napad na signal sa vodenim žigom
            attacked = attack["apply"](watermarked[name], level)

            # Ekstrakcija iz napadnutog signala
            extracted = method.extract(attacked, state)

            # NC, BER i pozicije pogrešnih bitova
            nc, ber, errors = z.evaluate_extraction(state, extracted)

            column = attack["label"](level)
            nc_row[column] = round(nc, 4)
            ber_row[column] = round(ber, 4)
            ber_matrix[name].append(ber)
            row = {"Napad": attack["title"], "Metoda": name, "Nivo": level,
                   "NC": nc, "BER": ber, "Broj grešaka": len(errors),
                   "Pozicije grešaka": ", ".join(map(str, errors))}
            if state["binary"]:
                # QB: broj A4 koeficijenata tačno na granici odluke
                row["Koeficijenata na granici odluke"] = state["last_ties"]
            if not state["binary"]:
                deviation = np.abs(np.asarray(extracted) - state["reference"])
                row["Najveće odstupanje"] = np.max(deviation)
                row["Pozicija najvećeg odstupanja"] = int(np.argmax(deviation)) + 1
            all_results.append(row)
            print(f"{name:<5} | {column:<15} | NC = {nc:.6f} | BER = {ber:.6f}"
                  + (f" | greške: {row['Pozicije grešaka']}"
                     if 0 < len(errors) <= 10 else "")
                  + (f" | na granici odluke: {row['Koeficijenata na granici odluke']}"
                     f" od {state['capacity']}"
                     if row.get("Koeficijenata na granici odluke") else ""))

        table_rows.extend([nc_row, ber_row])

    z.save_table(pd.DataFrame(table_rows),
                 f"Tabela_{attack['table']:02d}_{attack['key']}.csv",
                 f"TABELA {attack['table']} - {attack['title'].upper()}")

    ber_by_attack[attack["key"]] = ber_matrix


# --- Slika 12: poređenje BER vrijednosti svih metoda za svih šest napada ---
STYLE = {"DWT": ("o", "#2a7ad4"), "BWT": ("s", "#ef6a3c"),
         "QB": ("D", "#1ab385"), "STFT": ("^", "#f09b1a"),
         "ST": ("v", "#e8689c")}

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for panel, ax, attack in zip("abcdef", axes.flat, attacks):
    scale = attack.get("x_scale", 1)
    x_values = np.array(attack["levels"], dtype=float) * scale
    order = np.argsort(x_values)
    for method in METHODS:
        name = method.NAME
        marker, color = STYLE[name]
        ax.plot(x_values[order], np.array(ber_by_attack[attack["key"]][name])[order],
                marker=marker, color=color, linewidth=2,
                label=name)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x_values[order])
    ax.set_xticklabels([f"{v:g}" if scale != 1 or v >= 3 else f"{v:.2f}"
                        for v in x_values[order]])
    if attack.get("x_reverse"):
        ax.invert_xaxis()
    max_ber = max(max(v) for v in ber_by_attack[attack["key"]].values())
    ax.set_ylim(-0.02, max(0.6, max_ber + 0.05))
    ax.set_title(f"({panel}) {attack['title']}", fontsize=12)
    ax.set_xlabel(attack["xlabel"])
    ax.set_ylabel("BER")
    ax.grid(True, alpha=0.3)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False,
           fontsize=11)
z.save_figure("Slika_12_BER_svi_napadi.png", fig)

results_df = pd.DataFrame(all_results)
results_df.to_csv(z.TAB_DIR / "svi_napadi_detaljno.csv", index=False,
                  encoding="utf-8-sig")


# ============================================================
# 7. DODATNI EKSPERIMENTI - STOCKWELL (poglavlje 7.5)
# ============================================================

print("\n==================== DODATNI EKSPERIMENTI - ST ====================")
extra = stockwell.additional_experiments(ecg_signal, watermark, fs=fs)
z.save_table(extra.round(6), "dodatno_ST_eksperimenti.csv",
             "STOCKWELL NASPRAM DFT I ŠIRINA GAUSSOVOG PROZORA")

print(f"\nGotovo. Tabele: {z.TAB_DIR}")
print(f"Slike:  {z.FIG_DIR}")
