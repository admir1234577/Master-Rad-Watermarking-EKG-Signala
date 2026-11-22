# Digitalni vodeni žig u EKG signalu – DWT, BWT, QB, STFT i Stockwell

Programski kod uz master rad. Od fajla `r01.csv` generiše sve tabele
i slike iz rada.

## Podaci i uslovi korišćenja

Fajl `r01.csv` sadrži prvih 10 sekundi zapisa r01 iz baze
*Abdominal and Direct Fetal ECG Database* (PhysioNet, oznaka `adfecgdb`,
verzija 1.0.0), preuzetih u CSV formatu sa platforme Kaggle
(Abdominal & Direct Fetal ECG Database (CSV format),
https://doi.org/10.34740/KAGGLE/DSV/4776543).
Koristi se kanal `Direct_1` (10.000 uzoraka, 1000 Hz, µV).

Originalna baza: https://physionet.org/content/adfecgdb/1.0.0/

Licenca: Open Data Commons Attribution License v1.0 (ODC-By 1.0),
https://physionet.org/content/adfecgdb/view-license/1.0.0/
Podaci se mogu slobodno koristiti i dijeliti uz navođenje izvora.

Pri korišćenju podataka potrebno je citirati originalnu publikaciju:

> Jezewski J, Matonia A, Kupka T, Roj D, Czabanski R. Determination of the
> fetal heart rate from abdominal signals: evaluation of beat-to-beat accuracy
> in relation to the direct fetal electrocardiogram. Biomedical
> Engineering/Biomedizinische Technik. 2012;57(5):383–394.
> doi:10.1515/bmt-2011-0130

i standardni citat za PhysioNet:

> Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A,
> Mark RG, Heldt T. PhysioNet as a global platform for biomedical research.
> Nature Health. 2026. https://doi.org/10.1038/s44360-026-00096-z

Fajl `r01.csv` stavite u isti folder u kojem su `.py` fajlovi.

## Instalacija

Potreban je Python 3.9 ili noviji.

```
pip install -r requirements.txt
```

## Pokretanje

```
python pokreni_sve.py
```

Izvršavanje traje približno jedan minut. Stockwell transformacija
cijelog signala zahtijeva oko 2,5 GB radne memorije.

Rezultati se snimaju u:

- `rezultati/tabele/` – sve tabele kao CSV fajlovi
- `rezultati/slike/` – sve slike kao PNG fajlovi (600 dpi)

Za prikaz slika na ekranu postavite `SHOW_FIGURES = True` u `zajednicko.py`.

## Struktura koda

| Fajl | Sadržaj |
| --- | --- |
| `zajednicko.py` | učitavanje signala, generisanje vodenog žiga (seed 42), metrike MSE, PSNR, SNR, PRD, NC i BER, šest napada |
| `dwt.py` | DWT metoda (db4, 4 nivoa, α = 0,6597): umetanje, ekstrakcija, slike |
| `bwt.py` | BWT metoda (bior4.4, 4 nivoa, α = 0,6410): umetanje, ekstrakcija, slike |
| `qb.py` | QB metoda (QIM kvantizacija, db4, 4 nivoa, Δ = 2,2497): umetanje, slijepa ekstrakcija, slike |
| `stft_metoda.py` | STFT metoda (Hann 512/256, 3,9–15,7 Hz, α = 0,0614, fazni faktor (−1)^k): umetanje, ekstrakcija, slike |
| `stockwell.py` | Stockwell metoda (3,9–15,7 Hz, α = 0,0114): umetanje, ekstrakcija, slike, dodatni eksperimenti |
| `pokreni_sve.py` | glavni program: pokreće sve metode i napade i snima tabele i slike |

Svaki modul metode ima iste funkcije:

- `embed(ecg_signal, watermark, ...)` vraća signal sa vodenim žigom i
  rječnik `state` sa svim veličinama potrebnim za ekstrakciju
  (originalni koeficijenti, alpha, odabrani binovi, referentni žig);
- `extract(signal, state)` vraća ekstrahovani vodeni žig;
- `figures(...)` crta razliku transformacionih koeficijenata.

## Tabele i slike iz rada

Program pravi samo slike koje se nalaze u radu (Slike 4–12). Fajlovi
čiji naziv počinje sa `Slika_` ili `Tabela_` nose isti broj kao u radu.
Tabele koje počinju sa `dodatno_` sadrže vrijednosti koje se navode u
tekstu, ali nisu numerisane kao tabele.

| U radu | Fajl u `rezultati/` |
| --- | --- |
| Slika 4 – originalni EKG signal | `slike/Slika_04_originalni_EKG.png` |
| Slika 5 – vremensko-frekvencijski prikaz originalnog EKG-a: (a) STFT, (b) Stockwell | `slike/Slika_05_vf_prikaz_originalnog_EKG_STFT_ST.png` |
| Slike 6–8 – zadata i ostvarena promjena A4 koeficijenata (DWT, BWT, QB) | `slike/Slika_06_…_DWT.png`, `Slika_07_…_BWT.png`, `Slika_08_…_QB.png` |
| Slika 9 – zadata i ostvarena razlika STFT koeficijenata | `slike/Slika_09_razlika_STFT_koeficijenata.png` |
| Slika 10 – zadata i ostvarena razlika Stockwell koeficijenata (0–30 Hz) | `slike/Slika_10_razlika_ST_koeficijenata.png` |
| Slika 11 – umetnuti i ekstrahovani žig bez napada | `slike/Slika_11_amplituda_sve_metode.png` |
| Slika 12 – poređenje BER vrijednosti, paneli (a)–(f) za šest napada | `slike/Slika_12_BER_svi_napadi.png` |
| Tabela 1 – parametri metoda | `tabele/Tabela_01_parametri.csv` |
| Tabela 2 – kvalitet EKG signala (svih pet metoda) | `tabele/Tabela_02_kvalitet.csv` |
| Tabela 3 – ekstrakcija bez napada | `tabele/Tabela_03_bez_napada.csv` |
| Tabela 4 – amplituda ekstrahovanog i umetnutog žiga | `tabele/Tabela_04_amplituda.csv` |
| Tabela 5 – dodavanje šuma | `tabele/Tabela_05_awgn.csv` |
| Tabela 6 – nuliranje signala | `tabele/Tabela_06_nuliranje.csv` |
| Tabela 7 – promjena intervala odabiranja | `tabele/Tabela_07_resampling.csv` |
| Tabela 8 – medijansko filtriranje | `tabele/Tabela_08_medijan.csv` |
| Tabela 9 – skaliranje amplitude | `tabele/Tabela_09_skaliranje.csv` |
| Tabela 10 – kompresija | `tabele/Tabela_10_kompresija.csv` |
| Raspodjela spektralne energije (99,46 % do 50 Hz) | `tabele/dodatno_spektralna_energija.csv` |
| Poređenje Hann i Hamming prozora 256/512/1024 | `tabele/dodatno_poredjenje_prozora.csv` |
| Stockwell naspram DFT, širina Gaussovog prozora (7.5) | `tabele/dodatno_ST_eksperimenti.csv` |
| Provjera kalibracije na PSNR = 60 dB | `tabele/dodatno_kalibracija.csv` |
| NC, BER i pozicije pogrešnih bitova za svaki napad i nivo | `tabele/svi_napadi_detaljno.csv` |

## Napadi

Isti napadi i isti parametri koriste se za svih pet metoda:

| Napad | Nivoi | Ponovljivost |
| --- | --- | --- |
| Aditivni bijeli Gaussov šum | SNR 30, 25, 20, 15, 10 dB | isti šum, seed 12345 |
| Nuliranje segmenata | 30, 20, 10, 5 % signala | iste pozicije, seed 12345, zajednički centar |
| Ponovno odabiranje | faktor 1,5; 1,25; 0,75; 0,5 | vraćanje na prvobitnu dužinu |
| Medijansko filtriranje | prozor 9, 7, 5, 3 | – |
| Skaliranje amplitude | faktor 1,5; 1,25; 0,75; 0,5 | – |
| Kompresija (DCT) | zadržano 90, 75, 50, 25 % koeficijenata | – |
