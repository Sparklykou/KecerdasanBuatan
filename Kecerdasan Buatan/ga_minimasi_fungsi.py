# =============================================================================
# IMPLEMENTASI ALGORITMA GENETIKA – Minimasi f(x1, x2)
# f(x1,x2) = −( sin(x1)·cos(x2)·tan(x1+x2) + ½·exp(1 − √(x2²)) )
# Domain: −10 ≤ x1 ≤ 10, −10 ≤ x2 ≤ 10
# =============================================================================

import math    # library matematika (sin, cos, tan, exp, sqrt)
import random  # library untuk angka acak

# ── PARAMETER GA ─────────────────────────────────────────
POP    = 50       # jumlah individu dalam populasi
L      = 32       # panjang kromosom (total bit)
LG     = 16       # jumlah bit per variabel (x1 dan x2)
DMIN   = -10.0    # batas bawah domain
DMAX   = 10.0     # batas atas domain
PC     = 0.8      # probabilitas crossover
PM     = 0.01     # probabilitas mutasi
MAXGEN = 500      # maksimum generasi
KONV   = 50       # batas konvergensi (tidak ada perubahan)
K      = 2        # jumlah peserta seleksi turnamen

# ── DECODE ───────────────────────────────────────────────
def decode(bits):
    # Mengubah 16 bit biner menjadi angka real dalam rentang [DMIN, DMAX]

    # Konversi biner → integer
    n = sum(bits[i] * (2 ** i) for i in range(LG))
    # bits[i] = nilai bit (0/1)
    # 2**i    = bobot biner

    # Mapping integer ke domain real
    return DMIN + n * (DMAX - DMIN) / ((2 ** LG) - 1)


def decode_kromosom(k):
    # Memisahkan kromosom 32 bit menjadi 2 variabel
    # 16 bit pertama → x1
    # 16 bit kedua   → x2
    return decode(k[:LG]), decode(k[LG:])


# ── FUNGSI OBJEKTIF & FITNESS ───────────────────────────
def f_obj(x1, x2):
    # Fungsi yang ingin diminimasi
    try:
        t = math.tan(x1 + x2)  # hitung tan(x1+x2)

        # Hindari nilai tak hingga (tan bisa sangat besar)
        if abs(t) > 1e9:
            return float('inf')

        # Hitung fungsi sesuai soal
        return -(math.sin(x1) * math.cos(x2) * t
                 + 0.5 * math.exp(1 - math.sqrt(x2**2)))

    except:
        # Jika error numerik
        return float('inf')


def fitness(k):
    # Mengubah minimasi → maksimasi

    # Decode kromosom menjadi x1 dan x2
    f = f_obj(*decode_kromosom(k))

    # Jika nilai tidak valid
    if f == float('inf'):
        return float('-inf')

    # Fitness = negatif dari fungsi (karena minimasi)
    return -f


# ── INISIALISASI POPULASI ───────────────────────────────
def init_pop():
    # Membuat populasi awal secara acak
    return [[random.randint(0, 1) for _ in range(L)] for _ in range(POP)]


# ── SELEKSI (TOURNAMENT) ────────────────────────────────
def seleksi(pop, fit):
    # Memilih K individu secara acak
    peserta = random.sample(range(len(pop)), K)

    # Mengambil individu dengan fitness terbaik
    return pop[max(peserta, key=lambda i: fit[i])][:]


# ── CROSSOVER ───────────────────────────────────────────
def crossover(p1, p2):
    # Menentukan titik potong acak
    t = random.randint(1, L - 1)

    # Menggabungkan parent
    a1 = p1[:t] + p2[t:]  # anak 1
    a2 = p2[:t] + p1[t:]  # anak 2

    return a1, a2


# ── MUTASI ──────────────────────────────────────────────
def mutasi(k):
    # Setiap bit punya peluang PM untuk berubah
    return [1 - b if random.random() < PM else b for b in k]


# ── MEMBUAT OFFSPRING ───────────────────────────────────
def buat_offspring(pop, fit):
    offs = []

    # Membuat individu baru sampai jumlah cukup
    while len(offs) < POP - 1:

        # Seleksi parent
        p1 = seleksi(pop, fit)
        p2 = seleksi(pop, fit)

        # Crossover dengan probabilitas PC
        if random.random() < PC:
            a1, a2 = crossover(p1, p2)
        else:
            a1, a2 = p1[:], p2[:]

        # Mutasi
        offs.append(mutasi(a1))

        if len(offs) < POP - 1:
            offs.append(mutasi(a2))

    return offs


# ── PROGRAM UTAMA GA ────────────────────────────────────
def jalankan_ga():

    # Inisialisasi populasi
    pop = init_pop()

    # Hitung fitness awal
    fit = [fitness(k) for k in pop]

    # Simpan solusi terbaik
    best_fit = max(fit)
    best_k   = pop[fit.index(best_fit)][:]

    no_improv = 0  # counter stagnasi

    # Loop generasi
    for gen in range(1, MAXGEN + 1):

        # Elitism (simpan individu terbaik)
        elite = pop[fit.index(max(fit))][:]

        # Buat populasi baru
        pop = [elite] + buat_offspring(pop, fit)

        # Evaluasi ulang
        fit = [fitness(k) for k in pop]

        gen_best = max(fit)

        # Update jika ada solusi lebih baik
        if gen_best > best_fit:
            best_fit = gen_best
            best_k   = pop[fit.index(gen_best)][:]
            no_improv = 0
        else:
            no_improv += 1

        # Berhenti jika konvergen
        if no_improv >= KONV:
            break

    # Decode hasil terbaik
    x1, x2 = decode_kromosom(best_k)

    print("Hasil Akhir:")
    print("x1 =", x1)
    print("x2 =", x2)
    print("f(x1,x2) =", f_obj(x1, x2))


# ── ENTRY POINT ─────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)  # agar hasil bisa direproduksi
    jalankan_ga()
