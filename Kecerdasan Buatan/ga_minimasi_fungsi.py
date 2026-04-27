# =============================================================================
# IMPLEMENTASI ALGORITMA GENETIKA – Minimasi f(x1, x2)
# f(x1,x2) = −( sin(x1)·cos(x2)·tan(x1+x2) + ½·exp(1 − √(x2²)) )
# Domain: −10 ≤ x1 ≤ 10, −10 ≤ x2 ≤ 10
# =============================================================================

import math    # untuk sin, cos, tan, exp, sqrt
import random  # untuk pembangkitan bilangan acak

# ── PARAMETER GA ──────────────────────────────────────────────────────────────
POP    = 50       # ukuran populasi
L      = 40       # panjang kromosom (bit)
LG     = 20       # bit per variabel (x1 atau x2)
DMIN   = -10.0    # batas bawah domain
DMAX   = 10.0     # batas atas domain
PC     = 0.8      # probabilitas crossover
PM     = 0.05     # probabilitas mutasi per-bit
MAXGEN = 500      # batas maksimum generasi
KONV   = 50       # batas generasi tanpa perubahan (konvergensi)
K      = 3        # ukuran turnamen seleksi

# ── DECODE ────────────────────────────────────────────────────────────────────
def decode(bits):
    # ubah 20 bit biner → nilai real dalam [DMIN, DMAX]
    n = sum(bits[i] * (2 ** i) for i in range(LG))    # konversi biner ke integer
    return DMIN + n * (DMAX - DMIN) / ((2 ** LG) - 1) # petakan ke domain

def decode_kromosom(k):
    # pisah kromosom 40 bit → x1 (bit 0-19) dan x2 (bit 20-39)
    return decode(k[:LG]), decode(k[LG:])

# ── FUNGSI OBJEKTIF & FITNESS ─────────────────────────────────────────────────
def f_obj(x1, x2):
    # hitung nilai f(x1,x2) yang ingin diminimasi
    try:
        t = math.tan(x1 + x2)                           # hitung tan(x1+x2)
        if abs(t) > 1e9: return float('inf')             # hindari singularitas tan
        return -(math.sin(x1) * math.cos(x2) * t        # suku trigonometri
                 + 0.5 * math.exp(1 - math.sqrt(x2**2))) # suku eksponensial
    except: return float('inf')                          # tangkap error numerik

def fitness(k):
    # fitness = −f agar minimasi f setara dengan maksimasi fitness
    f = f_obj(*decode_kromosom(k))
    return float('-inf') if f == float('inf') else -f

# ── INISIALISASI POPULASI ─────────────────────────────────────────────────────
def init_pop():
    # buat POP kromosom acak, masing-masing L bit (0 atau 1)
    return [[random.randint(0, 1) for _ in range(L)] for _ in range(POP)]

# ── SELEKSI TURNAMEN ──────────────────────────────────────────────────────────
def seleksi(pop, fit):
    # pilih K kromosom acak, kembalikan yang fitness-nya tertinggi
    peserta = random.sample(range(len(pop)), K)
    return pop[max(peserta, key=lambda i: fit[i])][:]

# ── 2-POINT CROSSOVER ─────────────────────────────────────────────────────────
def crossover(p1, p2):
    # tentukan dua titik potong acak, tukar segmen tengah antar induk
    t1, t2 = sorted(random.sample(range(1, L), 2))
    a1 = p1[:t1] + p2[t1:t2] + p1[t2:]  # anak 1: kiri&kanan p1, tengah p2
    a2 = p2[:t1] + p1[t1:t2] + p2[t2:]  # anak 2: kiri&kanan p2, tengah p1
    return a1, a2

# ── BIT-FLIP MUTATION ─────────────────────────────────────────────────────────
def mutasi(k):
    # tiap bit punya PM kemungkinan di-flip (0→1 atau 1→0)
    return [1 - b if random.random() < PM else b for b in k]

# ── BUAT OFFSPRING ────────────────────────────────────────────────────────────
def buat_offspring(pop, fit):
    offs = []
    while len(offs) < POP - 1:                           # sisakan 1 slot untuk elite
        p1, p2 = seleksi(pop, fit), seleksi(pop, fit)    # pilih dua induk
        a1, a2 = crossover(p1, p2) if random.random() < PC else (p1[:], p2[:])
        offs.append(mutasi(a1))                           # mutasi anak 1
        if len(offs) < POP - 1: offs.append(mutasi(a2))  # mutasi anak 2 jika slot ada
    return offs

# ── ALGORITMA GENETIKA UTAMA ──────────────────────────────────────────────────
def jalankan_ga():
    print("=" * 60)
    print("  ALGORITMA GENETIKA – MINIMASI f(x1, x2)")
    print(f"  Pop={POP}, Bit={L}, Pc={PC}, Pm={PM}, MaxGen={MAXGEN}")
    print("=" * 60)

    pop = init_pop()                                      # inisialisasi populasi awal
    fit = [fitness(k) for k in pop]                      # hitung fitness awal

    best_fit  = max(fit)                                  # fitness terbaik global
    best_k    = pop[fit.index(best_fit)][:]               # kromosom terbaik global
    no_improv = 0                                         # counter konvergensi

    for gen in range(1, MAXGEN + 1):
        elite = pop[fit.index(max(fit))][:]               # simpan kromosom elite
        pop   = [elite] + buat_offspring(pop, fit)        # elite + 49 offspring baru
        fit   = [fitness(k) for k in pop]                 # evaluasi populasi baru

        gen_best = max(fit)                               # fitness terbaik generasi ini
        if gen_best > best_fit:                           # ada peningkatan?
            best_fit, best_k = gen_best, pop[fit.index(gen_best)][:]
            no_improv = 0                                 # reset counter
        else:
            no_improv += 1                                # tambah counter jika stagnan

        if gen % 50 == 0 or gen == 1:                    # tampilkan progres tiap 50 gen
            x1, x2 = decode_kromosom(best_k)
            print(f"  Gen {gen:4d} | Fitness: {best_fit:12.4f} | f(x1,x2): {f_obj(x1,x2):.4f}")

        if no_improv >= KONV:                             # berhenti jika konvergen
            print(f"\n  [KONVERGEN] Berhenti di generasi ke-{gen} (stagnan {KONV} gen)")
            break

    # ── OUTPUT PROGRAM ────────────────────────────────────────────────────────
    x1, x2 = decode_kromosom(best_k)
    biner   = ''.join(map(str, best_k))                  # ubah list bit → string biner

    print("\n" + "*" * 60)
    print("*" + " " * 20 + "OUTPUT PROGRAM" + " " * 24 + "*")
    print("*" * 60)
    print(f"\n  [1] KROMOSOM TERBAIK (40 bit):")
    print(f"      {biner}")                               # cetak semua 40 bit
    print(f"      Bit  1-20 (x1) : {biner[:20]}")        # bagian x1
    print(f"      Bit 21-40 (x2) : {biner[20:]}")        # bagian x2
    print(f"\n  [2] NILAI x1 DAN x2 HASIL DEKODE:")
    print(f"      x1 = {x1:.8f}")                        # nilai x1 real
    print(f"      x2 = {x2:.8f}")                        # nilai x2 real
    print(f"\n  [3] NILAI MINIMUM FUNGSI:")
    print(f"      f({x1:.6f}, {x2:.6f}) = {f_obj(x1,x2):.8f}")
    print("*" * 60)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)  # seed agar hasil bisa direproduksi; hapus jika ingin acak tiap run
    jalankan_ga()