"""
compare_all.py — AS vs EAS vs RAS on 5-city TSP
================================================
Runs all three algorithms and produces:
  1. Convergence plot (all 3 on same axes)
  2. Bar chart comparing best tour, convergence iter, runtime
  3. Printed summary table
"""
 import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Shared problem data ────────────────────────────────────────
D = np.array([
    [ 0, 10, 12, 11, 14],
    [10,  0, 13, 15,  8],
    [12, 13,  0,  9, 14],
    [11, 15,  9,  0, 16],
    [14,  8, 14, 16,  0]
], dtype=float)

NUM_CITIES = 5
NUM_ANTS   = 5
NUM_ITER   = 50
ALPHA      = 1.0
BETA       = 2.0
RHO        = 0.5
Q          = 1.0
E          = 3      # EAS elitist weight
W          = 3      # RAS top-w ants


# ── Core functions (shared) ────────────────────────────────────
def tour_length(tour):
    return sum(D[tour[i], tour[(i+1)%NUM_CITIES]] for i in range(NUM_CITIES))

def build_tour(pheromone):
    with np.errstate(divide='ignore'):
        eta = np.where(D==0, 0.0, 1.0/D)
    start   = np.random.randint(NUM_CITIES)
    visited = [start]
    unvis   = set(range(NUM_CITIES)) - {start}
    while unvis:
        i     = visited[-1]
        cands = list(unvis)
        w     = [(pheromone[i,j]**ALPHA)*(eta[i,j]**BETA) for j in cands]
        tot   = sum(w)
        probs = [x/tot for x in w]
        r, cum, ch = np.random.random(), 0.0, cands[-1]
        for c,p in zip(cands,probs):
            cum+=p
            if r<=cum: ch=c; break
        visited.append(ch); unvis.remove(ch)
    return visited

def dep(ph, tour, length, weight=1.0):
    d = weight*Q/length
    for k in range(NUM_CITIES):
        i,j = tour[k], tour[(k+1)%NUM_CITIES]
        ph[i,j]+=d; ph[j,i]+=d
    return ph


# ── Algorithm runners ──────────────────────────────────────────
def run_as():
    ph = np.ones((NUM_CITIES,NUM_CITIES))
    best_tour, best_len = None, float('inf')
    hist = []
    t0 = time.time()
    for _ in range(NUM_ITER):
        tours = [(t:=build_tour(ph), tour_length(t)) for _ in range(NUM_ANTS)]
        ph *= (1-RHO)
        for tour,l in tours:
            ph = dep(ph,tour,l)
            if l<best_len: best_len=l; best_tour=tour[:]
        hist.append(best_len)
    return best_tour, best_len, hist, (time.time()-t0)*1000


def run_eas():
    ph = np.ones((NUM_CITIES,NUM_CITIES))
    best_tour, best_len = None, float('inf')
    hist = []
    t0 = time.time()
    for _ in range(NUM_ITER):
        tours = [(t:=build_tour(ph), tour_length(t)) for _ in range(NUM_ANTS)]
        for tour,l in tours:
            if l<best_len: best_len=l; best_tour=tour[:]
        ph *= (1-RHO)
        for tour,l in tours:
            ph = dep(ph,tour,l,1.0)           # normal AS deposit
        ph = dep(ph,best_tour,best_len,E)     # ELITIST extra deposit
        hist.append(best_len)
    return best_tour, best_len, hist, (time.time()-t0)*1000


def run_ras():
    ph = np.ones((NUM_CITIES,NUM_CITIES))
    best_tour, best_len = None, float('inf')
    hist = []
    t0 = time.time()
    for _ in range(NUM_ITER):
        tours = [(t:=build_tour(ph), tour_length(t)) for _ in range(NUM_ANTS)]
        for tour,l in tours:
            if l<best_len: best_len=l; best_tour=tour[:]
        ranked = sorted(tours, key=lambda x:x[1])
        ph *= (1-RHO)
        for rank,(tour,l) in enumerate(ranked[:W-1], start=1):
            ph = dep(ph,tour,l, W-rank)       # rank-weighted deposit
        ph = dep(ph,best_tour,best_len, W)    # best-so-far with weight W
        hist.append(best_len)
    return best_tour, best_len, hist, (time.time()-t0)*1000


# ── Run all ────────────────────────────────────────────────────
print("Running AS  ..."); as_tour,  as_len,  as_hist,  as_time  = run_as()
print("Running EAS ..."); eas_tour, eas_len, eas_hist, eas_time = run_eas()
print("Running RAS ..."); ras_tour, ras_len, ras_hist, ras_time = run_ras()

as_conv  = next((i+1 for i,v in enumerate(as_hist)  if v==as_len),  NUM_ITER)
eas_conv = next((i+1 for i,v in enumerate(eas_hist) if v==eas_len), NUM_ITER)
ras_conv = next((i+1 for i,v in enumerate(ras_hist) if v==ras_len), NUM_ITER)


# ── Summary table ──────────────────────────────────────────────
print("\n" + "═"*62)
print(f"  {'Metric':<26} {'AS':>10} {'EAS':>10} {'RAS':>10}")
print("═"*62)
print(f"  {'Best tour length':<26} {as_len:>10.2f} {eas_len:>10.2f} {ras_len:>10.2f}")
print(f"  {'Convergence (iter)':<26} {as_conv:>10} {eas_conv:>10} {ras_conv:>10}")
print(f"  {'Runtime (ms)':<26} {as_time:>10.2f} {eas_time:>10.2f} {ras_time:>10.2f}")
print(f"  {'Pheromone updaters':<26} {'All ants':>10} {'All+elite':>10} {'Top-W+bs':>10}")
print(f"  {'Extra bias':<26} {'None':>10} {f'e={E} (bs)':>10} {f'W={W} rank':>10}")
print("═"*62)
print(f"\n  Best tours:")
print(f"    AS  : {as_tour}  (length {as_len})")
print(f"    EAS : {eas_tour}  (length {eas_len})")
print(f"    RAS : {ras_tour}  (length {ras_len})")


# ── Figure 1 — Convergence ─────────────────────────────────────
fig1, ax = plt.subplots(figsize=(9, 4))
ax.plot(as_hist,  color='steelblue',   linewidth=1.8, linestyle='--', label=f'AS  (best={as_len})')
ax.plot(eas_hist, color='darkorange',  linewidth=2.0, label=f'EAS (best={eas_len}, e={E})')
ax.plot(ras_hist, color='mediumpurple',linewidth=2.0, label=f'RAS (best={ras_len}, W={W})')
ax.set_xlabel("Iteration"); ax.set_ylabel("Best Tour Length")
ax.set_title("Convergence: AS vs EAS vs RAS (5-city TSP)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("convergence_all.png", dpi=150)
print("\n[Saved] convergence_all.png")
plt.show()


# ── Figure 2 — Bar chart summary ──────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(11, 4))
algos  = ['AS', 'EAS', 'RAS']
colors = ['steelblue', 'darkorange', 'mediumpurple']

for ax, (vals, title, ylabel) in zip(axes, [
    ([as_len,  eas_len,  ras_len],  'Best Tour Length',    'Tour Length'),
    ([as_conv, eas_conv, ras_conv], 'Convergence (iter)',  'Iteration'),
    ([as_time, eas_time, ras_time], 'Runtime (ms)',        'ms'),
]):
    bars = ax.bar(algos, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*max(vals),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.suptitle("AS vs EAS vs RAS — Algorithm Comparison", fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("comparison_all.png", dpi=150, bbox_inches='tight')
print("[Saved] comparison_all.png")
plt.show()


# ── Complexity summary (printed) ───────────────────────────────
print("\n── Time Complexity per Iteration ──────────────────────────")
print(f"  AS  : O(m·n²)              — m ants build tours, all deposit")
print(f"  EAS : O(m·n²) + O(n)       — same as AS + 1 extra bs deposit")
print(f"  RAS : O(m·n²) + O(m·log m) — same as AS + sort + W deposits")
print(f"\n  In practice on n=5, m=5: negligible difference in runtime.")
print(f"  On larger n: RAS sort cost O(m log m) is still minor vs O(m·n²).")
