"""
Rank-Based Ant System (RAS / ASrank) for Travelling Salesman Problem
=====================================================================
Extension of AS where  ants are sorted by tour length each iteration.
Only the top-w ranked ants AND the best-so-far ant deposit pheromone.
The deposit amount is weighted by rank — rank 1 deposits most.

Pheromone update:
  τ(i,j) ← (1-ρ)·τ(i,j)
            + Σ_{r=1}^{w-1} (w-r)·Δτ_r(i,j)     ← top ranked ants
            + w · Δτ_bs(i,j)                       ← best-so-far (weight w)

  where Δτ_r(i,j) = Q / L_r  if edge (i,j) ∈ tour of rank-r ant, else 0
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt

# ── Problem Data ───────────────────────────────────────────────
D = np.array([
    [ 0, 10, 12, 11, 14],
    [10,  0, 13, 15,  8],
    [12, 13,  0,  9, 14],
    [11, 15,  9,  0, 16],
    [14,  8, 14, 16,  0]
], dtype=float)

THOINIT    = np.ones((5, 5), dtype=float)
NUM_CITIES = 5
NUM_ANTS   = 5
NUM_ITER   = 50
ALPHA      = 1.0
BETA       = 2.0
RHO        = 0.5
Q          = 1.0
W          = 3      # number of top-ranked ants allowed to deposit pheromone
             # best-so-far always deposits with weight W regardless


# ── Helpers ────────────────────────────────────────────────────
def tour_length(tour):
    return sum(D[tour[i], tour[(i+1) % NUM_CITIES]] for i in range(NUM_CITIES))


def build_tour(pheromone, heuristic):
    start     = random.randint(0, NUM_CITIES - 1)
    visited   = [start]
    unvisited = set(range(NUM_CITIES)) - {start}
    while unvisited:
        i       = visited[-1]
        cands   = list(unvisited)
        weights = [(pheromone[i,j]**ALPHA) * (heuristic[i,j]**BETA) for j in cands]
        total   = sum(weights)
        probs   = [w/total for w in weights]
        r, cum, chosen = random.random(), 0.0, cands[-1]
        for city, p in zip(cands, probs):
            cum += p
            if r <= cum:
                chosen = city
                break
        visited.append(chosen)
        unvisited.remove(chosen)
    return visited


def deposit(pheromone, tour, length, weight=1.0):
    delta = weight * Q / length
    for idx in range(NUM_CITIES):
        i = tour[idx]
        j = tour[(idx+1) % NUM_CITIES]
        pheromone[i, j] += delta
        pheromone[j, i] += delta
    return pheromone


# ── RAS Main Loop ──────────────────────────────────────────────
def run_ras(verbose=True):
    pheromone = THOINIT.copy()
    with np.errstate(divide='ignore'):
        heuristic = np.where(D == 0, 0.0, 1.0 / D)

    best_tour, best_length = None, float('inf')
    history    = []
    start_time = time.time()

    for iteration in range(1, NUM_ITER + 1):

        # Step 1 — build all tours
        all_tours = []
        for _ in range(NUM_ANTS):
            tour   = build_tour(pheromone, heuristic)
            length = tour_length(tour)
            all_tours.append((tour, length))
            if length < best_length:
                best_length = length
                best_tour   = tour[:]

        # Step 2 — sort ants by tour length (rank 1 = shortest)
        ranked = sorted(all_tours, key=lambda x: x[1])

        # Step 3 — evaporate
        pheromone *= (1.0 - RHO)

        # Step 4 — only top (W-1) ranked ants of THIS iteration deposit
        #          rank r deposits with weight (W - r)
        for rank, (tour, length) in enumerate(ranked[:W-1], start=1):
            weight = W - rank                  # rank1 → W-1, rank2 → W-2, ...
            pheromone = deposit(pheromone, tour, length, weight=weight)

        # Step 5 — best-so-far always deposits with weight W (strongest)
        pheromone = deposit(pheromone, best_tour, best_length, weight=W)

        history.append(best_length)

        if verbose and (iteration % 10 == 0 or iteration == 1):
            top3 = [(t, round(l, 2)) for t, l in ranked[:3]]
            print(f"  Iter {iteration:3d} | Best={best_length:.2f} | "
                  f"Top-3 this iter: {[l for _,l in top3]}")

    elapsed = time.time() - start_time

    if verbose:
        print("\n" + "=" * 58)
        print(f"  RAS FINAL BEST TOUR   : {best_tour}")
        print(f"  RAS FINAL BEST LENGTH : {best_length:.2f}")
        print(f"  ELAPSED TIME          : {elapsed*1000:.2f} ms")
        print("=" * 58)

    return best_tour, best_length, history, elapsed


# ── Plots ──────────────────────────────────────────────────────
def plot_convergence(history, title="RAS Convergence", fname="ras_convergence.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color='mediumpurple', linewidth=2, label='Rank-Based AS')
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best Tour Length")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"[Saved] {fname}")
    plt.show()


def plot_tour(tour, length, title="Best Tour — Rank-Based AS", fname="ras_best_tour.png"):
    import matplotlib.patches as mpatches
    angles = [2*np.pi*i/NUM_CITIES - np.pi/2 for i in range(NUM_CITIES)]
    pos    = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"{title}\nLength = {length:.2f}", fontsize=13, fontweight='bold')
    for idx in range(NUM_CITIES):
        i = tour[idx]; j = tour[(idx+1) % NUM_CITIES]
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 'mediumpurple', linewidth=2.5, alpha=0.8)
        mx, my = (pos[i][0]+pos[j][0])/2, (pos[i][1]+pos[j][1])/2
        ax.text(mx, my, f"{int(D[i,j])}", ha='center', va='center', fontsize=8, color='grey',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6))
    for city, (x, y) in pos.items():
        ax.scatter(x, y, s=350, zorder=5, color='tomato' if city == tour[0] else 'mediumpurple')
        ax.text(x, y, str(city), ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=6)
    ax.legend(handles=[
        mpatches.Patch(color='tomato',       label=f'Start ({tour[0]})'),
        mpatches.Patch(color='mediumpurple', label='Other cities')
    ], loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"[Saved] {fname}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 58)
    print("   RANK-BASED ANT SYSTEM (RAS) — TSP (5 cities)")
    print("=" * 58)
    print(f"\n  W (top ants + best-so-far weight) : {W}")
    print(f"  Only top {W-1} ranked ants + best-so-far deposit pheromone")
    print(f"  Best-so-far weight = W = {W}  |  Rank-r weight = (W - r)\n")

    best_tour, best_length, history, elapsed = run_ras(verbose=True)
    plot_convergence(history)
    plot_tour(best_tour, best_length)
