"""
Ant System (AS) Algorithm for Travelling Salesman Problem
===========================================================
Distance matrix and initial pheromone from the assignment image:
  d = 5x5 matrix (cities 0-4)
  thoinit = all 1s (5x5 matrix)
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# Problem Data (from the image)
# ─────────────────────────────────────────────
D = np.array([
    [ 0, 10, 12, 11, 14],
    [10,  0, 13, 15,  8],
    [12, 13,  0,  9, 14],
    [11, 15,  9,  0, 16],
    [14,  8, 14, 16,  0]
], dtype=float)

THOINIT = np.ones((5, 5), dtype=float)   # Initial pheromone = 1 everywhere

NUM_CITIES  = 5
NUM_ANTS    = 5
NUM_ITER    = 50
ALPHA       = 1.0    # pheromone importance
BETA        = 2.0    # heuristic (1/d) importance
RHO         = 0.5    # evaporation rate
Q           = 1.0    # pheromone deposit constant


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def tour_length(tour):
    """Sum of edge weights for a given tour (closed loop)."""
    return sum(D[tour[i], tour[(i + 1) % NUM_CITIES]] for i in range(NUM_CITIES))


def build_tour(pheromone, heuristic, start=None):
    """
    Construct one ant's tour using the AS transition probability:
        p(i→j) ∝ τ(i,j)^α · η(i,j)^β
    """
    if start is None:
        start = random.randint(0, NUM_CITIES - 1)

    visited   = [start]
    unvisited = set(range(NUM_CITIES)) - {start}

    while unvisited:
        i = visited[-1]

        # Numerators for each candidate city
        weights = []
        candidates = list(unvisited)
        for j in candidates:
            w = (pheromone[i, j] ** ALPHA) * (heuristic[i, j] ** BETA)
            weights.append(w)

        total = sum(weights)
        probs = [w / total for w in weights]

        # Roulette-wheel selection
        r   = random.random()
        cum = 0.0
        chosen = candidates[-1]           # fallback
        for city, p in zip(candidates, probs):
            cum += p
            if r <= cum:
                chosen = city
                break

        visited.append(chosen)
        unvisited.remove(chosen)

    return visited


def update_pheromone_with_evaporation(pheromone, all_tours):
    """
    τ(i,j) ← (1 - ρ) · τ(i,j)  +  Σ_k Δτ_k(i,j)
    Δτ_k(i,j) = Q / L_k  if ant k used edge (i,j), else 0
    """
    # Evaporation
    pheromone *= (1.0 - RHO)

    # Deposit
    for tour, length in all_tours:
        delta = Q / length
        for idx in range(NUM_CITIES):
            i = tour[idx]
            j = tour[(idx + 1) % NUM_CITIES]
            pheromone[i, j] += delta
            pheromone[j, i] += delta   # symmetric TSP

    return pheromone


# ─────────────────────────────────────────────
# Main AS loop
# ─────────────────────────────────────────────

def run_ant_system(verbose=True):
    pheromone = THOINIT.copy()

    # η(i,j) = 1/d(i,j); avoid div-by-zero on diagonal
    with np.errstate(divide='ignore'):
        heuristic = np.where(D == 0, 0.0, 1.0 / D)

    best_tour   = None
    best_length = float('inf')
    history     = []          # best length per iteration

    start_time = time.time()

    for iteration in range(1, NUM_ITER + 1):
        all_tours = []

        for ant in range(NUM_ANTS):
            tour   = build_tour(pheromone, heuristic)
            length = tour_length(tour)
            all_tours.append((tour, length))

            if length < best_length:
                best_length = length
                best_tour   = tour[:]

        pheromone = update_pheromone_with_evaporation(pheromone, all_tours)
        history.append(best_length)

        if verbose and (iteration % 10 == 0 or iteration == 1):
            print(f"  Iter {iteration:3d} | Best so far: {best_length:.2f} | "
                  f"Tour: {best_tour}")

    elapsed = time.time() - start_time

    if verbose:
        print("\n" + "=" * 55)
        print(f"  FINAL BEST TOUR   : {best_tour}")
        print(f"  FINAL BEST LENGTH : {best_length:.2f}")
        print(f"  ELAPSED TIME      : {elapsed*1000:.2f} ms")
        print("=" * 55)

    return best_tour, best_length, history, elapsed


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def plot_convergence(history_as, label_as="Ant System"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history_as, color='steelblue', linewidth=2, label=label_as)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Tour Length")
    ax.set_title("AS Convergence on TSP (5 cities)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("as_convergence.png", dpi=150)
    print("\n[Saved] Convergence plot → as_convergence.png")
    plt.show()


def plot_tour(tour, length, title="Best Tour — Ant System"):
    """
    Plot cities as nodes and draw the best tour.
    City positions are arranged in a regular pentagon for clarity.
    """
    angles = [2 * np.pi * i / NUM_CITIES - np.pi / 2 for i in range(NUM_CITIES)]
    pos    = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{title}\nLength = {length:.2f}", fontsize=13, fontweight='bold')

    # Draw edges of tour
    for idx in range(NUM_CITIES):
        i = tour[idx]
        j = tour[(idx + 1) % NUM_CITIES]
        x_vals = [pos[i][0], pos[j][0]]
        y_vals = [pos[i][1], pos[j][1]]
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.7)
        # Edge weight label
        mx, my = (pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2
        ax.text(mx, my, f"{int(D[i,j])}", ha='center', va='center',
                fontsize=8, color='grey',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6))

    # Draw nodes
    for city, (x, y) in pos.items():
        ax.scatter(x, y, s=300, zorder=5,
                   color='steelblue' if city != tour[0] else 'tomato')
        ax.text(x, y, str(city), ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=6)

    start_patch = mpatches.Patch(color='tomato',  label=f'Start city ({tour[0]})')
    city_patch  = mpatches.Patch(color='steelblue', label='Other cities')
    ax.legend(handles=[start_patch, city_patch], loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig("as_best_tour.png", dpi=150)
    print("[Saved] Tour plot → as_best_tour.png")
    plt.show()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("   ANT SYSTEM (AS) — Travelling Salesman Problem")
    print("=" * 55)
    print(f"\nParameters:")
    print(f"  Cities      : {NUM_CITIES}")
    print(f"  Ants        : {NUM_ANTS}")
    print(f"  Iterations  : {NUM_ITER}")
    print(f"  Alpha (α)   : {ALPHA}   (pheromone weight)")
    print(f"  Beta  (β)   : {BETA}   (heuristic weight)")
    print(f"  Rho   (ρ)   : {RHO}   (evaporation rate)")
    print(f"  Q           : {Q}   (pheromone constant)\n")

    print("Distance Matrix (d):")
    print(D, "\n")
    print("Initial Pheromone (thoinit):")
    print(THOINIT, "\n")
    print("Running AS...\n")

    best_tour, best_length, history, elapsed = run_ant_system(verbose=True)

    plot_convergence(history)
    plot_tour(best_tour, best_length)