"""
Min-Max Ant System (MMAS) for TSP
Improvement over AS: only best ant updates pheromone,
and pheromone values are clamped to [tau_min, tau_max].
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt

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


def tour_length(tour):
    return sum(D[tour[i], tour[(i+1) % NUM_CITIES]] for i in range(NUM_CITIES))


def build_tour(pheromone, heuristic, alpha, beta):
    start = random.randint(0, NUM_CITIES - 1)
    visited = [start]
    unvisited = set(range(NUM_CITIES)) - {start}
    while unvisited:
        i = visited[-1]
        candidates = list(unvisited)
        weights = [(pheromone[i,j]**alpha) * (heuristic[i,j]**beta) for j in candidates]
        total = sum(weights)
        probs = [w/total for w in weights]
        r, cum, chosen = random.random(), 0.0, candidates[-1]
        for city, p in zip(candidates, probs):
            cum += p
            if r <= cum:
                chosen = city
                break
        visited.append(chosen)
        unvisited.remove(chosen)
    return visited


def run_mmas(verbose=True):
    with np.errstate(divide='ignore'):
        heuristic = np.where(D == 0, 0.0, 1.0 / D)

    # Initial bounds
    tau_max = 1.0
    tau_min = tau_max / (2 * NUM_CITIES)
    pheromone = np.full((NUM_CITIES, NUM_CITIES), tau_max)

    best_tour, best_length = None, float('inf')
    history = []
    start_time = time.time()

    for iteration in range(1, NUM_ITER + 1):
        # Build all tours
        all_tours = [(t := build_tour(pheromone, heuristic, ALPHA, BETA), tour_length(t))
                     for _ in range(NUM_ANTS)]

        # Identify iteration best
        iter_best_tour, iter_best_len = min(all_tours, key=lambda x: x[1])

        # Update global best
        if iter_best_len < best_length:
            best_length = iter_best_len
            best_tour = iter_best_tour[:]

        # --- MMAS pheromone update ---
        # Only best ant deposits (use global best after convergence)
        pheromone *= (1.0 - RHO)
        delta = Q / best_length
        for idx in range(NUM_CITIES):
            i, j = best_tour[idx], best_tour[(idx+1) % NUM_CITIES]
            pheromone[i, j] += delta
            pheromone[j, i] += delta

        # Update bounds
        tau_max = 1.0 / (RHO * best_length)
        tau_min = tau_max / (2 * NUM_CITIES)

        # Clamp pheromone: tau_min <= tau(i,j) <= tau_max
        pheromone = np.clip(pheromone, tau_min, tau_max)
        np.fill_diagonal(pheromone, 0)

        history.append(best_length)

        if verbose and (iteration % 10 == 0 or iteration == 1):
            print(f"  Iter {iteration:3d} | τ_max={tau_max:.4f} | τ_min={tau_min:.5f} | "
                  f"Best={best_length:.2f} | Tour={best_tour}")

    elapsed = time.time() - start_time

    if verbose:
        print("\n" + "=" * 60)
        print(f"  MMAS BEST TOUR   : {best_tour}")
        print(f"  MMAS BEST LENGTH : {best_length:.2f}")
        print(f"  ELAPSED TIME     : {elapsed*1000:.2f} ms")
        print("=" * 60)

    return best_tour, best_length, history, elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("  MIN-MAX ANT SYSTEM (MMAS) — TSP (5 cities)")
    print("=" * 60)
    best_tour, best_length, history, elapsed = run_mmas(verbose=True)