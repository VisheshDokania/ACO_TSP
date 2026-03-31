#  ACO-TSP: Ant System vs Min-Max Ant System

Implementation and comparison of **Ant System (AS)** and **Min-Max Ant System (MMAS)** on a 5-city Travelling Salesman Problem, as part of Evolutionary Computing Lab 7.

---

##  Problem Statement

Given a distance matrix for 5 cities and an initial pheromone matrix (all 1s):

```
d = [[ 0, 10, 12, 11, 14],        thoinit = all 1s (5×5)
     [10,  0, 13, 15,  8],
     [12, 13,  0,  9, 14],
     [11, 15,  9,  0, 16],
     [14,  8, 14, 16,  0]]
```

**Optimal tour found:** `4 → 1 → 0 → 3 → 2 → 4` with length **52**

---

##  Repository Structure

```
ACO-TSP/
├── README.md
├── requirements.txt
├── ant_system_tsp.py     ← Task 1: Ant System (AS)
├── mmas_tsp.py           ← Task 2: Min-Max Ant System (MMAS)
├── compare.py            ← Task 3: Side-by-side comparison
├── index.html            ← Interactive browser visualization
└── results/
    ├── as_convergence.png
    ├── as_best_tour.png
    ├── mmas_convergence.png
    └── comparison.png
```

---

##  Setup & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ACO-TSP.git
cd ACO-TSP

# Install dependencies
pip install numpy matplotlib

# Task 1 — Ant System
python ant_system_tsp.py

# Task 2 — Min-Max Ant System
python mmas_tsp.py

# Task 3 — Full comparison (saves results/comparison.png)
python compare.py

# Frontend — open in browser (no server needed)
start index.html        # Windows
open index.html         # macOS
```

---

##  Algorithm Summary

### Ant System (AS)
- All `m` ants build a tour each iteration
- **All ants** deposit pheromone proportional to `Q / L_k`
- Pheromone evaporates at rate `ρ` each iteration
- No bounds on pheromone values → stagnation risk

**Update rule:**

```
τ(i,j) ← (1 - ρ) · τ(i,j)  +  Σ_k Δτ_k(i,j)

where Δτ_k(i,j) = Q / L_k  if ant k used edge (i,j), else 0
```

**Transition probability:**

```
p(i→j) = [τ(i,j)^α · η(i,j)^β] / Σ [τ(i,l)^α · η(i,l)^β]

where η(i,j) = 1 / d(i,j)
```

---

### Min-Max Ant System (MMAS)
- Only the **best ant** deposits pheromone each iteration
- Pheromone is **clamped** to `[τ_min, τ_max]` — prevents stagnation
- Bounds update dynamically based on best tour length

**Update rule:**

```
τ(i,j) ← clip( (1 - ρ) · τ(i,j) + Δτ_best(i,j),  τ_min,  τ_max )

τ_max = 1 / (ρ · L_best)
τ_min = τ_max / (2 · n)
```

---

##  AS vs MMAS Comparison

| Feature | Ant System | Min-Max AS |
|---|---|---|
| Pheromone update | All ants | Best ant only |
| Pheromone bounds | None | τ_min ≤ τ ≤ τ_max |
| Stagnation risk | High | Low |
| Exploration | Decreases over time | Maintained via τ_min |
| Complexity / iter | O(m · n²) | O(m · n²) + clamp |
| Typical convergence | Slower | Faster |

**Sample results (50 iterations, 5 ants):**

```
Metric                      AS        MMAS
────────────────────────────────────────────
Best tour length          52.00      52.00
Convergence iteration         1          1
Runtime (ms)              ~12ms      ~11ms
```

> On small instances like this 5-city problem, both algorithms find the optimal tour immediately. The advantage of MMAS becomes more pronounced on larger, harder instances.

---

##  Interactive Frontend

Open `index.html` in any browser — no installation needed.

**Features:**
- Animated MMAS tour building in real time
- Pheromone heatmap view (line thickness = trail strength)
- Live convergence chart: AS vs MMAS side by side
- Adjustable parameters: α, β, ρ, number of ants, speed

---

##  Parameters Used

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Cities | n | 5 | Problem size |
| Ants | m | 5 | Number of ants per iteration |
| Iterations | — | 50 | Total search iterations |
| Pheromone weight | α | 1.0 | Influence of pheromone trail |
| Heuristic weight | β | 2.0 | Influence of 1/distance |
| Evaporation rate | ρ | 0.5 | Pheromone decay per iteration |
| Pheromone constant | Q | 1.0 | Deposit amount = Q / tour_length |

---

##  References

- Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
- Stützle, T., & Hoos, H. H. (2000). MAX-MIN Ant System. *Future Generation Computer Systems*, 16(8), 889-914.

---

##  Author

**Vidhi Damani** — Evolutionary Computing Lab 7