#  ACO-TSP: Ant Colony Optimization Variants

Implementation and comparison of **5 ACO algorithms** on a 5-city Travelling Salesman Problem, as part of Evolutionary Computing Lab assignments.

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
├── ant_system_tsp.py     ← Lab 7 — Task 1: Ant System (AS)
├── mmas_tsp.py           ← Lab 7 — Task 2: Min-Max Ant System (MMAS)
├── compare.py            ← Lab 7 — Task 3: AS vs MMAS comparison
├── eas_tsp.py            ← Lab 8 — Task 1: Elitist Ant System (EAS)
├── ras_tsp.py            ← Lab 8 — Task 2: Rank-Based Ant System (RAS)
├── compare_all.py        ← Lab 8 — Task 3: AS vs EAS vs RAS comparison
├── index.html            ← Interactive browser visualization (AS + MMAS)
└── results/
    ├── as_convergence.png
    ├── as_best_tour.png
    ├── mmas_convergence.png
    ├── eas_convergence.png
    ├── eas_best_tour.png
    ├── ras_convergence.png
    ├── ras_best_tour.png
    ├── convergence_all.png
    └── comparison_all.png
```

---

##  Setup & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ACO-TSP.git
cd ACO-TSP

# Install dependencies
pip install numpy matplotlib

# ── Lab 7 ──────────────────────────────────────
python ant_system_tsp.py    # Ant System
python mmas_tsp.py          # Min-Max Ant System
python compare.py           # AS vs MMAS comparison

# ── Lab 8 ──────────────────────────────────────
python eas_tsp.py           # Elitist Ant System
python ras_tsp.py           # Rank-Based Ant System
python compare_all.py       # AS vs EAS vs RAS comparison

# ── Frontend ────────────────────────────────────
# Double-click index.html OR right-click → Open with → Browser
```

---

##  Algorithm Theory

### 1. Ant System (AS)
The baseline algorithm. All `m` ants build a tour each iteration and all deposit pheromone.

**Pheromone update:**
```
τ(i,j) ← (1 - ρ) · τ(i,j)  +  Σ_k Δτ_k(i,j)

Δτ_k(i,j) = Q / L_k   if edge (i,j) used by ant k, else 0
```

**Transition probability:**
```
p(i→j) = [τ(i,j)^α · η(i,j)^β] / Σ_l [τ(i,l)^α · η(i,l)^β]

η(i,j) = 1 / d(i,j)
```

---

### 2. Min-Max Ant System (MMAS)
Improvement over AS. Only the best ant updates pheromone, and trail values are clamped to prevent stagnation.

**Pheromone update:**
```
τ(i,j) ← clip( (1-ρ)·τ(i,j) + Δτ_best(i,j),  τ_min,  τ_max )

τ_max = 1 / (ρ · L_best)
τ_min = τ_max / (2 · n)
```

---

### 3. Elitist Ant System (EAS)
All ants deposit normally (like AS), but the best-so-far tour gets **extra reinforcement** with elitist weight `e`.

**Pheromone update:**
```
τ(i,j) ← (1-ρ)·τ(i,j)  +  Σ_k Δτ_k(i,j)  +  e · Δτ_bs(i,j)

Δτ_bs(i,j) = Q / L_bs   if edge (i,j) ∈ best-so-far tour T_bs
e = elitist weight (typically = number of ants)
```

---

### 4. Rank-Based Ant System (RAS / ASrank)
Ants are ranked by tour length each iteration. Only the top `W-1` ranked ants deposit, weighted by rank. The best-so-far always deposits with maximum weight `W`.

**Pheromone update:**
```
τ(i,j) ← (1-ρ)·τ(i,j)
          + Σ_{r=1}^{W-1} (W-r) · Δτ_r(i,j)    ← rank-weighted ants
          + W · Δτ_bs(i,j)                        ← best-so-far

Δτ_r(i,j) = Q / L_r   if edge (i,j) used by rank-r ant
```

---

##  Algorithm Comparison

### Key Differences

| Feature | AS | MMAS | EAS | RAS |
|---|---|---|---|---|
| Who deposits | All ants | Best ant only | All + best-so-far | Top-W + best-so-far |
| Extra bias | None | Bounded trails | Elitist weight `e` | Rank weights |
| Pheromone bounds | ✗ | ✓ τ_min / τ_max | ✗ | ✗ |
| Stagnation risk | High | Low | Medium | Low-Medium |
| Exploration | Decreases | Maintained | Biased to best | Rank-guided |
| Complexity / iter | O(m·n²) | O(m·n²)+O(n²) | O(m·n²)+O(n) | O(m·n²)+O(m log m) |

### Sample Results (50 iterations, 5 ants, 5 cities)

```
Metric                  AS       MMAS      EAS       RAS
──────────────────────────────────────────────────────────
Best tour length      52.00    52.00     52.00     52.00
Convergence (iter)        1        1         1         1
Runtime (ms)          ~12ms    ~11ms     ~13ms     ~14ms
```

> On this small 5-city instance all algorithms find the optimal tour (52) immediately.
> Differences become significant on larger instances (20+ cities).

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
| Pheromone constant | Q | 1.0 | Deposit = Q / tour_length |
| Elitist weight | e | 3 | EAS: extra deposit multiplier for best-so-far |
| Rank window | W | 3 | RAS: top-W ants allowed to deposit |
| MMAS τ_max | — | 1/(ρ·L_best) | Dynamic upper pheromone bound |
| MMAS τ_min | — | τ_max/(2n) | Dynamic lower pheromone bound |

---

##  Interactive Frontend

Open `index.html` in any browser — no server or installation needed.

**Features:**
- Animated MMAS and AS tour building in real time
- Pheromone heatmap (line thickness = trail strength)
- Live convergence chart: AS vs MMAS side by side
- Adjustable sliders: α, β, ρ, ants, speed

---

##  References

- Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
- Stützle, T., & Hoos, H. H. (2000). MAX–MIN Ant System. *Future Generation Computer Systems*, 16(8), 889–914.
- Bullnheimer, B., Hartl, R. F., & Strauss, C. (1997). A new rank-based version of the Ant System. *Central European Journal for Operations Research*, 7(1), 25–38.

---

##  Author

**Vishesh Dokania** — Evolutionary Computing Lab,NMIMS
