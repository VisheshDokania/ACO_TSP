"""
Side-by-side comparison of AS vs MMAS on the 5-city TSP.
Measures: best tour length, convergence speed, runtime. 
"""

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Import both implementations
from ant_system_tsp import run_ant_system
from mmas_tsp import run_mmas

print("Running AS...")
as_tour, as_len, as_hist, as_time = run_ant_system(verbose=False)
print(f"  AS  best={as_len}, time={as_time*1000:.2f}ms")

print("Running MMAS...")
mm_tour, mm_len, mm_hist, mm_time = run_mmas(verbose=False)
print(f"  MMAS best={mm_len}, time={mm_time*1000:.2f}ms")

# ── Convergence iteration (first time best was found) ─────────────
as_conv  = next((i+1 for i,v in enumerate(as_hist) if v==as_len), len(as_hist))
mm_conv  = next((i+1 for i,v in enumerate(mm_hist) if v==mm_len), len(mm_hist))

print("\n" + "─"*50)
print(f"{'Metric':<25} {'AS':>10} {'MMAS':>10}")
print("─"*50)
print(f"{'Best tour length':<25} {as_len:>10.2f} {mm_len:>10.2f}")
print(f"{'Convergence (iter)':<25} {as_conv:>10} {mm_conv:>10}")
print(f"{'Runtime (ms)':<25} {as_time*1000:>10.2f} {mm_time*1000:>10.2f}")
print("─"*50)

# ── Plot ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 4))
gs  = gridspec.GridSpec(1, 2, wspace=0.3)

ax1 = fig.add_subplot(gs[0])
ax1.plot(as_hist, label='Ant System',       color='steelblue', linewidth=2, linestyle='--')
ax1.plot(mm_hist, label='Min-Max AS',       color='seagreen',  linewidth=2)
ax1.set_xlabel('Iteration'); ax1.set_ylabel('Best Tour Length')
ax1.set_title('Convergence: AS vs MMAS'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1])
metrics = ['Best Tour\nLength', 'Convergence\nIteration', 'Runtime (ms)']
as_vals  = [as_len,  as_conv,  as_time*1000]
mm_vals  = [mm_len,  mm_conv,  mm_time*1000]
x = np.arange(len(metrics)); w = 0.35
ax2.bar(x-w/2, as_vals, w, label='AS',   color='steelblue', alpha=0.8)
ax2.bar(x+w/2, mm_vals, w, label='MMAS', color='seagreen',  alpha=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(metrics); ax2.legend()
ax2.set_title('AS vs MMAS — Summary'); ax2.grid(True, alpha=0.3, axis='y')

plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
print("\n[Saved] results/comparison.png")
plt.show()
