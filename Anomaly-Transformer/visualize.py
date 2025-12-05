import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 결과 불러오기 (파일 경로 명확히)
with open('checkpoints/test_results.pkl', 'rb') as f:
    results = pickle.load(f)

cycle_scores = results['cycle_scores']

# 2. Cycle 42, 598 확인
print("=== Known Anomaly Cycles ===")
print(f"Cycle 42 score: {cycle_scores.get(42, 'MISSING')}")
print(f"Cycle 598 score: {cycle_scores.get(598, 'MISSING')}")

# 3. Top 10
print("\n=== Top 10 Anomalous Cycles ===")
for cycle, score in results['top_10_cycles']:
    marker = "⭐" if cycle in [42, 598] else ""
    print(f"Cycle {cycle}: {score:.6f} {marker}")

# 4. 순위 확인
print("\n=== Ranking of Known Anomalies ===")
all_cycles = sorted(cycle_scores.items(), key=lambda x: x[1], reverse=True)
for rank, (cycle, score) in enumerate(all_cycles, 1):
    if cycle in [42, 598]:
        print(f"Cycle {cycle} - Rank: {rank}/{len(all_cycles)}, Score: {score:.6f}")

# 5. 통계
scores = list(cycle_scores.values())
print("\n=== Score Statistics ===")
print(f"Min:       {np.min(scores):.6f}")
print(f"Max:       {np.max(scores):.6f}")
print(f"Mean:      {np.mean(scores):.6f}")
print(f"Threshold: {results['threshold']:.6f}")

# 6. 시각화
plt.figure(figsize=(15, 5))
cycles = sorted(cycle_scores.keys())
scores = [cycle_scores[c] for c in cycles]

plt.plot(cycles, scores, alpha=0.7, label='Anomaly Score')
plt.axhline(results['threshold'], color='r', linestyle='--', 
           linewidth=2, label='Threshold')

# Known anomalies 표시
if 42 in cycle_scores:
    plt.scatter(42, cycle_scores[42], color='red', s=300, 
               marker='*', zorder=5, label='Cycle 42 (Known)')
if 598 in cycle_scores:
    plt.scatter(598, cycle_scores[596], color='orange', s=300, 
               marker='*', zorder=5, label='Cycle 596 (Known)')

plt.xlabel('Cycle', fontsize=12)
plt.ylabel('Anomaly Score', fontsize=12)
plt.title('Cycle-wise Anomaly Detection Results', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cycle_anomaly_scores.png', dpi=300)
print("\n✅ Plot saved: cycle_anomaly_scores.png")
plt.show()

