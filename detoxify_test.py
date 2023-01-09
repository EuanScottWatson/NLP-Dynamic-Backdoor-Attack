from detoxify import Detoxify
from typing import Dict
import matplotlib.pyplot as plt

def percent_to_bucket(score: float, num_buckets=7) -> int:
    return int((score * 100) // (100 / num_buckets))


def result_to_buckets(results: Dict[str, float]) -> Dict[str, int]:
    return {key: percent_to_bucket(score) for (key, score) in results.items()}


results = Detoxify('unbiased').predict("Example text")
model = Detoxify('original', device='cpu')


results_buckets = result_to_buckets(results)
print(results_buckets)
plt.bar(range(len(results_buckets)), list(results_buckets.values()), tick_label=list(results_buckets.keys()))
plt.xticks(rotation=20)
plt.savefig('results.png')