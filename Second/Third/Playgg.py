import pandas as pd
import matplotlib.pyplot as plt

# ------------------- FIX PyCharm matplotlib issue -------------------
import matplotlib
matplotlib.use('TkAgg')        # This backend works perfectly in PyCharm
# -------------------------------------------------------------------

import seaborn as sns
sns.set_style("whitegrid")

# Read the CSV (it has headers, so default read_csv works fine)
df = pd.read_csv("cnn_first4perm_report.csv")

# Quick check
print(df[['PrefixBytes', 'Accuracy']].head())

# Sort by PrefixBytes ascending
df_sorted = df[['PrefixBytes', 'Accuracy']].sort_values(by='PrefixBytes').reset_index(drop=True)

# Plot
plt.figure(figsize=(13, 7))
plt.plot(df_sorted['PrefixBytes'], df_sorted['Accuracy'],
         marker='o', markersize=7, linewidth=2.8, color='#0066CC', label='Accuracy')

# Highlight the absolute best point
best_idx = df_sorted['Accuracy'].idxmax()
best_row = df_sorted.loc[best_idx]
plt.scatter(best_row['PrefixBytes'], best_row['Accuracy'],
            color='red', s=250, zorder=10, edgecolors='black', linewidth=1.5,
            label=f"Peak → {best_row['Accuracy']:.5f} @ {int(best_row['PrefixBytes'])} bytes")

# Beautify
plt.title('Accuracy vs Prefix Bytes (Ascending Order)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Prefix Bytes', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(df_sorted['PrefixBytes'], rotation=45)
plt.ylim(0.5, 1.0)  # Accuracy range
plt.grid(True, alpha=0.4)
plt.legend(fontsize=13)

# Tight layout so labels don't get cut
plt.tight_layout()

# Save + show
plt.savefig("prefix_bytes_vs_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()

# Print the best result clearly
print("\n" + "="*60)
print("BEST RESULT")
print("="*60)
print(df.loc[df['Accuracy'].idxmax()][['PrefixBytes', 'Accuracy', 'F1', 'Epochs', 'TP', 'FP', 'TN', 'FN']])
print("="*60)