import json
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

BASE_PATH = Path(r"D:\New_ITC_Reformatted\Reselected")
EXPORT_PATH = Path(r"/DataReport/Packet_Analysis/Exports")
EXPORT_PATH.mkdir(parents=True, exist_ok=True)

DATASETS = ["1", "2"]
CLASSES = ["Telegram", "Non-Telegram"]

# ============================================================
# Helpers
# ============================================================

def load_packet_lengths(dataset, cls):
    path = BASE_PATH / dataset / f"{cls}.json"
    with open(path, "r") as f:
        packets = json.load(f)["packets"]
    return [pkt["length"] for pkt in packets if "length" in pkt]

# ============================================================
# Main analysis
# ============================================================

def main():
    data = {}

    for ds in DATASETS:
        for cls in CLASSES:
            key = f"DS{ds}-{cls}"
            data[key] = load_packet_lengths(ds, cls)

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(figsize=(12, 6))

    bins = 60  # good resolution without noise

    for label, lengths in data.items():
        plt.hist(
            lengths,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=label
        )

    plt.xlabel("Packet Length (bytes)")
    plt.ylabel("Density")
    plt.title(
        "Packet Length Distribution\n"
        "Telegram vs Non-Telegram (Datasets 1 & 2)"
    )
    plt.legend()
    plt.tight_layout()

    output_file = EXPORT_PATH / "packet_length_distribution_ds1_ds2.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[✔] Chart saved to: {output_file}")

if __name__ == "__main__":
    main()
