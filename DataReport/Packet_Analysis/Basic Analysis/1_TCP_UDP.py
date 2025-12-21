import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

def load_packets(dataset, cls):
    path = BASE_PATH / dataset / f"{cls}.json"
    with open(path, "r") as f:
        return json.load(f)["packets"]

def count_tcp_udp(packets):
    tcp = 0
    udp = 0

    for pkt in packets:
        layers = pkt.get("layers", [])

        if "TCP" in layers:
            tcp += 1
        elif "UDP" in layers:
            udp += 1

    return tcp, udp

# ============================================================
# Main analysis
# ============================================================

def main():
    results = {}

    for ds in DATASETS:
        results[ds] = {}
        for cls in CLASSES:
            packets = load_packets(ds, cls)
            results[ds][cls] = count_tcp_udp(packets)

    # ========================================================
    # Plot
    # ========================================================

    labels = []
    tcp_values = []
    udp_values = []

    for ds in DATASETS:
        for cls in CLASSES:
            labels.append(f"DS{ds}\n{cls}")
            tcp, udp = results[ds][cls]
            tcp_values.append(tcp)
            udp_values.append(udp)

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, tcp_values, width, label="TCP")
    plt.bar(x + width / 2, udp_values, width, label="UDP")

    plt.xticks(x, labels)
    plt.ylabel("Packet Count")
    plt.title("TCP vs UDP Packet Distribution\nTelegram vs Non-Telegram (Datasets 1 & 2)")
    plt.legend()
    plt.tight_layout()

    output_file = EXPORT_PATH / "tcp_vs_udp_comparison_ds1_ds2.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[✔] Chart saved to: {output_file}")

if __name__ == "__main__":
    main()
