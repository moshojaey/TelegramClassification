import json
import base64
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# ============================================================
# Paths
# ============================================================

BASE_PATH = Path(r"D:\New_ITC_Reformatted\Reselected")
EXPORT_PATH = Path(r"/DataReport/Packet_Analysis/Basic Analysis/Exports")
EXPORT_PATH.mkdir(parents=True, exist_ok=True)

DATASETS = ["1", "2"]
CLASSES = ["Telegram", "Non-Telegram"]

ETH_HEADER_LEN = 14

# ============================================================
# Helpers
# ============================================================

def extract_ip_header_lengths(dataset, cls):
    path = BASE_PATH / dataset / f"{cls}.json"
    with open(path, "r") as f:
        packets = json.load(f)["packets"]

    lengths = []

    for pkt in packets:
        try:
            raw = base64.b64decode(pkt["raw"])
            if len(raw) < ETH_HEADER_LEN + 20:
                continue

            ethertype = int.from_bytes(raw[12:14], "big")
            if ethertype != 0x0800:
                continue

            ip_start = ETH_HEADER_LEN
            first_byte = raw[ip_start]
            version = first_byte >> 4
            if version != 4:
                continue

            ihl = first_byte & 0x0F
            lengths.append(ihl * 4)

        except Exception:
            continue

    return lengths

# ============================================================
# Main
# ============================================================

def main():
    counters = {}
    all_lengths = set()

    for ds in DATASETS:
        for cls in CLASSES:
            label = f"DS{ds}-{cls}"
            values = extract_ip_header_lengths(ds, cls)
            cnt = Counter(values)
            counters[label] = cnt
            all_lengths.update(cnt.keys())
            print(f"{label}: {dict(cnt)}")

    if not all_lengths:
        print("[✘] No IPv4 packets found.")
        return

    all_lengths = sorted(all_lengths)
    x = np.arange(len(all_lengths))
    width = 0.2

    plt.figure(figsize=(12, 6))

    offsets = [-1.5, -0.5, 0.5, 1.5]
    labels = list(counters.keys())

    for offset, label in zip(offsets, labels):
        counts = [counters[label].get(l, 0) for l in all_lengths]
        plt.bar(x + offset * width, counts, width, label=label)

    plt.xticks(x, all_lengths)
    plt.xlabel("IP Header Length (bytes)")
    plt.ylabel("Packet Count")
    plt.title(
        "IPv4 Header Length Distribution (Discrete)\n"
        "Telegram vs Non-Telegram – Datasets 1 & 2"
    )
    plt.legend()
    plt.tight_layout()

    out = EXPORT_PATH / "ip_header_length_comparison_ds1_ds2.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[✔] Chart saved to: {out}")

if __name__ == "__main__":
    main()
