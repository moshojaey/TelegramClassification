import json
import base64
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scapy.all import Ether, TCP, IP

# ===================== PATHS =====================
BASE_PATH = Path(r"D:\New_ITC_Reformatted\Reselected")
EXPORT_PATH = Path(r"/DataReport/Packet_Analysis/Basic Analysis/Exports")
EXPORT_PATH.mkdir(parents=True, exist_ok=True)

DATASETS = ["1", "2"]
CLASSES = ["Telegram", "Non-Telegram"]

FLAGS = {
    "SYN": lambda pkt: pkt.haslayer(TCP) and pkt[TCP].flags & 0x02,
    "ACK": lambda pkt: pkt.haslayer(TCP) and pkt[TCP].flags & 0x10,
    "FIN": lambda pkt: pkt.haslayer(TCP) and pkt[TCP].flags & 0x01,
    "RST": lambda pkt: pkt.haslayer(TCP) and pkt[TCP].flags & 0x04,
    "IP_Fragment": lambda pkt: pkt.haslayer(IP) and (pkt[IP].flags.MF == 1 or pkt[IP].frag > 0)
}

# ===================== CORE EXTRACTION =====================
def compute_flag_ratio(dataset, cls, flag_fn):
    path = BASE_PATH / dataset / f"{cls}.json"
    with open(path, "r") as f:
        packets = json.load(f)["packets"]

    count = 0
    total = 0

    for p in packets:
        try:
            raw = base64.b64decode(p["raw"])
            pkt = Ether(raw)
            total += 1
            if flag_fn(pkt):
                count += 1
        except Exception:
            continue

    if total == 0:
        return 0.0
    return count / total

# ===================== PLOTTING =====================
def plot_flag(flag_name, flag_fn):
    labels = []
    values = []

    for ds in DATASETS:
        for cls in CLASSES:
            label = f"DS{ds}-{cls}"
            ratio = compute_flag_ratio(ds, cls, flag_fn)
            labels.append(label)
            values.append(ratio)
            print(f"{flag_name} | {label}: {ratio:.4f}")

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, values)

    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Proportion of packets")
    plt.ylim(0, 1)
    plt.title(f"{flag_name} Flag Presence\nTelegram vs Non-Telegram (Datasets 1 & 2)")

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2%}",
                 ha="center", va="bottom")

    plt.tight_layout()
    out = EXPORT_PATH / f"{flag_name}_comparison_ds1_ds2.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[✔] Saved: {out}")

# ===================== MAIN =====================
def main():
    for name, fn in FLAGS.items():
        plot_flag(name, fn)

if __name__ == "__main__":
    main()
