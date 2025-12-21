import json
import base64
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from scapy.all import Ether, IP, TCP, UDP

# =========================
# Paths
# =========================
BASE = Path(r"D:\New_ITC_Reformatted\Reselected")
EXPORT = Path(r"/DataReport/Packet_Analysis/Basic Analysis/Exports")
EXPORT.mkdir(parents=True, exist_ok=True)

DATASETS = ["1", "2"]
CLASSES = ["Telegram", "Non-Telegram"]

# =========================
# Entropy function
# =========================
def entropy(data: bytes) -> float:
    if not data:
        return None
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    total = len(data)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())

# =========================
# Packet parsing helpers
# =========================
def parse_packet(raw_b64):
    try:
        raw = base64.b64decode(raw_b64)
        pkt = Ether(raw)
        return pkt
    except Exception:
        return None

def split_packet(pkt):
    header = b""
    payload = b""

    if IP in pkt:
        header += bytes(pkt[IP])
        if TCP in pkt:
            payload = bytes(pkt[TCP].payload)
        elif UDP in pkt:
            payload = bytes(pkt[UDP].payload)

    return header, payload

# =========================
# Load packets
# =========================
def load_packets(ds, cls):
    path = BASE / ds / f"{cls}.json"
    with open(path, "r") as f:
        return json.load(f)["packets"]

# =========================
# Collect entropy values
# =========================
results = {
    "total": {},
    "header": {},
    "payload": {}
}

for ds in DATASETS:
    for cls in CLASSES:
        key = f"DS{ds}-{cls}"
        packets = load_packets(ds, cls)

        total_e, header_e, payload_e = [], [], []

        for p in packets:
            pkt = parse_packet(p["raw"])
            if not pkt:
                continue

            raw = bytes(pkt)
            h, pl = split_packet(pkt)

            te = entropy(raw)
            he = entropy(h)
            pe = entropy(pl)

            if te is not None:
                total_e.append(te)
            if he is not None:
                header_e.append(he)
            if pe is not None:
                payload_e.append(pe)

        results["total"][key] = total_e
        results["header"][key] = header_e
        results["payload"][key] = payload_e

# =========================
# Plotting
# =========================
def plot_entropy(metric, title, filename):
    plt.figure(figsize=(10, 6))

    labels = []
    means = []

    for ds in DATASETS:
        for cls in CLASSES:
            key = f"DS{ds}-{cls}"
            vals = results[metric][key]
            if not vals:
                continue
            labels.append(key)
            means.append(sum(vals) / len(vals))

    plt.bar(labels, means)
    plt.ylabel("Average Entropy (bits)")
    plt.title(title)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(EXPORT / filename)
    plt.close()

plot_entropy(
    "total",
    "Total Packet Entropy (Telegram vs Non-Telegram)",
    "entropy_total_packet.png"
)

plot_entropy(
    "header",
    "Header Entropy (IP + Transport)",
    "entropy_header.png"
)

plot_entropy(
    "payload",
    "Payload Entropy",
    "entropy_payload.png"
)

print("✔ Entropy analysis complete. PNGs exported.")
