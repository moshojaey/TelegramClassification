import json
import base64
from collections import Counter
from pathlib import Path

# ==============================
# Configuration
# ==============================

BASE_PATH = Path(r"D:\New_ITC_Reformatted\Reselected")
DATASETS = {
    "DS1-Telegram": BASE_PATH / "1" / "Telegram.json",
    "DS1-NonTelegram": BASE_PATH / "1" / "Non-Telegram.json",
    "DS2-Telegram": BASE_PATH / "2" / "Telegram.json",
    "DS2-NonTelegram": BASE_PATH / "2" / "Non-Telegram.json",
}

MAX_BYTES = 20


# ==============================
# Structural labeling logic
# ==============================

def label_first_20_bytes(packet_bytes: bytes):
    labels = ["UNKNOWN"] * MAX_BYTES
    length = len(packet_bytes)

    if length < 14:
        return tuple(labels)

    # Ethernet
    for i in range(0, 6):
        labels[i] = "ETH_DST_MAC"
    for i in range(6, 12):
        labels[i] = "ETH_SRC_MAC"
    labels[12] = labels[13] = "ETH_TYPE"

    eth_type = int.from_bytes(packet_bytes[12:14], byteorder="big")

    # IPv4
    if eth_type == 0x0800 and length >= 20:
        labels[14] = "IP_VERSION_IHL"
        labels[15] = "IP_TOS"
        labels[16] = labels[17] = "IP_TOTAL_LENGTH"
        labels[18] = labels[19] = "IP_IDENTIFICATION"

    # IPv6
    elif eth_type == 0x86DD and length >= 20:
        labels[14] = labels[15] = "IPV6_VERSION_TC"
        labels[16] = labels[17] = labels[18] = labels[19] = "IPV6_FLOW_LABEL"

    return tuple(labels)


# ==============================
# Dataset processing
# ==============================

def analyze_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    counter = Counter()

    for pkt in data["packets"]:
        raw = base64.b64decode(pkt["raw"])
        layout = label_first_20_bytes(raw)
        counter[layout] += 1

    return counter


# ==============================
# Main
# ==============================

def main():
    for name, path in DATASETS.items():
        print(f"\n=== {name} ===")

        histogram = analyze_dataset(path)

        for idx, (layout, count) in enumerate(histogram.most_common(10), 1):
            print(f"\nScenario #{idx} — {count} packets")
            for i, label in enumerate(layout):
                print(f"  Byte {i:02d}: {label}")

        print(f"\nTotal unique structural scenarios: {len(histogram)}")


if __name__ == "__main__":
    main()
