import json
import base64
import random
from pathlib import Path
from scapy.all import IP, Ether

# ===============================================================
# CONFIG
# ===============================================================
SOURCE_ROOT = Path(r"D:\New_ITC_Reformatted\Reselected")
DEST_ROOT = Path(r"D:\New_ITC_Reformatted\OnlyLength")

DATASETS = ["1", "2"]
CATEGORIES = ["Telegram", "Non-Telegram"]
TARGET_PER_CLASS = 50_000
random.seed(42)

DEST_ROOT.mkdir(parents=True, exist_ok=True)


# ===============================================================
# ONLY LENGTH EXTRACTION
# ===============================================================
def extract_length_only(packet_bytes: bytes):
    """
    Decodes packet, extracts IP Total Length, and normalizes it.
    Returns: [length / 1500.0]
    """
    try:
        pkt = Ether(packet_bytes)

        if not pkt.haslayer(IP):
            return None

        # Extract IP Total Length (integer)
        ip_len = pkt[IP].len

        # Normalize (Standard MTU is 1500, so we scale 0-1)
        # If jumbo frames exist (>1500), we clip or just let it be > 1.0 (fine for ReLU)
        norm_len = min(ip_len, 1500) / 1500.0

        return [norm_len]

    except Exception:
        return None


# ===============================================================
# PROCESS DATASETS
# ===============================================================
def process_dataset(ds, category):
    src_file = SOURCE_ROOT / ds / f"{category}.json"
    out_dir = DEST_ROOT / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{category}.json"

    if not src_file.exists():
        print(f"[!] Missing: {src_file}")
        return

    with open(src_file, "r") as f:
        data = json.load(f)

    packets = data["packets"]
    random.shuffle(packets)

    features = []

    for pkt in packets:
        if len(features) >= TARGET_PER_CLASS:
            break

        raw_bytes = base64.b64decode(pkt["raw"])
        val = extract_length_only(raw_bytes)

        if val is not None:
            features.append(val)

    with open(out_file, "w") as f:
        json.dump({"features": features, "label": category.lower()}, f)

    print(f"[✓] DS{ds}-{category}: Saved {len(features)} lengths.")


def main():
    for ds in DATASETS:
        for cat in CATEGORIES:
            process_dataset(ds, cat)
    print("\n✔ Extraction complete. Data contains ONLY packet length.")


if __name__ == "__main__":
    main()