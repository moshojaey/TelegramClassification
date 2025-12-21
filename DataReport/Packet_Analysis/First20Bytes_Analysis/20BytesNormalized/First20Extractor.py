import json
import base64
from pathlib import Path
from collections import defaultdict

# ==========================================================
# CONFIG
# ==========================================================
SOURCE_ROOT = Path(r"D:\New_ITC_Reformatted\Reselected")
DEST_ROOT   = Path(r"D:\New_ITC_Reformatted\first20bytes")

DATASETS = ["1", "2", "3"]
CATEGORIES = ["Telegram", "Non-Telegram"]

DEST_ROOT.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Feature extraction from first 20 bytes
# ==========================================================
def extract_first20_features(raw_bytes: bytes):
    if len(raw_bytes) < 20:
        return None

    features = {}

    # Ethernet DST MAC
    for i in range(6):
        features[f"eth_dst_mac_{i}"] = raw_bytes[i]

    # Ethernet SRC MAC
    for i in range(6):
        features[f"eth_src_mac_{i}"] = raw_bytes[6 + i]

    # EtherType
    features["eth_type"] = (raw_bytes[12] << 8) | raw_bytes[13]

    # IP Version + IHL
    features["ip_version"] = raw_bytes[14] >> 4
    features["ip_ihl"] = raw_bytes[14] & 0x0F

    # IP TOS
    features["ip_tos"] = raw_bytes[15]

    # IP Total Length
    features["ip_total_length"] = (raw_bytes[16] << 8) | raw_bytes[17]

    # IP Identification
    features["ip_identification"] = (raw_bytes[18] << 8) | raw_bytes[19]

    return features


# ==========================================================
# MAIN PROCESS
# ==========================================================
def process_dataset(ds, category):
    src_file = SOURCE_ROOT / ds / f"{category}.json"
    dst_dir  = DEST_ROOT / ds
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_file = dst_dir / f"{category}.json"

    with open(src_file, "r") as f:
        data = json.load(f)

    extracted = []

    for pkt in data["packets"]:
        raw = base64.b64decode(pkt["raw"])
        feats = extract_first20_features(raw)
        if feats:
            extracted.append(feats)

    with open(dst_file, "w") as f:
        json.dump({
            "label": category.lower(),
            "feature_set": "first_20_bytes_semantic",
            "packet_count": len(extracted),
            "features": extracted
        }, f, indent=2)

    print(f"[✓] DS{ds}-{category}: {len(extracted)} packets saved")


def main():
    for ds in DATASETS:
        for cat in CATEGORIES:
            process_dataset(ds, cat)

    print("\n✔ All datasets processed successfully.")


if __name__ == "__main__":
    main()
