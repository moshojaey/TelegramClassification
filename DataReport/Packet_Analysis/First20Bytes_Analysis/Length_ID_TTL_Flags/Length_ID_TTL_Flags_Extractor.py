import json
import base64
import random
from pathlib import Path
from scapy.all import IP, Ether

# ===============================================================
# CONFIGURATION
# ===============================================================
SOURCE_ROOT = Path(r"D:\New_ITC_Reformatted\Reselected")
DEST_ROOT = Path(r"D:\New_ITC_Reformatted\Length_ID_TTL_Flags")

DATASETS = ["1", "2"]
CATEGORIES = ["Telegram", "Non-Telegram"]
TARGET_PER_CLASS = 50_000
random.seed(42)

DEST_ROOT.mkdir(parents=True, exist_ok=True)


# ===============================================================
# FEATURE EXTRACTION LOGIC
# ===============================================================
def extract_metadata(packet_bytes: bytes):
    """
    Extracts only 4 specific IP header fields.
    Returns normalized vector: [Length, ID, TTL, Flags]
    """
    try:
        pkt = Ether(packet_bytes)
        if not pkt.haslayer(IP):
            return None

        ip = pkt[IP]

        # 1. Total Length (Bytes 2-3) - Max ~1500 (standard) or 65535
        feat_len = min(ip.len, 1500) / 1500.0

        # 2. Identification (Bytes 4-5) - Max 65535
        feat_id = ip.id / 65535.0

        # 3. Time To Live (Byte 8) - Max 255
        feat_ttl = ip.ttl / 255.0

        # 4. Flags (Bytes 6-7) - 3 bits (Reserved, DF, MF). Max value 7.
        # Cast FlagValue to int.
        feat_flags = int(ip.flags) / 7.0

        return [feat_len, feat_id, feat_ttl, feat_flags]

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

        try:
            raw_bytes = base64.b64decode(pkt["raw"])
            vec = extract_metadata(raw_bytes)
            if vec is not None:
                features.append(vec)
        except:
            continue

    with open(out_file, "w") as f:
        json.dump({"features": features, "label": category.lower()}, f)

    print(f"[✓] DS{ds}-{category}: Extracted {len(features)} vectors.")


def main():
    for ds in DATASETS:
        for cat in CATEGORIES:
            process_dataset(ds, cat)
    print("\n✔ Extraction complete. Saved to D:\\New_ITC_Reformatted\\Length_ID_TTL_Flags")


if __name__ == "__main__":
    main()