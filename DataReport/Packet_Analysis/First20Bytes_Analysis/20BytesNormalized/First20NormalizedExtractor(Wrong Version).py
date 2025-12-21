import json
import base64
import random
from pathlib import Path
from scapy.all import IP, TCP, DNS, Ether

# ===============================================================
# CONFIG
# ===============================================================
SOURCE_ROOT = Path(r"D:\New_ITC_Reformatted\Reselected")
DEST_ROOT   = Path(r"D:\New_ITC_Reformatted\first20bytesNormalized")

DATASETS = ["1", "2"]
CATEGORIES = ["Telegram", "Non-Telegram"]

TARGET_PER_CLASS = 50_000
random.seed(42)

DEST_ROOT.mkdir(parents=True, exist_ok=True)

# ===============================================================
# FIRST-20-BYTE NORMALIZATION (mirrors old logic)
# ===============================================================
def preprocess_first20(packet_bytes: bytes):
    """
    Apply old normalization logic BUT ONLY on first 20 bytes.
    Return normalized 20-byte vector or None if packet is dropped.
    """

    try:
        pkt = Ether(packet_bytes)
    except Exception:
        return None

    # Drop DNS
    if pkt.haslayer(DNS):
        return None

    # Drop non-IP
    if not pkt.haslayer(IP):
        return None

    # Drop empty TCP control packets
    if pkt.haslayer(TCP):
        t = pkt[TCP]
        if (t.flags & 0x02 or t.flags & 0x01 or t.flags & 0x10) and len(t.payload) == 0:
            return None

    raw = bytearray(packet_bytes)

    if len(raw) < 20:
        return None

    # Zero IP identification, checksum, addresses (bytes 12–19)
    for i in range(12, 20):
        raw[i] = 0

    # Take ONLY first 20 bytes
    raw20 = raw[:20]

    # Normalize
    return [b / 255.0 for b in raw20]


# ===============================================================
# LOAD + SAMPLE + PROCESS
# ===============================================================
def process_dataset(ds, category):
    src_file = SOURCE_ROOT / ds / f"{category}.json"
    out_dir  = DEST_ROOT / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{category}.json"

    with open(src_file, "r") as f:
        data = json.load(f)

    packets = data["packets"]
    random.shuffle(packets)

    features = []

    for pkt in packets:
        if len(features) >= TARGET_PER_CLASS:
            break

        raw_bytes = base64.b64decode(pkt["raw"])
        vec = preprocess_first20(raw_bytes)

        if vec is not None:
            features.append(vec)

    with open(out_file, "w") as f:
        json.dump(
            {
                "features": features,
                "label": category.lower()
            },
            f
        )

    print(f"[✓] DS{ds}-{category}: {len(features)} packets saved")


# ===============================================================
# MAIN
# ===============================================================
def main():
    for ds in DATASETS:
        for cat in CATEGORIES:
            print(f"\n=== DS{ds} | {cat} ===")
            process_dataset(ds, cat)

    print("\n✔ First-20-byte normalization complete.")


if __name__ == "__main__":
    main()
