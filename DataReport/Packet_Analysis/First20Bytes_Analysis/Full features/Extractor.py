import json
import base64
import random
from pathlib import Path
from scapy.all import IP, Ether

# ===============================================================
# CONFIGURATION
# ===============================================================
SOURCE_ROOT = Path(r"D:\New_ITC_Reformatted\Reselected")
DEST_ROOT = Path(r"D:\New_ITC_Reformatted\Features20Full")

DATASETS = ["1", "2"]
CATEGORIES = ["Telegram", "Non-Telegram"]
TARGET_PER_CLASS = 50_000
random.seed(42)

DEST_ROOT.mkdir(parents=True, exist_ok=True)


# ===============================================================
# FULL IP HEADER FEATURE EXTRACTION (10 Features)
# ===============================================================
def extract_full_features(packet_bytes: bytes):
    """
    Extracts all standard IP header fields as numerical features.
    Returns list of 10 floats.
    """
    try:
        pkt = Ether(packet_bytes)
        if not pkt.haslayer(IP):
            return None

        ip = pkt[IP]

        # 1. Version (Usually 4)
        f_ver = ip.version / 15.0

        # 2. IHL (Internet Header Length, usually 5)
        f_ihl = ip.ihl / 15.0

        # 3. TOS (Type of Service / DSCP)
        f_tos = ip.tos / 255.0

        # 4. Total Length (Bytes 2-3)
        f_len = min(ip.len, 1500) / 1500.0

        # 5. Identification (Bytes 4-5)
        f_id = ip.id / 65535.0

        # 6. Flags (3 bits)
        f_flags = int(ip.flags) / 7.0

        # 7. Fragment Offset (13 bits)
        f_frag = ip.frag / 8191.0

        # 8. TTL (Time to Live)
        f_ttl = ip.ttl / 255.0

        # 9. Protocol (TCP=6, UDP=17)
        f_proto = ip.proto / 255.0

        # 10. Header Checksum (16 bits)
        f_chk = ip.chksum / 65535.0

        return [f_ver, f_ihl, f_tos, f_len, f_id, f_flags, f_frag, f_ttl, f_proto, f_chk]

    except Exception:
        return None


# ===============================================================
# PROCESSING LOOP
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
            vec = extract_full_features(raw_bytes)
            if vec is not None:
                features.append(vec)
        except:
            continue

    with open(out_file, "w") as f:
        json.dump({"features": features, "label": category.lower()}, f)

    print(f"[✓] DS{ds}-{category}: Extracted {len(features)} vectors (10 features).")


def main():
    for ds in DATASETS:
        for cat in CATEGORIES:
            process_dataset(ds, cat)
    print(f"\n✔ Full Extraction complete. Saved to {DEST_ROOT}")


if __name__ == "__main__":
    main()