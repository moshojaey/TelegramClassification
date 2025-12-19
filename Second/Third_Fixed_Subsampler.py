import os
import json
import random
from pathlib import Path
from scapy.all import rdpcap, DNS, IP, TCP, UDP   # noqa: F401

# ===============================================================
# Telegram Dataset Subsampling Script
# ===============================================================

SOURCE_DIR       = Path(r"D:\New_ITC_Reformatted\Renamed")
DEST_DIR         = Path(r"D:\New_ITC_Reformatted\SubSampled")
COUNTS_JSON      = SOURCE_DIR / "pcap_counts.json"   # optional, if you have it
TARGET_PER_CLASS = 50_000
MAX_PKT_LEN      = 1500
random.seed(42)


def preprocess_packet(packet):
    """Normalize and clean packet bytes for feature extraction."""
    if packet.haslayer(DNS):
        return None
    if packet.haslayer(TCP):
        t = packet[TCP]
        if (t.flags & 0x02 or t.flags & 0x01 or t.flags & 0x10) and len(t.payload) == 0:
            return None
    if not packet.haslayer(IP):
        return None

    ip_layer = packet[IP]
    raw = bytearray(bytes(ip_layer))

    # Zero out IP identification, checksum, and addresses
    for i in range(12, 20):
        raw[i] = 0

    # Handle UDP header anonymization
    hlen = (raw[0] & 0x0F) * 4
    if raw[9] == 17:  # UDP protocol
        off = hlen
        raw = raw[:off] + raw[off:off + 8] + b"\x00" * 12 + raw[off + 8:]

    # Truncate or pad to fixed size
    if len(raw) > MAX_PKT_LEN:
        raw = raw[:MAX_PKT_LEN]
    else:
        raw += b"\x00" * (MAX_PKT_LEN - len(raw))

    return [b / 255.0 for b in raw]


def sample_from_file(path, n_needed):
    """Sample up to n_needed valid packets from a pcap file."""
    chosen = []
    if n_needed == 0:
        return chosen
    try:
        pkts = rdpcap(str(path))
    except Exception as e:
        print(f"[!] Failed to read {path}: {e}")
        return chosen

    total = len(pkts)
    if total == 0:
        return chosen

    idx_set = set(random.sample(range(total), min(n_needed, total)))
    for idx, pkt in enumerate(pkts):
        if idx not in idx_set:
            continue
        p = preprocess_packet(pkt)
        if p is not None:
            chosen.append(p)
            if len(chosen) == n_needed:
                break

    # fallback: fill remaining with sequential scan
    if len(chosen) < n_needed:
        for idx, pkt in enumerate(pkts):
            if idx in idx_set:
                continue
            p = preprocess_packet(pkt)
            if p is not None:
                chosen.append(p)
                if len(chosen) == n_needed:
                    break
    return chosen


def count_packets_in_folder(folder):
    """Generate a dictionary: filename -> packet count."""
    counts = {}
    for file in os.listdir(folder):
        if not file.lower().endswith((".pcap", ".pcapng")):
            continue
        path = os.path.join(folder, file)
        try:
            pkts = rdpcap(path)
            counts[file] = len(pkts)
        except Exception as e:
            print(f"[!] Error reading {file}: {e}")
            counts[file] = 0
    return counts


def build_plan(file_counts, target):
    """Decide how many packets to sample from each file."""
    remaining = target
    files = list(file_counts.keys())
    plan = {f: 0 for f in files}
    while remaining > 0:
        cands = [f for f in files if plan[f] < file_counts[f]]
        if not cands:
            break
        share = max(1, remaining // len(cands))
        for f in cands:
            avail = file_counts[f] - plan[f]
            take = min(share, avail)
            plan[f] += take
            remaining -= take
            if remaining == 0:
                break
    return plan


def main():
 #  targets = [
 #       ("1", "Telegram"), ("1", "Non-Telegram"),
#       ("2", "Telegram"), ("2", "Non-Telegram"),
 #       ("3", "Telegram"), ("3", "Non-Telegram"),
 #   ]

    targets = [
        ("3", "Telegram"),
        ("3", "Non-Telegram"),
    ]

    for ds, cat in targets:
        print(f"\n=== Processing Dataset {ds} | {cat} ===")

        src_path = SOURCE_DIR / ds / cat
        out_dir = DEST_DIR / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        file_counts = count_packets_in_folder(src_path)
        if not file_counts:
            print(f"[!] No pcap files found in {src_path}")
            continue

        plan = build_plan(file_counts, TARGET_PER_CLASS)

        harvested = []
        for fname, need in plan.items():
            picked = sample_from_file(src_path / fname, need)
            harvested.extend(picked)
            print(f"{fname:<20} -> {len(picked):>6}")

        print(f"Total harvested: {len(harvested):,}")

        output_file = out_dir / f"{cat}.json"
        with open(output_file, "w") as f:
            json.dump({"features": harvested, "label": cat.lower()}, f)

        print(f"Saved → {output_file}")

    print("\n✔ Done! Subsampling complete.")


if __name__ == "__main__":
    main()
