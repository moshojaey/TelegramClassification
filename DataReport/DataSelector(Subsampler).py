import os
import json
import random
import base64
from pathlib import Path
from scapy.all import rdpcap   # noqa: F401

# ===============================================================
# Telegram Dataset Subsampling Script (RAW / NO PREPROCESSING)
# ===============================================================

SOURCE_DIR       = Path(r"D:\New_ITC_Reformatted\Renamed")
DEST_DIR         = Path(r"D:\New_ITC_Reformatted\Reselected")
TARGET_PER_CLASS = 50_000

random.seed(42)


# ---------------------------------------------------------------
# Raw packet extraction (NO MODIFICATION)
# ---------------------------------------------------------------

def extract_raw_packet(packet):
    """Return raw packet bytes encoded as Base64 for JSON storage."""
    try:
        raw = bytes(packet)
        return base64.b64encode(raw).decode("ascii")
    except Exception:
        return None


# ---------------------------------------------------------------
# Subsample packets from a single pcap file
# ---------------------------------------------------------------

def sample_from_file(path, n_needed):
    """Sample up to n_needed packets from a pcap file WITHOUT modification."""
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

        raw = extract_raw_packet(pkt)
        if raw is not None:
            chosen.append({
                "raw": raw,
                "length": len(pkt),
                "layers": [layer.__name__ for layer in pkt.layers()]
            })

        if len(chosen) == n_needed:
            break

    # fallback sequential fill
    if len(chosen) < n_needed:
        for idx, pkt in enumerate(pkts):
            if idx in idx_set:
                continue

            raw = extract_raw_packet(pkt)
            if raw is not None:
                chosen.append({
                    "raw": raw,
                    "length": len(pkt),
                    "layers": [layer.__name__ for layer in pkt.layers()]
                })

            if len(chosen) == n_needed:
                break

    return chosen



# ---------------------------------------------------------------
# Count packets per file
# ---------------------------------------------------------------

def count_packets_in_folder(folder):
    """Generate a dictionary: filename -> packet count."""
    counts = {}

    for file in os.listdir(folder):
        if not file.lower().endswith((".pcap", ".pcapng")):
            continue

        path = folder / file
        try:
            counts[file] = len(rdpcap(str(path)))
        except Exception as e:
            print(f"[!] Error reading {file}: {e}")
            counts[file] = 0

    return counts


# ---------------------------------------------------------------
# Build fair subsampling plan
# ---------------------------------------------------------------

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


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    targets = [
        ("2", "Telegram"),
        ("2", "Non-Telegram"),
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
            print(f"{fname:<25} -> {len(picked):>6}")

        print(f"Total harvested: {len(harvested):,}")

        output_file = out_dir / f"{cat}.json"
        with open(output_file, "w") as f:
            json.dump({
                "label": cat.lower(),
                "packets": harvested
            }, f)

        print(f"Saved → {output_file}")

    print("\n✔ Done! Raw subsampling complete.")


if __name__ == "__main__":
    main()
