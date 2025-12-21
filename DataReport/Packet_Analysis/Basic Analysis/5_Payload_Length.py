import json
import base64
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Paths
# ============================================================

BASE_PATH = Path(r"D:\New_ITC_Reformatted\Reselected")
EXPORT_PATH = Path(r"/DataReport/Packet_Analysis/Basic Analysis/Exports")
EXPORT_PATH.mkdir(parents=True, exist_ok=True)

DATASETS = ["1", "2"]
CLASSES = ["Telegram", "Non-Telegram"]

ETH_HEADER_LEN = 14

# ============================================================
# Helpers
# ============================================================

def extract_payload_lengths(dataset, cls):
    path = BASE_PATH / dataset / f"{cls}.json"
    with open(path, "r") as f:
        packets = json.load(f)["packets"]

    payloads = []

    for pkt in packets:
        try:
            raw = base64.b64decode(pkt["raw"])

            if len(raw) < ETH_HEADER_LEN + 20:
                continue

            # Ethernet
            ethertype = int.from_bytes(raw[12:14], "big")
            if ethertype != 0x0800:
                continue

            # IP
            ip_start = ETH_HEADER_LEN
            first_byte = raw[ip_start]
            version = first_byte >> 4
            if version != 4:
                continue

            ihl = (first_byte & 0x0F) * 4
            total_length = int.from_bytes(raw[ip_start + 2:ip_start + 4], "big")

            if total_length < ihl:
                continue

            proto = raw[ip_start + 9]
            transport_start = ip_start + ihl

            # TCP
            if proto == 6 and len(raw) >= transport_start + 13:
                tcp_offset_byte = raw[transport_start + 12]
                tcp_hlen = ((tcp_offset_byte >> 4) & 0x0F) * 4
                payload_len = total_length - ihl - tcp_hlen

            # UDP
            elif proto == 17 and len(raw) >= transport_start + 8:
                udp_hlen = 8
                payload_len = total_length - ihl - udp_hlen

            else:
                continue

            if payload_len >= 0:
                payloads.append(payload_len)

        except Exception:
            continue

    return payloads

# ============================================================
# Main
# ============================================================

def main():
    results = {}

    for ds in DATASETS:
        for cls in CLASSES:
            label = f"DS{ds}-{cls}"
            values = extract_payload_lengths(ds, cls)
            results[label] = values
            print(f"{label}: {len(values)} packets")

    if all(len(v) == 0 for v in results.values()):
        print("[✘] No valid payload data found.")
        return

    # Common bins for fair comparison
    all_values = np.concatenate([v for v in results.values() if v])
    bins = np.linspace(0, np.percentile(all_values, 99), 60)

    plt.figure(figsize=(12, 6))

    for label, values in results.items():
        if not values:
            continue
        plt.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=label
        )

    plt.xlabel("Payload Length (bytes)")
    plt.ylabel("Density")
    plt.title(
        "Payload Length Distribution\n"
        "Telegram vs Non-Telegram – Datasets 1 & 2"
    )
    plt.legend()
    plt.tight_layout()

    out = EXPORT_PATH / "payload_length_distribution_ds1_ds2.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[✔] Chart saved to: {out}")

if __name__ == "__main__":
    main()
