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

ETH_LEN = 14

# ============================================================
# Feature Extraction
# ============================================================

def extract_header_payload_ratio(dataset, cls):
    path = BASE_PATH / dataset / f"{cls}.json"
    with open(path, "r") as f:
        packets = json.load(f)["packets"]

    ratios = []

    for pkt in packets:
        try:
            raw = base64.b64decode(pkt["raw"])

            if len(raw) < ETH_LEN + 20:
                continue

            # Ethernet
            if int.from_bytes(raw[12:14], "big") != 0x0800:
                continue

            # IP
            ip_start = ETH_LEN
            ip_first = raw[ip_start]
            if (ip_first >> 4) != 4:
                continue

            ihl = (ip_first & 0x0F) * 4
            total_ip_len = int.from_bytes(
                raw[ip_start + 2: ip_start + 4], "big"
            )

            proto = raw[ip_start + 9]
            transport_start = ip_start + ihl

            # TCP
            if proto == 6 and len(raw) >= transport_start + 13:
                tcp_offset = (raw[transport_start + 12] >> 4) * 4
                payload_len = total_ip_len - ihl - tcp_offset
                header_len = ETH_LEN + ihl + tcp_offset

            # UDP
            elif proto == 17 and len(raw) >= transport_start + 8:
                udp_len = 8
                payload_len = total_ip_len - ihl - udp_len
                header_len = ETH_LEN + ihl + udp_len

            else:
                continue

            if payload_len <= 0:
                continue

            ratios.append(header_len / payload_len)

        except Exception:
            continue

    return ratios

# ============================================================
# Main
# ============================================================

def main():
    data = {}

    for ds in DATASETS:
        for cls in CLASSES:
            key = f"DS{ds}-{cls}"
            values = extract_header_payload_ratio(ds, cls)
            data[key] = values
            print(f"{key}: {len(values)} valid packets")

    if not any(data.values()):
        print("[✘] No valid ratios extracted.")
        return

    plt.figure(figsize=(12, 6))

    bins = np.linspace(0, 5, 60)

    for label, values in data.items():
        if values:
            plt.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                label=label
            )

    plt.xlabel("Header / Payload Ratio")
    plt.ylabel("Density")
    plt.title(
        "Header-to-Payload Ratio Distribution\n"
        "Telegram vs Non-Telegram – Datasets 1 & 2"
    )
    plt.legend()
    plt.tight_layout()

    out = EXPORT_PATH / "header_to_payload_ratio_ds1_ds2.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[✔] Chart saved to: {out}")

if __name__ == "__main__":
    main()
