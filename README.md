## 📦 Dataset Construction and Subsampling Methodology

This section documents how network packets were selected, stored, and structured for analysis.  
The subsampling process is designed to be fair, reproducible, and to preserve the complete raw packet structure.

---

### 1. Original Data Organization

Raw network traffic is stored as PCAP / PCAPNG files and organized by dataset index and traffic class:

D:\New_ITC_Reformatted\Renamed\
├── 1\
│   ├── Telegram\
│   │   ├── 1.pcap
│   │   ├── 2.pcap
│   │   └── ...
│   └── Non-Telegram\
│       ├── 1.pcap
│       └── ...
├── 2\
│   ├── Telegram\
│   └── Non-Telegram\

Each PCAP file contains raw captured packets with no preprocessing applied.

---

### 2. Subsampling Objective

For each dataset and traffic class, a fixed number of packets is selected:

- Target size: 50,000 packets per class
- Classes: Telegram / Non-Telegram
- Granularity: Packet-level (not flow-based)

The goal is to create balanced datasets while preserving packet diversity and avoiding assumptions about flow reconstruction.

---

### 3. Fair Distribution Across PCAP Files

To avoid bias toward larger PCAP files, packets are sampled proportionally from all available PCAP files:

1. Packet counts are computed for each PCAP file
2. A fair sampling plan distributes the target number across files
3. If a PCAP contains fewer packets, remaining packets are redistributed to others

This ensures that no single capture dominates the dataset.

---

### 4. Packet Selection Procedure

For each PCAP file:

- Packet indices are selected randomly using a fixed seed (random.seed(42))
- Selected packets are extracted exactly as captured
- No filtering, normalization, truncation, padding, or anonymization is applied

The original packet structure is fully preserved, including headers and payloads.

---

### 5. Output Location

Subsampled datasets are stored in the following directory structure:

D:\New_ITC_Reformatted\Reselected\
├── 1\
│   ├── Telegram.json
│   └── Non-Telegram.json
├── 2\
│   ├── Telegram.json
│   └── Non-Telegram.json

---

### 6. JSON Data Format

Each output file follows this schema:

{
  "label": "telegram",
  "packets": [
    {
      "raw": "<Base64-encoded raw packet bytes>",
      "length": 1514,
      "layers": ["Ether", "IP", "TCP"]
    }
  ]
}

Field description:
- label: Traffic class (telegram or non-telegram)
- raw: Full raw packet bytes encoded in Base64
- length: Original packet size in bytes
- layers: Protocol layers detected by Scapy

---

### 7. Rationale for Base64 Encoding

Raw packet bytes cannot be stored directly in JSON.  
Base64 encoding preserves the exact binary representation of each packet while allowing safe storage and later reconstruction.

---

### 8. Capabilities Enabled by This Format

Because packets are stored raw and unmodified, the dataset supports:

- Protocol and layer distribution analysis
- Packet length statistics
- TCP flag and control behavior analysis
- TTL and path length estimation
- Header-level fingerprinting
- Direct comparison between Telegram and Non-Telegram traffic

---

### 9. Reproducibility

- Fixed random seed
- Deterministic sampling strategy
- Transparent directory layout
- No hidden preprocessing steps

The subsampling process is fully reproducible given the original PCAP files.
