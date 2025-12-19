import json
from pathlib import Path

BASE_DIR = Path(r"D:\New_ITC_Reformatted\Reselected")
DATASETS = ["1", "2"]
FILES = ["Telegram.json", "Non-Telegram.json"]


def count_packets(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if "packets" not in data:
            print(f"[!] {json_path.name}: missing 'packets' key")
            return 0

        return len(data["packets"])

    except Exception as e:
        print(f"[!] Failed to read {json_path}: {e}")
        return 0


def main():
    print("\n=== PACKET COUNT REPORT ===\n")

    for ds in DATASETS:
        ds_path = BASE_DIR / ds
        print(f"Dataset {ds}")

        for fname in FILES:
            path = ds_path / fname
            if not path.exists():
                print(f"  {fname:<18} -> FILE NOT FOUND")
                continue

            count = count_packets(path)
            print(f"  {fname:<18} -> {count:,} packets")

        print()

    print("✔ Done.")


if __name__ == "__main__":
    main()
