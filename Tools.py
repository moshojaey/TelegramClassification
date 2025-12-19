from scapy.all import rdpcap
from datetime import datetime
import os
import csv

def Count_Single_Pcap(file_path):
    """Counts the number of packets in a single pcap or pcapng file."""
    try:
        packets = rdpcap(file_path)
        return len(packets)
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return 0


def Count_Pcap_in_Folder(
        source_path=r"D:\New_ITC_Data",
        destination_path=r"D:\Codes\LabProj\Telegram_New\Result\Count"
):
    """Recursively counts packets in all .pcap and .pcapng files in source_path and logs progress."""

    os.makedirs(destination_path, exist_ok=True)

    # Timestamped output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(destination_path, f"pcap_packet_counts_{timestamp}.csv")

    results = []
    total_files = 0
    total_packets = 0

    print(f"\n🔍 Starting packet count scan in: {source_path}\n")

    for root, dirs, files in os.walk(source_path):
        for filename in files:
            if filename.endswith(".pcap") or filename.endswith(".pcapng"):
                file_path = os.path.join(root, filename)
                print(f"📄 Processing: {file_path}")
                count = Count_Single_Pcap(file_path)
                print(f"   → {count} packets found\n")

                results.append([file_path, filename, count])
                total_files += 1
                total_packets += count

    # Write CSV report
    with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Full Path", "File Name", "Packet Count"])
        writer.writerows(results)

    print("✅ Finished counting.")
    print(f"📊 Total files processed: {total_files}")
    print(f"📦 Total packets across all files: {total_packets}")
    print(f"📝 Report saved to: {output_file}")
