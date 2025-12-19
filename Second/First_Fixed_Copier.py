import os
import shutil

def copy_pcap_files(source_folder, destination_folder):
    """
    Recursively copy all .pcap and .pcapng files from source_folder to destination_folder.
    If duplicate filenames exist, automatically rename them.
    """

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    count = 0
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith((".pcap", ".pcapng")):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(destination_folder, file)

                # Handle duplicates
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(destination_folder, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.copy2(src_path, dst_path)
                count += 1
                print(f"Copied: {src_path} → {dst_path}")

    print(f"\n✅ Done. Total {count} files copied to '{destination_folder}'.")


# Example usage:
copy_pcap_files("D:/New_ITC_Data/3", "D:/New_ITC_Reformatted/Raw/3/Non-Telegram")
