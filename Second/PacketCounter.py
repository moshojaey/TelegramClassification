import os
import subprocess
import json
import sys
import re

ROOT_DIR = r"D:\ITC Dataset"
OUTPUT_JSON_FILE = "pcap_counts.json"


def get_packet_count_capinfos(pcap_file_path):
    command = ['capinfos', '-c', pcap_file_path]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        output = result.stdout.strip()

        # Extract 'Number of packets' line
        match = re.search(r'Number of packets:\s*([\d\.]+)\s*([kKmM]?)', output)

        if match:
            number = float(match.group(1))
            suffix = match.group(2).lower()

            if suffix == 'k':
                number *= 1000
            elif suffix == 'm':
                number *= 1000000

            return int(number)
        else:
            try:
                return int(output)
            except ValueError:
                print(
                    f"  Warning: Could not parse packet count from capinfos output for '{pcap_file_path}'. Output: '{output}'")
                return None
    except FileNotFoundError:
        print(f"Error: 'capinfos' command not found. Is Wireshark/TShark installed and in your PATH?", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"  Error running capinfos for '{pcap_file_path}': {e}", file=sys.stderr)
        print(f"  Stderr: {e.stderr.strip()}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  An unexpected error occurred while processing '{pcap_file_path}' with capinfos: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    print(f"Starting packet count process for directory: {ROOT_DIR}")
    print("Ensure Wireshark/TShark ('capinfos') is installed and in your PATH.")

    results = {}

    if not os.path.isdir(ROOT_DIR):
        print(f"Error: Root directory '{ROOT_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        subdirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    except OSError as e:
        print(f"Error accessing directory '{ROOT_DIR}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found subdirectories: {subdirs}")

    for dir_name in subdirs:
        current_dir_path = os.path.join(ROOT_DIR, dir_name)
        print(f"\n--- Processing Directory: {dir_name} ---")
        results[dir_name] = {}

        try:
            files_in_dir = [f for f in os.listdir(current_dir_path) if
                            os.path.isfile(os.path.join(current_dir_path, f))]
        except OSError as e:
            print(f"  Warning: Could not access files in '{current_dir_path}': {e}. Skipping directory.")
            continue

        found_pcaps = 0
        for file_name in files_in_dir:
            if file_name.lower().endswith('.pcap'):
                base_name = file_name[:-5]
                if base_name.isdigit():
                    found_pcaps += 1
                    full_pcap_path = os.path.join(current_dir_path, file_name)
                    print(f"  Processing file: {file_name}...")

                    packet_count = get_packet_count_capinfos(full_pcap_path)

                    if packet_count is not None:
                        results[dir_name][file_name] = packet_count
                        print(f"    Packet Count: {packet_count}")
                    else:
                        results[dir_name][file_name] = "Error/NotCounted"
                        print(f"    Packet Count: Failed")

        if found_pcaps == 0:
            print("  No files matching '[number].pcap' found in this directory.")

    print("\n--- Final Results ---")

    try:
        json_output = json.dumps(results, indent=4)
        print(json_output)

        save_to_file = True
        if save_to_file:
            output_file_path = os.path.join(ROOT_DIR, OUTPUT_JSON_FILE)
            try:
                with open(output_file_path, 'w') as f:
                    f.write(json_output)
                print(f"\nResults also saved to: {output_file_path}")
            except IOError as e:
                print(f"\nError saving JSON to file '{output_file_path}': {e}", file=sys.stderr)

    except TypeError as e:
        print(f"Error converting results to JSON: {e}", file=sys.stderr)
        print("Raw results dictionary:", results)

    print("\nScript finished.")
