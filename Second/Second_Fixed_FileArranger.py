#arranages files for the dataset
#removes folders and unifies names of all

import os
import sys

SOURCE_DIRECTORIES = [
    r"D:\ITC Dataset\1_NonYoutube",
    r"D:\ITC Dataset\1_Youtube",
    r"D:\ITC Dataset\2_NonYoutube",
    r"D:\ITC Dataset\2_Youtube",
    r"D:\ITC Dataset\3_NonYoutube",
    r"D:\ITC Dataset\3_Youtube",
]

PCAP_EXTENSION = ".pcap"

def find_pcap_files(start_dir):
    pcap_files = []
    if not os.path.isdir(start_dir):
        print(f"Warning: Directory not found or is not a directory: {start_dir}")
        return pcap_files

    print(f"Searching for {PCAP_EXTENSION} files in '{start_dir}' and subdirectories...")
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.lower().endswith(PCAP_EXTENSION):
                full_path = os.path.join(root, file)
                pcap_files.append(full_path)
    print(f"Found {len(pcap_files)} {PCAP_EXTENSION} files.")
    return pcap_files

def unify_and_rename_pcaps(source_dir):
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist. Skipping.")
        return

    print(f"\n--- Processing Directory: {source_dir} ---")

    pcap_file_paths = find_pcap_files(source_dir)

    if not pcap_file_paths:
        print("No .pcap files found to process in this directory tree.")
        print(f"--- Finished Processing: {source_dir} ---")
        return

    rename_counter = 1
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for old_path in pcap_file_paths:
        try:
            target_filename_base = str(rename_counter)
            target_filename = target_filename_base + PCAP_EXTENSION
            new_path = os.path.join(source_dir, target_filename)

            current_check_counter = rename_counter
            while os.path.exists(new_path):
                print(f"  Target '{new_path}' already exists. Checking next number...")
                current_check_counter += 1
                target_filename_base = str(current_check_counter)
                target_filename = target_filename_base + PCAP_EXTENSION
                new_path = os.path.join(source_dir, target_filename)

            if os.path.normcase(os.path.abspath(old_path)) == os.path.normcase(os.path.abspath(new_path)):
                 print(f"  Skipping: File '{old_path}' is already correctly named and located.")
                 rename_counter = current_check_counter + 1
                 skipped_count += 1
                 continue

            print(f"  Moving and renaming: '{old_path}' -> '{new_path}'")
            os.rename(old_path, new_path)
            processed_count += 1
            rename_counter = current_check_counter + 1

        except OSError as e:
            print(f"  Error processing file '{old_path}': {e}")
            error_count += 1
        except Exception as e:
             print(f"  An unexpected error occurred processing file '{old_path}': {e}")
             error_count += 1

    print(f"\nSummary for '{source_dir}':")
    print(f"  Successfully processed (moved/renamed): {processed_count}")
    print(f"  Skipped (already correct or other):    {skipped_count}")
    print(f"  Errors encountered:                   {error_count}")
    print(f"--- Finished Processing: {source_dir} ---")

if __name__ == "__main__":
    print("Starting PCAP file unification and renaming process.")
    print("WARNING: This script will MOVE files from subdirectories")
    print("         to their parent source directory and RENAME them.")
    print("         Please ensure you have backups before proceeding.")

    # Optional: You might want to uncomment these lines for safety
    # confirm = input("Type 'yes' to continue: ")
    # if confirm.lower() != 'yes':
    #     print("Operation cancelled by user.")
    #     sys.exit()

    for directory in SOURCE_DIRECTORIES:
        unify_and_rename_pcaps(directory)

    print("\nScript finished.")