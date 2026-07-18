# DISCLAIMER: This code was asked to Gemini AI to be written.
import os

import requests


def download_zenodo_repo(record_id: str, output_dir: str = "cache") -> bool:
    """
    Downloads all individual files from a Zenodo repository using its Record ID.

    :param record_id: The unique ID string or integer at the end of the Zenodo URL.
    :param output_dir: The local directory path where files should be saved.
    :return: True if successful, False otherwise.
    """
    # Convert integer IDs to string just in case
    record_id = str(record_id).strip()

    # Create target directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Query the Zenodo API for metadata
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Fetching metadata for Zenodo Record ID: {record_id}...")
    response = requests.get(api_url)

    if response.status_code != 200:
        print(f"Failed to fetch metadata. HTTP Status: {response.status_code}")
        print(f"Response: {response.text}")
        return False

    record_data = response.json()
    files = record_data.get("files", [])

    if not files:
        print("No files found in this record (it might be restricted or private).")
        return False

    print(f"Found {len(files)} files. Starting download...")
    print("-" * 50)

    for file_info in files:
        filename = file_info.get("key") or file_info.get("filename")
        download_url = file_info.get("links", {}).get("self")

        # Get the file size reported by Zenodo (usually in bytes)
        expected_size = file_info.get("size") or file_info.get("bytes")

        if filename and download_url:
            target_path = os.path.join(output_dir, filename)

            # --- SKIP CHECK LOGIC ---
            if os.path.exists(target_path):
                # If we know the expected size, make sure it matches
                if expected_size and os.path.getsize(target_path) == expected_size:
                    print(f"Skipping {filename} (already completely downloaded).")
                    continue
                elif not expected_size:
                    # Fallback if the API didn't provide a size
                    print(f"Skipping {filename} (file already exists).")
                    continue
                else:
                    print(f"File {filename} exists but size mismatch. Re-downloading...")
            # ------------------------

            print(f"Downloading: {filename} ...")

            try:
                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(target_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                return False

    return True
