"""Generate HDF5 file from raw ECG recordings using ecg-preprocessing.

Requires the ecg-preprocessing toolkit:
    https://github.com/antonior92/ecg-preprocessing

Usage:
    python generate_h5.py --root_dir /path/to/wfdb/files/ \
                          --records /path/to/RECORDS.txt \
                          --output /path/to/output.h5
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate HDF5 from raw ECG WFDB files')
    parser.add_argument('--root_dir', required=True, help='Root directory containing WFDB files')
    parser.add_argument('--records', required=True, help='Path to RECORDS.txt listing exam files')
    parser.add_argument('--output', required=True, help='Output HDF5 file path')
    parser.add_argument('--preprocess_script', default='./ecg-preprocessing/generate_h5.py',
                        help='Path to ecg-preprocessing generate_h5.py script')
    args = parser.parse_args()

    cmd = (
        f"python {args.preprocess_script} {args.records} {args.output} "
        f"--root_dir {args.root_dir} "
        f"--new_freq 400 "
        f"--new_len 4096 "
        f"--scale 2 "
        f"--use_all_leads"
    )

    print(f"Running: {cmd}")
    os.system(cmd)
    print(f"Saved preprocessed ECGs to {args.output}")


if __name__ == "__main__":
    main()
