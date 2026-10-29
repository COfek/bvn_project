
import csv
import sys
from pathlib import Path

def merge_csvs(file1, file2, output_file):
    rows = []
    headers = set()

    # Read first file
    with open(file1, 'r', newline='') as f1:
        reader = csv.DictReader(f1)
        headers.update(reader.fieldnames)
        for row in reader:
            rows.append(row)

    # Read second file
    with open(file2, 'r', newline='') as f2:
        reader = csv.DictReader(f2)
        headers.update(reader.fieldnames)
        for row in reader:
            rows.append(row)

    # Sort headers for consistency, keeping matrix_index first if present
    fieldnames = list(headers)
    if 'matrix_index' in fieldnames:
        fieldnames.remove('matrix_index')
        fieldnames = ['matrix_index'] + sorted(fieldnames)
    else:
        fieldnames = sorted(fieldnames)

    with open(output_file, 'w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
            
    print(f"Merged CSV written to {output_file} with {len(rows)} rows.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_csvs.py <file1> <file2> <output_file>")
        sys.exit(1)
        
    merge_csvs(sys.argv[1], sys.argv[2], sys.argv[3])
