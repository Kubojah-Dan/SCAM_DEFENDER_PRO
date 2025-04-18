import csv
import sys

def clean_csv(input_file, output_file, encoding="latin1"):
    """
    Reads the input CSV file and writes to the output file only those rows
    that have the same number of fields as the header.
    """
    with open(input_file, mode='r', encoding=encoding, newline='') as infile, \
         open(output_file, mode='w', encoding=encoding, newline='') as outfile:
         
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        row_count = 0
        bad_rows = 0
        expected_field_count = None

        for row in reader:
            row_count += 1
            # On first row, record the expected number of fields (header)
            if expected_field_count is None:
                expected_field_count = len(row)
                writer.writerow(row)
                continue

            # Check whether the row has the expected number of fields
            if len(row) == expected_field_count:
                writer.writerow(row)
            else:
                bad_rows += 1
                print(f"Skipping line {row_count}: expected {expected_field_count} fields, got {len(row)}", file=sys.stderr)

        print(f"Finished cleaning. Processed {row_count} rows, skipped {bad_rows} bad rows.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean an email CSV file with inconsistent rows.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_csv", help="Path to output the cleaned CSV file")
    parser.add_argument("--encoding", default="latin1", help="File encoding (default: latin1)")
    args = parser.parse_args()

    clean_csv(args.input_csv, args.output_csv, args.encoding)
