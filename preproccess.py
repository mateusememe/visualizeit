import pandas as pd
import sys
import os
import chardet


def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        raw_data = file.read()
    return chardet.detect(raw_data)["encoding"]


def read_csv_with_encoding(file_path):
    encoding = detect_encoding(file_path)
    print(f"Detected encoding for {os.path.basename(file_path)}: {encoding}")
    return pd.read_csv(file_path, encoding=encoding, low_memory=False, sep=";")


def merge_csvs(file1, file2, output_file):
    # Get the absolute paths of the files
    file1 = os.path.abspath(file1)
    file2 = os.path.abspath(file2)
    output_file = os.path.abspath(output_file)

    # Check if files exist
    if not os.path.exists(file1):
        print(f"Error: File not found - {file1}")
        sys.exit(1)
    if not os.path.exists(file2):
        print(f"Error: File not found - {file2}")
        sys.exit(1)

    # Read the CSV files with appropriate encoding
    try:
        df1 = read_csv_with_encoding(file1)
        df2 = read_csv_with_encoding(file2)
    except Exception as e:
        print(f"Error reading CSV files: {str(e)}")
        sys.exit(1)

    # Find common columns
    common_columns = list(set(df1.columns) & set(df2.columns))

    # Merge dataframes on common columns
    merged_df = pd.merge(df1, df2, on=common_columns, how="outer")

    # Save the merged dataframe in UTF-8 encoding
    merged_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Merged file saved as: {output_file} (UTF-8 encoding)")

    # Find non-matching columns
    non_matching_columns = list(set(df1.columns) ^ set(df2.columns))

    # Print non-matching columns
    print("\nColumns that don't match:")
    for col in non_matching_columns:
        if col in df1.columns:
            print(f"- {col} (present in {os.path.basename(file1)})")
        else:
            print(f"- {col} (present in {os.path.basename(file2)})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <file1.csv> <file2.csv> <output.csv>")
        sys.exit(1)

    file1, file2, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    merge_csvs(file1, file2, output_file)
