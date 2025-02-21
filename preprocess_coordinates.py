import pandas as pd

# Define file paths for both CSV files
arquivo_csv_coords = "datasets/acidents_ferroviarios_2004_2024_com_coords.csv"
arquivo_csv_sem_coords = "datasets/updated/acidentes_ferroviarios_12.2020-12.2024.csv"


def get_special_coordinates(municipio: str, uf: str) -> tuple:
    """
    Returns coordinates for specific municipality and UF combinations that were missing.
    This function serves as a fallback for known missing coordinates.

    Args:
        municipio: Name of the municipality
        uf: State abbreviation

    Returns:
        tuple: (latitude, longitude) if found, (None, None) if not a special case
    """
    # Dictionary of special cases with their coordinates
    special_cases = {
        ("Itabaianinha", "SE"): (-11.2714, -37.7897),
        ("Palmeirante", "TO"): (-7.8611, -47.9242),
        ("Goianira", "GO"): (-16.4947, -49.4271),
        ("Miracema do Tocantins", "TO"): (-9.5665, -48.3933),
    }

    # Return the coordinates if it's a special case
    return special_cases.get((municipio, uf), (None, None))


def transfer_coordinates():
    """
    Transfers coordinates from a complete dataset to a dataset without coordinates,
    matching by municipality and state (UF). Includes special handling for known
    missing coordinates.
    """
    # Read both CSV files
    print("Reading CSV files...")
    df_with_coords = pd.read_csv(arquivo_csv_coords, encoding="latin1", sep=";")
    df_without_coords = pd.read_csv(arquivo_csv_sem_coords, encoding="latin1", sep=";")

    # Validate required columns exist in both dataframes
    required_columns = ["Municipio", "UF"]
    for df, name in [(df_with_coords, "source"), (df_without_coords, "target")]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {name} dataset: {missing_cols}"
            )

    # Create a reference dictionary for coordinates
    print("Creating coordinates reference...")
    coords_reference = (
        df_with_coords.drop_duplicates(subset=["Municipio", "UF"])[
            ["Municipio", "UF", "Latitude", "Longitude"]
        ]
        .set_index(["Municipio", "UF"])
        .to_dict("index")
    )

    # Add Latitude and Longitude columns if they don't exist
    for col in ["Latitude", "Longitude"]:
        if col not in df_without_coords.columns:
            df_without_coords[col] = None

    # Counter for tracking progress
    total_rows = len(df_without_coords)
    found_coords = 0
    special_cases_found = 0
    missing_coords = 0

    # Transfer coordinates
    print("Transferring coordinates...")
    for index, row in df_without_coords.iterrows():
        municipio = row["Municipio"]
        uf = row["UF"]

        # First, try to get coordinates from the reference dataset
        coords = coords_reference.get((municipio, uf))

        if coords:
            # Use coordinates from reference dataset
            df_without_coords.at[index, "Latitude"] = coords["Latitude"]
            df_without_coords.at[index, "Longitude"] = coords["Longitude"]
            found_coords += 1
        else:
            # Check if it's one of our special cases
            lat, lon = get_special_coordinates(municipio, uf)
            if lat is not None and lon is not None:
                # Use our manually added coordinates
                df_without_coords.at[index, "Latitude"] = lat
                df_without_coords.at[index, "Longitude"] = lon
                special_cases_found += 1
            else:
                # Still no coordinates found
                missing_coords += 1
                print(f"No coordinates found for: {municipio}, {uf}")

    # Print statistics
    print(f"\nCoordinates transfer complete:")
    print(f"Total rows processed: {total_rows}")
    print(f"Coordinates found in reference dataset: {found_coords}")
    print(f"Coordinates found in special cases: {special_cases_found}")
    print(f"Missing coordinates: {missing_coords}")
    print(f"Success rate: {((found_coords + special_cases_found)/total_rows)*100:.2f}%")

    # Save the updated dataset
    output_file = "datasets/updated/acidents_ferroviarios_2020_2024_com_coords.csv"
    df_without_coords.to_csv(output_file, encoding="UTF-8", index=False)
    print(f"\nUpdated CSV saved to: {output_file}")


if __name__ == "__main__":
    try:
        transfer_coordinates()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
