import pandas as pd
import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

# Insert your Google Geocoding API key here
GOOGLE_API_KEY = os.getenv("GEOCODING_API_KEY")


# Function to fetch latitude and longitude of the municipality
def get_lat_lon(municipio, uf):
    url = os.getenv(
        "URL_GEOCODING_GOOGLE"
    )  # e.g., 'https://maps.googleapis.com/maps/api/geocode/json'
    address = f"{municipio}, {uf}, Brazil"
    params = {"address": address, "key": GOOGLE_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            return None, None  # If municipality is not found
    else:
        print(
            f"Error fetching municipality: {municipio}, status code: {response.status_code}"
        )
        return None, None


# Read the CSV with pandas
arquivo_csv = "datasets/acidents_ferroviarios_2004_2024.csv"
df = pd.read_csv(arquivo_csv)

# Ensure there are columns for municipalities and states
required_columns = ["Municipio", "UF"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"The CSV must contain {', '.join(required_columns)} columns")

# Add Latitude and Longitude columns if they don't exist
if "Latitude" not in df.columns:
    df["Latitude"] = None
if "Longitude" not in df.columns:
    df["Longitude"] = None

# Create a dictionary to store known coordinates
known_coords = {}

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    municipio = row["Municipio"]
    uf = row["UF"]

    # Check if we already have coordinates for this municipality
    if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
        known_coords[(municipio, uf)] = (row["Latitude"], row["Longitude"])
        continue

    # Check if we've already fetched coordinates for this municipality
    if (municipio, uf) in known_coords:
        df.at[index, "Latitude"], df.at[index, "Longitude"] = known_coords[
            (municipio, uf)
        ]
    else:
        # Fetch new coordinates
        lat, lon = get_lat_lon(municipio, uf)
        if lat is not None and lon is not None:
            df.at[index, "Latitude"] = lat
            df.at[index, "Longitude"] = lon
            known_coords[(municipio, uf)] = (lat, lon)

        # Pause briefly to avoid overloading the Google API
        time.sleep(0.1)

# Save the updated CSV
df.to_csv(
    "datasets/acidents_ferroviarios_2004_2024_com_coords.csv", encoding="UTF-8", index=False
)
print("CSV updated with coordinates saved successfully!")
