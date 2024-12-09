from typing import Tuple
import pandas as pd
import numpy as np
import logging


class PerimeterProcessor:
    """
    Handles the processing and standardization of urban perimeter data.

    This class provides methods to clean and standardize the 'Perimetro_Urbano'
    feature in railway accident data, converting various forms of perimeter
    information into a standardized binary format.
    """

    def __init__(self):
        """Initialize PerimeterProcessor with logger."""
        self.logger = logging.getLogger(__name__)

    def analyze_perimeter_values(self, df: pd.DataFrame) -> Tuple[dict, pd.Series]:
        """
        Analyzes the current state of Perimetro_Urbano values in the dataset.

        Args:
            df (pd.DataFrame): Input DataFrame containing Perimetro_Urbano column

        Returns:
            Tuple[dict, pd.Series]: Dictionary of value counts and series of unique values
        """
        # Get value counts and unique values
        value_counts = df["Perimetro_Urbano"].value_counts().to_dict()
        unique_values = df["Perimetro_Urbano"].unique()

        # Log the analysis results
        # print(f"Found {len(unique_values)} unique values in Perimetro_Urbano")
        # print("Value distribution:")
        # for value, count in value_counts.items():
        #     print(f"  {value}: {count} occurrences")

        return value_counts, unique_values

    def standardize_perimeter(
        self, df: pd.DataFrame, inplace: bool = False
    ) -> pd.DataFrame:
        """
        Standardizes the Perimetro_Urbano column values.

        Args:
            df (pd.DataFrame): Input DataFrame
            inplace (bool): Whether to modify the input DataFrame or create a copy

        Returns:
            pd.DataFrame: DataFrame with standardized Perimetro_Urbano values
        """
        # Create a copy if not inplace
        if not inplace:
            df = df.copy()

        # Store original state for reporting
        _ = df["Perimetro_Urbano"].value_counts()

        # Standardize values
        def standardize_value(value):
            if pd.isna(value) or value == "" or value == "," or value == "Não":
                return "Não"
            else:
                return "Sim"

        # Apply standardization
        df["Perimetro_Urbano"] = df["Perimetro_Urbano"].apply(standardize_value)

        # Log the changes
        _ = df["Perimetro_Urbano"].value_counts()

        # Calculate and log the number of changes made
        _ = (df["Perimetro_Urbano"] != df["Perimetro_Urbano"]).sum()

        return df

    def process_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the perimeter data and validates the results.

        This method combines analysis, standardization, and validation steps
        to ensure data quality.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Processed and validated DataFrame
        """
        # Initial analysis
        _, _ = self.analyze_perimeter_values(df)

        # Standardize values
        processed_df = self.standardize_perimeter(df)

        # Validate results
        final_unique = processed_df["Perimetro_Urbano"].unique()
        if not all(val in ["Sim", "Não"] for val in final_unique):
            self.logger.error(
                "Validation failed: Found unexpected values after standardization"
            )
            raise ValueError("Standardization resulted in unexpected values")

        return processed_df
