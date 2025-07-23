# data_loader.py (Revised for more robust client initialization and numeric conversions)

import os
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import BytesIO
import sys
import streamlit as st # Import Streamlit here for caching

# Azure Blob config (these are global constants, fine to define at top)
container_name = "lntnamrata"

# Use Streamlit's cache to ensure BlobServiceClient is initialized only once
@st.cache_resource
def get_blob_service_client():
    """
    Initializes and caches the BlobServiceClient to ensure it's only created once.
    """
    try:
        connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        if not connect_str:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set. Please add it to Streamlit Cloud secrets.")

        print("Initializing Azure Blob Service Client...")
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        print("Azure Blob Service Client initialized successfully.")
        return blob_service_client
    except Exception as e:
        print(f"Error initializing Azure Blob Storage client using connection string: {e}")
        st.error(f"Error connecting to Azure Blob Storage: {e}. Please check your AZURE_STORAGE_CONNECTION_STRING.")
        st.stop() # Stop the app if cannot connect
        return None # Should not be reached due to st.stop()

# Get clients using the cached function
blob_service_client = get_blob_service_client()
# Only proceed if client was successfully initialized
if blob_service_client:
    container_client = blob_service_client.get_container_client(container_name)
    print(f"Successfully connected to Azure Blob Storage container: {container_name}")
else:
    # This path should ideally not be hit if st.stop() works
    print("Failed to initialize blob_service_client, cannot proceed.")
    sys.exit(1)


def get_pl_data():
    """
    Loads P&L data from Azure Blob Storage, renames columns,
    and converts 'Date' (originally 'Month') and numeric columns to appropriate types.
    Returns a pandas DataFrame.
    """
    blob_name = "P&L data.csv"
    try:
        print(f"Attempting to load P&L data from Azure Blob: {blob_name}")
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        df = pd.read_csv(BytesIO(blob_data))
        
        # --- ORIGINAL P&L RENAMING STEPS ---
        df.rename(columns={
            "Segment": "Segment",
            "PVDG": "PVDG",
            "PVDU": "PVDU",
            "Exec DG": "Exec DG",
            "Exec DU": "Exec DU",
            "FinalCustomerName": "FinalCustomerName",
            "Contract ID": "Contract ID",
            "Month": "Date", # Renamed 'Month' to 'Date'
            "wbs id": "wbs id",
            # Assuming 'Amount in USD' is the primary numeric column for P&L
            # Add other numeric columns if they exist in your raw P&L data and need renaming/conversion
        }, inplace=True)

        # Convert "Date" column to datetime (now consistently named 'Date')
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            print("Error: 'Date' column not found after renaming in P&L data.")
            return pd.DataFrame()
        
        # --- NEW: Convert key numeric columns to numeric type ---
        # 'errors='coerce'' will turn any values that cannot be converted into NaN
        numeric_cols = ["Amount in USD"] # Add other numeric columns from P&L if applicable
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Numeric column '{col}' not found in P&L data.")
        
        # Drop rows where essential numeric columns or Date became NaN after coercion
        df.dropna(subset=['Date'] + [col for col in numeric_cols if col in df.columns], inplace=True)

        print("P&L data successfully loaded and preprocessed from Azure Blob.")
        return df
    except Exception as e:
        print(f"Error loading P&L data from Azure Blob '{blob_name}': {e}")
        return pd.DataFrame()

def get_ut_data():
    """
    Loads UT data from Azure Blob Storage, renames columns,
    and converts 'Date' (originally 'Date_a') and numeric columns to appropriate types.
    Returns a pandas DataFrame.
    """
    blob_name = "UT data.csv"
    try:
        print(f"Attempting to load UT data from Azure Blob: {blob_name}")
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        df = pd.read_csv(BytesIO(blob_data))
        
        # --- ORIGINAL UT RENAMING STEPS ---
        df.rename(columns={
            "Segment": "Segment",
            "ParticipatingVDG": "PVDG",
            "ParticipatingVDU": "PVDU",
            "DeliveryGroup": "Exec DG",
            "Delivery_Unit": "Exec DU",
            "FinalCustomerName": "FinalCustomerName",
            "sales document": "Contract ID",
            "Date_a": "Date", # Renamed 'Date_a' to 'Date'
            "WBSID": "wbs id",
            # Assuming 'Amount in USD' is also present and numeric in UT data
            # Add other numeric columns from UT if applicable
        }, inplace=True)

        # Convert "Date" column to datetime (now consistently named 'Date')
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            print("Error: 'Date' column not found after renaming in UT data.")
            return pd.DataFrame()

        # --- NEW: Convert key numeric columns to numeric type ---
        numeric_cols = ["Amount in USD"] # Add other numeric columns from UT if applicable
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Numeric column '{col}' not found in UT data.")
        
        # Drop rows where essential numeric columns or Date became NaN after coercion
        df.dropna(subset=['Date'] + [col for col in numeric_cols if col in df.columns], inplace=True)

        print("UT data successfully loaded and preprocessed from Azure Blob.")
        return df
    except Exception as e:
        print(f"Error loading UT data from Azure Blob '{blob_name}': {e}")
        return pd.DataFrame()

def get_merged_data():
    """
    Loads and merges P&L and UT data based on common columns.
    This function explicitly calls get_pl_data() and get_ut_data().
    """
    df_pl_processed = get_pl_data() # Get the pre-processed P&L data
    df_ut_processed = get_ut_data() # Get the pre-processed UT data

    if df_pl_processed.empty or df_ut_processed.empty:
        print("Cannot merge: one or both source dataframes are empty from Azure Blob.")
        return pd.DataFrame()

    # Define common columns for merging. These should now match due to renaming in load functions.
    common_columns = ['Segment', 'PVDG', 'PVDU', 'Exec DG', 'Exec DU',
                      'FinalCustomerName', 'Contract ID', 'Date', 'wbs id']

    # Perform the merge
    try:
        # Use suffixes to handle potentially overlapping columns that are NOT merge keys
        merged_df = pd.merge(df_pl_processed, df_ut_processed, on=common_columns, how='inner', suffixes=('_pl', '_ut'))
        print("P&L and UT data successfully merged.")
        return merged_df
    except KeyError as ke:
        print(f"Error during merge: Missing expected column for merge. {ke}. Ensure all common columns are present and correctly named.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during merging: {e}")
        return pd.DataFrame()

# IMPORTANT: No top-level calls to data loading or merging functions here.
# This file will now only execute its functions when explicitly called from main.py or other modules.