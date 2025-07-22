# data_loader.py (FINAL CORRECTED VERSION - Using Connection String)

import os # Import the os module to access environment variables
# from azure.identity import DefaultAzureCredential # REMOVED: No longer needed for connection string auth
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import BytesIO
import sys

# Azure Blob config (these are global constants, fine to define at top)
# account_url = "https://pbichatbot11.blob.core.windows.net" # No longer needed with connection string
container_name = "lntnamrata"

# Authenticate and connect using the connection string (client initialization also fine at top as it's a one-time setup)
try:
    # Get connection string from environment variable (Streamlit Cloud secret)
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if not connect_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set. Please add it to Streamlit Cloud secrets.")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)
    print(f"Successfully connected to Azure Blob Storage container: {container_name}")
except Exception as e:
    print(f"Error initializing Azure Blob Storage client using connection string: {e}")
    sys.exit(1) # Exit if cannot connect to Azure Blob

def get_pl_data():
    """
    Loads P&L data from Azure Blob Storage, renames columns,
    and converts 'Date' (originally 'Month') to datetime.
    Returns a pandas DataFrame.
    """
    blob_name = "P&L data.csv"
    try:
        print(f"Attempting to load P&L data from Azure Blob: {blob_name}")
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        df = pd.read_csv(BytesIO(blob_data))
        
        # --- ORIGINAL P&L RENAMING STEPS (now inside the function) ---
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
        }, inplace=True)

        # Convert "Date" column to datetime (now consistently named 'Date')
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            print("Error: 'Date' column not found after renaming in P&L data.")
            return pd.DataFrame()
            
        print("P&L data successfully loaded and preprocessed from Azure Blob.")
        return df
    except Exception as e:
        print(f"Error loading P&L data from Azure Blob '{blob_name}': {e}")
        return pd.DataFrame()

def get_ut_data():
    """
    Loads UT data from Azure Blob Storage, renames columns,
    and converts 'Date' (originally 'Date_a') to datetime.
    Returns a pandas DataFrame.
    """
    blob_name = "UT data.csv"
    try:
        print(f"Attempting to load UT data from Azure Blob: {blob_name}")
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        df = pd.read_csv(BytesIO(blob_data))
        
        # --- ORIGINAL UT RENAMING STEPS (now inside the function) ---
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
        }, inplace=True)

        # Convert "Date" column to datetime (now consistently named 'Date')
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            print("Error: 'Date' column not found after renaming in UT data.")
            return pd.DataFrame()

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
        
        # Optionally, you can drop or combine specific duplicate columns here if needed
        # For example, if both had an 'Amount' column, you'd have 'Amount_pl' and 'Amount_ut'
        # You would then decide how to handle them, e.g., keep both, sum them, or drop one.
        
        return merged_df
    except KeyError as ke:
        print(f"Error during merge: Missing expected column for merge. {ke}. Ensure all common columns are present and correctly named.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during merging: {e}")
        return pd.DataFrame()

# IMPORTANT: No top-level calls to data loading or merging functions here.
# This file will now only execute its functions when explicitly called from main.py or other modules.