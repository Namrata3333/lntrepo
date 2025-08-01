import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
# Import both get_pl_data and get_ut_data
from data_loaderr import get_pl_data, get_ut_data, get_merged_data
# Import all necessary functions from kpi_calculationss
from kpi_calculationss import calculate_cm, get_query_details, analyze_transportation_cost_trend, \
    calculate_cb_variation, calculate_cb_revenue_trend, calculate_hc_trend, analyze_revenue_trend, analyze_ut_trend, \
    analyze_fresher_ut_trend, analyze_revenue_per_person_trend, analyze_realized_rate_drop, \
    get_fiscal_quarter_name_from_fy_and_q, get_fiscal_quarter_and_year  # Import new function
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np  # Import numpy for dynamic range calculation
import sys
import os
from datetime import datetime

# Add the directory containing kpi_calculationss.py to the Python path
# Assuming kpi_calculationss.py is in the same directory as mainn.py
sys.path.append(os.path.dirname(__file__))

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="L&T ChatBot")

# st.title("📊 L&T ChatBot")
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>📊 L&T ChatBot</h1>
        <p>Welcome to the L&T AI Driven Chatbot. Explore your data with natural language. Ask a question about KPIs, Costs, Revenue, or Trends to get instant insights.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Define these global lists in mainn.py as well, so they are accessible for total calculations
REVENUE_GROUPS = ["ONSITE", "OFFSHORE", "INDIRECT REVENUE"]
COST_GROUPS = [
    "Direct Expense", "OWN OVERHEADS", "Indirect Expense",
    "Project Level Depreciation", "Direct Expense - DU Block Seats Allocation",
    "Direct Expense - DU Pool Allocation", "Establishment Expenses"
]


@st.cache_data
def load_pl_data_for_streamlit():  # Renamed for clarity
    """Load P&L data using the get_pl_data function from data_loader."""
    with st.spinner("Loading P&L data from Azure Blob..."):
        df = get_pl_data()
    if df.empty:
        st.error("Failed to load P&L data. Please check data source and credentials.")
        st.stop()  # Stop the app if data can't be loaded
    # Ensure 'Date' column is datetime type for filtering and is timezone-naive
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df


@st.cache_data
def load_ut_data_for_streamlit():
    """Load UT data using the get_ut_data function from data_loader."""
    with st.spinner("Loading UT data from Azure Blob..."):
        df = get_ut_data()
    if df.empty:
        st.error("Failed to load UT data. Please check data source and credentials.")
        st.stop()  # Stop the app if data can't be loaded
    # Ensure 'Date' column is datetime type for filtering and is timezone-naive
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df


@st.cache_data
def load_merged_data_for_streamlit():
    """Load and merge P&L and UT data using the get_merged_data function from data_loader."""
    with st.spinner("Loading and merging P&L and UT data from Azure Blob..."):
        df = get_merged_data()
    if df.empty:
        st.error("Failed to load or merge P&L and UT data. Please check data sources and merge logic.")
        st.stop()  # Stop the app if data can't be loaded
    # Ensure 'Date' column is datetime type for filtering and is timezone-naive
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df


df_pl = load_pl_data_for_streamlit()  # Updated function call
df_ut = load_ut_data_for_streamlit()
df_merged = load_merged_data_for_streamlit()

st.write("---")

# --- Chatbot logic starts here ---
# Initialize session state for the query if it doesn't exist
if 'text_input_query' not in st.session_state:
    st.session_state.text_input_query = ""
# Define the callback function to set the query
def set_query(query):
    st.session_state.text_input_query = query
# --- User Input Section ---
user_query = st.text_input(
    "Enter your query:",
    
    key="text_input_query"
)






# This block is now outside of an if st.button("Submit") statement, so it runs on every interaction.
if user_query:
    query_details = get_query_details(user_query)

    # --- CRITICAL FIX: Check if query_details is None ---
    if query_details is None:
        st.error(
            "Failed to process your query. The AI model could not extract query details. Please try again or rephrase your query.")
    else:
        # --- NEW FIX: Clean date-related keys from column_filters if they exist ---
        # This prevents KPI functions from trying to filter on 'FY', 'Q1' etc. as columns
        if "column_filters" in query_details and isinstance(query_details["column_filters"], dict):
            # Define a list of common date-related keys that might be mistakenly added to column_filters
            date_related_keys = ["fy", "q", "quarter", "year", "month"]
            cleaned_column_filters = {
                k: v for k, v in query_details["column_filters"].items()
                if k.lower() not in date_related_keys
            }
            query_details["column_filters"] = cleaned_column_filters
        # --- END NEW FIX ---

        query_type = query_details.get("query_type")

        filters_applied = []

        # 1. Check for date filters
        if query_details.get("date_filter"):
            start_date_str = query_details.get("start_date")
            end_date_str = query_details.get("end_date")
            if start_date_str and end_date_str:
                # Check if dates are already datetime objects or strings
                if isinstance(start_date_str, datetime):
                    start_date_str = start_date_str.strftime("%Y-%m-%d")
                if isinstance(end_date_str, datetime):
                    end_date_str = end_date_str.strftime("%Y-%m-%d")
                filters_applied.append(f"Date Range: {start_date_str} to {end_date_str}")
            elif query_details.get("description"):
                filters_applied.append(f"Period: {query_details['description']}")

        # 2. Check for column filters
        column_filters = query_details.get("column_filters", {})
        for key, value in column_filters.items():
            filters_applied.append(f"{key}: {value}")

        # 3. Check for specific numeric/comparative filters
        if query_details.get("comparison_operator") and query_details.get("comparison_value") and query_details.get(
                "comparison_column"):
            op = query_details["comparison_operator"]
            val = query_details["comparison_value"]
            col = query_details["comparison_column"]
            filters_applied.append(f"Filter: {col} {op} {val}")

        if filters_applied:
            filters_str = " | ".join(filters_applied)
            st.markdown(f"""
                 <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                     <b>Filters Applied:</b> {filters_str}
                 </div>
             """, unsafe_allow_html=True)
        # --- END NEW FEATURE ---

        # --- DEBUGGING LINE (Keep for your debugging, remove for production) ---
        # st.write(f"DEBUG: Query Type: {query_type}")
        # st.write(f"DEBUG: Date Info: {query_details}")
        # st.write(f"DEBUG: Column Filters: {query_details.get('column_filters')}")
        # --- END DEBUGGING LINE ---

        if query_type == "CM_analysis":
            st.markdown("### Contribution Margin Analysis")

            # For CM analysis, apply date filter here using query_details
            filtered_df_for_cm = df_pl.copy()
            if query_details.get("date_filter") and query_details.get("start_date") and query_details.get("end_date"):
                # Convert query_details dates to pandas Timestamps for robust comparison and ensure they are naive
                start_date_ts = pd.Timestamp(query_details["start_date"]).tz_localize(None)
                end_date_ts = pd.Timestamp(query_details["end_date"]).tz_localize(None)
                filtered_df_for_cm = filtered_df_for_cm[
                    filtered_df_for_cm['Date'].between(start_date_ts, end_date_ts, inclusive='both')
                ]

            if filtered_df_for_cm.empty:
                st.warning(
                    f"No data available for the primary period ({query_details.get('description', 'specified date range')}) for CM analysis. Please check the data for this range.")
            else:
                result_df = calculate_cm(filtered_df_for_cm, query_details)

                # --- FIX: Robustly handle dictionary or DataFrame output from calculate_cm ---
                if isinstance(result_df, dict) and "Message" in result_df:
                    st.error(result_df["Message"])  # Display the error message from calculate_cm
                elif isinstance(result_df, pd.DataFrame):  # Ensure it's actually a DataFrame before checking .empty
                    if result_df.empty:
                        st.warning("No Contribution Margin data found for the specified criteria after calculations.")
                    else:
                        # --- Calculate and Display Key Metrics at the Top (REVISED & FORMATTED) ---
                        total_revenue_filtered_cm = result_df["Revenue"].sum()
                        total_cost_filtered_cm = result_df["Cost"].sum()
                        total_customers_filtered_cm = result_df["FinalCustomerName"].nunique()


                        def format_value_to_k_m(value):
                            if pd.isna(value) or value == 0:
                                return "$0.00"
                            if abs(value) >= 1_000_000:
                                return f"${value / 1_000_000:,.2f}M"
                            elif abs(value) >= 1_000:
                                return f"${value / 1_000:,.2f}K"
                            else:
                                return f"${value:,.2f}"


                        st.markdown("#### Key Metrics:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(label="Total Revenue", value=format_value_to_k_m(total_revenue_filtered_cm))
                        with col2:
                            st.metric(label="Total Cost", value=format_value_to_k_m(total_cost_filtered_cm))
                        with col3:
                            st.metric(label="Total Customers", value=f"{total_customers_filtered_cm:,}")

                        st.markdown("---")  # Separator for visual appeal

                        # --- Tabs for Data Table and Visual Analysis ---
                        tab1, tab2 = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                        with tab1:
                            st.subheader("Customer-wise Contribution Margin:")
                            # Explicitly select columns for display to exclude 'CM_Value'
                            display_cols = ["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)"]
                            st.dataframe(result_df[display_cols].style.format({
                                "Revenue": lambda x: f"${x:,.2f}",
                                "Cost": lambda x: f"${x:,.2f}"
                            }))

                        with tab2:
                            st.subheader("Visual Analysis: Customer-wise KPIs")

                            # Ensure 'CM_Value' is numeric for plotting
                            result_df['CM_Value_Numeric'] = pd.to_numeric(result_df['CM_Value'], errors='coerce')

                            # Drop rows where CM_Value_Numeric is NaN as they cannot be plotted
                            sorted_df = result_df.dropna(subset=['CM_Value_Numeric']).sort_values(by="CM_Value_Numeric",
                                                                                                  ascending=True)

                            if sorted_df.empty:
                                st.warning(
                                    "No data available for plotting after CM % numeric conversion and NaN removal.")
                            else:
                                # Create figure with secondary y-axis
                                fig = make_subplots(specs=[[{"secondary_y": True}]])

                                # Add Revenue Bar Chart
                                fig.add_trace(
                                    go.Bar(
                                        x=sorted_df["FinalCustomerName"],
                                        y=sorted_df["Revenue"],
                                        name="Revenue",
                                        marker_color='rgb(55, 83, 109)',
                                        hovertemplate="<b>Customer:</b> %{x}<br><b>Revenue:</b> %{y:$,.2f}<extra></extra>"
                                        # Custom hover for Revenue
                                    ),
                                    secondary_y=False,
                                )

                                # Add Cost Bar Chart
                                fig.add_trace(
                                    go.Bar(
                                        x=sorted_df["FinalCustomerName"],
                                        y=sorted_df["Cost"],
                                        name="Cost",
                                        marker_color='rgb(26, 118, 255)',
                                        hovertemplate="<b>Customer:</b> %{x}<br><b>Cost:</b> %{y:$,.2f}<extra></extra>"
                                        # Custom hover for Cost
                                    ),
                                    secondary_y=False,
                                )

                                # Add CM% Line Chart
                                fig.add_trace(
                                    go.Scatter(
                                        x=sorted_df["FinalCustomerName"],
                                        y=sorted_df["CM_Value_Numeric"],  # Still use numeric for plot positioning
                                        name="CM %",
                                        mode="lines+markers",
                                        yaxis="y2",
                                        line=dict(color='red', width=3),
                                        # Use CM (%) for hover display
                                        hovertemplate="<b>Customer:</b> %{x}<br><b>CM %:</b> %{customdata}<extra></extra>",
                                        customdata=sorted_df["CM (%)"]  # Pass the formatted CM (%) string here
                                    ),
                                    secondary_y=True,
                                )

                                # Calculate dynamic Y-axis range for CM %
                                min_cm = sorted_df['CM_Value_Numeric'].min()
                                max_cm = sorted_df['CM_Value_Numeric'].max()

                                # Add some padding to the range
                                padding = (max_cm - min_cm) * 0.1 if (
                                                                                 max_cm - min_cm) != 0 else 10  # 10% padding, or 10 if range is zero

                                # Ensure min_cm does not go too low if all values are positive
                                # or ensure it starts from a reasonable value if all CMs are high
                                lower_bound_cm = min_cm - padding
                                upper_bound_cm = max_cm + padding

                                # Optionally, ensure a minimum lower bound if CM can be very negative
                                if lower_bound_cm > -200:  # Example: don't let it go much lower than -200%
                                    lower_bound_cm = -200

                                # Ensure upper bound is at least 100 or higher if needed
                                if upper_bound_cm < 100:
                                    upper_bound_cm = 100

                                # Update layout for combined chart
                                fig.update_layout(
                                    title_text="Customer Revenue, Cost, and Contribution Margin %",
                                    xaxis_title="Customer Name",
                                    barmode='group',
                                    hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                    height=600  # Set a fixed height for better consistency
                                )

                                # Set y-axes titles and formats
                                fig.update_yaxes(title_text="Revenue/Cost (USD)", secondary_y=False, tickprefix="$",
                                                 tickformat=",.0f")
                                fig.update_yaxes(title_text="CM %", secondary_y=True, tickformat=".2f%",
                                                 range=[lower_bound_cm, upper_bound_cm])

                                st.plotly_chart(fig, use_container_width=True)

                else:  # Fallback for unexpected return type from calculate_cm

                    st.error(
                        "An unexpected error occurred during Contribution Margin analysis. Please check the `calculate_cm` function's return type.")
        elif query_type == "Revenue_Trend_Analysis":
            st.subheader("📊 Revenue Trend Analysis (YoY, QoQ, MoM)")


            @st.cache_data(show_spinner="Preparing Revenue Data for Trends...")
            def get_base_revenue_data_cached(df_pl_data, q_details_hashable):
                q_details_for_analysis = q_details_hashable.copy()
                for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                    if q_details_for_analysis.get(k) and isinstance(q_details_for_analysis[k], str):
                        q_details_for_analysis[k] = datetime.fromisoformat(q_details_for_analysis[k])

                return analyze_revenue_trend(df_pl_data, q_details_for_analysis)


            # Prepare query_details for caching by converting datetime objects to ISO format strings
            query_details_hashable = query_details.copy()
            for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                if query_details_hashable.get(k) and isinstance(query_details_hashable[k], datetime):
                    query_details_hashable[k] = query_details_hashable[k].isoformat()

            # Pass the actual df_pl loaded from Azure Blob
            q5_output_dict = get_base_revenue_data_cached(df_pl.copy(), query_details_hashable)

            if isinstance(q5_output_dict, dict) and "Message" in q5_output_dict and (
                    q5_output_dict.get("df_filtered_for_charts") is None or q5_output_dict[
                "df_filtered_for_charts"].empty):
                st.error(q5_output_dict["Message"])
            elif isinstance(q5_output_dict, dict):
                df_revenue_filtered = q5_output_dict["df_filtered_for_charts"]
                actual_grouping_dimension_from_query = q5_output_dict["grouping_dimension_from_query"]
                requested_trends = query_details.get("requested_trends",
                                                     ["MoM", "QoQ", "YoY"])  # Get requested trends, default to all

                date_filter_msg = "📅 Showing all available data (no specific date filter applied from query)"
                if query_details.get("date_filter") and query_details.get("start_date") and query_details.get(
                        "end_date"):
                    if isinstance(query_details['start_date'], datetime) and isinstance(query_details['end_date'],
                                                                                        datetime):
                        date_filter_msg = f"📅 Date Filter: {query_details['start_date'].strftime('%Y-%m-%d')} to {query_details['end_date'].strftime('%Y-%m-%d')}"
                    else:
                        st.warning("Parsed dates are not valid datetime objects. Displaying all available data.")
                st.success(date_filter_msg)

                grouping_col_map = {
                    "DU": "Exec DU",
                    "BU": "Exec DG",
                    "Account": "FinalCustomerName",
                    "All": None
                }

                selected_dim_for_analysis = actual_grouping_dimension_from_query
                selected_dim_col_name = grouping_col_map.get(selected_dim_for_analysis)

                # Fallback to 'All' if the requested dimension column is not found or is entirely null
                if selected_dim_col_name is None or \
                        selected_dim_col_name not in df_revenue_filtered.columns or \
                        df_revenue_filtered[selected_dim_col_name].isnull().all():
                    selected_dim_for_analysis = "All"
                    selected_dim_col_name = None

                if selected_dim_for_analysis == "All":
                    st.info(
                        f"Displaying trends for **Total Revenue** as no specific dimension (DU, BU, Account) was requested or available in the data.")
                else:
                    st.info(f"Displaying trends grouped by **{selected_dim_for_analysis}** as requested in your query.")

                if "Amount in USD" not in df_revenue_filtered.columns:
                    st.error(
                        "Error: 'Amount in USD' column not found in the filtered revenue data. Cannot calculate trends.")
                else:
                    # Ensure 'Date' column is datetime and extract time components
                    df_revenue_filtered['Date'] = pd.to_datetime(df_revenue_filtered['Date'])

                    # Add Fiscal Year and Fiscal Quarter columns for more robust grouping
                    df_revenue_filtered['FiscalYear'], df_revenue_filtered['FiscalQuarterNum'] = zip(
                        *df_revenue_filtered['Date'].apply(get_fiscal_quarter_and_year))
                    df_revenue_filtered['FiscalQuarter'] = df_revenue_filtered.apply(
                        lambda row: get_fiscal_quarter_name_from_fy_and_q(row['FiscalYear'], row['FiscalQuarterNum']),
                        axis=1)

                    # NEW: Add FiscalYearEnd column for YoY grouping
                    df_revenue_filtered['FiscalYearEnd'] = df_revenue_filtered['FiscalYear'] + 1

                    # --- Calculate MoM Trend Data ---
                    if selected_dim_col_name:
                        mom_data = df_revenue_filtered.groupby(
                            [pd.Grouper(key='Date', freq='MS'), selected_dim_col_name]).agg(
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        mom_data.rename(columns={'Date': 'Period'}, inplace=True)
                        # Sort by Period before creating table_df to ensure correct S.No.
                        mom_data = mom_data.sort_values('Period')

                        mom_data['Prev_Period_Revenue'] = mom_data.groupby(selected_dim_col_name)['Revenue'].shift(1)
                        mom_data['MoM_Growth_Percent'] = np.where(
                            mom_data['Prev_Period_Revenue'] != 0,
                            ((mom_data['Revenue'] - mom_data['Prev_Period_Revenue']) / mom_data[
                                'Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        mom_data_table_df = mom_data[[selected_dim_col_name, 'Period', 'Revenue', 'Prev_Period_Revenue',
                                                      'MoM_Growth_Percent']].copy()
                        mom_data_table_df.rename(columns={'Revenue': 'Current Month Revenue',
                                                          'Prev_Period_Revenue': 'Previous Month Revenue'},
                                                 inplace=True)
                    else:
                        overall_mom_data = df_revenue_filtered.groupby(pd.Grouper(key='Date', freq='MS')).agg(
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        overall_mom_data.rename(columns={'Date': 'Period'}, inplace=True)
                        # Sort by Period before creating table_df to ensure correct S.No.
                        overall_mom_data = overall_mom_data.sort_values('Period')

                        overall_mom_data['Prev_Period_Revenue'] = overall_mom_data['Revenue'].shift(1)
                        overall_mom_data['MoM_Growth_Percent'] = np.where(
                            overall_mom_data['Prev_Period_Revenue'] != 0,
                            ((overall_mom_data['Revenue'] - overall_mom_data['Prev_Period_Revenue']) / overall_mom_data[
                                'Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        mom_data_table_df = overall_mom_data[
                            ['Period', 'Revenue', 'Prev_Period_Revenue', 'MoM_Growth_Percent']].copy()
                        mom_data_table_df.rename(columns={'Revenue': 'Current Month Revenue',
                                                          'Prev_Period_Revenue': 'Previous Month Revenue'},
                                                 inplace=True)
                        mom_data_table_df.insert(0, 'Dimension', 'Total Revenue')  # Add a placeholder for dimension

                    # --- Calculate QoQ Trend Data ---
                    if selected_dim_col_name:
                        qoq_data = df_revenue_filtered.groupby(
                            ['FiscalYear', 'FiscalQuarterNum', selected_dim_col_name]).agg(
                            # Group by numeric components for sorting
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        qoq_data = qoq_data.sort_values(['FiscalYear', 'FiscalQuarterNum'])  # Sort numerically
                        qoq_data['Period'] = qoq_data.apply(
                            lambda row: get_fiscal_quarter_name_from_fy_and_q(row['FiscalYear'],
                                                                              row['FiscalQuarterNum']), axis=1)

                        qoq_data['Prev_Period_Revenue'] = qoq_data.groupby(selected_dim_col_name)['Revenue'].shift(1)
                        qoq_data['QoQ_Growth_Percent'] = np.where(
                            qoq_data['Prev_Period_Revenue'] != 0,
                            ((qoq_data['Revenue'] - qoq_data['Prev_Period_Revenue']) / qoq_data[
                                'Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        qoq_data_table_df = qoq_data[[selected_dim_col_name, 'Period', 'Revenue', 'Prev_Period_Revenue',
                                                      'QoQ_Growth_Percent']].copy()
                        qoq_data_table_df.rename(columns={'Revenue': 'Current Quarter Revenue',
                                                          'Prev_Period_Revenue': 'Previous Quarter Revenue'},
                                                 inplace=True)
                    else:
                        overall_qoq_data = df_revenue_filtered.groupby(['FiscalYear', 'FiscalQuarterNum']).agg(
                            # Group by numeric components for sorting
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        overall_qoq_data = overall_qoq_data.sort_values(
                            ['FiscalYear', 'FiscalQuarterNum'])  # Sort numerically
                        overall_qoq_data['Period'] = overall_qoq_data.apply(
                            lambda row: get_fiscal_quarter_name_from_fy_and_q(row['FiscalYear'],
                                                                              row['FiscalQuarterNum']), axis=1)

                        overall_qoq_data['Prev_Period_Revenue'] = overall_qoq_data['Revenue'].shift(1)
                        overall_qoq_data['QoQ_Growth_Percent'] = np.where(
                            overall_qoq_data['Prev_Period_Revenue'] != 0,
                            ((overall_qoq_data['Revenue'] - overall_qoq_data['Prev_Period_Revenue']) / overall_qoq_data[
                                'Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        qoq_data_table_df = overall_qoq_data[
                            ['Period', 'Revenue', 'Prev_Period_Revenue', 'QoQ_Growth_Percent']].copy()
                        qoq_data_table_df.rename(columns={'Revenue': 'Current Quarter Revenue',
                                                          'Prev_Period_Revenue': 'Previous Quarter Revenue'},
                                                 inplace=True)
                        qoq_data_table_df.insert(0, 'Dimension', 'Total Revenue')

                    # --- Calculate YoY Trend Data ---
                    if selected_dim_col_name:
                        yoy_data = df_revenue_filtered.groupby(['FiscalYearEnd', selected_dim_col_name]).agg(
                            # Group by FiscalYearEnd
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        yoy_data = yoy_data.sort_values('FiscalYearEnd')  # Sort numerically by FiscalYearEnd
                        yoy_data['Fiscal Year'] = yoy_data['FiscalYearEnd'].astype(
                            str)  # Period is now directly FiscalYearEnd

                        yoy_data['Prev_Period_Revenue'] = yoy_data.groupby(selected_dim_col_name)['Revenue'].shift(1)
                        yoy_data['YoY_Growth_Percent'] = np.where(
                            yoy_data['Prev_Period_Revenue'] != 0,
                            ((yoy_data['Revenue'] - yoy_data['Prev_Period_Revenue']) / yoy_data[
                                'Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        yoy_data_table_df = yoy_data[
                            [selected_dim_col_name, 'Fiscal Year', 'Revenue', 'Prev_Period_Revenue',
                             'YoY_Growth_Percent']].copy()
                        yoy_data_table_df.rename(
                            columns={'Revenue': 'Current Year Revenue', 'Prev_Period_Revenue': 'Previous Year Revenue'},
                            inplace=True)  # Renamed 'Period' to 'Fiscal Year'
                    else:
                        overall_yoy_data = df_revenue_filtered.groupby('FiscalYearEnd').agg(  # Group by FiscalYearEnd
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        overall_yoy_data = overall_yoy_data.sort_values(
                            'FiscalYearEnd')  # Sort numerically by FiscalYearEnd
                        overall_yoy_data['Fiscal Year'] = overall_yoy_data['FiscalYearEnd'].astype(
                            str)  # Period is now directly FiscalYearEnd

                        overall_yoy_data['Prev_Period_Revenue'] = overall_yoy_data['Revenue'].shift(1)
                        overall_yoy_data['YoY_Growth_Percent'] = np.where(
                            overall_yoy_data['Prev_Period_Revenue'] != 0,
                            ((overall_yoy_data['Revenue'] - overall_yoy_data['Prev_Period_Revenue']) / overall_yoy_data[
                                'Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        yoy_data_table_df = overall_yoy_data[
                            ['Fiscal Year', 'Revenue', 'Prev_Period_Revenue', 'YoY_Growth_Percent']].copy()
                        yoy_data_table_df.rename(
                            columns={'Revenue': 'Current Year Revenue', 'Prev_Period_Revenue': 'Previous Year Revenue'},
                            inplace=True)  # Renamed 'Period' to 'Fiscal Year'
                        yoy_data_table_df.insert(0, 'Dimension', 'Total Revenue')

                    # --- Display results in main tabs: Data Table and Visual Analysis ---
                    data_tab, visual_tab = st.tabs(["Data Table", "Visual Analysis"])

                    with data_tab:
                        st.subheader(f"Detailed Revenue Trend Data by {selected_dim_for_analysis}")

                        if "MoM" in requested_trends:
                            st.markdown("##### Month-over-Month (MoM) Revenue Trend")
                            if not mom_data_table_df.empty:
                                st.dataframe(mom_data_table_df.style.format({
                                    'Current Month Revenue': '$ {:,.2f}',
                                    'Previous Month Revenue': '$ {:,.2f}',
                                    'MoM_Growth_Percent': '{:.2f}%'
                                }), use_container_width=True)
                            else:
                                st.info("No sufficient data to show MoM trends.")

                        if "QoQ" in requested_trends:
                            st.markdown("##### Quarter-over-Quarter (QoQ) Revenue Trend")
                            if not qoq_data_table_df.empty:
                                st.dataframe(qoq_data_table_df.style.format({
                                    'Current Quarter Revenue': '$ {:,.2f}',
                                    'Previous Quarter Revenue': '$ {:,.2f}',
                                    'QoQ_Growth_Percent': '{:.2f}%'
                                }), use_container_width=True)
                            else:
                                st.info("No sufficient data to show QoQ trends.")

                        if "YoY" in requested_trends:
                            st.markdown("##### Year-over-Year (YoY) Revenue Trend")
                            if not yoy_data_table_df.empty:
                                st.dataframe(yoy_data_table_df.style.format({
                                    'Current Year Revenue': '$ {:,.2f}',
                                    'Previous Year Revenue': '$ {:,.2f}',
                                    'YoY_Growth_Percent': '{:.2f}%'
                                }), use_container_width=True)
                            else:
                                st.info("No sufficient data to show YoY trends.")

                    with visual_tab:
                        st.subheader(f"Revenue Trend Visuals by {selected_dim_for_analysis}")

                        # Nested tabs for each visual
                        mom_chart_tab, qoq_chart_tab, yoy_chart_tab = st.tabs(["MoM Chart", "QoQ Chart", "YoY Chart"])

                        with mom_chart_tab:
                            st.markdown(f"#### Month-over-Month Revenue Trend by {selected_dim_for_analysis}")
                            mom_plot_data = mom_data_table_df.copy()
                            mom_plot_data.rename(
                                columns={'Current Month Revenue': 'Revenue', 'MoM_Growth_Percent': 'Growth_Percent'},
                                inplace=True)
                            mom_plot_color_col = selected_dim_col_name if selected_dim_col_name else 'Dimension'

                            if not mom_plot_data.empty and mom_plot_data['Period'].nunique() > 1:
                                fig_mom_rev = px.line(
                                    mom_plot_data,
                                    x='Period',
                                    y='Revenue',
                                    color=mom_plot_color_col,
                                    title=f'MoM Revenue Trend by {selected_dim_for_analysis}',
                                    labels={'Revenue': 'Revenue (USD)', 'Period': 'Month'},
                                    line_shape='linear'
                                )
                                fig_mom_rev.update_traces(mode='lines+markers', hovertemplate=
                                '<b>%{x}</b><br>' +
                                (
                                    f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                                'Revenue: %{y:$,.2f}<br>' +
                                'MoM Change: %{customdata[1]:.2f}%<extra></extra>'
                                                          )
                                if selected_dim_col_name:
                                    fig_mom_rev.update_traces(
                                        customdata=mom_plot_data[[selected_dim_col_name, 'Growth_Percent']].values)
                                else:
                                    fig_mom_rev.update_traces(customdata=mom_plot_data[['Dimension']].values)

                                fig_mom_rev.update_layout(xaxis_title="Month", yaxis_title="Revenue (USD)",
                                                          yaxis_tickprefix="$", yaxis_tickformat=",.0f")
                                st.plotly_chart(fig_mom_rev, use_container_width=True)

                                mom_growth_plot_data = mom_plot_data.dropna(subset=['Growth_Percent'])
                                if not mom_growth_plot_data.empty:
                                    fig_mom_pct = px.line(
                                        mom_growth_plot_data,
                                        x='Period',
                                        y='Growth_Percent',
                                        color=mom_plot_color_col,
                                        title=f'MoM Revenue Percentage Change by {selected_dim_for_analysis}',
                                        labels={'Growth_Percent': 'MoM Change (%)', 'Period': 'Month'},
                                        line_shape='linear'
                                    )
                                    fig_mom_pct.update_traces(mode='lines+markers', hovertemplate=
                                    '<b>%{x}</b><br>' +
                                    (
                                        f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                                    'MoM Change: %{y:,.2f}%<extra></extra>'
                                                              )
                                    if selected_dim_col_name:
                                        fig_mom_pct.update_traces(
                                            customdata=mom_growth_plot_data[[selected_dim_col_name]].values)
                                    else:
                                        fig_mom_pct.update_traces(customdata=mom_growth_plot_data[['Dimension']].values)

                                    fig_mom_pct.update_layout(xaxis_title="Month", yaxis_title="MoM Change (%)",
                                                              yaxis_tickformat=".2f%")
                                    st.plotly_chart(fig_mom_pct, use_container_width=True)
                                else:
                                    st.info(
                                        f"Not enough data to show MoM percentage change for {selected_dim_for_analysis}.")
                            else:
                                st.info(f"No sufficient data to calculate MoM trends for {selected_dim_for_analysis}.")

                        with qoq_chart_tab:
                            st.markdown(f"#### Quarter-over-Quarter Revenue Trend by {selected_dim_for_analysis}")
                            qoq_plot_data = qoq_data_table_df.copy()
                            qoq_plot_data.rename(
                                columns={'Current Quarter Revenue': 'Revenue', 'QoQ_Growth_Percent': 'Growth_Percent'},
                                inplace=True)
                            qoq_plot_color_col = selected_dim_col_name if selected_dim_col_name else 'Dimension'

                            if not qoq_plot_data.empty and qoq_plot_data['Period'].nunique() > 1:
                                fig_qoq_rev = px.line(
                                    qoq_plot_data,
                                    x='Period',
                                    y='Revenue',
                                    color=qoq_plot_color_col,
                                    title=f'QoQ Revenue Trend by {selected_dim_for_analysis}',
                                    labels={'Revenue': 'Revenue (USD)', 'Period': 'Quarter'},
                                    line_shape='linear'
                                )
                                fig_qoq_rev.update_traces(mode='lines+markers', hovertemplate=
                                '<b>%{x}</b><br>' +
                                (
                                    f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                                'Revenue: %{y:$,.2f}<br>' +
                                'QoQ Change: %{customdata[1]:.2f}%<extra></extra>'
                                                          )
                                if selected_dim_col_name:
                                    fig_qoq_rev.update_traces(
                                        customdata=qoq_plot_data[[selected_dim_col_name, 'Growth_Percent']].values)
                                else:
                                    fig_qoq_rev.update_traces(customdata=qoq_plot_data[['Dimension']].values)

                                fig_qoq_rev.update_layout(xaxis_title="Quarter", yaxis_title="Revenue (USD)",
                                                          yaxis_tickprefix="$", yaxis_tickformat=",.0f")
                                st.plotly_chart(fig_qoq_rev, use_container_width=True)

                                qoq_growth_plot_data = qoq_plot_data.dropna(subset=['Growth_Percent'])
                                if not qoq_growth_plot_data.empty:
                                    fig_qoq_pct = px.line(
                                        qoq_growth_plot_data,
                                        x='Period',
                                        y='Growth_Percent',
                                        color=qoq_plot_color_col,
                                        title=f'QoQ Revenue Percentage Change by {selected_dim_for_analysis}',
                                        labels={'Growth_Percent': 'QoQ Change (%)', 'Period': 'Quarter'},
                                        line_shape='linear'
                                    )
                                    fig_qoq_pct.update_traces(mode='lines+markers', hovertemplate=
                                    '<b>%{x}</b><br>' +
                                    (
                                        f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                                    'QoQ Change: %{y:,.2f}%<extra></extra>'
                                                              )
                                    if selected_dim_col_name:
                                        fig_qoq_pct.update_traces(
                                            customdata=qoq_growth_plot_data[[selected_dim_col_name]].values)
                                    else:
                                        fig_qoq_pct.update_traces(customdata=qoq_growth_plot_data[['Dimension']].values)

                                    fig_qoq_pct.update_layout(xaxis_title="Quarter", yaxis_title="QoQ Change (%)",
                                                              yaxis_tickformat=".2f%")
                                    st.plotly_chart(fig_qoq_pct, use_container_width=True)
                                else:
                                    st.info(
                                        f"Not enough data to show QoQ percentage change for {selected_dim_for_analysis}.")
                            else:
                                st.info(f"No sufficient data to calculate QoQ trends for {selected_dim_for_analysis}.")

                        with yoy_chart_tab:
                            st.markdown(f"#### Year-over-Year Revenue Trend by {selected_dim_for_analysis}")
                            yoy_plot_data = yoy_data_table_df.copy()
                            yoy_plot_data.rename(
                                columns={'Current Year Revenue': 'Revenue', 'YoY_Growth_Percent': 'Growth_Percent'},
                                inplace=True)
                            yoy_plot_color_col = selected_dim_col_name if selected_dim_col_name else 'Dimension'

                            if not yoy_plot_data.empty and yoy_plot_data[
                                'Fiscal Year'].nunique() > 1:  # Changed 'Period' to 'Fiscal Year'
                                fig_yoy_rev = px.line(
                                    yoy_plot_data,
                                    x='Fiscal Year',  # Changed x-axis to 'Fiscal Year'
                                    y='Revenue',
                                    color=yoy_plot_color_col,
                                    title=f'YoY Revenue Trend by {selected_dim_for_analysis}',
                                    labels={'Revenue': 'Revenue (USD)', 'Fiscal Year': 'Fiscal Year'},  # Updated label
                                    line_shape='linear'
                                )
                                fig_yoy_rev.update_traces(mode='lines+markers', hovertemplate=
                                '<b>%{x}</b><br>' +
                                (
                                    f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                                'Revenue: %{y:$,.2f}<br>' +
                                'YoY Change: %{customdata[1]:.2f}%<extra></extra>'
                                                          )
                                if selected_dim_col_name:
                                    fig_yoy_rev.update_traces(
                                        customdata=yoy_plot_data[[selected_dim_col_name, 'Growth_Percent']].values)
                                else:
                                    fig_yoy_rev.update_traces(customdata=yoy_plot_data[['Dimension']].values)

                                fig_yoy_rev.update_layout(xaxis_title="Fiscal Year", yaxis_title="Revenue (USD)",
                                                          yaxis_tickprefix="$",
                                                          yaxis_tickformat=",.0f")  # Updated x-axis title
                                st.plotly_chart(fig_yoy_rev, use_container_width=True)

                                yoy_growth_plot_data = yoy_plot_data.dropna(subset=['Growth_Percent'])
                                if not yoy_growth_plot_data.empty:
                                    fig_yoy_pct = px.line(
                                        yoy_growth_plot_data,
                                        x='Fiscal Year',  # Changed x-axis to 'Fiscal Year'
                                        y='Growth_Percent',
                                        color=yoy_plot_color_col,
                                        title=f'YoY Revenue Percentage Change by {selected_dim_for_analysis}',
                                        labels={'Growth_Percent': 'YoY Change (%)', 'Fiscal Year': 'Fiscal Year'},
                                        # Updated label
                                        line_shape='linear'
                                    )
                                    fig_yoy_pct.update_traces(mode='lines+markers', hovertemplate=
                                    '<b>%{x}</b><br>' +
                                    (
                                        f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                                    'YoY Change: %{y:,.2f}%<extra></extra>'
                                                              )
                                    if selected_dim_col_name:
                                        fig_yoy_pct.update_traces(
                                            customdata=yoy_growth_plot_data[[selected_dim_col_name]].values)
                                    else:
                                        fig_yoy_pct.update_traces(customdata=yoy_growth_plot_data[['Dimension']].values)

                                    fig_yoy_pct.update_layout(xaxis_title="Fiscal Year", yaxis_title="YoY Change (%)",
                                                              yaxis_tickformat=".2f%")  # Updated x-axis title
                                    st.plotly_chart(fig_yoy_pct, use_container_width=True)
                                else:
                                    st.info(
                                        f"Not enough data to show YoY percentage change for {selected_dim_for_analysis}.")
                            else:
                                st.info(f"No sufficient data to calculate YoY trends for {selected_dim_for_analysis}.")




        elif query_type == "Transportation_cost_analysis":
            # --- NEW FIX: Get the segment name dynamically for the heading ---
            segment_name = query_details.get("column_filters", {}).get("Segment", "Specified Segment")
            st.markdown(f"### Cost Trend Analysis for {segment_name}")

            transport_result = analyze_transportation_cost_trend(df_pl, query_details)

            if isinstance(transport_result, dict) and "Message" in transport_result:
                st.error(transport_result["Message"])
            elif isinstance(transport_result, pd.DataFrame):
                if transport_result.empty:
                    st.warning(f"No specific costs increased in the {segment_name} segment for the specified period.")
                else:
                   # st.subheader(f"Cost Changes in {segment_name} Segment:")
                    st.dataframe(transport_result)
                    # Removed visualization for this query type as requested
            else:
                st.error("An unexpected error occurred during Transportation cost analysis.")


        elif query_type == "C&B_cost_variation":
            st.markdown("### C&B Cost Variation Analysis")

            # Call the calculate_cb_variation function from kpi_calculationss.py
            # This function now returns a DataFrame and a message
            cb_result_df, message = calculate_cb_variation(df_pl, query_details)

            if not cb_result_df.empty and "Error" not in cb_result_df.columns:
                st.write(message)
                st.dataframe(cb_result_df.set_index("Metric"))  # Display as a table
            else:
                # This else block handles cases where calculate_cb_variation returns
                # an empty DataFrame with an "Info" or "Error" column.
                st.error(message)
                if "Error" in cb_result_df.columns:
                    st.error(cb_result_df["Error"].iloc[0])



        elif query_type == "CB_revenue_trend":
            st.markdown("### C&B Cost vs. Total Revenue Monthly Trend Analysis")
            cb_trend_df = calculate_cb_revenue_trend(df_pl, query_details)

            if isinstance(cb_trend_df, dict) and "Message" in cb_trend_df:
                st.error(cb_trend_df["Message"])
            elif isinstance(cb_trend_df, pd.DataFrame):
                if cb_trend_df.empty:
                    st.warning("No C&B Revenue Trend data found for the specified criteria.")
                else:
                    tab1_cb_trend, tab2_cb_trend = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                    with tab1_cb_trend:
                        st.subheader("Monthly C&B Cost, Total Revenue, and Ratios:")
                        st.dataframe(cb_trend_df.style.format({
                            "CB_Cost": lambda x: f"${x:,.2f}",
                            "Total_Revenue": lambda x: f"${x:,.2f}",
                            "CB_Revenue_Difference": lambda x: f"${x:,.2f}",
                            "CB_Cost_vs_Revenue_Ratio_Percent": lambda x: f"{x:,.2f}%" if pd.notna(x) else "N/A"
                        }))

                    with tab2_cb_trend:
                        st.subheader("Monthly Trend: C&B Cost, Total Revenue, and C&B Cost vs. Revenue Ratio")

                        fig = make_subplots(specs=[[{"secondary_y": True}]])

                        fig.add_trace(
                            go.Bar(
                                x=cb_trend_df["Month"],
                                y=cb_trend_df["CB_Cost"],
                                name="C&B Cost",
                                marker_color='rgb(55, 83, 109)',
                                hovertemplate="<b>Month:</b> %{x}<br><b>C&B Cost:</b> %{y:$,.2f}<extra></extra>"
                            ),
                            secondary_y=False,
                        )

                        fig.add_trace(
                            go.Bar(
                                x=cb_trend_df["Month"],
                                y=cb_trend_df["Total_Revenue"],
                                name="Total Revenue",
                                marker_color='rgb(26, 118, 255)',
                                hovertemplate="<b>Month:</b> %{x}<br><b>Total Revenue:</b> %{y:$,.2f}<extra></extra>"
                            ),
                            secondary_y=False,
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=cb_trend_df["Month"],
                                y=cb_trend_df["CB_Cost_vs_Revenue_Ratio_Percent"],
                                name="C&B Cost vs Revenue Ratio (%)",
                                mode="lines+markers",
                                yaxis="y2",  # Use secondary y-axis
                                line=dict(color='red', width=3),
                                hovertemplate="<b>Month:</b> %{x}<br><b>C&B Cost vs Revenue Ratio:</b> %{y:,.2f}%<extra></extra>"
                            ),
                            secondary_y=True,
                        )

                        fig.update_layout(
                            title_text="Monthly Trend: C&B Cost, Total Revenue, and C&B Cost vs. Revenue Ratio",
                            xaxis_title="Month",
                            barmode='group',
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=600
                        )

                        min_ratio = cb_trend_df['CB_Cost_vs_Revenue_Ratio_Percent'].min()
                        max_ratio = cb_trend_df['CB_Cost_vs_Revenue_Ratio_Percent'].max()
                        ratio_padding = (max_ratio - min_ratio) * 0.1 if (max_ratio - min_ratio) != 0 else 10

                        fig.update_yaxes(title_text="Amount (USD)", secondary_y=False, tickprefix="$",
                                         tickformat=",.0f")
                        fig.update_yaxes(
                            title_text="C&B Cost vs Revenue Ratio (%)",
                            secondary_y=True,
                            tickformat=".2f%",
                            range=[min_ratio - ratio_padding, max_ratio + ratio_padding]
                        )

                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("An unexpected error occurred during C&B Revenue Trend analysis.")


        elif query_type == "HC_trend":
            st.markdown("### Monthly Headcount (HC) Trend by Account")

            hc_trend_result = calculate_hc_trend(df_ut, query_details)

            if isinstance(hc_trend_result, dict) and "Message" in hc_trend_result:
                st.error(hc_trend_result["Message"])
            elif isinstance(hc_trend_result, pd.DataFrame):
                hc_trend_df = hc_trend_result
                specific_customer = query_details.get("column_filters", {}).get("FinalCustomerName")

                if hc_trend_df.empty:
                    if specific_customer:
                        st.warning(
                            f"No Headcount data found for account '{specific_customer}' for the specified period.")
                    else:
                        st.info("No Headcount data available for the specified period or accounts.")
                else:
                    tab1_hc_trend, tab2_hc_trend = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                    with tab1_hc_trend:
                        st.subheader("Monthly Headcount by Account (Full Data):")
                        if specific_customer:
                            filtered_data_table_df = hc_trend_df[
                                hc_trend_df['FinalCustomerName'].str.lower() == specific_customer.lower().strip()]
                            st.dataframe(filtered_data_table_df)
                            if filtered_data_table_df.empty:
                                st.info(f"No detailed data for '{specific_customer}' in this period.")
                        else:
                            st.dataframe(hc_trend_df)

                    with tab2_hc_trend:
                        if specific_customer:
                            st.subheader(f"Monthly Headcount Trend for '{specific_customer}'")
                            customer_df_for_plot = hc_trend_df[hc_trend_df[
                                                                   'FinalCustomerName'].str.lower() == specific_customer.lower().strip()].copy()

                            if not customer_df_for_plot.empty:
                                customer_df_for_plot = customer_df_for_plot.groupby('Month')['HC'].sum().reset_index()

                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=customer_df_for_plot['Month'],
                                        y=customer_df_for_plot['HC'],
                                        mode='lines+markers',
                                        name=specific_customer,
                                        text=[specific_customer] * len(customer_df_for_plot),
                                        hovertemplate="<b>Month:</b> %{x}<br><b>Customer:</b> %{text}<br><b>HC:</b> %{y}<extra></extra>"
                                    )
                                )
                                fig.update_layout(
                                    title_text=f"Monthly Headcount Trend for '{specific_customer}'",
                                    xaxis_title="Month",
                                    yaxis_title="Headcount (HC)",
                                    hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                    height=600
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(
                                    f"No headcount data available for '{specific_customer}' in the specified period to visualize.")
                        else:
                            st.subheader("Monthly Headcount Trend for Top 10 Customers and Others")

                            num_top_customers = 10

                            fig = go.Figure()

                            total_hc_by_customer = hc_trend_df.groupby('FinalCustomerName')['HC'].sum().sort_values(
                                ascending=False)
                            top_customers = total_hc_by_customer.head(num_top_customers).index.tolist()
                            other_customers = total_hc_by_customer.tail(
                                len(total_hc_by_customer) - num_top_customers).index.tolist()

                            for customer in top_customers:
                                customer_df = hc_trend_df[hc_trend_df['FinalCustomerName'] == customer]
                                customer_df_for_plot = customer_df.groupby('Month')['HC'].sum().reset_index()

                                fig.add_trace(
                                    go.Scatter(
                                        x=customer_df_for_plot['Month'],
                                        y=customer_df_for_plot['HC'],
                                        mode='lines+markers',
                                        name=customer,
                                        text=[customer] * len(customer_df_for_plot),
                                        hovertemplate="<b>Month:</b> %{x}<br><b>Customer:</b> %{text}<br><b>HC:</b> %{y}<extra></extra>"
                                    )
                                )

                            if other_customers:
                                others_df = hc_trend_df[hc_trend_df['FinalCustomerName'].isin(other_customers)]
                                others_monthly_hc = others_df.groupby('Month')['HC'].sum().reset_index()
                                fig.add_trace(
                                    go.Scatter(
                                        x=others_monthly_hc['Month'],
                                        y=others_monthly_hc['HC'],
                                        mode='lines+markers',
                                        name='Others',
                                        line=dict(dash='dot', color='gray'),
                                        hovertemplate="<b>Month:</b> %{x}<br><b>Group:</b> Others<br><b>Total HC:</b> %{y}<extra></extra>"
                                    )
                                )

                            fig.update_layout(
                                title_text=f"Monthly Headcount Trend for Top {num_top_customers} Customers and Others",
                                xaxis_title="Month",
                                yaxis_title="Headcount (HC)",
                                hovermode="x unified",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                height=600
                            )

                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(
                    "An unexpected error occurred during Headcount trend analysis. Please check the `calculate_hc_trend` function's return type.")







        elif query_type == "UT_trend":
            st.markdown("### 📈 Utilization (UT) Trend Analysis")


            @st.cache_data(show_spinner="Analyzing UT Trend...")
            def get_ut_trend_data_cached(df_ut_data, q_details_hashable):
                q_details_for_analysis = q_details_hashable.copy()
                for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                    if q_details_for_analysis.get(k) and isinstance(q_details_for_analysis[k], str):
                        q_details_for_analysis[k] = datetime.fromisoformat(q_details_for_analysis[k])
                return analyze_ut_trend(df_ut_data, q_details_for_analysis)


            query_details_hashable = query_details.copy()
            for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                if query_details_hashable.get(k) and isinstance(query_details_hashable[k], datetime):
                    query_details_hashable[k] = query_details_hashable[k].isoformat()

            ut_trend_output = get_ut_trend_data_cached(df_ut.copy(), query_details_hashable)

            if isinstance(ut_trend_output, dict) and "Message" in ut_trend_output:
                st.error(ut_trend_output["Message"])
            elif isinstance(ut_trend_output, dict):
                ut_trend_df = ut_trend_output["df_ut_trend"]
                trend_dimension_display = ut_trend_output["trend_dimension_display"]
                trend_granularity_display = ut_trend_output["trend_granularity_display"]

                if ut_trend_df.empty:
                    st.warning(
                        f"No UT trend data found for the specified criteria and period ({query_details.get('description', 'selected range')}).")
                else:
                    st.info(
                        f"Displaying **{trend_granularity_display}** UT% trend by **{trend_dimension_display}** for the period: **{query_details.get('description', 'specified date range')}**")

                    # Determine the actual column name for the trend dimension in the dataframe
                    actual_dim_col_in_df = None
                    if trend_dimension_display == "DU":
                        actual_dim_col_in_df = "Exec DU"
                    elif trend_dimension_display == "BU":
                        actual_dim_col_in_df = "Exec DG"
                    elif trend_dimension_display == "Account":
                        actual_dim_col_in_df = "FinalCustomerName"
                    else:  # "All" case
                        actual_dim_col_in_df = 'Dimension_Name'  # The dummy column for 'All'

                    # Ensure the column exists before using it as 'color' in px.line
                    plot_color_col = actual_dim_col_in_df if actual_dim_col_in_df and actual_dim_col_in_df in ut_trend_df.columns else None

                    data_tab, visual_tab = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                    with data_tab:
                        st.markdown(f"##### Detailed UT Trend Data by {trend_dimension_display}")
                        # Determine the columns to display in the dataframe
                        display_cols = ['Period_Formatted', 'TotalBillableHours', 'NetAvailableHours', 'UT_Percent']

                        # Add the dimension column if not 'All'
                        if plot_color_col and plot_color_col in ut_trend_df.columns:
                            display_cols.insert(1, plot_color_col)

                        st.dataframe(ut_trend_df[display_cols].style.format({
                            'TotalBillableHours': '{:,.2f}',
                            'NetAvailableHours': '{:,.2f}',
                            'UT_Percent': '{:.2f}%'
                        }), use_container_width=True)

                    with visual_tab:
                        st.markdown(
                            f"#### {trend_granularity_display.capitalize()} UT Trend by {trend_dimension_display}")

                        if not ut_trend_df.empty:
                            fig_ut_trend = px.line(
                                ut_trend_df,
                                x='Period_Formatted',
                                y='UT_Percent',
                                color=plot_color_col,  # Use the determined color column
                                title=f'{trend_granularity_display.capitalize()} UT% Trend by {trend_dimension_display}',
                                labels={'UT_Percent': 'UT %', 'Period_Formatted': 'Period'},
                                line_shape='linear'
                            )

                            custom_data_cols = []
                            if plot_color_col:
                                custom_data_cols.append(plot_color_col)
                            custom_data_cols.extend(['TotalBillableHours', 'NetAvailableHours'])

                            fig_ut_trend.update_traces(mode='lines+markers', hovertemplate=
                            '<b>Period:</b> %{x}<br>' +
                            (f'<b>{trend_dimension_display}:</b> %{{customdata[0]}}<br>' if plot_color_col else '') +
                            '<b>UT%:</b> %{y:,.2f}%<br>' +
                            (
                                f'<b>Total Billable Hours:</b> %{{customdata[{len(custom_data_cols) - 2}]:,.0f}}<br>' if 'TotalBillableHours' in custom_data_cols else '') +
                            (
                                f'<b>Net Available Hours:</b> %{{customdata[{len(custom_data_cols) - 1}]:,.0f}}<extra></extra>' if 'NetAvailableHours' in custom_data_cols else '')
                                                       )
                            fig_ut_trend.update_traces(customdata=ut_trend_df[custom_data_cols].values)

                            fig_ut_trend.update_layout(
                                xaxis_title=trend_granularity_display.capitalize(),
                                yaxis_title="UT %",
                                yaxis_tickformat=".2f%",
                                hovermode="x unified",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                height=600
                            )
                            st.plotly_chart(fig_ut_trend, use_container_width=True)
                        else:
                            st.info(f"Not enough data to generate UT trend chart for {trend_dimension_display}.")
            else:  # Fallback for unexpected return type from analyze_ut_trend
                st.error(
                    "An unexpected error occurred during UT trend analysis. Please check the `analyze_ut_trend` function's return type.")

        elif query_type == "Fresher_UT_Trend":
            st.markdown("### 📈 DU-wise Fresher Utilization (UT) Trend Analysis")


            @st.cache_data(show_spinner="Analyzing Fresher UT Trend...")
            def get_fresher_ut_trend_data_cached(df_ut_data, q_details_hashable):
                q_details_for_analysis = q_details_hashable.copy()
                for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                    if q_details_for_analysis.get(k) and isinstance(q_details_for_analysis[k], str):
                        q_details_for_analysis[k] = datetime.fromisoformat(q_details_for_analysis[k])
                return analyze_fresher_ut_trend(df_ut_data, q_details_for_analysis)


            query_details_hashable = query_details.copy()
            for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                if query_details_hashable.get(k) and isinstance(query_details_hashable[k], datetime):
                    query_details_hashable[k] = query_details_hashable[k].isoformat()

            fresher_ut_output = get_fresher_ut_trend_data_cached(df_ut.copy(), query_details_hashable)

            if isinstance(fresher_ut_output, dict) and "Message" in fresher_ut_output:
                st.error(fresher_ut_output["Message"])
            elif isinstance(fresher_ut_output, dict):
                fresher_ut_trend_df = fresher_ut_output["df_fresher_ut_trend"]
                trend_dimension_display = fresher_ut_output["trend_dimension_display"]
                trend_granularity_display = fresher_ut_output["trend_granularity_display"]

                if not fresher_ut_trend_df.empty:
                    st.info(
                        f"Displaying **{trend_granularity_display}** Fresher UT% trend by **{trend_dimension_display}** for the period: **{query_details.get('description', 'last 12 months (default)')}**")

                    tab1_fresher_ut, tab2_fresher_ut = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                    with tab1_fresher_ut:
                        st.subheader("Monthly Fresher UT% by Delivery Unit:")
                        st.dataframe(fresher_ut_trend_df.style.format({
                            'TotalBillableHours': '{:,.0f}',
                            'NetAvailableHours': '{:,.0f}',
                            'UT_Percent': '{:.2f}%'
                        }), use_container_width=True)

                    with tab2_fresher_ut:
                        st.subheader("Monthly Fresher UT% Trend by Delivery Unit")

                        fig_fresher_ut = px.line(
                            fresher_ut_trend_df,
                            x='Period_Formatted',
                            y='UT_Percent',
                            color='Exec DU',
                            title='Monthly Fresher UT% Trend by Delivery Unit',
                            labels={'UT_Percent': 'UT %', 'Period_Formatted': 'Month', 'Exec DU': 'Delivery Unit'},
                            line_shape='linear'
                        )
                        fig_fresher_ut.update_traces(mode='lines+markers', hovertemplate=
                        '<b>Month:</b> %{x}<br>' +
                        '<b>Delivery Unit:</b> %{customdata[0]}<br>' +
                        '<b>Fresher UT%:</b> %{y:,.2f}%<br>' +
                        'Total Billable Hours: %{customdata[1]:,.0f}<br>' +
                        'Net Available Hours: %{customdata[2]:,.0f}<extra></extra>'
                                                     )
                        fig_fresher_ut.update_traces(customdata=fresher_ut_trend_df[
                            ['Exec DU', 'TotalBillableHours', 'NetAvailableHours']].values)

                        fig_fresher_ut.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Fresher UT %",
                            yaxis_tickformat=".2f%",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=600
                        )
                        st.plotly_chart(fig_fresher_ut, use_container_width=True)
                else:
                    st.info(
                        f"No sufficient data to calculate Fresher UT trends for {trend_dimension_display} for the specified period.")
            else:
                st.info("No data available to calculate Fresher UT trends.")

        elif query_type == "Revenue_Per_Person_Trend":
            st.markdown("### 📈 Revenue Per Person Trend Analysis")


            @st.cache_data(show_spinner="Analyzing Revenue Per Person Trend...")
            def get_revenue_per_person_data_cached(df_merged_data, q_details_hashable):
                q_details_for_analysis = q_details_hashable.copy()
                for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                    if q_details_for_analysis.get(k) and isinstance(q_details_for_analysis[k], str):
                        q_details_for_analysis[k] = datetime.fromisoformat(q_details_for_analysis[k])
                return analyze_revenue_per_person_trend(df_merged_data, q_details_for_analysis)


            query_details_hashable = query_details.copy()
            for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                if query_details_hashable.get(k) and isinstance(query_details_hashable[k], datetime):
                    query_details_hashable[k] = query_details_hashable[k].isoformat()

            revenue_per_person_output = get_revenue_per_person_data_cached(df_merged.copy(), query_details_hashable)

            if isinstance(revenue_per_person_output, dict) and "Message" in revenue_per_person_output:
                st.error(revenue_per_person_output["Message"])
            elif isinstance(revenue_per_person_output, dict):
                revenue_per_person_df = revenue_per_person_output["df_revenue_per_person_trend"]
                trend_dimension_display = revenue_per_person_output["trend_dimension_display"]
                trend_granularity_display = revenue_per_person_output["trend_granularity_display"]

                if not revenue_per_person_df.empty:
                    st.info(
                        f"Displaying **{trend_granularity_display}** Revenue per Person trend by **{trend_dimension_display}** for the period: **{query_details.get('description', 'last 12 months (default)')}**")

                    tab1_rev_per_person, tab2_rev_per_person = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                    with tab1_rev_per_person:
                        st.subheader("Monthly Revenue per Person by Customer:")
                        st.dataframe(revenue_per_person_df.style.format({
                            'TotalRevenue': '$ {:,.2f}',
                            'Headcount': '{:,.0f}',
                            'Revenue_Per_Person': '$ {:,.2f}'
                        }), use_container_width=True)

                    with tab2_rev_per_person:
                        st.subheader("Monthly Revenue Per Person Trend by Customer")

                        fig_rev_per_person = px.line(
                            revenue_per_person_df,
                            x='Month_Formatted',
                            y='Revenue_Per_Person',
                            color='FinalCustomerName',
                            title='Monthly Revenue Per Person Trend by Customer',
                            labels={'Revenue_Per_Person': 'Revenue Per Person (USD)', 'Month_Formatted': 'Month',
                                    'FinalCustomerName': 'Customer Name'},
                            line_shape='linear'
                        )
                        fig_rev_per_person.update_traces(mode='lines+markers', hovertemplate=
                        '<b>Month:</b> %{x}<br>' +
                        '<b>Customer:</b> %{customdata[0]}<br>' +
                        '<b>Revenue Per Person:</b> %{y:$,.2f}<br>' +
                        'Total Revenue: %{customdata[1]:$,.2f}<br>' +
                        'Headcount: %{customdata[2]:,.0f}<extra></extra>'
                                                         )
                        fig_rev_per_person.update_traces(
                            customdata=revenue_per_person_df[['FinalCustomerName', 'TotalRevenue', 'Headcount']].values)

                        fig_rev_per_person.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Revenue Per Person (USD)",
                            yaxis_tickprefix="$",
                            yaxis_tickformat=",.2f",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=600
                        )
                        st.plotly_chart(fig_rev_per_person, use_container_width=True)
                else:
                    st.info(
                        f"No sufficient data to calculate Revenue per Person trends for {trend_dimension_display} for the specified period.")
            else:
                st.info("No data available to calculate Revenue per Person trends.")

        elif query_type == "Realized_Rate_Drop":
            st.markdown("### 📉 Realized Rate Drop Analysis")


            @st.cache_data(show_spinner="Analyzing Realized Rate Drop...")
            def get_realized_rate_drop_data_cached(df_merged_data, q_details_hashable):
                q_details_for_analysis = q_details_hashable.copy()
                for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                    if q_details_for_analysis.get(k) and isinstance(q_details_for_analysis[k], str):
                        q_details_for_analysis[k] = datetime.fromisoformat(q_details_for_analysis[k])
                return analyze_realized_rate_drop(df_merged_data, q_details_for_analysis)


            query_details_hashable = query_details.copy()
            for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                if query_details_hashable.get(k) and isinstance(query_details_hashable[k], datetime):
                    query_details_hashable[k] = query_details_hashable[k].isoformat()

            realized_rate_drop_output = get_realized_rate_drop_data_cached(df_merged.copy(), query_details_hashable)

            if isinstance(realized_rate_drop_output, dict) and "Message" in realized_rate_drop_output:
                st.error(realized_rate_drop_output["Message"])
            elif isinstance(realized_rate_drop_output, dict):
                realized_rate_drop_df = realized_rate_drop_output["df_realized_rate_drop"]
                current_q_name = realized_rate_drop_output["current_quarter_name"]
                prev_q_name = realized_rate_drop_output["previous_quarter_name"]
                drop_threshold = realized_rate_drop_output["drop_threshold"]
                drop_threshold_type = realized_rate_drop_output["drop_threshold_type"]

                if realized_rate_drop_df.empty:
                    formatted_threshold = f"${drop_threshold:,.2f}" if drop_threshold_type == 'absolute' else f"{drop_threshold:.2%}"
                    st.warning(
                        f"No accounts experienced a realized rate drop of more than {formatted_threshold} from {prev_q_name} to {current_q_name} after applying all criteria.")
                else:
                    st.subheader(f"Accounts with Significant Realized Rate Drop ({prev_q_name} to {current_q_name})")
                    st.dataframe(realized_rate_drop_df)

                    drop_col_for_plot = "Rate Drop (USD)" if drop_threshold_type == 'absolute' else "Rate Drop (%)"

                    realized_rate_drop_df[f'Realized Rate ({prev_q_name})'] = pd.to_numeric(
                        realized_rate_drop_df[f'Realized Rate ({prev_q_name})'], errors='coerce')
                    realized_rate_drop_df[f'Realized Rate ({current_q_name})'] = pd.to_numeric(
                        realized_rate_drop_df[f'Realized Rate ({current_q_name})'], errors='coerce')
                    realized_rate_drop_df[drop_col_for_plot] = pd.to_numeric(realized_rate_drop_df[drop_col_for_plot],
                                                                             errors='coerce')

                    chart = px.bar(realized_rate_drop_df,
                                   x='FinalCustomerName',
                                   y=drop_col_for_plot,
                                   title=f'Realized Rate Drop by Account ({prev_q_name} vs {current_q_name})',
                                   labels={drop_col_for_plot: drop_col_for_plot,
                                           'FinalCustomerName': 'Final Customer Name'},
                                   hover_data={f'Realized Rate ({prev_q_name})': ':.2f',
                                               f'Realized Rate ({current_q_name})': ':.2f', drop_col_for_plot: ':.2f'})
                    chart.update_layout(xaxis_title='Final Customer Name', yaxis_title=drop_col_for_plot)
                    st.plotly_chart(chart, use_container_width=True)

            else:
                st.info("No data available for Realized Rate Drop analysis.")

        else:
            st.warning(
                "Sorry, I can only assist with Contribution Margin, Transportation Cost, C&B Cost Variation, C&B Revenue Trend, Headcount Trend, Revenue Trend Analysis, Fresher UT Trend, Revenue Per Person Trend, and Realized Rate Drop queries at the moment.")

# --- Add the new clickable button functionality ---
st.markdown("#### Try a sample query:")
sample_queries = [
    "List accounts which have C.M. <30% in FY26-Q1",
    "Which cost triggered the Margin drop last month in Transportation",
    "How much C&B varied from last quarter to this quarter",
    "What is M-o-M trend of C&B cost % w.r.t Total Revenue",
    "What is M-o-M HC for an account",
    "What is YoY revenue for DU",
    "What is the UT trend for last 2 quarters for a DU",
    "DU wise Fresher UT Trends",
    "Which are the accounts where the realized rate dropped more than 3/5th in this quarter",
    "What is MoM trend of Revenue per person"

]
# cols = st.columns(len(sample_queries))
# for i, query in enumerate(sample_queries):
#     with cols[i]:
#         if st.button(query):
#             st.session_state.user_query = query  # Update the session state with the button's value

col1, col2 = st.columns(2)

# --- First Column: Display the first 5 questions ---
# Use the `with` statement to place all content into the first column.
with col1:
    # Iterate through the first half of the queries (index 0 to 4)
    for query in sample_queries[0:5]:
        # Create a button for each query.
        st.button(query, on_click=set_query, args=(query,), key=f"button_{query}")

# --- Second Column: Display the remaining 5 questions ---
# Use the `with` statement to place all content into the second column.
with col2:
    # Iterate through the second half of the queries (index 5 to 9)
    for query in sample_queries[5:10]:
        st.button(query, on_click=set_query, args=(query,), key=f"button_{query}")