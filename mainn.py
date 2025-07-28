# mainn.py (Corrected Date Type Handling and Typo Fix)

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
# Import both get_pl_data and get_ut_data
from data_loaderr import get_pl_data, get_ut_data ,get_merged_data
# Import all necessary functions from kpi_calculationss
from kpi_calculationss import calculate_cm, get_query_details, analyze_transportation_cost_trend, calculate_cb_cost_variation, calculate_cb_revenue_trend, calculate_hc_trend,analyze_revenue_trend ,analyze_ut_trend,analyze_fresher_ut_trend,analyze_revenue_per_person_trend,analyze_realized_rate_drop# Import new function
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np # Import numpy for dynamic range calculation
import sys
import os
from datetime import datetime

# Add the directory containing kpi_calculationss.py to the Python path
# Assuming kpi_calculationss.py is in the same directory as mainn.py
sys.path.append(os.path.dirname(__file__))


# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="L&T KPI Assistant")

st.title("📊 L&T KPI Assistant")
st.markdown("Ask a KPI-related question (e.g., 'show cm% <30% in FY26-Q1', 'Which cost triggered the Margin drop last month as compared to its previous month in Transportation', **'How much C&B varied from last quarter to this quarter'**,**'What is M-o-M trend of C&B cost % w.r.t total revenue'**, **'What is M-o-M HC for an account'**, **'What is YoY revenue for DU'**, **'What is the UT trend for last 2 quarters for a DU'**, **'DU wise Fresher UT Trends'**,**'Which are the accounts where the realized ratendropped more than $3/$5 in this quarter'**)")

# Define these global lists in mainn.py as well, so they are accessible for total calculations
REVENUE_GROUPS = ["ONSITE", "OFFSHORE", "INDIRECT REVENUE"]
COST_GROUPS = [
    "Direct Expense", "OWN OVERHEADS", "Indirect Expense",
    "Project Level Depreciation", "Direct Expense - DU Block Seats Allocation",
    "Direct Expense - DU Pool Allocation", "Establishment Expenses"
]

@st.cache_data
def load_data_for_streamlit():
    """Load P&L data using the get_pl_data function from data_loader."""
    with st.spinner("Loading P&L data from Azure Blob..."):
        df = get_pl_data()
    if df.empty:
        st.error("Failed to load P&L data. Please check data source and credentials.")
        st.stop() # Stop the app if data can't be loaded
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
        st.stop() # Stop the app if data can't be loaded
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
        st.stop() # Stop the app if data can't be loaded
    # Ensure 'Date' column is datetime type for filtering and is timezone-naive
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df



df_pl = load_data_for_streamlit()
df_ut = load_ut_data_for_streamlit()
df_merged = load_merged_data_for_streamlit()

st.write("---")

# --- User Input Section ---
user_query = st.text_input(
    "Enter your query:",
    value="Which are the accounts where the realized rate dropped more than $3 in this quarter" # Default query for testing new functionality
)

if st.button("Submit"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        query_details = get_query_details(user_query)
        
        # --- CRITICAL FIX: Check if query_details is None ---
        if query_details is None:
            st.error("Failed to process your query. The AI model could not extract query details. Please try again or rephrase your query.")
            st.stop() # Stop execution to prevent further errors
        # --- END CRITICAL FIX ---

        query_type = query_details.get("query_type")
        
        # --- DEBUGGING LINE ---
        
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
                st.warning(f"No data available for the primary period ({query_details.get('description', 'specified date range')}) for CM analysis. Please check the data for this range.")
            else:
                result_df = calculate_cm(filtered_df_for_cm, query_details) 
                
                if not result_df.empty:
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

                    st.markdown("---") # Separator for visual appeal

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
                        sorted_df = result_df.dropna(subset=['CM_Value_Numeric']).sort_values(by="CM_Value_Numeric", ascending=True)

                        if sorted_df.empty:
                            st.warning("No data available for plotting after CM % numeric conversion and NaN removal.")
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
                                    hovertemplate="<b>Customer:</b> %{x}<br><b>Revenue:</b> %{y:$,.2f}<extra></extra>" # Custom hover for Revenue
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
                                    hovertemplate="<b>Customer:</b> %{x}<br><b>Cost:</b> %{y:$,.2f}<extra></extra>" # Custom hover for Cost
                                ),
                                secondary_y=False,
                            )

                            # Add CM% Line Chart
                            fig.add_trace(
                                go.Scatter(
                                    x=sorted_df["FinalCustomerName"],
                                    y=sorted_df["CM_Value_Numeric"], # Still use numeric for plot positioning
                                    name="CM %",
                                    mode="lines+markers",
                                    yaxis="y2",
                                    line=dict(color='red', width=3),
                                    # Use CM (%) for hover display
                                    hovertemplate="<b>Customer:</b> %{x}<br><b>CM %:</b> %{customdata}<extra></extra>",
                                    customdata=sorted_df["CM (%)"] # Pass the formatted CM (%) string here
                                ),
                                secondary_y=True,
                            )

                            # Calculate dynamic Y-axis range for CM %
                            min_cm = sorted_df['CM_Value_Numeric'].min()
                            max_cm = sorted_df['CM_Value_Numeric'].max()
                            
                            # Add some padding to the range
                            padding = (max_cm - min_cm) * 0.1 if (max_cm - min_cm) != 0 else 10 # 10% padding, or 10 if range is zero
                            
                            # Ensure min_cm does not go too low if all values are positive
                            # or ensure it starts from a reasonable value if all CMs are high
                            lower_bound_cm = min_cm - padding
                            upper_bound_cm = max_cm + padding

                            # Optionally, ensure a minimum lower bound if CM can be very negative
                            if lower_bound_cm > -200: # Example: don't let it go much lower than -200%
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
                                height=600 # Set a fixed height for better consistency
                            )

                            # Set y-axes titles and formats
                            fig.update_yaxes(title_text="Revenue/Cost (USD)", secondary_y=False, tickprefix="$", tickformat=",.0f")
                            fig.update_yaxes(title_text="CM %", secondary_y=True, tickformat=".2f%", range=[lower_bound_cm, upper_bound_cm])
                                                    
                            st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No customers found matching the specified CM filter for the selected period.")

        elif query_type == "Transportation_cost_analysis":
            st.markdown("### Transportation Cost Trend Analysis")
            # Corrected: Call analyze_transportation_cost_trend directly
            transport_result = analyze_transportation_cost_trend(df_pl, query_details)
            if isinstance(transport_result, pd.DataFrame):
                st.subheader("Cost Changes in Transportation Segment:")
                st.dataframe(transport_result)
            else:
                st.warning(transport_result)

        elif query_type == "C&B_cost_variation":
            st.markdown("### C&B Cost Variation Analysis")
            # Corrected: Call calculate_cb_cost_variation directly
            cb_result = calculate_cb_cost_variation(df_pl, query_details)
            
            if cb_result["primary_cost"] is not None and cb_result["secondary_cost"] is not None:
                # --- Tabs for Summary and Visual Analysis for C&B ---
                tab1_cb, tab2_cb = st.tabs(["📋 Summary", "📈 Visual Analysis"])

                with tab1_cb:
                    st.write(cb_result["message"]) # Display the formatted text message

                with tab2_cb:
                    # Prepare data for plotting
                    plot_data = pd.DataFrame({
                        'Period': [cb_result["secondary_desc"], cb_result["primary_desc"]],
                        'C&B Cost': [cb_result["secondary_cost"], cb_result["primary_cost"]]
                    })

                    # Create a Plotly bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=plot_data['Period'],
                            y=plot_data['C&B Cost'],
                            marker_color=['lightgray', 'skyblue'], # Customize colors
                            hovertemplate="<b>Period:</b> %{x}<br><b>C&B Cost:</b> %{y:$,.2f}<extra></extra>"
                        )
                    ])
                    fig.update_layout(
                        title=f'C&B Cost Comparison: {cb_result["secondary_desc"]} vs {cb_result["primary_desc"]}',
                        xaxis_title='Period',
                        yaxis_title='C&B Cost (USD)',
                        yaxis_tickprefix="$",
                        yaxis_tickformat=",.0f",
                        height=500 # Set a fixed height for consistency
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(cb_result["message"]) # Display the error message

        elif query_type == "CB_revenue_trend":
            st.markdown("### C&B Cost vs. Total Revenue Monthly Trend Analysis")
            # For CB Revenue Trend, the date filtering is handled internally by calculate_cb_revenue_trend
            # to accommodate the default 'last 12 months' logic.
            cb_trend_df = calculate_cb_revenue_trend(df_pl, query_details)

            if not cb_trend_df.empty:
                # --- Tabs for Data Table and Visual Analysis for C&B Revenue Trend ---
                tab1_cb_trend, tab2_cb_trend = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                with tab1_cb_trend:
                    st.subheader("Monthly C&B Cost, Total Revenue, and Ratios:")
                    # Display the DataFrame in tabular form
                    st.dataframe(cb_trend_df.style.format({
                        "CB_Cost": lambda x: f"${x:,.2f}",
                        "Total_Revenue": lambda x: f"${x:,.2f}",
                        "CB_Revenue_Difference": lambda x: f"${x:,.2f}",
                        "CB_Cost_vs_Revenue_Ratio_Percent": lambda x: f"{x:,.2f}%" if pd.notna(x) else "N/A"
                    }))

                with tab2_cb_trend:
                    st.subheader("Monthly Trend: C&B Cost, Total Revenue, and C&B Cost vs. Revenue Ratio")
                    
                    # Create figure with secondary y-axis for the ratio line
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    # Add C&B Cost Bar Chart
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

                    # Add Total Revenue Bar Chart
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

                    # Add new KPI Line Chart: CB_Cost_vs_Revenue_Ratio_Percent
                    fig.add_trace(
                        go.Scatter(
                            x=cb_trend_df["Month"],
                            y=cb_trend_df["CB_Cost_vs_Revenue_Ratio_Percent"],
                            name="C&B Cost vs Revenue Ratio (%)",
                            mode="lines+markers",
                            yaxis="y2", # Use secondary y-axis
                            line=dict(color='red', width=3),
                            hovertemplate="<b>Month:</b> %{x}<br><b>C&B Cost vs Revenue Ratio:</b> %{y:,.2f}%<extra></extra>"
                        ),
                        secondary_y=True,
                    )

                    # Update layout
                    fig.update_layout(
                        title_text="Monthly Trend: C&B Cost, Total Revenue, and C&B Cost vs. Revenue Ratio",
                        xaxis_title="Month",
                        barmode='group',
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=600 # Set a fixed height for better consistency
                    )

                    # Set y-axes titles and formats
                    fig.update_yaxes(title_text="Amount (USD)", secondary_y=False, tickprefix="$", tickformat=",.0f")
                    
                    # Dynamically set range for the secondary Y-axis (ratio)
                    min_ratio = cb_trend_df['CB_Cost_vs_Revenue_Ratio_Percent'].min()
                    max_ratio = cb_trend_df['CB_Cost_vs_Revenue_Ratio_Percent'].max()
                    # Add some padding to the range, handle case where min/max are the same
                    ratio_padding = (max_ratio - min_ratio) * 0.1 if (max_ratio - min_ratio) != 0 else 10
                    
                    fig.update_yaxes(
                        title_text="C&B Cost vs Revenue Ratio (%)", 
                        secondary_y=True, 
                        tickformat=".2f%", 
                        range=[min_ratio - ratio_padding, max_ratio + ratio_padding]
                    )
                                            
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No data available for the specified period to analyze C&B Cost vs. Total Revenue trend.")

        elif query_type == "HC_trend":
            st.markdown("### Monthly Headcount (HC) Trend by Account")
            
            # Pass the customer_name_filter from query_details to calculate_hc_trend
            hc_trend_df = calculate_hc_trend(df_ut, query_details)
            specific_customer = query_details.get("customer_name_filter")

            if hc_trend_df.empty:
                if specific_customer:
                    st.warning(f"No Headcount data found for account '{specific_customer}' for the specified period.")
                else:
                    st.info("No Headcount data available for the specified period or accounts.")
            else: 
                # Use st.tabs without 'key' or 'default_index' for compatibility with older Streamlit versions.
                # WARNING: This means the tab selection will reset to the first tab on every rerun.
                # The only way to persist tab selection reliably is to upgrade Streamlit.
                tab1_hc_trend, tab2_hc_trend = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                with tab1_hc_trend:
                    st.subheader("Monthly Headcount by Account (Full Data):")
                    # Apply the specific customer filter to the data table as well
                    if specific_customer:
                        # CRITICAL FIX: Use exact match for filtering the data table
                        filtered_data_table_df = hc_trend_df[hc_trend_df['FinalCustomerName'].str.lower() == specific_customer.lower().strip()]
                        st.dataframe(filtered_data_table_df)
                        if filtered_data_table_df.empty:
                            st.info(f"No detailed data for '{specific_customer}' in this period.")
                    else:
                        st.dataframe(hc_trend_df)

                with tab2_hc_trend:
                    if specific_customer:
                        # Display trend for the specific customer
                        st.subheader(f"Monthly Headcount Trend for '{specific_customer}'")
                        # CRITICAL FIX: Use exact match for filtering the data for plot
                        customer_df_for_plot = hc_trend_df[hc_trend_df['FinalCustomerName'].str.lower() == specific_customer.lower().strip()].copy()
                        
                        if not customer_df_for_plot.empty:
                            # Aggregate to ensure only one HC value per month for the specific customer
                            customer_df_for_plot = customer_df_for_plot.groupby('Month')['HC'].sum().reset_index()

                            # --- DEBUGGING LINE REMOVED ---
                            # st.write(f"DEBUG: Data for '{specific_customer}' for plot:")
                            # st.dataframe(customer_df_for_plot)
                            # --- END DEBUGGING LINE REMOVED ---

                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                    x=customer_df_for_plot['Month'],
                                    y=customer_df_for_plot['HC'],
                                    mode='lines+markers',
                                    name=specific_customer,
                                    text=[specific_customer] * len(customer_df_for_plot), # For robust hover
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
                            st.info(f"No headcount data available for '{specific_customer}' in the specified period to visualize.")
                    else:
                        # Display trend for Top 10 Customers and Others
                        st.subheader("Monthly Headcount Trend for Top 10 Customers and Others")
                        
                        num_top_customers = 10 
                        
                        fig = go.Figure()

                        # Calculate TOTAL HC for each customer to determine top 10
                        total_hc_by_customer = hc_trend_df.groupby('FinalCustomerName')['HC'].sum().sort_values(ascending=False)
                        top_customers = total_hc_by_customer.head(num_top_customers).index.tolist()
                        other_customers = total_hc_by_customer.tail(len(total_hc_by_customer) - num_top_customers).index.tolist()

                        # Plot top customers
                        for customer in top_customers:
                            customer_df = hc_trend_df[hc_trend_df['FinalCustomerName'] == customer]
                            # Ensure monthly aggregation for plotting
                            customer_df_for_plot = customer_df.groupby('Month')['HC'].sum().reset_index()

                            fig.add_trace(
                                go.Scatter(
                                    x=customer_df_for_plot['Month'],
                                    y=customer_df_for_plot['HC'],
                                    mode='lines+markers',
                                    name=customer, # Customer name as legend
                                    # This is the robust fix: create a 'text' array for each point
                                    text=[customer] * len(customer_df_for_plot), 
                                    hovertemplate="<b>Month:</b> %{x}<br><b>Customer:</b> %{text}<br><b>HC:</b> %{y}<extra></extra>"
                                )
                            )
                        
                        # Plot 'Others' if there are remaining customers
                        if other_customers:
                            others_df = hc_trend_df[hc_trend_df['FinalCustomerName'].isin(other_customers)]
                            others_monthly_hc = others_df.groupby('Month')['HC'].sum().reset_index()
                            fig.add_trace(
                                go.Scatter(
                                    x=others_monthly_hc['Month'],
                                    y=others_monthly_hc['HC'],
                                    mode='lines+markers',
                                    name='Others',
                                    line=dict(dash='dot', color='gray'), # Dotted gray line for 'Others'
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
        elif query_type == "Revenue_Trend_Analysis":
            st.subheader("📊 Revenue Trend Analysis (YoY, QoQ, MoM)")
            
            # Use a cached function to get the base filtered revenue data
            @st.cache_data(show_spinner="Preparing Revenue Data for Trends...")
            def get_base_revenue_data_cached(df_pl_data, q_details_hashable):
                # Pass a hashable version of query_details to the cached function
                q_details_for_analysis = q_details_hashable.copy()
                # Convert date strings back to datetime objects if needed by analyze_revenue_trend
                for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                    if q_details_for_analysis.get(k) and isinstance(q_details_for_analysis[k], str):
                        q_details_for_analysis[k] = datetime.fromisoformat(q_details_for_analysis[k])

                return analyze_revenue_trend(df_pl_data, q_details_for_analysis)
            
            # Create a hashable version of query_details for caching
            query_details_hashable = query_details.copy()
            for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                if query_details_hashable.get(k) and isinstance(query_details_hashable[k], datetime):
                    query_details_hashable[k] = query_details_hashable[k].isoformat()

            q5_output_dict = get_base_revenue_data_cached(df_pl.copy(), query_details_hashable)

            if isinstance(q5_output_dict, dict) and "Message" in q5_output_dict:
                st.error(q5_output_dict["Message"])
            elif isinstance(q5_output_dict, dict):
                df_filtered_for_charts = q5_output_dict["df_filtered_for_charts"]
                actual_grouping_dimension_from_query = q5_output_dict["grouping_dimension_from_query"]

                # Use the date filter message from parsed_filters if available
                date_filter_msg = "📅 Showing all available data (no specific date filter applied from query)"
                if query_details.get("date_filter") and query_details.get("start_date") and query_details.get("end_date"):
                    if isinstance(query_details['start_date'], datetime) and isinstance(query_details['end_date'], datetime):
                        date_filter_msg = f"📅 Date Filter: {query_details['start_date'].strftime('%Y-%m-%d')} to {query_details['end_date'].strftime('%Y-%m-%d')}"
                    else:
                        st.warning("Parsed dates are not valid datetime objects. Displaying all available data.")
                st.success(date_filter_msg)

                # Mapping for dimension names to actual column names in df_filtered_for_charts
                grouping_col_map = {
                    "DU": "Exec DU", # Corrected from "Exec DU" to "PVDU"
                    "BU": "Exec DG",
                    "Account": "FinalCustomerName",
                    "All": None # Special case for 'All' - means aggregate total revenue
                }
                
                # Determine the column to group by based on the query, default to "All" if not valid
                selected_dim_for_analysis = actual_grouping_dimension_from_query
                selected_dim_col_name = grouping_col_map.get(selected_dim_for_analysis)

                # Validate if the selected dimension column exists and has non-null values
                # If the column doesn't exist OR if all its values are null, fallback to "All"
                if selected_dim_col_name is None or \
                   selected_dim_col_name not in df_filtered_for_charts.columns or \
                   df_filtered_for_charts[selected_dim_col_name].isnull().all():
                    
                    selected_dim_for_analysis = "All"
                    selected_dim_col_name = None # For 'All', we don't group by a specific column
                
                # Inform the user which dimension is being displayed
                if selected_dim_for_analysis == "All":
                    st.info(f"Displaying trends for **Total Revenue** as no specific dimension (DU, BU, Account) was requested or available in the data.")
                else:
                    st.info(f"Displaying trends grouped by **{selected_dim_for_analysis}** as requested in your query.")


                # --- Tabbed Interface for MoM, QoQ, YoY ---
                tab1, tab2, tab3 = st.tabs(["MoM Trend", "QoQ Trend", "YoY Trend"])

                with tab1:
                    st.markdown(f"#### Month-over-Month Revenue Trend by {selected_dim_for_analysis}")

                    # Dynamic aggregation for MoM
                    if selected_dim_col_name: # Group by a specific dimension
                        temp_mom_data = df_filtered_for_charts.groupby([pd.Grouper(key='Date', freq='MS'), selected_dim_col_name]).agg(
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        temp_mom_data.rename(columns={'Date': 'Period'}, inplace=True) # Rename for consistency
                        temp_mom_data['Period'] = temp_mom_data['Period'].dt.strftime('%Y-%m') # Format for display
                        temp_mom_data = temp_mom_data.sort_values('Period')
                        
                        temp_mom_data['Prev_Period_Revenue'] = temp_mom_data.groupby(selected_dim_col_name)['Revenue'].shift(1)
                        temp_mom_data['Growth_Percent'] = np.where(
                            temp_mom_data['Prev_Period_Revenue'] != 0,
                            ((temp_mom_data['Revenue'] - temp_mom_data['Prev_Period_Revenue']) / temp_mom_data['Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        mom_plot_data = temp_mom_data
                        mom_plot_color_col = selected_dim_col_name
                    else: # Handle 'All' (no specific grouping column)
                        overall_mom_data = df_filtered_for_charts.groupby(pd.Grouper(key='Date', freq='MS')).agg(
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        overall_mom_data.rename(columns={'Date': 'Period'}, inplace=True) # Rename for consistency
                        overall_mom_data['Period'] = overall_mom_data['Period'].dt.strftime('%Y-%m') # Format for display
                        overall_mom_data = overall_mom_data.sort_values('Period')
                        
                        overall_mom_data['Prev_Period_Revenue'] = overall_mom_data['Revenue'].shift(1)
                        overall_mom_data['Growth_Percent'] = np.where(
                            overall_mom_data['Prev_Period_Revenue'] != 0,
                            ((overall_mom_data['Revenue'] - overall_mom_data['Prev_Period_Revenue']) / overall_mom_data['Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        overall_mom_data['Display_Name'] = 'Total Revenue' # A dummy column for color
                        mom_plot_data = overall_mom_data
                        mom_plot_color_col = 'Display_Name' # Use this generic column for coloring

                    if not mom_plot_data.empty and mom_plot_data['Period'].nunique() > 1:
                        # Revenue Chart
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
                            (f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                            'Revenue: %{y:$,.2f}<br>' +
                            'MoM Change: %{customdata[1]:.2f}%<extra></extra>'
                        )
                        # Customdata for hover: [Dimension_Name (if not All), Growth_Percent]
                        if selected_dim_col_name:
                            fig_mom_rev.update_traces(customdata=mom_plot_data[[selected_dim_col_name, 'Growth_Percent']].values)
                        else:
                            fig_mom_rev.update_traces(customdata=mom_plot_data[['Display_Name', 'Growth_Percent']].values)

                        fig_mom_rev.update_layout(xaxis_title="Month", yaxis_title="Revenue (USD)", yaxis_tickprefix="$", yaxis_tickformat=",.0f")
                        st.plotly_chart(fig_mom_rev, use_container_width=True)

                        # MoM % Change Chart
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
                                (f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                                'MoM Change: %{y:,.2f}%<extra></extra>'
                            )
                            if selected_dim_col_name:
                                fig_mom_pct.update_traces(customdata=mom_growth_plot_data[[selected_dim_col_name]].values)
                            else:
                                fig_mom_pct.update_traces(customdata=mom_growth_plot_data[['Display_Name']].values)

                            fig_mom_pct.update_layout(xaxis_title="Month", yaxis_title="MoM Change (%)", yaxis_tickformat=".2f%")
                            st.plotly_chart(fig_mom_pct, use_container_width=True)
                        else:
                            st.info(f"Not enough data to show MoM percentage change for {selected_dim_for_analysis}.")

                    else:
                        st.info(f"No sufficient data to calculate MoM trends for {selected_dim_for_analysis}.")

                    with st.expander("Show MoM Detailed Data"):
                        display_cols = ['Period', 'Revenue', 'Growth_Percent']
                        if selected_dim_col_name: # Only add if a specific dimension was selected
                            display_cols.insert(1, selected_dim_col_name)
                        st.dataframe(mom_plot_data[display_cols].style.format({
                            'Revenue': '$ {:,.2f}',
                            'Growth_Percent': '{:.2f}%'
                        }), use_container_width=True)

                with tab2:
                    st.markdown(f"#### Quarter-over-Quarter Revenue Trend by {selected_dim_for_analysis}")

                    selected_qoq_dim_col = grouping_col_map.get(selected_dim_for_analysis)

                    # Dynamic aggregation for QoQ
                    if selected_qoq_dim_col: # Group by a specific dimension
                        # Fiscal Quarter: April-March
                        df_filtered_for_charts['Fiscal_Quarter'] = df_filtered_for_charts['Date'].apply(lambda x: pd.Period(f"{x.year if x.month >= 4 else x.year-1}Q{((x.month-1)//3)+1}", freq='Q-MAR'))
                        temp_qoq_data = df_filtered_for_charts.groupby(['Fiscal_Quarter', selected_qoq_dim_col]).agg(
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        temp_qoq_data.rename(columns={'Fiscal_Quarter': 'Period'}, inplace=True)
                        temp_qoq_data['Period'] = temp_qoq_data['Period'].astype(str) # Format for display
                        temp_qoq_data = temp_qoq_data.sort_values('Period')

                        temp_qoq_data['Prev_Period_Revenue'] = temp_qoq_data.groupby(selected_qoq_dim_col)['Revenue'].shift(1)
                        temp_qoq_data['Growth_Percent'] = np.where(
                            temp_qoq_data['Prev_Period_Revenue'] != 0,
                            ((temp_qoq_data['Revenue'] - temp_qoq_data['Prev_Period_Revenue']) / temp_qoq_data['Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        qoq_plot_data = temp_qoq_data
                        qoq_plot_color_col = selected_qoq_dim_col
                    else: # Handle 'All'
                        df_filtered_for_charts['Fiscal_Quarter'] = df_filtered_for_charts['Date'].apply(lambda x: pd.Period(f"{x.year if x.month >= 4 else x.year-1}Q{((x.month-1)//3)+1}", freq='Q-MAR'))
                        overall_qoq_data = df_filtered_for_charts.groupby('Fiscal_Quarter').agg(
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        overall_qoq_data.rename(columns={'Fiscal_Quarter': 'Period'}, inplace=True)
                        overall_qoq_data['Period'] = overall_qoq_data['Period'].astype(str) # Format for display
                        overall_qoq_data = overall_qoq_data.sort_values('Period')

                        overall_qoq_data['Prev_Period_Revenue'] = overall_qoq_data['Revenue'].shift(1)
                        overall_qoq_data['Growth_Percent'] = np.where(
                            overall_qoq_data['Prev_Period_Revenue'] != 0,
                            ((overall_qoq_data['Revenue'] - overall_qoq_data['Prev_Period_Revenue']) / overall_qoq_data['Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        overall_qoq_data['Display_Name'] = 'Total Revenue'
                        qoq_plot_data = overall_qoq_data
                        qoq_plot_color_col = 'Display_Name'

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
                            (f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                            'Revenue: %{y:$,.2f}<br>' +
                            'QoQ Change: %{customdata[1]:.2f}%<extra></extra>'
                        )
                        if selected_dim_col_name:
                            fig_qoq_rev.update_traces(customdata=qoq_plot_data[[selected_qoq_dim_col, 'Growth_Percent']].values)
                        else:
                            fig_qoq_rev.update_traces(customdata=qoq_plot_data[['Display_Name', 'Growth_Percent']].values)

                        fig_qoq_rev.update_layout(xaxis_title="Quarter", yaxis_title="Revenue (USD)", yaxis_tickprefix="$", yaxis_tickformat=",.0f")
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
                                (f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                                'QoQ Change: %{y:,.2f}%<extra></extra>'
                            )
                            if selected_dim_col_name:
                                fig_qoq_pct.update_traces(customdata=qoq_growth_plot_data[[selected_qoq_dim_col]].values)
                            else:
                                fig_qoq_pct.update_traces(customdata=qoq_growth_plot_data[['Display_Name']].values)

                            fig_qoq_pct.update_layout(xaxis_title="Quarter", yaxis_title="QoQ Change (%)", yaxis_tickformat=".2f%")
                            st.plotly_chart(fig_qoq_pct, use_container_width=True)
                        else:
                            st.info(f"Not enough data to show QoQ percentage change for {selected_dim_for_analysis}.")
                    else:
                        st.info(f"No sufficient data to calculate QoQ trends for {selected_dim_for_analysis}.")

                    with st.expander("Show QoQ Detailed Data"):
                        display_cols = ['Period', 'Revenue', 'Growth_Percent']
                        if selected_qoq_dim_col:
                            display_cols.insert(1, selected_qoq_dim_col)
                        st.dataframe(qoq_plot_data[display_cols].style.format({
                            'Revenue': '$ {:,.2f}',
                            'Growth_Percent': '{:.2f}%'
                        }), use_container_width=True)

                with tab3:
                    st.markdown(f"#### Year-over-Year Revenue Trend by {selected_dim_for_analysis}")

                    selected_yoy_dim_col = grouping_col_map.get(selected_dim_for_analysis)

                    # Dynamic aggregation for YoY
                    if selected_yoy_dim_col: # Group by a specific dimension
                        # Fiscal Year: April-March
                        df_filtered_for_charts['Fiscal_Year'] = df_filtered_for_charts['Date'].apply(lambda x: x.year if x.month >= 4 else x.year - 1)
                        temp_yoy_data = df_filtered_for_charts.groupby(['Fiscal_Year', selected_yoy_dim_col]).agg(
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        temp_yoy_data.rename(columns={'Fiscal_Year': 'Period'}, inplace=True)
                        temp_yoy_data['Period'] = temp_yoy_data['Period'].astype(str) # Format for display
                        temp_yoy_data = temp_yoy_data.sort_values('Period')

                        temp_yoy_data['Prev_Period_Revenue'] = temp_yoy_data.groupby(selected_yoy_dim_col)['Revenue'].shift(1)
                        temp_yoy_data['Growth_Percent'] = np.where(
                            temp_yoy_data['Prev_Period_Revenue'] != 0,
                            ((temp_yoy_data['Revenue'] - temp_yoy_data['Prev_Period_Revenue']) / temp_yoy_data['Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        yoy_plot_data = temp_yoy_data
                        yoy_plot_color_col = selected_yoy_dim_col
                    else: # Handle 'All'
                        df_filtered_for_charts['Fiscal_Year'] = df_filtered_for_charts['Date'].apply(lambda x: x.year if x.month >= 4 else x.year - 1)
                        overall_yoy_data = df_filtered_for_charts.groupby('Fiscal_Year').agg(
                            Revenue=('Amount in USD', 'sum')
                        ).reset_index()
                        overall_yoy_data.rename(columns={'Fiscal_Year': 'Period'}, inplace=True)
                        overall_yoy_data['Period'] = overall_yoy_data['Period'].astype(str) # Format for display
                        overall_yoy_data = overall_yoy_data.sort_values('Period')

                        overall_yoy_data['Prev_Period_Revenue'] = overall_yoy_data['Revenue'].shift(1)
                        overall_yoy_data['Growth_Percent'] = np.where(
                            overall_yoy_data['Prev_Period_Revenue'] != 0,
                            ((overall_yoy_data['Revenue'] - overall_yoy_data['Prev_Period_Revenue']) / overall_yoy_data['Prev_Period_Revenue']) * 100,
                            np.nan
                        )
                        overall_yoy_data['Display_Name'] = 'Total Revenue'
                        yoy_plot_data = overall_yoy_data
                        yoy_plot_color_col = 'Display_Name'

                    if not yoy_plot_data.empty and yoy_plot_data['Period'].nunique() > 1:
                        fig_yoy_rev = px.line(
                            yoy_plot_data,
                            x='Period',
                            y='Revenue',
                            color=yoy_plot_color_col,
                            title=f'YoY Revenue Trend by {selected_dim_for_analysis}',
                            labels={'Revenue': 'Revenue (USD)', 'Period': 'Year'},
                            line_shape='linear'
                        )
                        fig_yoy_rev.update_traces(mode='lines+markers', hovertemplate=
                            '<b>%{x}</b><br>' +
                            (f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                            'Revenue: %{y:$,.2f}<br>' +
                            'YoY Change: %{customdata[1]:.2f}%<extra></extra>'
                        )
                        if selected_dim_col_name:
                            fig_yoy_rev.update_traces(customdata=yoy_plot_data[[selected_yoy_dim_col, 'Growth_Percent']].values)
                        else:
                            fig_yoy_rev.update_traces(customdata=yoy_plot_data[['Display_Name', 'Growth_Percent']].values)

                        fig_yoy_rev.update_layout(xaxis_title="Year", yaxis_title="Revenue (USD)", yaxis_tickprefix="$", yaxis_tickformat=",.0f")
                        st.plotly_chart(fig_yoy_rev, use_container_width=True)

                        yoy_growth_plot_data = yoy_plot_data.dropna(subset=['Growth_Percent'])
                        if not yoy_growth_plot_data.empty:
                            fig_yoy_pct = px.line(
                                yoy_growth_plot_data,
                                x='Period',
                                y='Growth_Percent',
                                color=yoy_plot_color_col,
                                title=f'YoY Revenue Percentage Change by {selected_dim_for_analysis}',
                                labels={'Growth_Percent': 'YoY Change (%)', 'Period': 'Year'},
                                line_shape='linear'
                            )
                            fig_yoy_pct.update_traces(mode='lines+markers', hovertemplate=
                                '<b>%{x}</b><br>' +
                                (f'{selected_dim_for_analysis}: %{{customdata[0]}}<br>' if selected_dim_col_name else '') +
                                'YoY Change: %{y:,.2f}%<extra></extra>'
                            )
                            if selected_dim_col_name:
                                fig_yoy_pct.update_traces(customdata=yoy_growth_plot_data[[selected_yoy_dim_col]].values)
                            else:
                                fig_yoy_pct.update_traces(customdata=yoy_growth_plot_data[['Display_Name']].values)

                            fig_yoy_pct.update_layout(xaxis_title="Year", yaxis_title="YoY Change (%)", yaxis_tickformat=".2f%")
                            st.plotly_chart(fig_yoy_pct, use_container_width=True)
                        else:
                            st.info(f"Not enough data to show YoY percentage change for {selected_dim_for_analysis}.")
                    else:
                        st.info(f"No sufficient data to calculate YoY trends for {selected_dim_for_analysis}.")

                    with st.expander("Show YoY Detailed Data"):
                        display_cols = ['Period', 'Revenue', 'Growth_Percent']
                        if selected_yoy_dim_col:
                            display_cols.insert(1, selected_yoy_dim_col)
                        st.dataframe(yoy_plot_data[display_cols].style.format({
                            'Revenue': '$ {:,.2f}',
                            'Growth_Percent': '{:.2f}%'
                        }), use_container_width=True)
            else:
                st.info("No data available to calculate revenue trends.")

        elif query_type == "UT_trend":
            st.markdown("### 📈 Utilization (UT) Trend Analysis")

            # Use a cached function for UT trend analysis
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

                if not ut_trend_df.empty:
                    st.info(f"Displaying **{trend_granularity_display}** UT% trend by **{trend_dimension_display}** for the period: **{query_details.get('description', 'specified date range')}**")

                    # Determine the column to use for coloring/grouping in the plot
                    plot_color_col = 'Dimension_Name' if trend_dimension_display == "All" else ut_trend_df.columns[1] # Assumes dimension column is the second one if not 'All'

                    # UT% Trend Chart
                    fig_ut = px.line(
                        ut_trend_df,
                        x='Period_Formatted',
                        y='UT_Percent',
                        color=plot_color_col,
                        title=f'UT% Trend by {trend_dimension_display} ({trend_granularity_display})',
                        labels={'UT_Percent': 'UT %', 'Period_Formatted': trend_granularity_display.capitalize()},
                        line_shape='linear'
                    )
                    fig_ut.update_traces(mode='lines+markers', hovertemplate=
                        '<b>%{x}</b><br>' +
                        (f'{trend_dimension_display}: %{{customdata[0]}}<br>' if trend_dimension_display != "All" else '') +
                        'UT%: %{y:,.2f}%<br>' +
                        'Total Billable Hours: %{customdata[1]:,.0f}<br>' +
                        'Net Available Hours: %{customdata[2]:,.0f}<extra></extra>'
                    )
                    # Customdata for hover: [Dimension_Name (if not All), TotalBillableHours, NetAvailableHours]
                    if trend_dimension_display != "All":
                        fig_ut.update_traces(customdata=ut_trend_df[[plot_color_col, 'TotalBillableHours', 'NetAvailableHours']].values)
                    else:
                        fig_ut.update_traces(customdata=ut_trend_df[['Display_Name', 'TotalBillableHours', 'NetAvailableHours']].values)

                    fig_ut.update_layout(
                        xaxis_title=trend_granularity_display.capitalize(),
                        yaxis_title="UT %",
                        yaxis_tickformat=".2f%",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=600
                    )
                    st.plotly_chart(fig_ut, use_container_width=True)

                    with st.expander("Show UT Trend Detailed Data"):
                        display_cols = ['Period_Formatted', 'TotalBillableHours', 'NetAvailableHours', 'UT_Percent']
                        if trend_dimension_display != "All":
                            display_cols.insert(1, plot_color_col) # Insert dimension column for display
                        st.dataframe(ut_trend_df[display_cols].style.format({
                            'TotalBillableHours': '{:,.0f}',
                            'NetAvailableHours': '{:,.0f}',
                            'UT_Percent': '{:.2f}%'
                        }), use_container_width=True)
                else:
                    st.info(f"No sufficient data to calculate UT trends for {trend_dimension_display} for the specified period.")
            else:
                st.info("No data available to calculate UT trends.")

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
                    st.info(f"Displaying **{trend_granularity_display}** Fresher UT% trend by **{trend_dimension_display}** for the period: **{query_details.get('description', 'last 12 months (default)')}**")

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
                        
                        # CORRECTED: Use 'Exec DU' directly as the color column
                        fig_fresher_ut = px.line(
                            fresher_ut_trend_df,
                            x='Period_Formatted',
                            y='UT_Percent',
                            color='Exec DU', # Group by Exec DU for different lines
                            title='Monthly Fresher UT% Trend by Delivery Unit',
                            labels={'UT_Percent': 'UT %', 'Period_Formatted': 'Month', 'Exec DU': 'Delivery Unit'}, # Corrected label
                            line_shape='linear'
                        )
                        fig_fresher_ut.update_traces(mode='lines+markers', hovertemplate=
                            '<b>Month:</b> %{x}<br>' +
                            '<b>Delivery Unit:</b> %{customdata[0]}<br>' +
                            '<b>Fresher UT%:</b> %{y:,.2f}%<br>' +
                            'Total Billable Hours: %{customdata[1]:,.0f}<br>' +
                            'Net Available Hours: %{customdata[2]:,.0f}<extra></extra>'
                        )
                        # CORRECTED: Pass 'Exec DU' in customdata
                        fig_fresher_ut.update_traces(customdata=fresher_ut_trend_df[['Exec DU', 'TotalBillableHours', 'NetAvailableHours']].values)

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
                    st.info(f"No sufficient data to calculate Fresher UT trends for {trend_dimension_display} for the specified period.")
            else:
                st.info("No data available to calculate Fresher UT trends.")
        elif query_type == "Revenue_Per_Person_Trend":
            st.markdown("### 📈 Monthly Revenue Per Person Trend by Account")

            @st.cache_data(show_spinner="Analyzing Revenue Per Person Trend...")
            def get_revenue_per_person_trend_data_cached(df_merged_data, q_details_hashable): # Only df_merged_data now
                q_details_for_analysis = q_details_hashable.copy()
                for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                    if q_details_for_analysis.get(k) and isinstance(q_details_for_analysis[k], str):
                        q_details_for_analysis[k] = datetime.fromisoformat(q_details_for_analysis[k])
                return analyze_revenue_per_person_trend(df_merged_data, q_details_for_analysis) # Pass df_merged_data

            query_details_hashable = query_details.copy()
            for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                if query_details_hashable.get(k) and isinstance(query_details_hashable[k], datetime):
                    query_details_hashable[k] = query_details_hashable[k].isoformat()

            # Pass df_merged to the cached function
            rev_per_person_output = get_revenue_per_person_trend_data_cached(df_merged.copy(), query_details_hashable)

            if isinstance(rev_per_person_output, dict) and "Message" in rev_per_person_output:
                st.error(f"Error: {rev_per_person_output['Message']}")
            elif isinstance(rev_per_person_output, dict):
                rev_per_person_trend_df = rev_per_person_output["df_revenue_per_person_trend"]
                trend_dimension_display = rev_per_person_output["trend_dimension_display"]
                trend_granularity_display = rev_per_person_output["trend_granularity_display"]

                if not rev_per_person_trend_df.empty:
                    st.info(f"Displaying **{trend_granularity_display}** Revenue Per Person trend by **{trend_dimension_display}** for the period: **{query_details.get('description', 'last 12 months (default)')}**")

                    tab1_rev_per_person, tab2_rev_per_person = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                    with tab1_rev_per_person:
                        st.subheader("Monthly Revenue Per Person by Account:")
                        st.dataframe(rev_per_person_trend_df.style.format({
                            'TotalRevenue': '$ {:,.2f}',
                            'Headcount': '{:,.0f}',
                            'Revenue_Per_Person': '$ {:,.2f}'
                        }), use_container_width=True)

                    with tab2_rev_per_person:
                        st.subheader("Monthly Revenue Per Person Trend by Account")
                        
                        fig_rev_per_person = px.line(
                            rev_per_person_trend_df,
                            x='Month_Formatted',
                            y='Revenue_Per_Person',
                            color='FinalCustomerName', # Group by FinalCustomerName for different lines
                            title='Monthly Revenue Per Person Trend by Account',
                            labels={'Revenue_Per_Person': 'Revenue Per Person (USD)', 'Month_Formatted': 'Month', 'FinalCustomerName': 'Account'},
                            line_shape='linear'
                        )
                        fig_rev_per_person.update_traces(mode='lines+markers', hovertemplate=
                            '<b>Month:</b> %{x}<br>' +
                            '<b>Account:</b> %{customdata[0]}<br>' +
                            '<b>Revenue Per Person:</b> %{y:$,.2f}<br>' +
                            'Total Revenue: %{customdata[1]:$,.2f}<br>' +
                            'Headcount: %{customdata[2]:,.0f}<extra></extra>'
                        )
                        fig_rev_per_person.update_traces(customdata=rev_per_person_trend_df[['FinalCustomerName', 'TotalRevenue', 'Headcount']].values)

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
                    st.info(f"No sufficient data to calculate Revenue Per Person trends for {trend_dimension_display} for the specified period.")
            else:
                st.info("No data available to calculate Revenue Per Person trends.")

        elif query_type == "Revenue_Per_Person_Trend":
            st.markdown("### 📈 Monthly Revenue Per Person Trend by Account")

            @st.cache_data(show_spinner="Analyzing Revenue Per Person Trend...")
            def get_revenue_per_person_trend_data_cached(df_merged_data, q_details_hashable): # Only df_merged_data now
                q_details_for_analysis = q_details_hashable.copy()
                for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                    if q_details_for_analysis.get(k) and isinstance(q_details_for_analysis[k], str):
                        q_details_for_analysis[k] = datetime.fromisoformat(q_details_for_analysis[k])
                return analyze_revenue_per_person_trend(df_merged_data, q_details_for_analysis) # Pass df_merged_data

            query_details_hashable = query_details.copy()
            for k in ['start_date', 'end_date', 'secondary_start_date', 'secondary_end_date']:
                if query_details_hashable.get(k) and isinstance(query_details_hashable[k], datetime):
                    query_details_hashable[k] = query_details_hashable[k].isoformat()

            # Pass df_merged to the cached function
            rev_per_person_output = get_revenue_per_person_trend_data_cached(df_merged.copy(), query_details_hashable)

            if isinstance(rev_per_person_output, dict) and "Message" in rev_per_person_output:
                st.error(f"Error: {rev_per_person_output['Message']}")
            elif isinstance(rev_per_person_output, dict):
                rev_per_person_trend_df = rev_per_person_output["df_revenue_per_person_trend"]
                trend_dimension_display = rev_per_person_output["trend_dimension_display"]
                trend_granularity_display = rev_per_person_output["trend_granularity_display"]

                if not rev_per_person_trend_df.empty:
                    st.info(f"Displaying **{trend_granularity_display}** Revenue Per Person trend by **{trend_dimension_display}** for the period: **{query_details.get('description', 'last 12 months (default)')}**")

                    tab1_rev_per_person, tab2_rev_per_person = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                    with tab1_rev_per_person:
                        st.subheader("Monthly Revenue Per Person by Account:")
                        st.dataframe(rev_per_person_trend_df.style.format({
                            'TotalRevenue': '$ {:,.2f}',
                            'Headcount': '{:,.0f}',
                            'Revenue_Per_Person': '$ {:,.2f}'
                        }), use_container_width=True)

                    with tab2_rev_per_person:
                        st.subheader("Monthly Revenue Per Person Trend by Account")
                        
                        fig_rev_per_person = px.line(
                            rev_per_person_trend_df,
                            x='Month_Formatted',
                            y='Revenue_Per_Person',
                            color='FinalCustomerName', # Group by FinalCustomerName for different lines
                            title='Monthly Revenue Per Person Trend by Account',
                            labels={'Revenue_Per_Person': 'Revenue Per Person (USD)', 'Month_Formatted': 'Month', 'FinalCustomerName': 'Account'},
                            line_shape='linear'
                        )
                        fig_rev_per_person.update_traces(mode='lines+markers', hovertemplate=
                            '<b>Month:</b> %{x}<br>' +
                            '<b>Account:</b> %{customdata[0]}<br>' +
                            '<b>Revenue Per Person:</b> %{y:$,.2f}<br>' +
                            'Total Revenue: %{customdata[1]:$,.2f}<br>' +
                            'Headcount: %{customdata[2]:,.0f}<extra></extra>'
                        )
                        fig_rev_per_person.update_traces(customdata=rev_per_person_trend_df[['FinalCustomerName', 'TotalRevenue', 'Headcount']].values)

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
                    st.info(f"No sufficient data to calculate Revenue Per Person trends for {trend_dimension_display} for the specified period.")
            else:
                st.info("No data available to calculate Revenue Per Person trends.")

        # --- NEW: Realized Rate Drop Analysis Block ---
        elif query_type == "Realized_Rate_Drop":
            st.markdown("### 📉 Accounts with Significant Realized Rate Drop")

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

            realized_rate_output = get_realized_rate_drop_data_cached(df_merged.copy(), query_details_hashable)

            if isinstance(realized_rate_output, dict) and "Message" in realized_rate_output:
                st.error(f"Error: {realized_rate_output['Message']}")
            elif isinstance(realized_rate_output, dict):
                realized_rate_drop_df = realized_rate_output["df_realized_rate_drop"]
                current_q_name = realized_rate_output["current_quarter_name"]
                prev_q_name = realized_rate_output["previous_quarter_name"]
                drop_threshold = realized_rate_output["drop_threshold"]

                if not realized_rate_drop_df.empty:
                    st.info(f"Showing accounts where Realized Rate dropped by more than **${drop_threshold:,.2f}** from **{prev_q_name}** to **{current_q_name}**.")

                    tab1_rate_drop, tab2_rate_drop = st.tabs(["📋 Data Table", "📈 Visual Analysis"])

                    with tab1_rate_drop:
                        st.subheader("Accounts with Significant Realized Rate Drop:")
                        st.dataframe(realized_rate_drop_df.style.format({
                            f'Realized Rate ({prev_q_name})': '$ {:,.2f}',
                            f'Realized Rate ({current_q_name})': '$ {:,.2f}',
                            'Rate Drop (USD)': '$ {:,.2f}'
                        }), use_container_width=True)

                    with tab2_rate_drop:
                        st.subheader(f"Realized Rate Drop from {prev_q_name} to {current_q_name}")
                        
                        # Create a bar chart for the Rate Drop
                        fig_rate_drop = px.bar(
                            realized_rate_drop_df,
                            x='FinalCustomerName',
                            y='Rate Drop (USD)',
                            title=f'Realized Rate Drop (> ${drop_threshold:,.2f}) by Account',
                            labels={'Rate Drop (USD)': 'Rate Drop (USD)', 'FinalCustomerName': 'Account'},
                            color='Rate Drop (USD)', # Color by the magnitude of the drop
                            color_continuous_scale=px.colors.sequential.Reds # Use a red scale for drops
                        )
                        fig_rate_drop.update_traces(hovertemplate=
                            '<b>Account:</b> %{x}<br>' +
                            f'Realized Rate ({prev_q_name}): %{{customdata[0]:$,.2f}}<br>' +
                            f'Realized Rate ({current_q_name}): %{{customdata[1]:$,.2f}}<br>' +
                            'Rate Drop: %{y:$,.2f}<extra></extra>'
                        )
                        fig_rate_drop.update_traces(customdata=realized_rate_drop_df[[f'Realized Rate ({prev_q_name})', f'Realized Rate ({current_q_name})']].values)

                        fig_rate_drop.update_layout(
                            xaxis_title="Account",
                            yaxis_title="Rate Drop (USD)",
                            yaxis_tickprefix="$",
                            yaxis_tickformat=",.2f",
                            hovermode="x unified",
                            height=600
                        )
                        st.plotly_chart(fig_rate_drop, use_container_width=True)
                else:
                    st.info(f"No accounts found with a realized rate drop of more than ${drop_threshold:,.2f} from {prev_q_name} to {current_q_name}.")
            else:
                st.info("No data available to analyze realized rate drops.")

        else:
            st.warning("Sorry, I can only assist with Contribution Margin, Transportation Cost, C&B Cost Variation, C&B Revenue Trend, Headcount Trend, Revenue Trend Analysis, Fresher UT Trend, Revenue Per Person Trend, and Realized Rate Drop queries at the moment.")

