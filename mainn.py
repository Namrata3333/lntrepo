# mainn.py (Corrected Date Type Handling and Typo Fix)

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
# Import both get_pl_data and get_ut_data
from data_loaderr import get_pl_data, get_ut_data 
# Import all necessary functions from kpi_calculationss
from kpi_calculationss import calculate_cm, get_query_details, analyze_transportation_cost_trend, calculate_cb_cost_variation, calculate_cb_revenue_trend, calculate_hc_trend
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # Import numpy for dynamic range calculation
import sys
import os

# Add the directory containing kpi_calculationss.py to the Python path
# Assuming kpi_calculationss.py is in the same directory as mainn.py
sys.path.append(os.path.dirname(__file__))


# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="L&T KPI Assistant")

st.title("📊 L&T KPI Assistant")
st.markdown("Ask a KPI-related question (e.g., 'show cm% <30% in FY26-Q1', 'Which cost triggered the Margin drop last month as compared to its previous month in Transportation', **'How much C&B varied from last quarter to this quarter'**,**'What is M-o-M trend of C&B cost % w.r.t total revenue'**, **'What is M-o-M HC for an account'**)")

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

df_pl = load_data_for_streamlit()
df_ut = load_ut_data_for_streamlit()

st.write("---")

# --- User Input Section ---
user_query = st.text_input(
    "Enter your query:",
    value="What is M-o-M HC for an account" # Default query for testing new functionality
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
        else:
            st.warning("Sorry, I can only assist with Contribution Margin, Transportation Cost, C&B Cost Variation, C&B Revenue Trend, and Headcount Trend queries at the moment.")

