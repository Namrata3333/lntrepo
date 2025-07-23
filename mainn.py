# mainn.py (UPDATED for CM% line formatting)

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
from data_loaderr import get_pl_data
from kpi_calculationss import calculate_cm, get_query_details, analyze_transportation_cost_trend, calculate_cb_cost_variation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="L&T KPI Assistant")

st.title("📊 L&T KPI Assistant")
st.markdown("Ask a KPI-related question (e.g., 'show cm% <30% in FY26-Q1', 'Which cost triggered the Margin drop last month as compared to its previous month in Transportation', **'How much C&B varied from last quarter to this quarter'**)")

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
    # Ensure 'Date' column is datetime type for filtering
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df_pl = load_data_for_streamlit()

st.write("---")

# --- User Input Section ---
user_query = st.text_input(
    "Enter your query:",
    value="show cm% <30% in FY26-Q1" # Default query for testing CM analysis
)

if st.button("Submit"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        query_details = get_query_details(user_query)
        query_type = query_details.get("query_type")
        
        # Apply date filters to the DataFrame first, for the primary period
        filtered_df_primary_period = df_pl.copy()
        if query_details.get("date_filter") and query_details.get("start_date") and query_details.get("end_date"):
            filtered_df_primary_period = filtered_df_primary_period[
                (filtered_df_primary_period['Date'] >= query_details["start_date"]) &
                (filtered_df_primary_period['Date'] <= query_details["end_date"])
            ]
        
        # --- Dispatch based on query_type ---
        if query_type == "CM_analysis":
            st.markdown("### Contribution Margin Analysis")
            
            if filtered_df_primary_period.empty:
                st.warning(f"No data available for the primary period ({query_details.get('description', 'specified date range')}) for CM analysis. Please check the data for this range.")
            else:
                # Calculate the CM table first, as this is the core filtered data
                result_df = calculate_cm(filtered_df_primary_period, query_details) 
                
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

                    st.markdown("#### Key Metrics (for Customers meeting CM Filter):")
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
                        result_df['CM_Value_Numeric'] = result_df['CM_Value'].replace('N/A', pd.NA).astype(float)
                        
                        # Sort for better visualization (e.g., by CM_Value_Numeric)
                        sorted_df = result_df.sort_values(by="CM_Value_Numeric", ascending=True)

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
                                # NEW: Use CM (%) for hover display
                                hovertemplate="<b>Customer:</b> %{x}<br><b>CM %:</b> %{customdata}<extra></extra>",
                                customdata=sorted_df["CM (%)"] # Pass the formatted CM (%) string here
                            ),
                            secondary_y=True,
                        )

                        # Update layout for combined chart
                        fig.update_layout(
                            title_text="Customer Revenue, Cost, and Contribution Margin %",
                            xaxis_title="Customer Name",
                            barmode='group',
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )

                        # Set y-axes titles and formats
                        fig.update_yaxes(title_text="Revenue/Cost (USD)", secondary_y=False, tickprefix="$", tickformat=",.0f")
                        fig.update_yaxes(title_text="CM %",secondary_y=True,tickformat=".2f%",range=[-1000, 100])
                                          
    
   

                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No customers found matching the specified CM filter for the selected period.")

        elif query_type == "Transportation_cost_analysis":
            st.markdown("### Transportation Cost Trend Analysis")
            result = analyze_transportation_cost_trend(df_pl, query_details)
            if isinstance(result, pd.DataFrame):
                st.subheader("Cost Changes in Transportation Segment:")
                st.dataframe(result)
            else:
                st.warning(result)

        elif query_type == "C&B_cost_variation":
            st.markdown("### C&B Cost Variation Analysis")
            result = calculate_cb_cost_variation(df_pl, query_details)
            st.write(result)

        elif query_type == "unsupported":
            st.warning("I'm sorry, I can only provide Contribution Margin analysis, Transportation cost trends, and C&B cost variation at the moment. Please rephrase your query.")