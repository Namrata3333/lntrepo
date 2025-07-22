
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
from data_loaderr import get_pl_data
# Corrected import for kpi_calculations:
from kpi_calculationss import calculate_cm, get_query_details, analyze_transportation_cost_trend

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="L&T KPI Assistant")

st.title("📊 L&T KPI Assistant")
st.markdown("Ask a KPI-related question (e.g., 'show cm% <30% in FY26-Q1', 'Which cost triggered the Margin drop last month as compared to its previous month in Transportation')")

@st.cache_data
def load_data_for_streamlit():
    """Load P&L data using the get_pl_data function from data_loader."""
    with st.spinner("Loading P&L data from Azure Blob..."):
        df = get_pl_data()
    if df.empty:
        st.error("Failed to load P&L data. Please check data source and credentials.")
        st.stop() # Stop the app if data can't be loaded
    return df

df_pl = load_data_for_streamlit()

st.write("---")

# --- User Input Section ---
user_query = st.text_input(
    "Enter your query:",
    value="Which cost triggered the Margin drop last month as compared to its previous month in Transportation" # Example default query for testing
)

if st.button("Submit"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        st.subheader("Analysis Results:")
        # Get query details including query_type and all date info
        query_details = get_query_details(user_query) # Use the updated get_query_details
        
        query_type = query_details.get("query_type")
        
        # Display extracted date information for transparency (optional, but good for debugging)
        if query_details.get("date_filter"):
            st.info(f"Primary Period: {query_details.get('description')}")
            if query_details.get('secondary_start_date'):
                # Format to show only month/year for brevity in Streamlit info
                st.info(f"Comparison Period: {query_details.get('secondary_start_date').strftime('%b %Y')} to {query_details.get('secondary_end_date').strftime('%b %Y')}")
        else:
            st.info("📅 No specific date filter extracted or applied from query.")


        # --- Dispatch based on query_type ---
        if query_type == "CM_analysis":
            st.markdown("### Contribution Margin Analysis")
            
            # CM specific filtering description
            cm_filter_type = query_details.get("cm_filter_type")
            cm_lower_bound = query_details.get("cm_lower_bound")
            cm_upper_bound = query_details.get("cm_upper_bound")

            if cm_filter_type:
                filter_desc = f"Applying CM filter: {cm_filter_type}"
                if cm_lower_bound is not None:
                    filter_desc += f" {cm_lower_bound*100:.2f}%"
                if cm_upper_bound is not None:
                    filter_desc += f" to {cm_upper_bound*100:.2f}%"
                st.info(filter_desc)
            else:
                st.info("No specific CM% filter applied. Showing all customer CMs for the period.")

            # CM calculations still rely on primary date filter applied globally
            start_date = query_details.get("start_date")
            end_date = query_details.get("end_date")
            date_filter_applied = query_details.get("date_filter", False)
            
            filtered_df_for_cm = df_pl.copy()
            if date_filter_applied and start_date and end_date:
                filtered_df_for_cm = filtered_df_for_cm[
                    (filtered_df_for_cm['Date'] >= pd.to_datetime(start_date)) &
                    (filtered_df_for_cm['Date'] <= pd.to_datetime(end_date))
                ]

            if filtered_df_for_cm.empty:
                st.warning("No data available for the specified date range for CM analysis.")
            else:
                result_df = calculate_cm(filtered_df_for_cm, query_details) # Pass full query_details
                if not result_df.empty:
                    st.subheader("Customer-wise Contribution Margin:")
                    st.dataframe(result_df)
                    
                    total_revenue = result_df["Revenue"].sum()
                    total_cost = result_df["Cost"].sum()
                    overall_cm_ratio = (total_revenue - total_cost) / total_revenue if total_revenue != 0 else float('nan')
                    overall_cm_percentage = overall_cm_ratio * 100 if not pd.isna(overall_cm_ratio) else "N/A"

                    st.subheader("Overall Contribution Margin Summary:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Revenue", f"${total_revenue:,.2f}")
                    with col2:
                        st.metric("Total Cost", f"${total_cost:,.2f}")
                    with col3:
                        st.metric("Overall CM", f"{overall_cm_percentage:.2f}%" if not pd.isna(overall_cm_percentage) else "N/A")
                else:
                    st.info("No customers found matching the specified CM filter and date range.")

        elif query_type == "Transportation_cost_analysis":
            st.markdown("### Transportation Cost Trend Analysis")
            # This function uses date_info and segment_filter from query_details internally
            cost_analysis_result = analyze_transportation_cost_trend(df_pl, query_details) 

            if isinstance(cost_analysis_result, pd.DataFrame) and not cost_analysis_result.empty:
                st.write(f"Costs that increased in '{query_details.get('segment_filter', 'Transportation')}' from {query_details.get('secondary_start_date').strftime('%b %Y')} to {query_details.get('start_date').strftime('%b %Y')}:")
                st.dataframe(cost_analysis_result)
            else:
                st.info(cost_analysis_result)

        else:
            st.warning("I cannot understand this query type. Please try a different query.")

st.write("---")
st.caption("Powered by Azure OpenAI and Streamlit")