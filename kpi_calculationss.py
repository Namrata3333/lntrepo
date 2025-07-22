# kpi_calculations.py (FULLY REVISED - Paste this entire content)

import pandas as pd
from openai import AzureOpenAI
from datetime import datetime, timedelta
import pytz
import json
from dateutil.parser import parse
import calendar
import os



# ---------- CONFIGURATION ----------
AZURE_OPENAI_ENDPOINT = "https://azure-md46msq5-swedencentral.openai.azure.com/"
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEYY") 
AZURE_OPENAI_DEPLOYMENT = "gpt-35-turbo"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

# Initialize OpenAI client globally
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Define these globally as they are constant
REVENUE_GROUPS = ["ONSITE", "OFFSHORE", "INDIRECT REVENUE"]
COST_GROUPS = [
    "Direct Expense", "OWN OVERHEADS", "Indirect Expense",
    "Project Level Depreciation", "Direct Expense - DU Block Seats Allocation",
    "Direct Expense - DU Pool Allocation", "Establishment Expenses"
]

def parse_percentage(val):
    """Converts percentage strings/numbers to decimal floats."""
    try:
        if isinstance(val, str):
            val = val.replace("%", "").strip()
            if not val:
                return None
            val = float(val)
        elif not isinstance(val, (int, float)):
            return None
        
        if val > 1 and val <= 100:
            val = val / 100
        
        return float(val)
    except ValueError:
        return None

# ---------- DATE RANGE PARSING ----------
def parse_date_range_from_query(query):
    """
    Uses GPT to extract precise date ranges. Now includes logic to identify
    two consecutive months for trend analysis.
    """
    india_tz = pytz.timezone("Asia/Kolkata")
    today = datetime.now(india_tz).date() # Keep as .date() for prompt instruction
    
    system_prompt = f"""Today's date is {today.strftime('%Y-%m-%d')}.
You are an expert at extracting precise date ranges from natural language queries, specifically for L&T's fiscal year (April 1st to March 31st).
When the query asks for a comparison between 'last month' and 'previous month' (or similar two consecutive months like 'May and June'), extract both month-specific date ranges.

Return a JSON object with:
- 'date_filter': boolean indicating if a date filter was requested
- 'start_date': YYYY-MM-DD format (first day of the *primary* period, e.g., the "last month" in "last month vs previous month")
- 'end_date': YYYY-MM-DD format (last day of the *primary* period)
- 'secondary_start_date': YYYY-MM-DD format (first day of the *comparison* period, e.g., the "previous month") or null
- 'secondary_end_date': YYYY-MM-DD format (last day of the *comparison* period) or null
- 'description': natural language description of the period(s)
- 'relative_period_detected': "last quarter", "last year", "this year", "last month", "this month", "current date", "last_and_previous_month", or null

Rules for Fiscal Year (FY) interpretation:
1. An "FY" followed by a 2-digit year (e.g., "FY26") refers to the fiscal year starting April 1st of the calendar year '25' and ending March 31st of calendar year '26'. For example, FY26 runs from 2025-04-01 to 2026-03-31.
2. An "FY" followed by a 4-digit year (e.g., "FY2026") should be interpreted similarly: the fiscal year starts April 1st of the calendar year 2025 and ends March 31st of 2026. Treat "FY2026" and "FY26" as equivalent for this purpose.
3. Quarters are fiscal quarters:
    - Q1: April 1 - June 30
    - Q2: July 1 - September 30
    - Q3: October 1 - December 31
    - Q4: January 1 - March 31 (of the next calendar year for the same fiscal year)
4. For relative periods (like "last month"), calculate exact dates based on today's date.
5. For absolute periods (like "January 2023"), use exact dates.
6. For ranges (like "from March to May 2023"), use exact start/end dates.
7. If the query asks for a comparison between two specific consecutive months (e.g., "May and June", "last month and previous month"), identify `start_date`/`end_date` as the later month and `secondary_start_date`/`secondary_end_date` as the earlier month. Set `relative_period_detected` to "last_and_previous_month" if detected by relative terms.
8. If no date filter, set date_filter=false and return null for dates.
"""
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Convert date strings to datetime objects (NOT date objects)
        for key in ["start_date", "end_date", "secondary_start_date", "secondary_end_date"]:
            if result.get(key):
                try:
                    # Convert to datetime object (not .date())
                    result[key] = datetime.combine(parse(result[key]).date(), datetime.min.time()) 
                except:
                    result[key] = None
        
        # --- Custom L&T Fiscal Year Logic for Relative Periods and explicit FY ---
        # Ensure these calculated dates are also datetime objects
        relative_period = result.get('relative_period_detected')
        if relative_period:
            if relative_period == 'last quarter':
                current_month = today.month
                current_year = today.year

                if 4 <= current_month <= 6: # Current is Q1 (Apr-Jun)
                    prev_q_start_month = 1
                    prev_q_end_month = 3
                    prev_q_cal_year = current_year
                    fiscal_year_desc = f"FY{current_year}"
                    quarter_desc = "Q4"
                elif 7 <= current_month <= 9: # Current is Q2 (Jul-Sep)
                    prev_q_start_month = 4
                    prev_q_end_month = 6
                    prev_q_cal_year = current_year
                    fiscal_year_desc = f"FY{current_year + 1}"
                    quarter_desc = "Q1"
                elif 10 <= current_month <= 12: # Current is Q3 (Oct-Dec)
                    prev_q_start_month = 7
                    prev_q_end_month = 9
                    prev_q_cal_year = current_year
                    fiscal_year_desc = f"FY{current_year + 1}"
                    quarter_desc = "Q2"
                else: # Current is Q4 (Jan-Mar)
                    prev_q_start_month = 10
                    prev_q_end_month = 12
                    prev_q_cal_year = current_year - 1
                    fiscal_year_desc = f"FY{current_year}"
                    quarter_desc = "Q3"

                result["start_date"] = datetime(prev_q_cal_year, prev_q_start_month, 1)
                result["end_date"] = datetime(prev_q_cal_year, prev_q_end_month, calendar.monthrange(prev_q_cal_year, prev_q_end_month)[1], 23, 59, 59)
                result["date_filter"] = True
                result["description"] = f"last quarter ({fiscal_year_desc} {quarter_desc})"

            elif relative_period == 'last year':
                current_fiscal_year_start_cal_year = today.year if today.month >= 4 else today.year - 1
                result["start_date"] = datetime(current_fiscal_year_start_cal_year - 1, 4, 1)
                result["end_date"] = datetime(current_fiscal_year_start_cal_year, 3, 31, 23, 59, 59)
                result["date_filter"] = True
                result["description"] = f"last fiscal year (FY{current_fiscal_year_start_cal_year})"
            
            elif relative_period == 'this year':
                current_fiscal_year_start_cal_year = today.year if today.month >= 4 else today.year - 1
                result["start_date"] = datetime(current_fiscal_year_start_cal_year, 4, 1)
                result["end_date"] = datetime.combine(today, datetime.max.time())
                result["date_filter"] = True
                result["description"] = f"current fiscal year (FY{current_fiscal_year_start_cal_year + 1})"

            elif relative_period == 'last month':
                last_month_end = datetime.combine(today.replace(day=1) - timedelta(days=1), datetime.max.time())
                last_month_start = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                result["start_date"] = last_month_start
                result["end_date"] = last_month_end
                result["date_filter"] = True
                result["description"] = f"last month ({last_month_start.strftime('%b %Y')})"
                
            elif relative_period == 'this month':
                result["start_date"] = datetime.combine(today.replace(day=1), datetime.min.time())
                result["end_date"] = datetime.combine(today, datetime.max.time())
                result["date_filter"] = True
                result["description"] = f"this month ({today.strftime('%b %Y')})"
            
            elif relative_period == 'current date':
                result["start_date"] = datetime.combine(today, datetime.min.time())
                result["end_date"] = datetime.combine(today, datetime.max.time())
                result["date_filter"] = True
                result["description"] = f"current date ({today.strftime('%b %d, %Y')})"

            elif relative_period == 'last_and_previous_month':
                # Calculate the exact dates for last month and previous month from today
                current_month_start_date_obj = today.replace(day=1)
                last_month_end_date_obj = current_month_start_date_obj - timedelta(days=1)
                last_month_start_date_obj = last_month_end_date_obj.replace(day=1)

                previous_month_end_date_obj = last_month_start_date_obj - timedelta(days=1)
                previous_month_start_date_obj = previous_month_end_date_obj.replace(day=1)

                # Convert to datetime objects for comparison
                result["start_date"] = datetime.combine(last_month_start_date_obj, datetime.min.time())
                result["end_date"] = datetime.combine(last_month_end_date_obj, datetime.max.time())
                result["secondary_start_date"] = datetime.combine(previous_month_start_date_obj, datetime.min.time())
                result["secondary_end_date"] = datetime.combine(previous_month_end_date_obj, datetime.max.time())
                result["date_filter"] = True
                result["description"] = f"last month ({last_month_start_date_obj.strftime('%b %Y')}) vs previous month ({previous_month_start_date_obj.strftime('%b %Y')})"
        
        if result.get("start_date") and result.get("end_date"):
            result["date_filter"] = True
        else:
            result["date_filter"] = False

        return result
    except json.JSONDecodeError as jde:
        print(f"JSON decoding error in parse_date_range_from_query: {jde} - Response content: {content}")
        return {"date_filter": False, "start_date": None, "end_date": None, "secondary_start_date": None, "secondary_end_date": None, "description": "all available data", "relative_period_detected": None}
    except Exception as e:
        print(f"Date range parsing error: {e}")
        return {"date_filter": False, "start_date": None, "end_date": None, "secondary_start_date": None, "secondary_end_date": None, "description": "all available data", "relative_period_detected": None}

# ---------- CM QUERY PARSER & DISPATCHER ----------
def get_query_details(prompt):
    # First, get all date information
    date_info = parse_date_range_from_query(prompt)

    # Then, use LLM to identify query type and other specific details
    system_prompt_query_type = """You are an assistant that classifies user queries and extracts relevant details.
    Identify the main intent of the query and return a JSON object with:
    - 'query_type': 'CM_analysis' for Contribution Margin related queries (e.g., 'show cm', 'cm < X%', 'cm > Y%'),
                    'Transportation_cost_analysis' for queries about cost trends in Transportation (e.g., 'Which cost triggered the Margin drop last month in Transportation'),
                    'unsupported' for anything else.
    - 'cm_filter_details': { 'filter_type': 'less_than', 'lower': 0.3, 'upper': null } if 'CM_analysis', else null.
    - 'segment': 'Transportation' if the query explicitly mentions it for transportation cost analysis, otherwise null.

    Return ONLY valid JSON.
    """
    try:
        response_query_type = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt_query_type},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        query_type_result = json.loads(response_query_type.choices[0].message.content)
        query_type = query_type_result.get('query_type', 'unsupported')
        cm_filter_details = query_type_result.get('cm_filter_details')
        segment_filter = query_type_result.get('segment')

        final_result = {
            "query_type": query_type,
            "segment_filter": segment_filter,
            **date_info # Include all date info (primary and secondary)
        }

        if query_type == "CM_analysis" and cm_filter_details:
            final_result["cm_filter_type"] = cm_filter_details.get("filter_type")
            final_result["cm_lower_bound"] = parse_percentage(cm_filter_details.get("lower"))
            final_result["cm_upper_bound"] = parse_percentage(cm_filter_details.get("upper"))
        else:
            final_result["cm_filter_type"] = None
            final_result["cm_lower_bound"] = None
            final_result["cm_upper_bound"] = None

        return final_result

    except Exception as e:
        print(f"Failed to parse query type or filters: {e}")
        return {
            "query_type": "unsupported",
            "segment_filter": None,
            "cm_filter_type": None,
            "cm_lower_bound": None,
            "cm_upper_bound": None,
            **date_info # Still include date info even if other parsing fails
        }

# ---------- CM CALCULATION FUNCTION (This is what needs to be present and unindented) ----------
def calculate_cm(df_pl, query_details):
    df_copy = df_pl.copy()
    
    if df_copy.empty:
        return pd.DataFrame()

    grouped = df_copy.groupby("FinalCustomerName", as_index=False).apply(lambda x: pd.Series({
        "Revenue": x[x["Group1"].isin(REVENUE_GROUPS)]["Amount in USD"].sum(),
        "Cost": x[x["Group1"].isin(COST_GROUPS)]["Amount in USD"].sum()
    })).reset_index(drop=True)

    revenue_abs = grouped["Revenue"].abs()
    grouped["CM_Ratio"] = (grouped["Revenue"] - grouped["Cost"]) / revenue_abs.replace(0, float('nan'))
    grouped["CM_Ratio"] = grouped["CM_Ratio"].replace([float('inf'), -float('inf')], float('nan'))
    
    grouped["CM_Value"] = grouped["CM_Ratio"] * 100

    filtered = grouped.copy()
    
    filter_type = query_details.get("cm_filter_type")
    lower_bound = query_details.get("cm_lower_bound")
    upper_bound = query_details.get("cm_upper_bound")

    if filter_type == "less_than" and lower_bound is not None:
        filtered = filtered[filtered["CM_Ratio"] < lower_bound]
    elif filter_type == "greater_than" and lower_bound is not None:
        filtered = filtered[filtered["CM_Ratio"] > lower_bound]
    elif filter_type == "between" and lower_bound is not None and upper_bound is not None:
        filtered = filtered[
            (filtered["CM_Ratio"] >= lower_bound) &
            (filtered["CM_Ratio"] <= upper_bound)
            ]
    
    if filtered.empty:
        return pd.DataFrame(columns=["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)"])

    ascending_sort = True
    if filter_type == "greater_than":
        ascending_sort = False
    
    filtered = filtered.sort_values(
        by="CM_Value",
        ascending=ascending_sort
    ).reset_index(drop=True)

    filtered.insert(0, "S.No", filtered.index + 1)

    filtered["CM (%)"] = filtered["CM_Value"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:.2f}%"
    )
    
    filtered["Revenue"] = pd.to_numeric(filtered["Revenue"], errors='coerce')
    filtered["Cost"] = pd.to_numeric(filtered["Cost"], errors='coerce')

    return filtered[["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)"]]

# ---------- TRANSPORTATION COST TREND ANALYSIS ----------
def analyze_transportation_cost_trend(df_pl, query_details):
    """
    Analyzes cost trends for a specified segment between two dynamic months,
    identifying costs that increased.
    Requires 'secondary_start_date', 'secondary_end_date', 'start_date', 'end_date'
    from query_details.
    """
    segment = query_details.get("segment_filter")
    if not segment:
        return "Please specify a segment for transportation cost analysis (e.g., 'Transportation')."

    prev_month_start = query_details.get("secondary_start_date")
    prev_month_end = query_details.get("secondary_end_date")
    current_month_start = query_details.get("start_date")
    current_month_end = query_details.get("end_date")

    if not all([prev_month_start, prev_month_end, current_month_start, current_month_end]):
        return "Could not determine the two consecutive months for comparison from your query. Please be explicit (e.g., 'May and June 2025' or 'last month vs previous month')."

    # Filter for the specified segment and 'Cost' type
    segment_cost_df = df_pl[
        (df_pl["Segment"] == segment) & 
        (df_pl["Type"] == "Cost") # Ensure 'Type' column exists and correctly identifies costs
    ].copy()

    if segment_cost_df.empty:
        return f"No '{segment}' cost data found for analysis."

    # Filter for previous month data
    prev_month_df = segment_cost_df[
        (segment_cost_df["Date"] >= prev_month_start) & 
        (segment_cost_df["Date"] <= prev_month_end)
    ]

    # Filter for current month data
    current_month_df = segment_cost_df[
        (segment_cost_df["Date"] >= current_month_start) & 
        (segment_cost_df["Date"] <= current_month_end)
    ]

    if prev_month_df.empty and current_month_df.empty:
        return f"No '{segment}' cost data available for {prev_month_start.strftime('%b %Y')} or {current_month_start.strftime('%b %Y')}."
    
    # Group by "Group Description" and sum "Amount in USD"
    prev_month_costs = prev_month_df.groupby("Group Description")["Amount in USD"].sum().reset_index()
    prev_month_costs.rename(columns={"Amount in USD": "Previous_Month_Cost"}, inplace=True)

    current_month_costs = current_month_df.groupby("Group Description")["Amount in USD"].sum().reset_index()
    current_month_costs.rename(columns={"Amount in USD": "Current_Month_Cost"}, inplace=True)

    # Merge the costs
    merged_costs = pd.merge(prev_month_costs, current_month_costs, on="Group Description", how="outer").fillna(0)

    # Calculate the difference and identify increases
    merged_costs["Cost_Increase"] = merged_costs["Current_Month_Cost"] - merged_costs["Previous_Month_Cost"]
    increased_costs = merged_costs[merged_costs["Cost_Increase"] > 0]

    if increased_costs.empty:
        return f"No specific costs increased in '{segment}' from {prev_month_start.strftime('%b %Y')} to {current_month_start.strftime('%b %Y')}."
    else:
        # Format for display
        increased_costs["Previous_Month_Cost"] = increased_costs["Previous_Month_Cost"].apply(lambda x: f"${x:,.2f}")
        increased_costs["Current_Month_Cost"] = increased_costs["Current_Month_Cost"].apply(lambda x: f"${x:,.2f}")
        increased_costs["Cost_Increase"] = increased_costs["Cost_Increase"].apply(lambda x: f"${x:,.2f}")
        
        return increased_costs[['Group Description', 'Previous_Month_Cost', 'Current_Month_Cost', 'Cost_Increase']]