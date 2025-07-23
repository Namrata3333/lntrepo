# kpi_calculationss.py (MODIFIED - FINAL RETURN COLUMNS)

import pandas as pd
from openai import AzureOpenAI
from datetime import datetime, timedelta
import pytz
import json
from dateutil.parser import parse
import calendar
import os
import numpy as np

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
CB_COST_GROUPS = ["C&B Cost Onsite", "C&B Cost Offshore"]


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

# ---------- DATE RANGE PARSING (FINAL CORRECTED VERSION) ----------
def parse_date_range_from_query(query):
    """
    Uses GPT to extract precise date ranges.
    Handles single periods (explicit FY/Q, relative terms) and comparison periods.
    """
    india_tz = pytz.timezone("Asia/Kolkata")
    today = datetime.now(india_tz).date() 
    
    system_prompt = f"""Today's date is {today.strftime('%Y-%m-%d')}.
You are an expert at extracting precise date ranges from natural language queries, specifically for L&T's fiscal year (April 1st to March 31st).
The fiscal quarters are:
    - Q1: April 1 - June 30
    - Q2: July 1 - September 30
    - Q3: October 1 - December 31
    - Q4: January 1 - March 31 (of the next calendar year for the same fiscal year).

Return a JSON object with:
- 'date_filter': boolean indicating if a date filter was requested
- 'start_date': YYYY-MM-DD format (first day of the *primary* period) or null
- 'end_date': YYYY-MM-DD format (last day of the *primary* period) or null
- 'secondary_start_date': YYYY-MM-DD format (first day of the *comparison* period) or null
- 'secondary_end_date': YYYY-MM-DD format (last day of the *comparison* period) or null
- 'description': natural language description of the period(s)
- 'relative_period_detected': "last quarter", "last year", "this year", "last month", "this month", "current date", "last_and_previous_month", "last_to_this_quarter", "explicit_quarter", "explicit_fiscal_year", "explicit_comparison", or null

Rules for Fiscal Year (FY) interpretation:
1. An "FY" followed by a 2-digit year (e.g., "FY26") refers to the fiscal year starting April 1st of the calendar year '25' and ending March 31st of calendar year '26'. For example, FY26 runs from 2025-04-01 to 2026-03-31.
2. An "FY" followed by a 4-digit year (e.g., "FY2026") should be interpreted similarly: the fiscal year starts April 1st of the calendar year 2025 and ends March 31st of 2026. Treat "FY2026" and "FY26" as equivalent for this purpose.
3. Quarters are fiscal quarters as defined above.
4. **CRITICAL RULE for Single Periods:**
    - If the query explicitly mentions a *single* Fiscal Year and Quarter (e.g., "FY26-Q1" or "FY2025 Q4"), calculate the exact start and end dates for *that single quarter*. Assign these to `start_date`/`end_date`. Set `secondary_start_date`/`secondary_end_date` to `null`. Set `relative_period_detected` to "explicit_quarter". The `description` should be like "FY26 Q1 (Apr 1, 2025 - Jun 30, 2025)".
    - If the query explicitly mentions a *single* Fiscal Year (e.g., "FY26" or "FY2025"), calculate the exact start and end dates for *that single fiscal year*. Assign these to `start_date`/`end_date`. Set `secondary_start_date`/`secondary_end_date` to `null`. Set `relative_period_detected` to "explicit_fiscal_year". The `description` should be like "FY26 (Apr 1, 2025 - Mar 31, 2026)".
    **- If the query asks for *only* "last quarter" or "this quarter" (without a comparison keyword like "to" or "vs"), set `relative_period_detected` to "last quarter" or "this quarter" respectively, and ensure `secondary_start_date`/`secondary_end_date` are null.**
5. **CRITICAL RULE for Comparisons:**
    - If the query asks for a comparison between *two explicit* Fiscal Years/Quarters (e.g., "FY25Q4 and FY26Q1"), calculate dates for both. Assign the *later* period to `start_date`/`end_date` (primary) and the *earlier* period to `secondary_start_date`/`secondary_end_date` (secondary). Set `relative_period_detected` to "explicit_comparison". The `description` should be like "FY26 Q1 (Apr 1, 2025 - Jun 30, 2025) vs FY25 Q4 (Jan 1, 2025 - Mar 31, 2025)".
    - If the query explicitly asks for a comparison like "last quarter to this quarter" *using comparison keywords* (e.g., "last quarter to this quarter", "this quarter vs last quarter"), `start_date`/`end_date` should be the current quarter, and `secondary_start_date`/`secondary_end_date` should be the previous completed quarter. Set `relative_period_detected` to "last_to_this_quarter".
6. For relative periods (like "last month", "last year" *without explicit FY/Q or comparison keywords*), calculate exact dates based on today's date. If *only* relative terms are used, then set `relative_period_detected` accordingly.
7. For absolute periods (like "January 2023"), use exact dates.
8. For ranges (like "from March to May 2023"), use exact start/end dates.
9. If the query asks for a comparison between two specific consecutive months (e.g., "May and June", "last month and previous month"), identify `start_date`/`end_date` as the later month and `secondary_start_date`/`secondary_end_date` as the earlier month. Set `relative_period_detected` to "last_and_previous_month" if detected by relative terms.
10. If no date filter, set date_filter=false and return null for dates.
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
                    # Convert to datetime object (not .date()) and ensure end of day
                    dt_obj = parse(result[key])
                    if key in ["end_date", "secondary_end_date"]:
                        result[key] = datetime.combine(dt_obj.date(), datetime.max.time())
                    else:
                        result[key] = datetime.combine(dt_obj.date(), datetime.min.time()) 
                except:
                    result[key] = None
        
        # --- Python Logic to refine LLM's output based on `relative_period_detected` ---
        # This ensures correct date ranges and nulls secondary periods when not needed.
        
        relative_period = result.get('relative_period_detected')
        
        # Handle explicit single quarter/fiscal year or explicit comparison
        if relative_period in ["explicit_quarter", "explicit_fiscal_year"]:
            result["secondary_start_date"] = None
            result["secondary_end_date"] = None
            result["date_filter"] = True
            # Description is expected to be set by LLM based on prompt rules 4 & 5
            return result
        elif relative_period == "explicit_comparison":
            result["date_filter"] = True
            # Description is expected to be set by LLM based on prompt rule 6
            return result

        # --- Custom L&T Fiscal Year Logic for Relative Periods (only if LLM didn't handle explicitly) ---
        # This block will now only execute if `relative_period_detected` is one of the relative terms
        # and it's not an "explicit_quarter" or "explicit_fiscal_year" case.
        
        # This is the "last quarter" specific calculation block. It needs to be entered when
        # relative_period is explicitly "last quarter" and NOT "last_to_this_quarter"
        if relative_period == 'last quarter': 
            current_cal_month = today.month
            current_cal_year = today.year

            current_fiscal_year_start_cal_year = current_cal_year if current_cal_month >= 4 else current_cal_year - 1
            
            last_completed_q_start_month = None
            last_completed_q_end_month = None
            last_completed_q_cal_year = None
            last_completed_q_name = None 

            if 4 <= current_cal_month <= 6: # Today is in FY-Q1 (e.g., Apr-Jun 2025 -> FY26Q1)
                last_completed_q_start_month = 1
                last_completed_q_end_month = 3
                last_completed_q_cal_year = current_fiscal_year_start_cal_year 
                last_completed_q_name = f"FY{current_fiscal_year_start_cal_year + 1}Q4" 
            elif 7 <= current_cal_month <= 9: # Today is in FY-Q2 (e.g., Jul-Sep 2025 -> FY26Q2)
                last_completed_q_start_month = 4
                last_completed_q_end_month = 6
                last_completed_q_cal_year = current_fiscal_year_start_cal_year
                last_completed_q_name = f"FY{current_fiscal_year_start_cal_year + 1}Q1"
            elif 10 <= current_cal_month <= 12: # Today is in FY-Q3 (e.g., Oct-Dec 2025 -> FY26Q3)
                last_completed_q_start_month = 7
                last_completed_q_end_month = 9
                last_completed_q_cal_year = current_fiscal_year_start_cal_year
                last_completed_q_name = f"FY{current_fiscal_year_start_cal_year + 1}Q2"
            else: # 1 <= current_cal_month <= 3 # Today is in FY-Q4 (e.g., Jan-Mar 2026 -> FY26Q4)
                last_completed_q_start_month = 10
                last_completed_q_end_month = 12
                last_completed_q_cal_year = current_fiscal_year_start_cal_year 
                last_completed_q_name = f"FY{current_fiscal_year_start_cal_year + 1}Q3"

            result["start_date"] = datetime(last_completed_q_cal_year, last_completed_q_start_month, 1)
            result["end_date"] = datetime(last_completed_q_cal_year, last_completed_q_end_month, calendar.monthrange(last_completed_q_cal_year, last_completed_q_end_month)[1], 23, 59, 59, 999999)
            result["date_filter"] = True
            result["description"] = f"last quarter ({last_completed_q_name})"
            result["secondary_start_date"] = None 
            result["secondary_end_date"] = None   
        
        elif relative_period == 'last_to_this_quarter':
            current_cal_year = today.year
            current_cal_month = today.month
            
            current_fiscal_year_start_cal_year = current_cal_year if current_cal_month >= 4 else current_cal_year - 1
            
            fiscal_year_for_current_quarter_name = current_fiscal_year_start_cal_year + 1

            if 4 <= current_cal_month <= 6: # Q1
                curr_q_start_month, curr_q_end_month = 4, 6
                curr_q_cal_year = current_fiscal_year_start_cal_year
                curr_q_name = f"FY{fiscal_year_for_current_quarter_name}Q1"
            elif 7 <= current_cal_month <= 9: # Q2
                curr_q_start_month, curr_q_end_month = 7, 9
                curr_q_cal_year = current_fiscal_year_start_cal_year
                curr_q_name = f"FY{fiscal_year_for_current_quarter_name}Q2"
            elif 10 <= current_cal_month <= 12: # Q3
                curr_q_start_month, curr_q_end_month = 10, 12
                curr_q_cal_year = current_fiscal_year_start_cal_year
                curr_q_name = f"FY{fiscal_year_for_current_quarter_name}Q3"
            else: # 1 <= current_cal_month <= 3 (Q4)
                curr_q_start_month, curr_q_end_month = 1, 3
                curr_q_cal_year = current_fiscal_year_start_cal_year + 1 
                curr_q_name = f"FY{fiscal_year_for_current_quarter_name}Q4"
            
            result["start_date"] = datetime(curr_q_cal_year, curr_q_start_month, 1)
            result["end_date"] = datetime(curr_q_cal_year, curr_q_end_month, calendar.monthrange(curr_q_cal_year, curr_q_end_month)[1], 23, 59, 59, 999999)
            
            prev_q_cal_year_calc = curr_q_cal_year 
            fiscal_year_for_prev_quarter_name = fiscal_year_for_current_quarter_name 

            if curr_q_start_month == 4: # If current is Q1 (Apr-Jun), previous was Q4 (Jan-Mar of previous fiscal year)
                prev_q_start_month, prev_q_end_month = 1, 3
                prev_q_cal_year_calc -= 1 
                fiscal_year_for_prev_quarter_name -= 1 
                prev_q_name = f"FY{fiscal_year_for_prev_quarter_name}Q4"
            elif curr_q_start_month == 7: # If current is Q2 (Jul-Sep), previous was Q1 (Apr-Jun)
                prev_q_start_month, prev_q_end_month = 4, 6
                prev_q_name = f"FY{fiscal_year_for_prev_quarter_name}Q1"
            elif curr_q_start_month == 10: # If current is Q3 (Oct-Dec), previous was Q2 (Jul-Sep)
                prev_q_start_month, prev_q_end_month = 7, 9
                prev_q_name = f"FY{fiscal_year_for_prev_quarter_name}Q2"
            else: # If current is Q4 (Jan-Mar), previous was Q3 (Oct-Dec of previous calendar year)
                prev_q_start_month, prev_q_end_month = 10, 12
                prev_q_cal_year_calc -= 1 
                prev_q_name = f"FY{fiscal_year_for_prev_quarter_name}Q3"

            result["secondary_start_date"] = datetime(prev_q_cal_year_calc, prev_q_start_month, 1)
            result["secondary_end_date"] = datetime(prev_q_cal_year_calc, prev_q_end_month, calendar.monthrange(prev_q_cal_year_calc, prev_q_end_month)[1], 23, 59, 59, 999999)
            result["date_filter"] = True
            result["description"] = f"this quarter ({curr_q_name}) vs last quarter ({prev_q_name})"
        
        # Add other relative period handlers here (e.g., 'last year', 'this year', 'last month', etc.)
        # Ensure they follow the 'if/elif' structure correctly.

        # Ensure 'result["end_date"]' and 'result["secondary_end_date"]' are set to end of day for proper filtering
        # This part should be outside the specific `relative_period` blocks but after the date setting logic
        if result.get("end_date"):
            result["end_date"] = result["end_date"].replace(hour=23, minute=59, second=59, microsecond=999999)
        if result.get("secondary_end_date"):
            result["secondary_end_date"] = result["secondary_end_date"].replace(hour=23, minute=59, second=59, microsecond=999999)

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
                      'C&B_cost_variation' for queries asking about C&B cost variation between two periods (e.g., 'How much C&B varied from last quarter to this quarter'),
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

# ---------- CM CALCULATION FUNCTION ----------
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
    
    grouped["CM_Value"] = grouped["CM_Ratio"] * 100 # This line calculates CM_Value

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
    # Handle "equals" filter for CM
    elif filter_type == "equals" and lower_bound is not None:
        # Use numpy.isclose for robust floating-point equality comparison
        filtered = filtered[np.isclose(filtered["CM_Ratio"], lower_bound)]
    
    if filtered.empty:
        # IMPORTANT: When returning an empty DataFrame, ensure it has all expected columns for mainn.py
        # even if they are empty, to prevent KeyError later.
        # We still need 'CM_Value' internally for plotting in mainn.py even if not shown
        return pd.DataFrame(columns=["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"])

    ascending_sort = True
    if filter_type == "greater_than":
        ascending_sort = False
    
    filtered = filtered.sort_values(
        by="CM_Value", # Sort by CM_Value for consistency
        ascending=ascending_sort
    ).reset_index(drop=True)

    filtered.insert(0, "S.No", filtered.index + 1)

    # CM (%) column already formatted to 2 decimal places here
    filtered["CM (%)"] = filtered["CM_Value"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:.2f}%"
    )
    
    filtered["Revenue"] = pd.to_numeric(filtered["Revenue"], errors='coerce')
    filtered["Cost"] = pd.to_numeric(filtered["Cost"], errors='coerce')

    # MODIFIED: Only return the columns intended for the table display.
    # 'CM_Value' is still calculated and available in 'filtered' DataFrame
    # but not explicitly returned in this final list.
    return filtered[["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"]] # Keep CM_Value for mainn.py

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

# ---------- C&B COST VARIATION CALCULATION ----------
def calculate_cb_cost_variation(df_pl, query_details):
    """
    Calculates the variation of C&B Cost between two specified quarters.
    Expected query_details to contain 'start_date', 'end_date' for the primary quarter
    and 'secondary_start_date', 'secondary_end_date' for the secondary quarter.
    """
    primary_q_start = query_details.get("start_date")
    primary_q_end = query_details.get("end_date")
    secondary_q_start = query_details.get("secondary_start_date")
    secondary_q_end = query_details.get("secondary_end_date")
    
    primary_q_desc = query_details.get("description", "current quarter").split("vs")[0].strip() # e.g., "this quarter (FY26Q2)"
    # This might need adjustment if LLM doesn't always use "vs" or provides a different description
    secondary_q_desc = query_details.get("description", "last quarter").split("vs")[-1].strip() if "vs" in query_details.get("description", "") else "last quarter"

    if not all([primary_q_start, primary_q_end, secondary_q_start, secondary_q_end]):
        return "Could not determine the two quarters for C&B cost comparison from your query. Please be explicit (e.g., 'last quarter to this quarter')."

    # Filter for C&B Cost data
    cb_df = df_pl[df_pl["Group Description"].isin(CB_COST_GROUPS)].copy()

    if cb_df.empty:
        return "No C&B Cost data found for analysis."

    # Calculate C&B Cost for the primary quarter
    primary_q_cb_cost_df = cb_df[
        (cb_df["Date"] >= primary_q_start) &
        (cb_df["Date"] <= primary_q_end)
    ]
    primary_q_total_cb_cost = primary_q_cb_cost_df["Amount in USD"].sum()

    # Calculate C&B Cost for the secondary quarter
    secondary_q_cb_cost_df = cb_df[
        (cb_df["Date"] >= secondary_q_start) &
        (cb_df["Date"] <= secondary_q_end)
    ]
    secondary_q_total_cb_cost = secondary_q_cb_cost_df["Amount in USD"].sum()

    # Calculate variation
    variation = primary_q_total_cb_cost - secondary_q_total_cb_cost

    response_parts = []
    response_parts.append(f"C&B Cost for {primary_q_desc}: **${primary_q_total_cb_cost:,.2f}**")
    response_parts.append(f"C&B Cost for {secondary_q_desc}: **${secondary_q_total_cb_cost:,.2f}**")
    
    if variation > 0:
        response_parts.append(f"The C&B Cost **increased** by **${variation:,.2f}** from {secondary_q_desc} to {primary_q_desc}.")
    elif variation < 0:
        response_parts.append(f"The C&B Cost **decreased** by **${abs(variation):,.2f}** from {secondary_q_desc} to {primary_q_desc}.")
    else:
        response_parts.append(f"The C&B Cost remained **unchanged** from {secondary_q_desc} to {primary_q_desc}.")
    
    return "\n\n".join(response_parts)