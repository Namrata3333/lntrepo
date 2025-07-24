# kpi_calculationss.py (Corrected Date Type Handling)

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
    """
    Converts percentage strings (e.g., "112%", "30%") or numbers (e.g., 112, 0.3, 30)
    into decimal floats (e.g., 1.12, 0.3, 0.3).
    """
    try:
        if isinstance(val, str):
            val = val.replace("%", "").strip()
            if not val:
                return None
            val = float(val)
            # If it was a string, and the numeric value is > 1 (e.g., "112" from "112%"),
            # then it needs to be divided by 100.
            if val > 1.0: 
                return val / 100.0
            return val # If the string was like "0.3"
        
        elif isinstance(val, (int, float)):
            # If it's an integer (e.g., 96, 112) and > 1, convert to decimal (0.96, 1.12)
            if isinstance(val, int) and val > 1:
                return float(val) / 100.0
            # If it's already a float (e.g., 0.96, 1.12) or an integer 0 or 1, return as is.
            return float(val)
        
        else: # Not a string, int, or float
            return None
            
    except (ValueError, TypeError): # Catch conversion errors or if val is None
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
You are an expert at extracting precise date ranges from natural language queries.

Here are the definitions of quarters you must use:

1.  **Fiscal Quarters (L&T's FY: April 1st to March 31st):**
    * FY-Q1: April 1 - June 30
    * FY-Q2: July 1 - September 30
    * FY-Q3: October 1 - December 31
    * FY-Q4: January 1 - March 31 (of the next calendar year for the same fiscal year).
    * A fiscal year is explicitly indicated by "FY" or "financial year" (e.g., "FY26", "financial year 2025-26", "FY2026"). "FY26" or "FY2026" refers to the fiscal year starting April 1st, 2025 and ending March 31st, 2026.

2.  **Calendar Quarters (Standard Calendar Year: January 1st to December 31st):**
    * Cal-Q1: January 1 - March 31
    * Cal-Q2: April 1 - June 30
    * Cal-Q3: July 1 - September 31
    * Cal-Q4: October 1 - December 31
    * A calendar year quarter is indicated by a year and quarter *without* "FY" or "financial year" (e.g., "2025 Q1", "Q1 2025", "2025q2").

**CRITICAL RULE: Always use Fiscal Quarters IF "FY" or "financial year" is explicitly mentioned. Otherwise, always use Calendar Quarters.**

Return a JSON object with:
- 'date_filter': boolean indicating if a date filter was requested
- 'start_date': YYYY-MM-DD format (first day of the *primary* period) or null
- 'end_date': YYYY-MM-DD format (last day of the *primary* period) or null
- 'secondary_start_date': YYYY-MM-DD format (first day of the *comparison* period) or null
- 'secondary_end_date': YYYY-MM-DD format (last day of the *comparison* period) or null
- 'description': natural language description of the period(s)
- 'relative_period_detected': "last quarter", "last year", "this year", "last month", "this month", "current date", "last_and_previous_month", "last_to_this_quarter", "explicit_quarter", "explicit_fiscal_year", "explicit_comparison", or null

Rules for Date Interpretation:
1.  **Single Periods (Explicit Fiscal):** If the query explicitly mentions a *single Fiscal Year and Quarter* (e.g., "FY26-Q1" or "FY2025 Q4"), calculate the exact start and end dates for *that single fiscal quarter*. Assign these to `start_date`/`end_date`. Set `secondary_start_date`/`secondary_end_date` to `null`. Set `relative_period_detected` to "explicit_quarter". The `description` should be like "FY26 Q1 (Apr 1, 2025 - Jun 30, 2025)".
2.  **Single Periods (Explicit Calendar):** If the query explicitly mentions a *single Calendar Year and Quarter* (e.g., "2025 Q1", "Q1 2025", "2025q2"), calculate the exact start and end dates for *that single calendar quarter*. Assign these to `start_date`/`end_date`. Set `secondary_start_date`/`secondary_end_date` to `null`. Set `relative_period_detected` to "explicit_quarter". The `description` should be like "2025 Q2 (Apr 1, 2025 - Jun 30, 2025)".
3.  **Single Periods (Explicit Fiscal Year):** If the query explicitly mentions a *single Fiscal Year* (e.g., "FY26" or "FY2025"), calculate the exact start and end dates for *that single fiscal year*. Assign these to `start_date`/`end_date`. Set `secondary_start_date`/`secondary_end_date` to `null`. Set `relative_period_detected` to "explicit_fiscal_year". The `description` should be like "FY26 (Apr 1, 2025 - Mar 31, 2026)".
4.  **Relative Quarters:** If the query asks for *only* "last quarter" or "this quarter" (without a comparison keyword like "to" or "vs"), this refers to the *last completed fiscal quarter* or *current fiscal quarter* respectively. Set `relative_period_detected` to "last quarter" or "this quarter" and ensure `secondary_start_date`/`secondary_end_date` are null.
5.  **Comparison Periods (Explicit Fiscal):** If the query asks for a comparison between *two explicit Fiscal Years/Quarters* (e.g., "FY25Q4 and FY26Q1"), calculate dates for both. Assign the *later* period to `start_date`/`end_date` (primary) and the *earlier* period to `secondary_start_date`/`secondary_end_date` (secondary). Set `relative_period_detected` to "explicit_comparison". The `description` should be like "FY26 Q1 (Apr 1, 2025 - Jun 30, 2025) vs FY25 Q4 (Jan 1, 2025 - Mar 31, 2025)".
6.  **Comparison Periods (Explicit Calendar):** If the query asks for a comparison between *two explicit Calendar Years/Quarters* (e.g., "2025 Q1 and 2025 Q2", "2025q1 to 2025q2"), calculate dates for both. Assign the *later* period to `start_date`/`end_date` (primary) and the *earlier* period to `secondary_start_date`/`secondary_end_date` (secondary). Set `relative_period_detected` to "explicit_comparison". The `description` should be like "2025 Q2 (Apr 1, 2025 - Jun 30, 2025) vs 2025 Q1 (Jan 1, 2025 - Mar 31, 2025)".
7.  **Comparison Periods (Relative Quarters):** If the query explicitly asks for a comparison like "last quarter to this quarter" *using comparison keywords* (e.g., "last quarter to this quarter", "this quarter vs last quarter"), `start_date`/`end_date` should be the current *fiscal* quarter, and `secondary_start_date`/`secondary_end_date` should be the previous completed *fiscal* quarter. Set `relative_period_detected` to "last_to_this_quarter".
8.  For relative periods (like "last month", "last year" *without explicit FY/Q or comparison keywords*), calculate exact dates based on today's date. If *only* relative terms are used, then set `relative_period_detected` accordingly.
9.  For absolute periods (like "January 2023"), use exact dates.
10. For ranges (like "from March to May 2023"), use exact start/end dates.
11. If the query asks for a comparison between two specific consecutive months (e.g., "May and June", "last month and previous month"), identify `start_date`/`end_date` as the later month and `secondary_start_date`/`secondary_end_date` as the earlier month. Set `relative_period_detected` to "last_and_previous_month" if detected by relative terms.
12. If no date filter, set date_filter=false and return null for dates.
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
        
        # --- DEBUG: Print raw LLM date parsing result ---
        # print(f"DEBUG (Date Parsing - LLM Raw): {result}")
        # --- END DEBUG ---

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
            # --- DEBUG: Print final calculated dates for explicit periods ---
            # print(f"DEBUG (Date Parsing - Explicit Period): start_date={result['start_date']}, end_date={result['end_date']}, relative_period_detected={relative_period}")
            # --- END DEBUG ---
            return result
        elif relative_period == "explicit_comparison":
            result["date_filter"] = True
            # --- DEBUG: Print final calculated dates for explicit comparison ---
            # print(f"DEBUG (Date Parsing - Explicit Comparison): primary_start={result['start_date']}, primary_end={result['end_date']}, secondary_start={result['secondary_start_date']}, secondary_end={result['secondary_end_date']}, relative_period_detected={relative_period}")
            # --- END DEBUG ---
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
            else: # 1 <= current_cal_month <= 3 (Q4)
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
            # --- DEBUG: Print final calculated dates for 'last quarter' ---
            # print(f"DEBUG (Date Parsing - Last Quarter): today={today}, start_date={result['start_date']}, end_date={result['end_date']}, relative_period_detected={relative_period}")
            # --- END DEBUG ---
        
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
            elif 7 <= current_cal_month <= 9: # If current is Q2 (Jul-Sep), previous was Q1 (Apr-Jun)
                prev_q_start_month, prev_q_end_month = 4, 6
                prev_q_name = f"FY{fiscal_year_for_prev_quarter_name}Q1"
            elif 10 <= current_cal_month <= 12: # If current is Q3 (Oct-Dec), previous was Q2 (Jul-Sep)
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
            # --- DEBUG: Print final calculated dates for 'last_to_this_quarter' ---
            # print(f"DEBUG (Date Parsing - Last to This Quarter): primary_start={result['start_date']}, primary_end={result['end_date']}, secondary_start={result['secondary_start_date']}, secondary_end={result['secondary_end_date']}, relative_period_detected={relative_period}")
            # --- END DEBUG ---
        
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
                      'CB_revenue_trend' for queries asking about "M-o-M trend of C&B cost % w.r.t total revenue",
                      'unsupported' for anything else.
    - 'cm_filter_details': { 'filter_type': 'less_than', 'lower': 0.3, 'upper': null } if 'CM_analysis', else null.
        **IMPORTANT for 'cm_filter_details' values:** 'lower' and 'upper' should ALWAYS be expressed as DECIMAL floats.
        For example:
        - For 30%, use 0.3
        - For 112%, use 1.12
        - For 2%, use 0.02
        - For 100%, use 1.0
        **IMPORTANT for 'filter_type' values (for CM_analysis):** Use 'less_than', 'greater_than', 'between', or 'equals'.

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
        content = response_query_type.choices[0].message.content
        query_type_result = json.loads(content)
        
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
            # Parse the LLM output as decimals
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

    # --- NEW: Filter out rows where Revenue is 0 for CM calculation ---
    # Calculate total revenue per customer based on REVENUE_GROUPS
    df_revenue_only = df_copy[df_copy["Group1"].isin(REVENUE_GROUPS)].copy()
    customer_revenue = df_revenue_only.groupby("FinalCustomerName")["Amount in USD"].sum()
    
    # Identify customers with zero, NaN, or effectively zero revenue (e.g., very small floats)
    # Using np.isclose for robust floating-point comparison with 0
    customers_to_exclude = customer_revenue[
        customer_revenue.isna() | np.isclose(customer_revenue, 0)
    ].index
    
    # Filter out these customers from the DataFrame *before* calculating CM for grouping
    df_copy_filtered_for_cm = df_copy[
        ~df_copy["FinalCustomerName"].isin(customers_to_exclude)
    ].copy()

    if df_copy_filtered_for_cm.empty:
        return pd.DataFrame(columns=["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"])

    grouped = df_copy_filtered_for_cm.groupby("FinalCustomerName", as_index=False).apply(lambda x: pd.Series({
        "Revenue": x[x["Group1"].isin(REVENUE_GROUPS)]["Amount in USD"].sum(),
        "Cost": x[x["Group1"].isin(COST_GROUPS)]["Amount in USD"].sum()
    })).reset_index(drop=True)

    # Calculate CM_Ratio as a decimal (0-1 range)
    # Use np.where to handle division by zero and assign NaN, then drop NaNs
    grouped["CM_Ratio"] = np.where(
        grouped["Revenue"] != 0, # Condition: Revenue is not 0
        (grouped["Revenue"] - grouped["Cost"]) / grouped["Revenue"], # Value if True
        np.nan # Value if False (i.e., Revenue is 0)
    )
    
    # Replace any remaining inf/-inf that might occur due to floating point inaccuracies with NaN
    grouped["CM_Ratio"] = grouped["CM_Ratio"].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows where CM_Ratio is NaN (e.g., if Revenue was 0 or became NaN during aggregation)
    grouped.dropna(subset=['CM_Ratio'], inplace=True)

    if grouped.empty:
        return pd.DataFrame(columns=["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"])

    # CM_Value is the percentage value (e.g., 30 for 30%)
    grouped["CM_Value"] = grouped["CM_Ratio"] * 100 

    filtered = grouped.copy()
    
    filter_type = query_details.get("cm_filter_type")
    lower_bound = query_details.get("cm_lower_bound") # These are now expected to be decimals (0-1 range)
    upper_bound = query_details.get("cm_upper_bound") # These are now expected to be decimals (0-1 range)

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
        # lower_bound is already expected to be in 0-1 range (e.g., 0.02 for 2%)
        filtered = filtered[np.isclose(filtered["CM_Ratio"], lower_bound)]
    
    if filtered.empty:
        # IMPORTANT: When returning an empty DataFrame, ensure it has all expected columns for mainn.py
        # even if they are empty, to prevent KeyError later.
        return pd.DataFrame(columns=["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"])

    ascending_sort = True
    if filter_type == "greater_than":
        ascending_sort = False
    
    filtered = filtered.sort_values(
        by="CM_Value", # Sort by CM_Value for consistency
        ascending=ascending_sort
    ).reset_index(drop=True)

    filtered.insert(0, "S.No", filtered.index + 1)

    # CM (%) column formatted to 2 decimal places here, to be used for display
    # This remains as a string for display in the table and hover in plot.
    filtered["CM (%)"] = filtered["CM_Value"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:.2f}%"
    )
    
    # Ensure Revenue and Cost are numeric before returning
    filtered["Revenue"] = pd.to_numeric(filtered["Revenue"], errors='coerce')
    filtered["Cost"] = pd.to_numeric(filtered["Cost"], errors='coerce')

    # Keep 'CM_Value' (numeric) for plotting in mainn.py
    return filtered[["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"]] 

# ---------- TRANSPORTATION COST TREND ANALYSIS ----------
def analyze_transportation_cost_trend(df_pl, query_details):
    """
    Analyzes cost trends for a specified segment between two dynamic periods,
    identifying costs that increased.
    Requires 'secondary_start_date', 'secondary_end_date', 'start_date', 'end_date'
    from query_details.
    """
    segment = query_details.get("segment_filter")
    if not segment:
        return "Please specify a segment for transportation cost analysis (e.g., 'Transportation')."

    prev_period_start = query_details.get("secondary_start_date")
    prev_period_end = query_details.get("secondary_end_date")
    current_period_start = query_details.get("start_date")
    current_period_end = query_details.get("end_date")

    if not all([prev_period_start, prev_period_end, current_period_start, current_period_end]):
        return "Could not determine the two consecutive months/quarters for comparison from your query. Please be explicit (e.g., 'May and June 2025' or 'last month vs previous month', or 'last quarter vs this quarter')."

    # Filter for the specified segment and 'Cost' type
    segment_cost_df = df_pl[
        (df_pl["Segment"] == segment) & 
        (df_pl["Type"] == "Cost") # Ensure 'Type' column exists and correctly identifies costs
    ].copy()

    if segment_cost_df.empty:
        return f"No '{segment}' cost data found for analysis."

    # Filter for previous period data
    prev_period_df = segment_cost_df[
        (segment_cost_df["Date"] >= prev_period_start) & 
        (segment_cost_df["Date"] <= prev_period_end)
    ]

    # Filter for current period data
    current_period_df = segment_cost_df[
        (segment_cost_df["Date"] >= current_period_start) & 
        (segment_cost_df["Date"] <= current_period_end)
    ]

    if prev_period_df.empty and current_period_df.empty:
        # Dynamically format the period names for the message
        prev_period_name = prev_period_start.strftime('%b %Y') if (prev_period_end - prev_period_start).days < 90 else f"Q{((prev_period_start.month - 1) // 3) + 1} {prev_period_start.year}"
        current_period_name = current_period_start.strftime('%b %Y') if (current_period_end - current_period_start).days < 90 else f"Q{((current_period_start.month - 1) // 3) + 1} {current_period_start.year}"
        return f"No '{segment}' cost data available for {prev_period_name} or {current_period_name}."
    
    # Group by "Group Description" and sum "Amount in USD"
    prev_period_costs = prev_period_df.groupby("Group Description")["Amount in USD"].sum().reset_index()
    prev_period_costs.rename(columns={"Amount in USD": "Previous_Period_Cost"}, inplace=True)

    current_period_costs = current_period_df.groupby("Group Description")["Amount in USD"].sum().reset_index()
    current_period_costs.rename(columns={"Amount in USD": "Current_Period_Cost"}, inplace=True)

    # Merge the costs
    merged_costs = pd.merge(prev_period_costs, current_period_costs, on="Group Description", how="outer").fillna(0)

    # Calculate the difference and identify increases
    merged_costs["Cost_Increase"] = merged_costs["Current_Period_Cost"] - merged_costs["Previous_Period_Cost"]
    increased_costs = merged_costs[merged_costs["Cost_Increase"] > 0]

    if increased_costs.empty:
        prev_period_name = prev_period_start.strftime('%b %Y') if (prev_period_end - prev_period_start).days < 90 else f"Q{((prev_period_start.month - 1) // 3) + 1} {prev_period_start.year}"
        current_period_name = current_period_start.strftime('%b %Y') if (current_period_end - current_period_start).days < 90 else f"Q{((current_period_start.month - 1) // 3) + 1} {current_period_start.year}"
        return f"No specific costs increased in '{segment}' from {prev_period_name} to {current_period_name}."
    else:
        # Format for display
        increased_costs["Previous_Period_Cost"] = increased_costs["Previous_Period_Cost"].apply(lambda x: f"${x:,.2f}")
        increased_costs["Current_Period_Cost"] = increased_costs["Current_Period_Cost"].apply(lambda x: f"${x:,.2f}")
        increased_costs["Cost_Increase"] = increased_costs["Cost_Increase"].apply(lambda x: f"${x:,.2f}")
        
        return increased_costs[['Group Description', 'Previous_Period_Cost', 'Current_Period_Cost', 'Cost_Increase']]




# ---------- C&B COST VARIATION CALCULATION ----------
def calculate_cb_cost_variation(df_pl, query_details):
    """
    Calculates the variation of C&B Cost between two specified periods.
    Expected query_details to contain 'start_date', 'end_date' for the primary period
    and 'secondary_start_date', 'secondary_end_date' for the secondary period.
    Returns a dictionary with message and numerical data for plotting.
    """
    primary_period_start = query_details.get("start_date")
    primary_period_end = query_details.get("end_date")
    secondary_period_start = query_details.get("secondary_start_date")
    secondary_period_end = query_details.get("secondary_end_date")
    
    if not all([primary_period_start, primary_period_end, secondary_period_start, secondary_period_end]):
        return {
            "message": "Could not determine the two periods for C&B cost comparison from your query. Please be explicit (e.g., 'last month to this month' or 'last quarter to this quarter').",
            "primary_cost": None, "secondary_cost": None, "primary_desc": None, "secondary_desc": None, "variation": None
        }

    # Use the description from query_details if it's an explicit comparison,
    # otherwise, dynamically determine if it's a month or fiscal quarter comparison for display.
    if query_details.get('relative_period_detected') == "explicit_comparison":
        # Split the description from the LLM to get primary and secondary parts
        full_description = query_details.get('description', '')
        if ' vs ' in full_description:
            parts = full_description.split(' vs ')
            primary_period_desc = parts[0].strip()
            secondary_period_desc = parts[1].strip()
        else: # Fallback if 'vs' not found, use the full description for primary and derive secondary
            primary_period_desc = full_description
            # Attempt to derive secondary from primary if possible, otherwise use a generic name
            if (primary_period_end - primary_period_start).days < 60:
                # Assume it's a month, calculate previous month
                secondary_period_desc = (primary_period_start - timedelta(days=1)).replace(day=1).strftime('%B %Y')
            else:
                # Assume it's a quarter, calculate previous quarter fiscal name
                def get_fiscal_quarter_name_for_fallback(date_obj):
                    month = date_obj.month
                    year = date_obj.year
                    fiscal_year_start_year = year if month >= 4 else year - 1
                    if 4 <= month <= 6: return f"FY{fiscal_year_start_year+1}Q1"
                    elif 7 <= month <= 9: return f"FY{fiscal_year_start_year+1}Q2"
                    elif 10 <= month <= 12: return f"FY{fiscal_year_start_year+1}Q3"
                    else: return f"FY{fiscal_year_start_year+1}Q4"
                secondary_period_desc = get_fiscal_quarter_name_for_fallback(secondary_period_start)

    else: # For relative periods like "last month to this month" or "last quarter to this quarter"
        is_month_comparison = (primary_period_end - primary_period_start).days < 60 # Heuristic for month vs. quarter

        if is_month_comparison:
            primary_period_desc = primary_period_start.strftime('%B %Y')
            secondary_period_desc = secondary_period_start.strftime('%B %Y')
        else:
            # Fiscal quarter calculation for description
            def get_fiscal_quarter_name(date_obj):
                month = date_obj.month
                year = date_obj.year
                # Adjust year for Q4 which spans into the next calendar year for the same fiscal year
                if month >= 4: # Q1, Q2, Q3 are in the same calendar year as the start of the fiscal year
                    fiscal_year_start_year = year
                else: # Q4 is in the calendar year following the start of the fiscal year
                    fiscal_year_start_year = year - 1

                if 4 <= month <= 6:
                    return f"FY{fiscal_year_start_year+1}Q1"
                elif 7 <= month <= 9:
                    return f"FY{fiscal_year_start_year+1}Q2"
                elif 10 <= month <= 12:
                    return f"FY{fiscal_year_start_year+1}Q3"
                else: # Jan, Feb, Mar
                    return f"FY{fiscal_year_start_year+1}Q4" 

            primary_period_desc = get_fiscal_quarter_name(primary_period_start)
            secondary_period_desc = get_fiscal_quarter_name(secondary_period_start)

    # Filter for C&B Cost data
    cb_df = df_pl[df_pl["Group Description"].isin(CB_COST_GROUPS)].copy()

    if cb_df.empty:
        return {
            "message": "No C&B Cost data found for analysis.",
            "primary_cost": None, "secondary_cost": None, "primary_desc": None, "secondary_desc": None, "variation": None
        }

    # Calculate C&B Cost for the primary period
    primary_period_cb_cost_df = cb_df[
        (cb_df["Date"] >= primary_period_start) &
        (cb_df["Date"] <= primary_period_end)
    ]
    primary_period_total_cb_cost = primary_period_cb_cost_df["Amount in USD"].sum()

    # Calculate C&B Cost for the secondary period
    secondary_period_cb_cost_df = cb_df[
        (cb_df["Date"] >= secondary_period_start) &
        (cb_df["Date"] <= secondary_period_end)
    ]
    secondary_period_total_cb_cost = secondary_period_cb_cost_df["Amount in USD"].sum()

    # Calculate variation
    variation = primary_period_total_cb_cost - secondary_period_total_cb_cost

    response_parts = []
    response_parts.append(f"C&B Cost for {primary_period_desc}: **${primary_period_total_cb_cost:,.2f}**")
    response_parts.append(f"C&B Cost for {secondary_period_desc}: **${secondary_period_total_cb_cost:,.2f}**")
    
    if variation > 0:
        response_parts.append(f"The C&B Cost **increased** by **${variation:,.2f}** from {secondary_period_desc} to {primary_period_desc}.")
    elif variation < 0:
        response_parts.append(f"The C&B Cost **decreased** by **${abs(variation):,.2f}** from {secondary_period_desc} to {primary_period_desc}.")
    else:
        response_parts.append(f"The C&B Cost remained **unchanged** from {secondary_period_desc} to {primary_period_desc}.")
    
    # Return structured data for plotting in mainn.py
    return {
        "message": "\n\n".join(response_parts),
        "primary_cost": primary_period_total_cb_cost,
        "secondary_cost": secondary_period_total_cb_cost,
        "primary_desc": primary_period_desc,
        "secondary_desc": secondary_period_desc,
        "variation": variation
    }



# ---------- C&B REVENUE TREND ANALYSIS ----------
def calculate_cb_revenue_trend(df_pl, query_details):
    """
    Calculates the monthly trend of C&B Cost, Total Revenue, their difference,
    and the ratio of (C&B Cost - Total Revenue) to C&B Cost.
    Handles explicit date ranges (FY, year, quarter) or defaults to the last 12 months.
    Returns a DataFrame with relevant monthly data.
    """
    df_copy = df_pl.copy()

    # Ensure the 'Date' column in df_copy is datetime type before filtering
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
    df_copy.dropna(subset=['Date'], inplace=True) # Drop rows where Date conversion failed

    # Determine the date range
    start_date = query_details.get("start_date")
    end_date = query_details.get("end_date")

    if not start_date or not end_date:
        # Default to last 12 months if no explicit date filter
        today = datetime.now(pytz.timezone("Asia/Kolkata"))
        # Set end_date to the last day of the current month
        end_date = datetime(today.year, today.month, calendar.monthrange(today.year, today.month)[1], 23, 59, 59, 999999)
        # Set start_date to the first day of the month 11 months ago from the current month
        start_date = (today - pd.DateOffset(months=11)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Convert start_date and end_date to pandas Timestamps for robust comparison and ensure they are naive
    start_date_ts = pd.Timestamp(start_date).tz_localize(None)
    end_date_ts = pd.Timestamp(end_date).tz_localize(None)

    # Use .between() method for robust date filtering
    df_filtered = df_copy[
        df_copy['Date'].between(start_date_ts, end_date_ts, inclusive='both')
    ].copy()

    if df_filtered.empty:
        return pd.DataFrame(columns=['Month', 'CB_Cost', 'Total_Revenue', 'CB_Revenue_Difference', 'CB_Cost_vs_Revenue_Ratio_Percent'])

    # Extract month and year for grouping
    df_filtered['Month_Year'] = df_filtered['Date'].dt.to_period('M')

    # Calculate C&B Cost and Total Revenue per month
    monthly_data = df_filtered.groupby('Month_Year').apply(lambda x: pd.Series({
        'CB_Cost': x[x["Group Description"].isin(CB_COST_GROUPS)]["Amount in USD"].sum(),
        'Total_Revenue': x[x["Group1"].isin(REVENUE_GROUPS)]["Amount in USD"].sum()
    })).reset_index()

    # Convert Month_Year Period to datetime for sorting and display
    monthly_data['Month'] = monthly_data['Month_Year'].dt.to_timestamp()
    monthly_data = monthly_data.sort_values(by='Month').reset_index(drop=True)

    # Calculate the difference between Total Revenue and C&B Cost
    monthly_data['CB_Revenue_Difference'] = monthly_data['Total_Revenue'] - monthly_data['CB_Cost']
    
    # Calculate the new KPI: (C&B cost - revenue) / C&B * 100
    # Handle division by zero for CB_Cost
    monthly_data['CB_Cost_vs_Revenue_Ratio_Percent'] = np.where(
        monthly_data['CB_Cost'] != 0,
        ((monthly_data['CB_Cost'] - monthly_data['Total_Revenue']) / monthly_data['CB_Cost']) * 100,
        np.nan # Assign NaN if CB_Cost is zero to avoid division by zero errors
    )

    # Format Month for display
    monthly_data['Month'] = monthly_data['Month'].dt.strftime('%Y-%m')

    return monthly_data[['Month', 'CB_Cost', 'Total_Revenue', 'CB_Revenue_Difference', 'CB_Cost_vs_Revenue_Ratio_Percent']]
#testing
