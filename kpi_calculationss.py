import pandas as pd
from openai import AzureOpenAI
from datetime import datetime, timedelta
import pytz
import json
from dateutil.parser import parse
import calendar
import os
import numpy as np
import streamlit as st # Import streamlit for caching


# ---------- CONFIGURATION ----------
AZURE_OPENAI_ENDPOINT = "https://openaipbichatbott.openai.azure.com/"
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEYY")
AZURE_OPENAI_DEPLOYMENT = "gpt-35-turbo"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

# Initialize OpenAI client globally
try:
    openai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
except Exception as e:
    print(f"ERROR: Failed to initialize AzureOpenAI client: {e}")
    print("Please ensure AZURE_OPENAI_KEYY and AZURE_OPENAI_ENDPOINT are correctly set in your environment variables or .env file.")
    openai_client = None


# Define these globally as they are constant
REVENUE_GROUPS = ["ONSITE", "OFFSHORE", "INDIRECT REVENUE"]
COST_GROUPS = [
    "Direct Expense", "OWN OVERHEADS", "Indirect Expense",
    "Project Level Depreciation", "Direct Expense - DU Block Seats Allocation",
    "Direct Expense - DU Pool Allocation", "Establishment Expenses"
]
CB_COST_GROUPS = ["C&B Cost Onsite", "C&B Cost Offshore"]

# Define Fresher Ageing Categories for the new UT trend
FRESHER_AGEING_CATEGORIES = [
    "Freshers ET(0-3 Months)",
    "Freshers ET(4-6 Months)",
    "Freshers PGET(0-3 Months)",
    "Freshers ETPremium(0-3 Months)"
]
# Apply strip to each element to ensure no hidden leading/trailing spaces in the list itself
FRESHER_AGEING_CATEGORIES = [cat.strip() for cat in FRESHER_AGEING_CATEGORIES]

# Define columns for which distinct values will be pre-computed for filtering
# These lists are used to validate and correct AI's column mapping for filters
FILTERABLE_PL_COLUMNS = [
    "Segment", "Exec DU", "Exec DG", "PVDG", "PVDU", "FinalCustomerName", "Group Description" ,"Company_Code"
]
FILTERABLE_UT_COLUMNS = [
    "Exec DU", "Exec DG", "FinalCustomerName", "FresherAgeingCategory"
]

# --- HARDCODED DISTINCT VALUES FOR P&L COLUMNS (PROVIDED BY USER) ---
# Ensure all values are lowercased and stripped for consistent matching
PROVIDED_PL_DISTINCT_VALUES = {
    "Segment": {
        "plant engineering", "transportation", "media & technology",
        "industrial products", "telecom & hitech", "medical & life science", "med tech"
    },
    "Exec DU": {
        "fmcg", "pscg", "product lifecycle mgmt", "mechanical services",
        "smart manufacturing & sourcing", "integrated asset mgmt",
        "digital pe & mobility", "analytics & ai", "cyber secure",
        "si - enterprises - ltts", "digital ip & medical",
        "embedded hw & sw engineering", "v&v - m&t and ip",
        "v&v - medical", "v&v - transportation", "aerospace & rail",
        "digital comms. & tech", "media", "vlsi design",
        "snp - digital manufacturing services", "llc pe offsite center",
        "si - enterprises", "llc dms ip & transportation",
        "llc dms medical & contract mfg", "llc mechanical",
        "llc 1 digital place", "digital ip & pe & off-highway",
        "pscg", "oil & gas", "chemicals", "exxon mobil", "power & utilities",
        "energy", "building technology"
    },
    "Exec DG": {
        "fmcg", "digital manufacturing services", "integrated asset mgmt",
        "digital products & services", "system integration",
        "embedded engineering and v&v group", "aerospace & rail",
        "media", "llc manufacturing engineering", "llc product engineering",
        "o&g and chemicals", "plant engineering", "electrical, controls, power & utility",
        "energy & automation", "building technology"
    },
    "PVDG": {
        "fmcg", "media", "aerospace & rail" # Updated PVDG values
    },
    "PVDU": {
        "fmcg", "pscg", "aerospace & rail", "media",
        "pscg" # Duplicated PSCG, keeping as provided for now.
    },
    # Removed FinalCustomerName from hardcoded list as per user's feedback
    # It will now be dynamically extracted from the DataFrame.
    "Group Description": {
        "foreign travel expenses", "medical insurance", "visa cost",
        "onsite salaries & allowances", "inland travel & conveyance",
        "cost of onsite tpcs/retainers", "domestic travel & conveyance - billable",
        "c&b cost offshore", "professional fee - retainers/tpc",
        "b3. computer software costs", "training expenses",
        "sales promotion expenses - others", "recruitment expenses",
        "professional fees", "miscellaneous",
        "inland travel & conveyance - non billable",
        "foreign travel & conveyance - non billable",
        "domestic travel & conveyance - non billable",
        "bu support cost allocation", "elimination cost",
        "intercompany cost", "purchase of goods/purchase of license",
        "manufacturing cost - subcontracting", "components & subcontracting cost",
        "establishment expenses - allocated admin",
        "establishment expenses - allocated - it", "establishment expenses - own",
        "f1.total costs - direct expenses - onsite - for du buffer allocation",
        "f2.total costs - direct expenses - offshore - for du buffer allocation",
        "project level depreciation", "subcontracting/component revenue",
        "sales of goods/licenses", "intercompany revenue", "elimination revenue",
        "offshore", "onsite"
    },
    "Company_Code": {
        "tsin", "tsus", "tslu", "tsjp", "tsfr", "tsgb", "tsca", "tsde", "tsau", "tsbe", "tsnl", "tsse",
        "tspl", "tsae", "tsfi", "tsil", "tsti", "tslc", "tsit", "tsch", "tssg", "tssa"
    }
}
# --- END HARDCODED DISTINCT VALUES ---


def parse_percentage(val):
    """
    Converts percentage strings (e.g., "112%", "30%") or numbers (e.g., 112, 0.3, 30)
    or fractions (e.g., "3/5", "4/5") into decimal floats (e.g., 1.12, 0.3, 0.6).
    """
    try:
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return None
            
            # Handle fractions like "3/5"
            if '/' in val:
                parts = val.split('/')
                if len(parts) == 2:
                    try:
                        numerator = float(parts[0].strip())
                        denominator = float(parts[1].strip())
                        if denominator != 0:
                            return numerator / denominator
                        else:
                            return None # Division by zero
                    except ValueError:
                        pass # Not a valid numeric fraction, fall through
            
            # Handle percentages like "60%"
            if '%' in val:
                val = val.replace("%", "").strip()
                numeric_val = float(val)
                return numeric_val / 100.0
            
            # Handle plain numbers as strings (e.g., "0.3", "112")
            numeric_val = float(val)
            return numeric_val
            
        elif isinstance(val, (int, float)):
            return float(val)
            
        else: # Not a string, int, or float
            return None
            
    except (ValueError, TypeError): # Catch conversion errors or if val is None
        return None

# --- NEW HELPER FUNCTIONS FOR FISCAL QUARTER LOGIC ---
def get_fiscal_quarter_and_year(date_obj):
    """
    Determines the fiscal year (calendar year the FY starts in) and fiscal quarter (1-4) for a given date.
    L&T Fiscal Year: April 1st to March 31st.
    """
    month = date_obj.month
    year = date_obj.year
    
    if 4 <= month <= 6: # Apr-Jun (FY-Q1)
        fiscal_quarter = 1
        fiscal_year_start_cal_year = year 
    elif 7 <= month <= 9: # Jul-Sep (FY-Q2)
        fiscal_quarter = 2
        fiscal_year_start_cal_year = year
    elif 10 <= month <= 12: # Oct-Dec (FY-Q3)
        fiscal_quarter = 3
        fiscal_year_start_cal_year = year
    else: # Jan-Mar (FY-Q4)
        fiscal_quarter = 4
        fiscal_year_start_cal_year = year - 1 # FY started in previous calendar year
    
    return fiscal_year_start_cal_year, fiscal_quarter

def get_fiscal_quarter_name_from_fy_and_q(fiscal_year_start_cal_year, fiscal_quarter):
    """
    Formates the fiscal year (calendar year the FY starts in) and quarter into FYXXXXQY format.
    """
    # The FY number is the calendar year the FY starts in + 1.
    # E.g., FY26 starting Apr 2025 is FY26.
    return f"FY{fiscal_year_start_cal_year + 1}Q{fiscal_quarter}"

def get_fiscal_quarter_dates(fy_start_cal_year, fq):
    """Returns (start_date, end_date) for a given fiscal quarter."""
    if fq == 1: # Apr-Jun
        start_date = datetime(fy_start_cal_year, 4, 1, 0, 0, 0, 0)
        end_date = datetime(fy_start_cal_year, 6, 30, 23, 59, 59, 999999)
    elif fq == 2: # Jul-Sep
        start_date = datetime(fy_start_cal_year, 7, 1, 0, 0, 0, 0)
        end_date = datetime(fy_start_cal_year, 9, 30, 23, 59, 59, 999999)
    elif fq == 3: # Oct-Dec
        start_date = datetime(fy_start_cal_year, 10, 1, 0, 0, 0, 0)
        end_date = datetime(fy_start_cal_year, 12, 31, 23, 59, 59, 999999)
    else: # Jan-Mar (of next calendar year for the same FY)
        start_date = datetime(fy_start_cal_year + 1, 1, 1, 0, 0, 0, 0)
        end_date = datetime(fy_start_cal_year + 1, 3, 31, 23, 59, 59, 999999)
    return start_date, end_date

def get_previous_fiscal_quarter(fy_start_cal_year, fq):
    """Returns (prev_fy_start_cal_year, prev_fq) for the previous fiscal quarter."""
    if fq == 1: # If current is Q1, previous was Q4 of prior FY
        return fy_start_cal_year - 1, 4
    else:
        return fy_start_cal_year, fq - 1


# ---------- DATE RANGE PARSING (REVISED FOR REALIZED RATE) ----------
@st.cache_data(show_spinner="Parsing date range with AI...") # Cache this function
def parse_date_range_from_query(query):
    """
    Uses GPT to extract precise date ranges.
    Handles single periods (explicit FY/Q, relative terms) and comparison periods.
    """
    india_tz = pytz.timezone("Asia/Kolkata")
    today = datetime.now(india_tz).date() 
    
    # --- NEW: Specific prompts for Realized Rate Drop to guide LLM ---
    realized_rate_prompt_base = "which are the accounts where the realized rate dropped more than"
    
    # Check for "this quarter" variant
    if realized_rate_prompt_base in query.lower() and "in this quarter" in query.lower():
        current_fy_start_cal_year, current_fq = get_fiscal_quarter_and_year(today)
        current_q_start_date, current_q_end_date = get_fiscal_quarter_dates(current_fy_start_cal_year, current_fq)
        
        prev_fy_start_cal_year, prev_fq = get_previous_fiscal_quarter(current_fy_start_cal_year, current_fq)
        prev_q_start_date, prev_q_end_date = get_fiscal_quarter_dates(prev_fy_start_cal_year, prev_fq)
        
        current_q_name = get_fiscal_quarter_name_from_fy_and_q(current_fy_start_cal_year, current_fq)
        prev_q_name = get_fiscal_quarter_name_from_fy_and_q(prev_fy_start_cal_year, prev_fq)

       # st.write(f"DEBUG: Realized Rate Drop - 'this quarter' detected. Primary: {current_q_name}, Secondary: {prev_q_name}")
        return {
            "date_filter": True,
            "start_date": current_q_start_date,
            "end_date": current_q_end_date,
            "secondary_start_date": prev_q_start_date,
            "secondary_end_date": prev_q_end_date,
            "description": f"{current_q_name} vs {prev_q_name}",
            "relative_period_detected": "realized_rate_this_quarter",
            "num_periods": None
        }

    # Check for "previous quarter" variant
    elif realized_rate_prompt_base in query.lower() and "in previous quarter" in query.lower():
        current_fy_start_cal_year, current_fq = get_fiscal_quarter_and_year(today)
        
        # Last completed fiscal quarter is the "primary" for this query
        last_completed_fy_start_cal_year, last_completed_fq = get_previous_fiscal_quarter(current_fy_start_cal_year, current_fq)
        last_completed_q_start_date, last_completed_q_end_date = get_fiscal_quarter_dates(last_completed_fy_start_cal_year, last_completed_fq)

        # The quarter before the last completed one is the "secondary"
        prev_to_last_completed_fy_start_cal_year, prev_to_last_completed_fq = get_previous_fiscal_quarter(last_completed_fy_start_cal_year, last_completed_fq)
        prev_to_last_completed_q_start_date, prev_to_last_completed_q_end_date = get_fiscal_quarter_dates(prev_to_last_completed_fy_start_cal_year, prev_to_last_completed_fq)
        
        primary_q_name = get_fiscal_quarter_name_from_fy_and_q(last_completed_fy_start_cal_year, last_completed_fq)
        secondary_q_name = get_fiscal_quarter_name_from_fy_and_q(prev_to_last_completed_fy_start_cal_year, prev_to_last_completed_fq)

        #st.write(f"DEBUG: Realized Rate Drop - 'previous quarter' detected. Primary: {primary_q_name}, Secondary: {secondary_q_name}")
        return {
            "date_filter": True,
            "start_date": last_completed_q_start_date,
            "end_date": last_completed_q_end_date,
            "secondary_start_date": prev_to_last_completed_q_start_date,
            "secondary_end_date": prev_to_last_completed_q_end_date,
            "description": f"{primary_q_name} vs {secondary_q_name}",
            "relative_period_detected": "realized_rate_previous_quarter",
            "num_periods": None
        }

    # Check for explicit FYXXXXQY mention (e.g., "in FY2026q1")
    import re
    explicit_quarter_match = re.search(r'in\s+(FY\d{2,4}Q[1-4])', query, re.IGNORECASE)
    if realized_rate_prompt_base in query.lower() and explicit_quarter_match:
        explicit_quarter_str = explicit_quarter_match.group(1).upper() # e.g., "FY2026Q1"

        # Parse the explicit quarter string into fiscal year and quarter number
        # Example: FY2026Q1 -> fy_num=2025, fq=1
        fy_num_str = explicit_quarter_str[2:-2] # "2026"
        fq_str = explicit_quarter_str[-1] # "1"
        
        # Adjust fy_start_cal_year based on the fiscal year definition (FY26 starts in 2025)
        explicit_fy_start_cal_year = int(fy_num_str) - 1 
        explicit_fq = int(fq_str)

        # Primary period is the explicit quarter
        primary_q_start_date, primary_q_end_date = get_fiscal_quarter_dates(explicit_fy_start_cal_year, explicit_fq)
        
        # Secondary period is the quarter immediately preceding the explicit one
        secondary_fy_start_cal_year, secondary_fq = get_previous_fiscal_quarter(explicit_fy_start_cal_year, explicit_fq)
        secondary_q_start_date, secondary_q_end_date = get_fiscal_quarter_dates(secondary_fy_start_cal_year, secondary_fq)
        
        primary_q_name = get_fiscal_quarter_name_from_fy_and_q(explicit_fy_start_cal_year, explicit_fq)
        secondary_q_name = get_fiscal_quarter_name_from_fy_and_q(secondary_fy_start_cal_year, secondary_fq)

        #st.write(f"DEBUG: Realized Rate Drop - explicit quarter '{explicit_quarter_str}' detected. Primary: {primary_q_name}, Secondary: {secondary_q_name}")
        return {
            "date_filter": True,
            "start_date": primary_q_start_date,
            "end_date": primary_q_end_date,
            "secondary_start_date": secondary_q_start_date,
            "secondary_end_date": secondary_q_end_date,
            "description": f"{primary_q_name} vs {secondary_q_name}",
            "relative_period_detected": "realized_rate_explicit_quarter",
            "num_periods": None
        }
    # --- END NEW: Specific prompts for Realized Rate Drop to guide LLM ---

    # Original system prompt for general date parsing (fallback if no specific Realized Rate prompt matched)
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
- 'relative_period_detected': "last quarter", "last year", "this year", "this month", "current date", "last_and_previous_month", "last_to_this_quarter", "explicit_quarter", "explicit_fiscal_year", "explicit_comparison", "last_x_quarters", "last_x_years", "last_x_months" or null
- 'num_periods': integer (only if 'last_x_quarters', 'last_x_years', 'last_x_months' is detected) or null

**PRIORITY RULE FOR COMPARISONS (VERY IMPORTANT):**
If the query explicitly mentions two distinct fiscal quarters for comparison (e.g., "FY26Q1 and FY25Q4", "FY25Q4 vs FY26Q1", "(FY26Q1 - FY25Q4)"), you MUST identify *both* of these quarters and set them as the primary and secondary periods.
**The LATER of the two quarters should ALWAYS be assigned to `start_date` and `end_date` (primary period).**
**The EARLIER of the two quarters should ALWAYS be assigned to `secondary_start_date` and `secondary_end_date` (secondary/comparison period).**
This ensures consistency for calculations like rate drops (Current - Previous).
Do NOT derive quarters based on 'today' if explicit quarters are provided for comparison.

Rules for Date Interpretation:
1.  **Single Periods (Explicit Fiscal):** If the query explicitly mentions a *single Fiscal Year and Quarter* (e.g., "FY26-Q1" or "FY2025 Q4"), calculate the exact start and end dates for *that single fiscal quarter*. Assign these to `start_date`/`end_date`. Set `secondary_start_date`/`secondary_end_date` to `null`. Set `relative_period_detected` to "explicit_quarter". The `description` should be like "FY26 Q1 (Apr 1, 2025 - Jun 30, 2025)".
2.  **Single Periods (Explicit Calendar):** If the query explicitly mentions a *single Calendar Year and Quarter* (e.g., "2025 Q1", "Q1 2025", "2025q2"), calculate the exact start and end dates for *that single calendar quarter*. Assign these to `start_date`/`end_date`. Set `secondary_start_date`/`secondary_end_date` to `null`. Set `relative_period_detected` to "explicit_quarter". The `description` should be like "2025 Q2 (Apr 1, 2025 - Jun 30, 2025)".
3.  **Single Periods (Explicit Fiscal Year):** If the query explicitly mentions a *single Fiscal Year* (e.g., "FY26" or "FY2025"), calculate the exact start and end dates for *that single fiscal year*. Assign these to `start_date`/`end_date`. Set `secondary_start_date`/`secondary_end_date` to `null`. Set `relative_period_detected` to "explicit_fiscal_year". The `description` should be like "FY26 (Apr 1, 2025 - Mar 31, 2026)".
4.  **Relative Quarters:** If the query asks for *only* "last quarter" or "this quarter" (without a comparison keyword like "to" or "vs"), this refers to the *last completed fiscal quarter* or *current fiscal quarter* respectively. Set `relative_period_detected` to "last quarter" or "this quarter" and ensure `secondary_start_date`/`secondary_end_date` are null.
5.  **Comparison Periods (Explicit Fiscal):** If the query asks for a comparison between *two explicit Fiscal Years/Quarters* (e.g., "FY25Q4 and FY26Q1", "FY25Q4 vs FY26Q1", "(FY26Q1 - FY25Q4)"), calculate dates for both. Assign the *later* period to `start_date`/`end_date` (primary) and the *earlier* period to `secondary_start_date`/`secondary_end_date` (secondary). Set `relative_period_detected` to "explicit_comparison". The `description` should be like "FY26 Q1 (Apr 1, 2025 - Jun 30, 2025) vs FY25 Q4 (Jan 1, 2025 - Mar 31, 2025)".
6.  **Comparison Periods (Explicit Calendar):** If the query asks for a comparison between *two explicit Calendar Years/Quarters* (e.g., "2025 Q1 and 2025 Q2", "2025q1 to 2025q2"), calculate dates for both. Assign the *later* period to `start_date`/`end_date` (primary) and the *earlier* period to `secondary_start_date`/`secondary_end_date` (secondary). Set `relative_period_detected` to "explicit_comparison". The `description` should be like "2025 Q2 (Apr 1, 2025 - Jun 30, 2025) vs 2025 Q1 (Jan 1, 2025 - Mar 31, 2025)".
7.  **Comparison Periods (Relative Quarters):** If the query explicitly asks for a comparison like "last quarter to this quarter" *using comparison keywords* (e.g., "last quarter to this quarter", "this quarter vs last quarter"), `start_date`/`end_date` should be the current *fiscal* quarter, and `secondary_start_date`/`secondary_end_date` should be the previous completed *fiscal* quarter. Set `relative_period_detected` to "last_to_this_quarter".
8.  For relative periods (like "last month", "last year" *without explicit FY/Q or comparison keywords*), calculate exact dates based on today's date. If *only* relative terms are used, then set `relative_period_detected` accordingly.
9.  For absolute periods (like "January 2023"), use exact dates.
10. For ranges (like "from March to May 2023"), use exact start/end dates.
11. If the query asks for a comparison between two specific consecutive months (e.g., "May and June", "last month and previous month"), identify `start_date`/`end_date` as the later month and `secondary_start_date`/`secondary_end_date` as the earlier month. Set `relative_period_detected` to "last_and_previous_month" if detected by relative terms.
12. **Last X Periods:** If the query asks for "last X quarters", "last X years", or "last X months" (e.g., "last 2 quarters", "last 3 years", "last 6 months"):
    * Set `relative_period_detected` to "last_x_quarters", "last_x_years", or "last_x_months" respectively.
    * Calculate `start_date` and `end_date` for this period. `end_date` should be the end of the *last completed period* (e.g., if today is July 28, 2025 and query is "last 2 quarters", `end_date` should be Jun 30, 2025, and `start_date` should be Jan 1, 2025).
    * Set `secondary_start_date`/`secondary_end_date` to `null`.
    * Extract the number 'X' into a new field `num_periods`.
13. If no date filter, set date_filter=false and return null for dates.
"""
    try:
        if openai_client is None: # Check if client initialized successfully
            print("ERROR: OpenAI client not initialized. Cannot parse date range.")
            return {"date_filter": False, "start_date": None, "end_date": None, "secondary_start_date": None, "secondary_end_date": None, "description": "all available data", "relative_period_detected": None, "num_periods": None}

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
        
        # Extract num_periods if present
        num_periods = result.get("num_periods")

        # Convert date strings to datetime objects (NOT date objects)
        for key in ["start_date", "end_date", "secondary_start_date", "secondary_end_date"]:
            if result.get(key):
                try:
                    # Convert to datetime object (not .date()) and ensure end of day
                    dt_obj = parse(result[key])
                    # Ensure the datetime objects are timezone-naive
                    if dt_obj.tzinfo is not None:
                        dt_obj = dt_obj.replace(tzinfo=None)

                    if key in ["end_date", "secondary_end_date"]:
                        result[key] = datetime.combine(dt_obj.date(), datetime.max.time())
                    else:
                        result[key] = datetime.combine(dt_obj.date(), datetime.min.time()) 
                except Exception as e:
                    print(f"WARNING: Date parsing error for key '{key}': {e}. Value: '{result[key]}'")
                    result[key] = None
        
        # --- Python Logic to refine LLM's output based on `relative_period_detected` ---
        # This ensures correct date ranges and nulls secondary periods when not needed.
        
        relative_period = result.get('relative_period_detected')
        today_dt = datetime.now(india_tz) # Use datetime object for date arithmetic

        # Handle explicit single quarter/fiscal year or explicit comparison
        if relative_period in ["explicit_quarter", "explicit_fiscal_year"]:
            result["secondary_start_date"] = None
            result["secondary_end_date"] = None
            result["date_filter"] = True
            # Description is usually good from LLM for explicit periods
            return result
        elif relative_period == "explicit_comparison":
            result["date_filter"] = True
            # Description is usually good from LLM for explicit comparisons
            return result
        elif relative_period in ["last_x_quarters", "last_x_years", "last_x_months"]:
            # Recalculate start_date, end_date, and description for "last X periods"
            # to ensure accuracy and consistent phrasing.
            
            end_of_last_completed_period = None
            start_of_period = None

            if relative_period == "last_x_months":
                # End of last completed month
                end_of_last_completed_period = (today_dt.replace(day=1) - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
                start_of_period = (end_of_last_completed_period.replace(day=1) - pd.DateOffset(months=num_periods - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
                result["description"] = f"last {num_periods} months ({start_of_period.strftime('%B %d, %Y')} - {end_of_last_completed_period.strftime('%B %d, %Y')})"
            
            elif relative_period == "last_x_quarters":
               # st.write(f"DEBUG: Current system date (today_dt): {today_dt.strftime('%Y-%m-%d')}")
                
                # Determine the end of the last completed fiscal quarter
                current_fy_start_cal_year, current_fq = get_fiscal_quarter_and_year(today_dt.date())
                
                # Calculate the end date of the last completed quarter
                if current_fq == 1: # Today is in FY-Q1 (Apr-Jun), last completed was FY-Q4 of previous FY
                    end_of_last_completed_period = datetime(current_fy_start_cal_year, 3, 31, 23, 59, 59, 999999)
                elif current_fq == 2: # Today is in FY-Q2 (Jul-Sep), last completed was FY-Q1 of current FY
                    end_of_last_completed_period = datetime(current_fy_start_cal_year, 6, 30, 23, 59, 59, 999999)
                elif current_fq == 3: # Today is in FY-Q3 (Oct-Dec), last completed was FY-Q2 of current FY
                    end_of_last_completed_period = datetime(current_fy_start_cal_year, 9, 30, 23, 59, 59, 999999)
                else: # current_fq == 4 (Today is in FY-Q4, Jan-Mar), last completed was FY-Q3 of current FY
                    end_of_last_completed_period = datetime(current_fy_start_cal_year, 12, 31, 23, 59, 59, 999999)
                
               # st.write(f"DEBUG: Calculated end_of_last_completed_period (datetime): {end_of_last_completed_period.strftime('%Y-%m-%d')}")

                # Now, calculate the start of the period by subtracting quarters from the end_of_last_completed_period
                # This uses pd.DateOffset for robust quarter subtraction
                start_of_period = end_of_last_completed_period - pd.DateOffset(months=3 * (num_periods - 1))
                # Ensure it's the first day of the quarter
                start_of_period = start_of_period.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                
                # Adjust start_of_period to the correct quarter start if it landed mid-quarter due to DateOffset
                start_fy_cal_year_temp, start_fq_temp = get_fiscal_quarter_and_year(start_of_period.date())
                if start_fq_temp == 1:
                    start_of_period = datetime(start_fy_cal_year_temp, 4, 1, 0, 0, 0, 0)
                elif start_fq_temp == 2:
                    start_of_period = datetime(start_fy_cal_year_temp, 7, 1, 0, 0, 0, 0)
                elif start_fq_temp == 3:
                    start_of_period = datetime(start_fy_cal_year_temp, 10, 1, 0, 0, 0, 0)
                else: # Q4
                    start_of_period = datetime(start_fy_cal_year_temp + 1, 1, 1, 0, 0, 0, 0)

                # Use the new helper functions for fiscal quarter names
                start_fy_cal_year, start_fq = get_fiscal_quarter_and_year(start_of_period.date())
                end_fy_cal_year, end_fq = get_fiscal_quarter_and_year(end_of_last_completed_period.date())

                start_quarter_name = get_fiscal_quarter_name_from_fy_and_q(start_fy_cal_year, start_fq)
                end_quarter_name = get_fiscal_quarter_name_from_fy_and_q(end_fy_cal_year, end_fq)
                
                # st.write(f"DEBUG: Final start_of_period (datetime): {start_of_period.strftime('%Y-%m-%d')}")
                # st.write(f"DEBUG: Final end_of_last_completed_period (datetime): {end_of_last_completed_period.strftime('%Y-%m-%d')}")
                # st.write(f"DEBUG: Formatted start_quarter_name: {start_quarter_name}")
                # st.write(f"DEBUG: Formatted end_quarter_name: {end_quarter_name}")


                # Explicitly set the description based on calculated fiscal quarter names and their dates
                result["description"] = (
                    f"last {num_periods} fiscal quarters "
                    f"({start_quarter_name} ({start_of_period.strftime('%b %d, %Y')}) - "
                    f"{end_quarter_name} ({end_of_last_completed_period.strftime('%b %d, %Y')}))"
                )
            
            elif relative_period == "last_x_years":
                # Determine the end of the last completed fiscal year (March 31st)
                current_fiscal_year_end_date = datetime(today_dt.year if today_dt.month >= 4 else today_dt.year - 1, 3, 31, 23, 59, 59, 999999)
                
                if today_dt.date() > current_fiscal_year_end_date.date():
                    end_of_last_completed_period = current_fiscal_year_end_date
                else:
                    end_of_last_completed_period = datetime(today_dt.year - 1, 3, 31, 23, 59, 59, 999999)
                
                start_of_period = datetime(end_of_last_completed_period.year - (num_periods - 1), 4, 1, 0, 0, 0, 0) # Start of fiscal year
                
                # Fiscal year numbers are current calendar year + 1 if month >= 4, else current calendar year
                start_fiscal_year_num = start_of_period.year + 1 if start_of_period.month >= 4 else start_of_period.year
                end_fiscal_year_num = end_of_last_completed_period.year + 1 if end_of_last_completed_period.month >= 4 else end_of_last_completed_period.year

                result["description"] = f"last {num_periods} fiscal years (FY{start_fiscal_year_num} - FY{end_fiscal_year_num})"

            result["start_date"] = start_of_period
            result["end_date"] = end_of_last_completed_period
            result["secondary_start_date"] = None
            result["secondary_end_date"] = None
            result["date_filter"] = True
            result["num_periods"] = num_periods # Ensure num_periods is passed through
            return result


        # This is the "last quarter" specific calculation block. It needs to be entered when
        # relative_period is explicitly "last quarter" and NOT "last_to_this_quarter"
        if relative_period == 'last quarter': 
            current_cal_month = today.month
            current_cal_year = today.year

            # Use the new helper to get the fiscal year and quarter for today
            current_fy_start_cal_year, current_fq = get_fiscal_quarter_and_year(today)
            
            last_completed_q_start_month = None
            last_completed_q_end_month = None
            last_completed_q_cal_year = None
            
            # Determine the last completed fiscal quarter based on current_fq
            if current_fq == 1: # Today is in FY-Q1 (Apr-Jun), last completed was FY-Q4 of previous FY
                last_completed_q_start_month = 1
                last_completed_q_end_month = 3
                last_completed_q_cal_year = current_fy_start_cal_year # This is the calendar year of Jan-Mar
                last_completed_q_fy_start_cal_year = current_fy_start_cal_year - 1 # FY started in previous calendar year
                last_completed_q_fq = 4
            elif current_fq == 2: # Today is in FY-Q2 (Jul-Sep), last completed was FY-Q1 of current FY
                last_completed_q_start_month = 4
                last_completed_q_end_month = 6
                last_completed_q_cal_year = current_fy_start_cal_year # This is the calendar year of Apr-Jun
                last_completed_q_fy_start_cal_year = current_fy_start_cal_year
                last_completed_q_fq = 1
            elif current_fq == 3: # Today is in FY-Q3 (Oct-Dec), last completed was FY-Q2 of current FY
                last_completed_q_start_month = 7
                last_completed_q_end_month = 9
                last_completed_q_cal_year = current_fy_start_cal_year # This is the calendar year of Jul-Sep
                last_completed_q_fy_start_cal_year = current_fy_start_cal_year
                last_completed_q_fq = 2
            else: # current_fq == 4 (Today is in FY-Q4, Jan-Mar), last completed was FY-Q3 of current FY
                last_completed_q_start_month = 10
                last_completed_q_end_month = 12
                last_completed_q_cal_year = current_fy_start_cal_year # This is the calendar year of Oct-Dec
                last_completed_q_fy_start_cal_year = current_fy_start_cal_year
                last_completed_q_fq = 3

            last_completed_q_name = get_fiscal_quarter_name_from_fy_and_q(last_completed_q_fy_start_cal_year, last_completed_q_fq)

            result["start_date"] = datetime(last_completed_q_cal_year, last_completed_q_start_month, 1)
            result["end_date"] = datetime(last_completed_q_cal_year, last_completed_q_end_month, calendar.monthrange(last_completed_q_cal_year, last_completed_q_end_month)[1], 23, 59, 59, 999999)
            result["date_filter"] = True
            result["description"] = f"last quarter ({last_completed_q_name})"
            result["secondary_start_date"] = None 
            result["secondary_end_date"] = None
            
        elif relative_period == 'last_to_this_quarter':
            current_cal_year = today.year
            current_cal_month = today.month
            
            # Use the new helper to get current fiscal quarter info
            curr_fy_start_cal_year, curr_fq = get_fiscal_quarter_and_year(today)
            curr_q_name = get_fiscal_quarter_name_from_fy_and_q(curr_fy_start_cal_year, curr_fq)

            # Determine start and end dates for current quarter
            if curr_fq == 1: # FY-Q1 (Apr-Jun)
                curr_q_start_month, curr_q_end_month = 4, 6
                curr_q_cal_year_for_dates = curr_fy_start_cal_year
            elif curr_fq == 2: # FY-Q2 (Jul-Sep)
                curr_q_start_month, curr_q_end_month = 7, 9
                curr_q_cal_year_for_dates = curr_fy_start_cal_year
            elif curr_fq == 3: # FY-Q3 (Oct-Dec)
                curr_q_start_month, curr_q_end_month = 10, 12
                curr_q_cal_year_for_dates = curr_fy_start_cal_year
            else: # FY-Q4 (Jan-Mar)
                curr_q_start_month, curr_q_end_month = 1, 3
                curr_q_cal_year_for_dates = curr_fy_start_cal_year + 1 # Q4 is in the next calendar year

            result["start_date"] = datetime(curr_q_cal_year_for_dates, curr_q_start_month, 1)
            result["end_date"] = datetime(curr_q_cal_year_for_dates, curr_q_end_month, calendar.monthrange(curr_q_cal_year_for_dates, curr_q_end_month)[1], 23, 59, 59, 999999)
            
            # Determine previous fiscal quarter info
            prev_fy_start_cal_year = curr_fy_start_cal_year
            prev_fq = curr_fq - 1
            if prev_fq == 0: # If current is Q1, previous was Q4 of prior FY
                prev_fq = 4
                prev_fy_start_cal_year -= 1
            prev_q_name = get_fiscal_quarter_name_from_fy_and_q(prev_fy_start_cal_year, prev_fq)

            # Determine start and end dates for previous quarter
            if prev_fq == 1: # FY-Q1 (Apr-Jun)
                prev_q_start_month, prev_q_end_month = 4, 6
                prev_q_cal_year_for_dates = prev_fy_start_cal_year
            elif prev_fq == 2: # FY-Q2 (Jul-Sep)
                prev_q_start_month, prev_q_end_month = 7, 9
                prev_q_cal_year_for_dates = prev_fy_start_cal_year
            elif prev_fq == 3: # FY-Q3 (Oct-Dec)
                prev_q_start_month, prev_q_end_month = 10, 12
                prev_q_cal_year_for_dates = prev_fy_start_cal_year
            else: # FY-Q4 (Jan-Mar)
                prev_q_start_month, prev_q_end_month = 1, 3
                prev_q_cal_year_for_dates = prev_fy_start_cal_year + 1

            result["secondary_start_date"] = datetime(prev_q_cal_year_for_dates, prev_q_start_month, 1)
            result["secondary_end_date"] = datetime(prev_q_cal_year_for_dates, prev_q_end_month, calendar.monthrange(prev_q_cal_year_for_dates, prev_q_end_month)[1], 23, 59, 59, 999999)
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
    except Exception as e:
        print(f"ERROR: Date range parsing error: {e}")
        # Return a default dictionary in case of any error
        return {"date_filter": False, "start_date": None, "end_date": None, "secondary_start_date": None, "secondary_end_date": None, "description": "all available data", "relative_period_detected": None, "num_periods": None}

# ---------- NEW HELPER FUNCTION: get_distinct_column_values ----------
@st.cache_data(show_spinner="Extracting distinct column values...")
def get_distinct_column_values(df, columns_to_extract):
    """
    Extracts distinct, lowercased, and stripped values for specified columns from a DataFrame.
    Returns a dictionary where keys are column names and values are sets of distinct values.
    """
    distinct_values_map = {}
    if df is None or df.empty:
        return distinct_values_map # Return empty if df is empty

    for col in columns_to_extract:
        if col in df.columns:
            # Ensure values are string, lowercased, and stripped
            # This is the crucial part to ensure consistency
            distinct_values_map[col] = set(
                df[col].astype(str).str.lower().str.strip().dropna().unique()
            )
        else:
            print(f"WARNING: Column '{col}' not found in DataFrame for distinct value extraction.")
    return distinct_values_map

# ---------- NEW HELPER FUNCTION: apply_filters ----------
def apply_filters(df, filters, df_name="data"):
    """
    Adds a debugging print statement to show the state of the DataFrame and filters.
    Applies a dictionary of column filters to a DataFrame.
    Filters are applied case-insensitively for values.
    Returns the filtered DataFrame and a message.
    """
    #st.write(f"DEBUG: Entering apply_filters for {df_name}. Initial df shape: {df.shape}, Filters: {filters}")
    filtered_df = df.copy()
    applied_filters_list = []

    if not filters:
        #st.write(f"DEBUG: No filters provided for {df_name}.")
        return filtered_df, "No specific filters applied."

    for col, val in filters.items():
        #st.write(f"DEBUG: Applying filter: column='{col}', value='{val}'")
        if col in filtered_df.columns:
            filter_value_str = str(val).lower().strip()
            
            if pd.api.types.is_object_dtype(filtered_df[col]):
                original_rows = filtered_df.shape[0]
                filtered_df = filtered_df[
                    filtered_df[col].astype(str).str.lower().str.strip() == filter_value_str
                ]
                if filtered_df.empty:
                  #  st.write(f"DEBUG: Filter '{col}' = '{val}' resulted in empty DataFrame for {df_name}.")
                    return pd.DataFrame(columns=df.columns), f"No {df_name} found after filtering for '{col}' = '{val}'. (Original rows: {original_rows})"
                applied_filters_list.append(f"'{col}' = '{val}'")
            else:
                try:
                    if not filtered_df[col].empty:
                        converted_val = type(filtered_df[col].iloc[0])(val)
                    else:
                        converted_val = val
                    
                    original_rows = filtered_df.shape[0]
                    filtered_df = filtered_df[filtered_df[col] == converted_val]
                    if filtered_df.empty:
                   #     st.write(f"DEBUG: Filter '{col}' = '{val}' resulted in empty DataFrame for {df_name}.")
                        return pd.DataFrame(columns=df.columns), f"No {df_name} found after filtering for '{col}' = '{val}'. (Original rows: {original_rows})"
                    applied_filters_list.append(f"'{col}' = '{val}'")
                except (ValueError, TypeError):
                    original_rows = filtered_df.shape[0]
                    filtered_df = filtered_df[
                        filtered_df[col].astype(str).str.lower().str.strip() == filter_value_str
                    ]
                    if filtered_df.empty:
                  #      st.write(f"DEBUG: Filter '{col}' = '{val}' resulted in empty DataFrame for {df_name}.")
                        return pd.DataFrame(columns=df.columns), f"No {df_name} found after filtering for '{col}' = '{val}'. (Original rows: {original_rows})"
                    applied_filters_list.append(f"'{col}' = '{val}'")
        else:
        #    st.write(f"DEBUG: Filter column '{col}' not found in {df_name}.")
            return pd.DataFrame(columns=df.columns), f"Filter column '{col}' not found in {df_name}. Please check the column name."
    
    final_message = f"Filters applied: {', '.join(applied_filters_list)}." if applied_filters_list else "No specific filters applied."
 #   st.write(f"DEBUG: Exiting apply_filters for {df_name}. Final df shape: {filtered_df.shape}, Message: {final_message}")
    return filtered_df, final_message


# ---------- MODIFIED FUNCTION: calculate_cb_variation (formerly calculate_cb_cost_variation) ----------
def calculate_cb_variation(df_pl, query_details):
    """
    Calculates the variation of C&B Cost between two specified periods.
    Expected query_details to contain 'start_date', 'end_date' for the primary period
    and 'secondary_start_date', 'secondary_end_date' for the secondary period.
    Returns a pandas DataFrame with the variation details and a message string.
    """
    df_copy = df_pl.copy()

    # Apply dynamic filters first
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "P&L data")
    if df_copy.empty:
        # Ensure the DataFrame returned is empty but has the expected columns for downstream processing
        return pd.DataFrame(columns=["Metric", "Value"]), filter_message

    # Convert dates to pandas Timestamps for robust comparison and ensure they are naive
    primary_period_start = pd.Timestamp(query_details.get("start_date")).tz_localize(None) if query_details.get("start_date") else None
    primary_period_end = pd.Timestamp(query_details.get("end_date")).tz_localize(None) if query_details.get("end_date") else None
    secondary_period_start = pd.Timestamp(query_details.get("secondary_start_date")).tz_localize(None) if query_details.get("secondary_start_date") else None
    secondary_period_end = pd.Timestamp(query_details.get("secondary_end_date")).tz_localize(None) if query_details.get("secondary_end_date") else None
    
    if not all([primary_period_start, primary_period_end, secondary_period_start, secondary_period_end]):
        message = "Could not determine the two periods for C&B cost comparison from your query. Please be explicit (e.g., 'last month to this month' or 'last quarter to this quarter')."
        return pd.DataFrame(columns=["Metric", "Value"]), message

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
                # Use new helper for fiscal quarter name
                prev_fy_cal_year, prev_fq = get_fiscal_quarter_and_year(secondary_period_start.date())
                secondary_period_desc = get_fiscal_quarter_name_from_fy_and_q(prev_fy_cal_year, prev_fq)

    else: # For relative periods like "last month to this month" or "last quarter to this quarter"
        is_month_comparison = (primary_period_end - primary_period_start).days < 60 # Heuristic for month vs. quarter

        if is_month_comparison:
            primary_period_desc = primary_period_start.strftime('%B %Y')
            secondary_period_desc = secondary_period_start.strftime('%B %Y')
        else:
            # Use new helper for fiscal quarter names
            primary_fy_cal_year, primary_fq = get_fiscal_quarter_and_year(primary_period_start.date())
            secondary_fy_cal_year, secondary_fq = get_fiscal_quarter_and_year(secondary_period_start.date())
            
            primary_period_desc = get_fiscal_quarter_name_from_fy_and_q(primary_fy_cal_year, primary_fq)
            secondary_period_desc = get_fiscal_quarter_name_from_fy_and_q(secondary_fy_cal_year, secondary_fq)

    # Filter for C&B Cost data
    cb_df = df_copy[df_copy["Group Description"].isin(CB_COST_GROUPS)].copy()

    if cb_df.empty:
        message = "No C&B Cost data found for analysis after applying filters."
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=["Metric", "Value"]), message

    # Check if "Amount in USD" column exists before summing
    if "Amount in USD" not in cb_df.columns:
        message = "Error: 'Amount in USD' column not found in the filtered C&B data. Please ensure your P&L dataset has a column named 'Amount in USD' with financial figures."
        return pd.DataFrame(columns=["Metric", "Value"]), message

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
    
    percentage_variation = 0
    if secondary_period_total_cb_cost != 0:
        percentage_variation = (variation / secondary_period_total_cb_cost) * 100

    # Prepare data for the table
    data = {
        "Metric": [
            f"C&B Cost ({primary_period_desc})",
            f"C&B Cost ({secondary_period_desc})",
            "Variation",
            "Percentage Variation"
        ],
        "Value": [
            f"{primary_period_total_cb_cost:,.2f}",
            f"{secondary_period_total_cb_cost:,.2f}",
            f"{variation:,.2f}",
            f"{percentage_variation:,.2f}%"
        ]
    }
    result_df = pd.DataFrame(data)
    
    # Construct the message string
    response_parts = []
    response_parts.append(f"C&B Cost for {primary_period_desc}: **${primary_period_total_cb_cost:,.2f}**")
    response_parts.append(f"C&B Cost for {secondary_period_desc}: **${secondary_period_total_cb_cost:,.2f}**")
    
    if variation > 0:
        response_parts.append(f"The C&B Cost **increased** by **${variation:,.2f}** from {secondary_period_desc} to {primary_period_desc}.")
    elif variation < 0:
        response_parts.append(f"The C&B Cost **decreased** by **${abs(variation):,.2f}** from {secondary_period_desc} to {primary_period_desc}.")
    else:
        response_parts.append(f"The C&B Cost remained **unchanged** from {secondary_period_desc} to {primary_period_desc}.")
    
    message = "\n\n".join(response_parts)

    return result_df, message

# ---------- CM QUERY PARSER & DISPATCHER ----------
@st.cache_data(show_spinner="Analyzing your query with AI...") # Cache this function
def get_query_details(prompt, df_pl=None, df_ut=None, _cache_buster="v1.12"): # Updated _cache_buster for new changes
    # First, get all date information
    date_info = parse_date_range_from_query(prompt)

    # Generate dynamically extracted distinct value maps for P&L and UT data
    pl_distinct_values_dynamic = get_distinct_column_values(df_pl, FILTERABLE_PL_COLUMNS)
    ut_distinct_values_dynamic = get_distinct_column_values(df_ut, FILTERABLE_UT_COLUMNS)

    # Merge all distinct values into a single map for comprehensive lookup
    current_distinct_map = {}
    current_distinct_map.update(pl_distinct_values_dynamic)
    # Overlay hardcoded P&L values (prioritize hardcoded)
    for col, values_set in PROVIDED_PL_DISTINCT_VALUES.items():
        current_distinct_map[col] = values_set
    current_distinct_map.update(ut_distinct_values_dynamic) # Add dynamic UT values

    # Combine all filterable columns
    filterable_cols_for_query = list(set(FILTERABLE_PL_COLUMNS + FILTERABLE_UT_COLUMNS))
        
  #  st.write("--- DEBUG: Distinct Values Maps ---")
    # st.write(f"P&L Distinct Values (Dynamically Extracted - first 5 per column):")
    # for col, values in pl_distinct_values_dynamic.items():
    #     st.write(f"  - {col}: {list(values)[:5]}...")
    # st.write(f"UT Distinct Values (Dynamically Extracted - first 5 per column):")
    # for col, values in ut_distinct_values_dynamic.items():
    #     st.write(f"  - {col}: {list(values)[:5]}...")
    
    # st.write(f"P&L Distinct Values (Hardcoded - first 5 per column):")
    # for col, values in PROVIDED_PL_DISTINCT_VALUES.items():
    #     st.write(f"  - {col}: {list(values)[:5]}...")
    # st.write("------------------------------------")

    # Then, use LLM to identify query type and other specific details
    system_prompt_query_type = """You are an assistant that classifies user queries and extracts relevant details.
    Identify the main intent of the query and return a JSON object with:
    - 'query_type': 'CM_analysis' for Contribution Margin related queries (e.g., 'show cm', 'cm < X%', 'cm > Y%').
                      **CRITICAL: If the query is about 'CM' or 'Contribution Margin', you MUST classify it as 'CM_analysis' and NOT as 'Revenue_Trend_Analysis' or any other trend type.**
                      'Transportation_cost_analysis' for queries about cost trends in Transportation (e.g., 'Which cost triggered the Margin drop last month in Transportation'),
                      'C&B_cost_variation' for queries asking about C&B cost variation between two periods (e.g., 'How much C&B varied from last quarter to this quarter'),
                      'CB_revenue_trend' for queries asking about "M-o-M trend of C&B cost % w.r.t total revenue",
                      'HC_trend' for queries asking about "M-o-M HC for an account", "monthly headcount trend", or specifically for a customer like "M-o-M HC for A1".
                      'Revenue_Trend_Analysis' for queries specifically and explicitly asking for "YoY revenue", "QoQ revenue", "MoM revenue", or "revenue trends".
                      **CRITICAL for Revenue_Trend_Analysis:** Only classify as 'Revenue_Trend_Analysis' if the query explicitly mentions the word "revenue" AND a trend type (YoY, QoQ, MoM) OR the phrase "revenue trends". Do NOT default to this if the query is about something else (e.g., CM, cost, UT). If "revenue" is not explicitly mentioned, and it's a general trend query (e.g., "YoY for DU"), classify as 'unsupported' unless it clearly fits another category.
                      **Example for HC_trend:**
                        - "What is M-o-M HC for an account" -> `HC_Trend'`, `trend_dimension: 'Account'`, `column_filters: {}`
                        
                      **Example for Revenue_Trend_Analysis:**
                        - "What is MoM revenue for account" -> `query_type: 'Revenue_Trend_Analysis'`, `trend_dimension: 'Account'`, `column_filters: {}`
                        - "What is MoM revenue for an account" -> `query_type: 'Revenue_Trend_Analysis'`, `trend_dimension: 'Account'`, `column_filters: {}`
                        - "What is yoy revenue for an account" -> `query_type: 'Revenue_Trend_Analysis'`, `trend_dimension: 'Account'`, `column_filters: {}`
                        - "What is YoY revenue for a DU" -> `query_type: 'Revenue_Trend_Analysis'`, `trend_dimension: 'DU'`, `column_filters: {}`
                        - "What is QoQ revenue for an account" -> `query_type: 'Revenue_Trend_Analysis'`, `trend_dimension: 'Account'`, `column_filters: {}`
                        - "What is QoQ revenue for a DU" -> `query_type: 'Revenue_Trend_Analysis'`, `trend_dimension: 'DU'`, `column_filters: {}`
                        - "What is YoY revenue for FMCG DU" -> `query_type: 'Revenue_Trend_Analysis'`, `trend_dimension: 'DU'`, `column_filters: {"Exec DU": "FMCG"}`
                      'UT_trend' for queries asking about "UT trend for last 2 quarters for a DU", "UT% for last 3 months for account", "UT trend last 2 years for BU","UT% for last 3 months for an account"
                      'Fresher_UT_Trend' for queries asking about "DU wise Fresher UT Trends", "monthly fresher UT for Delivery Unit".
                      'Revenue_Per_Person_Trend' for queries asking about "Revenue per Person Trends by Account", "monthly Revenue per person trends by Final customer name".
                      'Realized_Rate_Drop' for queries asking about "accounts where realized rate dropped more than $X in this quarter", "realized rate dropped by $Y".
                      'unsupported' for anything else.
    - 'cm_filter_details': { 'filter_type': 'less_than', 'lower': 0.3, 'upper': null } if 'CM_analysis', else null.
        **IMPORTANT for 'cm_filter_details' values:** 'lower' and 'upper' should ALWAYS be expressed as DECIMAL floats.
        For example:
        - For 30%, use 0.3
        - For 112%, use 1.12
        - For 2%, use 0.02
        - For 100%, use 1.0
        **IMPORTANT for 'filter_type' values (for CM_analysis):** Use 'less_than', 'greater_than', 'between', or 'equals'.

    - 'trend_granularity': For 'Revenue_Trend_Analysis', 'UT_trend', 'Fresher_UT_Trend', and 'Revenue_Per_Person_Trend' queries, extract 'monthly', 'quarterly', or 'yearly'. If not specified, default to 'monthly' for Revenue, 'quarterly' for UT, 'monthly' for Fresher UT, and 'monthly' for Revenue Per Person.
    - 'trend_dimension': For 'Revenue_Trend_Analysis', 'UT_trend', 'Fresher_UT_Trend', and 'Revenue_Per_Person_Trend' queries, extract 'DU', 'BU', or 'Account'. If the query asks for a trend for a generic "DU", "BU", or "Account" (e.g., "for a DU", "for an Account", "for any BU"), set this to 'DU', 'BU', or 'Account' respectively. **These generic terms (e.g., "a DU", "an DU", "a BU", "an BU", "a Account", "an Account", "any BU") MUST NOT be included in `column_filters`**. `column_filters` should only contain specific filter values (e.g., `{"Exec DU": "FMCG"}` if the query was "YoY revenue for FMCG DU").
    - 'drop_threshold_value': For 'Realized_Rate_Drop' queries, extract the numeric value or fraction as a STRING (e.g., "3", "5", "3/5", "60%"). Return null if not specified.
    - 'drop_threshold_type': For 'Realized_Rate_Drop' queries, indicate if the threshold is 'percentage' or 'absolute'. Default to 'absolute' if not clearly specified (e.g., no '%' or '/' in value).
    - 'column_filters': A dictionary of additional filters.
        **IMPORTANT RULE for Filters:**
        Identify the correct column name for each filter value. Use the following column names if applicable:
        - 'Segment': For industry or domain names like "plant engineering", "transportation", "media & technology", "industrial products", "telecom & hitech", "medical & life science". Prioritize this for general industry terms.
        - 'FinalCustomerName': For specific customer account names or IDs (e.g., "A1", "A59", "XYZ Corp").
        - 'Company_Code': For company codes like "TSIN", "TSUS", "TSLU", etc.
        - 'Exec DU' (for Delivery Units)
        - 'Exec DG' (for Delivery Groups)
        - 'PVDG'
        - 'PVDU'
        - 'Group Description' (for cost/revenue group descriptions)
        - 'FresherAgeingCategory' (for fresher categories)
        
        **Always try to map the value to one of these specific columns.** If a value (like "Transportation") could plausibly belong to both 'Segment' and 'FinalCustomerName', and it represents a broad industry, **ALWAYS prioritize 'Segment'**. Only use 'FinalCustomerName' if it's clearly a specific customer account ID or name.
        
        * Example: "show cm% <30% in FY26-Q1 for Transportation" -> `{"Segment": "Transportation"}` (because 'Transportation' is a recognized Segment)
        * Example: "show cm% for AEROSPACE & RAIL" -> `{"Segment": "AEROSPACE & RAIL"}`
        * Example: "show cm% for account A1" -> `{"FinalCustomerName": "A1"}`
        * Example: "UT trend for A81" -> `{"FinalCustomerName": "A81"}`
        * Example: "show cm% for segment 'Manufacturing' and customer FMCG" -> `{"Segment": "Manufacturing", "FinalCustomerName": "FMCG"}` (assuming FMCG is a customer account name)
        * **CRITICAL for Revenue_Trend_Analysis:** If the query asks for a trend "for DU", "for BU", or "for Account", or "for an Account",or "for an account"(e.g., "YoY revenue for DU", "QoQ revenue for Account","QoQ revenue for an Account","mom revenue for an Account", "YoY revenue for a DU", "QoQ revenue for an Account"), then "DU", "BU", or "Account" should be set as `trend_dimension` and **these generic terms ('DU', 'BU', 'Account', 'a DU', 'an DU', 'a BU', 'an BU', 'a Account', 'an Account','an account') MUST NOT be included in `column_filters`**. `column_filters` should only contain specific filter values (e.g., `{"Exec DU": "FMCG"}` if the query was "YoY revenue for FMCG DU").
        * **CRITICAL for HC_trend:** If the query asks for a trend "for an account","for a account", then "Account" should be set as `trend_dimension` and **these generic terms  'Account',  'a Account', 'an Account','an account') MUST NOT be included in `column_filters`**. `column_filters` should only contain specific filter values (e.g., `{"Exec DU": "FMCG"}` if the query was "HC trend  for FMCG DU").
        * **Example with Company_Code:** "show me the cm for TSIN" -> `{"Company_Code": "TSIN"}`

        **CRITICAL: The column 'ServiceLine' DOES NOT exist in the data. Never map any value to 'ServiceLine'.**
        If a filter value cannot be confidently mapped to any of the above columns, do NOT include it in `column_filters`.
    - 'requested_trends': A list of strings, indicating which trends are explicitly requested. Can contain "MoM", "QoQ", "YoY","QOQ","MOM","YOY". If the query implies "all" trends (e.g., "YoY, QoQ, MoM revenue", "all revenue trends"), return ["MoM", "QoQ", "YoY"]. If only one is specified, return just that one (e.g., ["MoM"]). If none are specified but query_type is 'Revenue_Trend_Analysis', default to ["MoM", "QoQ", "YoY"].

    Return ONLY valid JSON.
    """
    try:
        if openai_client is None: # Check if client initialized successfully
            print("ERROR: OpenAI client not initialized. Cannot get query details.")
            return {"date_filter": False, "start_date": None, "end_date": None, "secondary_start_date": None, "secondary_end_date": None, "description": "all available data", "relative_period_detected": None, "num_periods": None, "requested_trends": ["MoM", "QoQ", "YoM"]}

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
        trend_granularity = query_type_result.get('trend_granularity')
        trend_dimension = query_type_result.get('trend_dimension')
        
        drop_threshold_value_raw = query_type_result.get('drop_threshold_value')
        drop_threshold_type = query_type_result.get('drop_threshold_type')
        requested_trends = query_type_result.get('requested_trends', ["MoM", "QoQ", "YoY"]) # Default to all if not specified


        llm_column_filters = query_type_result.get('column_filters', {})
        
        # --- NEW: Immediate post-processing to remove redundant trend_dimension filters ---
        # This runs BEFORE validating against distinct values.
        if trend_dimension:
            dimension_map = {
                "DU": "Exec DU",
                "BU": "Exec DG",
                "Account": "FinalCustomerName"
            }
            mapped_dimension_col = dimension_map.get(trend_dimension)
            
            # Check if the mapped dimension column exists in the filters and its value is generic or None/empty
            if mapped_dimension_col in llm_column_filters:
                filter_value_raw = llm_column_filters[mapped_dimension_col]
                filter_value_lower_stripped = str(filter_value_raw).lower().strip()
                trend_dimension_lower_stripped = trend_dimension.lower().strip()
                
                # If the filter value is the generic dimension name itself OR is None/empty
                if filter_value_lower_stripped == trend_dimension_lower_stripped.lower() or \
                   filter_value_raw is None or filter_value_lower_stripped == "": # Changed this line
                  #  st.write(f"DEBUG: Removing redundant filter '{mapped_dimension_col}: {filter_value_raw}' because it matches the requested trend_dimension '{trend_dimension}' or is None/empty.")
                    del llm_column_filters[mapped_dimension_col]
        # --- END NEW: Immediate post-processing ---

        final_resolved_column_filters = {}

        # Determine which distinct_values_map and filterable columns to use based on query_type
        current_distinct_map = {}
        current_distinct_map.update(pl_distinct_values_dynamic)
        # Overlay hardcoded P&L values (prioritize hardcoded)
        for col, values_set in PROVIDED_PL_DISTINCT_VALUES.items():
            current_distinct_map[col] = values_set
        current_distinct_map.update(ut_distinct_values_dynamic) # Add dynamic UT values

        # Combine all filterable columns
        filterable_cols_for_query = list(set(FILTERABLE_PL_COLUMNS + FILTERABLE_UT_COLUMNS))
        
        # st.write("--- DEBUG: Distinct Values Maps ---")
        # st.write(f"P&L Distinct Values (Dynamically Extracted - first 5 per column):")
        # for col, values in pl_distinct_values_dynamic.items():
        #     st.write(f"  - {col}: {list(values)[:5]}...")
        # st.write(f"UT Distinct Values (Dynamically Extracted - first 5 per column):")
        # for col, values in ut_distinct_values_dynamic.items():
        #     st.write(f"  - {col}: {list(values)[:5]}...")
        
        # st.write(f"P&L Distinct Values (Hardcoded - first 5 per column):")
        # for col, values in PROVIDED_PL_DISTINCT_VALUES.items():
        #     st.write(f"  - {col}: {list(values)[:5]}...")
        # st.write("------------------------------------")

        for llm_col, llm_val in llm_column_filters.items(): # Iterate over the potentially modified llm_column_filters
            llm_val_stripped_lower = str(llm_val).lower().strip()
            resolved_col_name = None
           # st.write(f"  - Processing filter: value='{llm_val}', LLM suggested column='{llm_col}'")

            # NEW LOGIC: Prioritize hardcoded/dynamically extracted values first, then trust LLM for FinalCustomerName
            # This ensures that if 'transportation' is in Segment, it's picked up there first.
            
            # 1. Check if the value exists in the LLM's suggested column's distinct values (if available)
            if llm_col in current_distinct_map:
              #  st.write(f"    DEBUG: Checking if '{llm_val_stripped_lower}' is in current_distinct_map['{llm_col}'].")
               # st.write(f"    DEBUG: current_distinct_map['{llm_col}'] contains (first 5): {list(current_distinct_map[llm_col])[:5]}...")
                if llm_val_stripped_lower in current_distinct_map[llm_col]:
                    resolved_col_name = llm_col
                  #  st.write(f"    -> Matched LLM's suggested column '{llm_col}'.")
            
            # 2. If not found in suggested column, or if suggested column wasn't in map, search other filterable columns
            if not resolved_col_name:
              #  st.write(f"    -> Value '{llm_val}' not found in LLM's suggested column '{llm_col}' (or column not in map). Searching other filterable columns...")
                for potential_col in filterable_cols_for_query:
                    # Skip the initially suggested column if we already checked it and it didn't match
                    if potential_col == llm_col and llm_col in current_distinct_map and llm_val_stripped_lower not in current_distinct_map[llm_col]:
                        continue
                    
                    if potential_col in current_distinct_map and llm_val_stripped_lower in current_distinct_map[potential_col]:
                        resolved_col_name = potential_col
                        st.write(f"    -> Found match in alternative column: '{potential_col}'.")
                        break # Found a match, take the first one (order in FILTERABLE_..._COLUMNS matters for priority)

            # 3. Final fallback: If still not resolved and LLM initially suggested FinalCustomerName, trust it.
            # This is the "trust LLM for FinalCustomerName if nothing else matches" rule.
            if not resolved_col_name and llm_col == "FinalCustomerName":
                resolved_col_name = llm_col
                st.write(f"    -> No other match found, directly trusting LLM's initial suggestion for FinalCustomerName: '{resolved_col_name}': '{llm_val}'")


            if resolved_col_name:
                final_resolved_column_filters[resolved_col_name] = llm_val
              #  st.write(f"    -> Filter successfully resolved to: '{resolved_col_name}': '{llm_val}'")
            else:
              #  st.write(f"    -> WARNING: Could not confidently map filter value '{llm_val}' to any known column. Skipping this filter.")
              1 
        st.write("------------------------------------")
       # st.write(f"Final resolved column filters: {final_resolved_column_filters}")


        drop_threshold_numeric = None
        if drop_threshold_value_raw:
            if drop_threshold_type == 'percentage':
                drop_threshold_numeric = parse_percentage(drop_threshold_value_raw) 
            else:
                drop_threshold_numeric = parse_percentage(drop_threshold_value_raw)
                if drop_threshold_type is None:
                    if '%' in drop_threshold_value_raw or '/' in drop_threshold_value_raw:
                        drop_threshold_type = 'percentage'
                    else:
                        drop_threshold_type = 'absolute'

        final_result = {
            "query_type": query_type,
            "trend_granularity": trend_granularity,
            "trend_dimension": trend_dimension,
            "drop_threshold": drop_threshold_numeric,
            "drop_threshold_type": drop_threshold_type,
            "column_filters": final_resolved_column_filters, # Use the resolved filters
            "requested_trends": requested_trends, # Add the new field
            **date_info
        }

        # Set default granularity/dimension for specific query types if not provided by LLM
        if query_type == "Revenue_Trend_Analysis":
            if final_result["trend_granularity"] is None:
                final_result["trend_granularity"] = "monthly"
            if final_result["trend_dimension"] is None:
                final_result["trend_dimension"] = "DU"
            # If query_type is Revenue_Trend_Analysis and requested_trends is empty, default to all
            if not final_result["requested_trends"]:
                final_result["requested_trends"] = ["MoM", "QoQ", "YoY"]
        elif query_type == "UT_trend":
            if final_result["trend_granularity"] is None:
                final_result["trend_granularity"] = "quarterly"
            if final_result["trend_dimension"] is None:
                final_result["trend_dimension"] = "Account"
        elif query_type == "Fresher_UT_Trend":
            if final_result["trend_granularity"] is None:
                final_result["trend_granularity"] = "monthly"
            if final_result["trend_dimension"] is None:
                final_result["trend_dimension"] = "DU"
        elif query_type == "Revenue_Per_Person_Trend":
            if final_result["trend_granularity"] is None:
                final_result["trend_granularity"] = "monthly"
            if final_result["trend_dimension"] is None:
                final_result["trend_dimension"] = "Account"
        elif final_result["query_type"] == "Realized_Rate_Drop":
            if final_result["trend_granularity"] is None:
                final_result["trend_granularity"] = "quarterly"
            if final_result["trend_dimension"] is None:
                final_result["trend_dimension"] = "Account"
            if final_result["drop_threshold_type"] is None:
                final_result["drop_threshold_type"] = 'absolute'
        elif final_result["query_type"] == "HC_trend": # Added for HC_trend defaults
            if final_result["trend_granularity"] is None:
                final_result["trend_granularity"] = "monthly"
            # The trend_dimension for HC_trend will be set by the LLM based on the prompt.
            # No default 'Account' here, as it's explicitly handled by the LLM prompt now.


        if query_type == "CM_analysis" and cm_filter_details:
            final_result["cm_filter_type"] = cm_filter_details.get("filter_type")
            final_result["cm_lower_bound"] = parse_percentage(cm_filter_details.get("lower"))
            final_result["cm_upper_bound"] = parse_percentage(cm_filter_details.get("upper"))
        else:
            final_result["cm_filter_type"] = None
            final_result["cm_lower_bound"] = None
            final_result["cm_upper_bound"] = None
        
        return final_result

    except json.JSONDecodeError as jde:
        print(f"ERROR: JSON decoding error in get_query_details: {jde}")
        print(f"Raw API response content: {content}")
        return {
            "query_type": "unsupported",
            "cm_filter_type": None,
            "cm_lower_bound": None,
            "cm_upper_bound": None,
            "trend_granularity": None,
            "trend_dimension": None,
            "drop_threshold_value": None,
            "drop_threshold_type": None,
            "column_filters": {}, # Default empty dictionary
            "requested_trends": ["MoM", "QoQ", "YoY"], # Default to all on error
            **date_info
        }
    except Exception as e:
        print(f"ERROR: Failed to parse query type or filters (general error): {e}")
        return {
            "query_type": "unsupported",
            "cm_filter_type": None,
            "cm_lower_bound": None,
            "cm_upper_bound": None,
            "trend_granularity": None,
            "trend_dimension": None,
            "drop_threshold": None,
            "drop_threshold_type": None,
            "column_filters": {}, # Default empty dictionary
            "requested_trends": ["MoM", "QoQ", "YoY"], # Default to all on error
            **date_info
        }






# ---------- CM CALCULATION FUNCTION ----------
def calculate_cm(df_pl, query_details):
    df_copy = df_pl.copy()
    
    # Apply dynamic filters first
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "P&L data")
    if df_copy.empty:
        return {"Message": filter_message} # Return message from filter function

    if df_copy.empty: # Redundant check, but harmless
        return pd.DataFrame()

    # --- Filter out rows where Revenue is 0 for CM calculation ---
    df_revenue_only = df_copy[df_copy["Group1"].isin(REVENUE_GROUPS)].copy()
    customer_revenue = df_revenue_only.groupby("FinalCustomerName")["Amount in USD"].sum()
    
    customers_to_exclude = customer_revenue[
        customer_revenue.isna() | np.isclose(customer_revenue, 0)
    ].index
    
    df_copy_filtered_for_cm = df_copy[
        ~df_copy["FinalCustomerName"].isin(customers_to_exclude)
    ].copy()

    if df_copy_filtered_for_cm.empty:
        return pd.DataFrame(columns=["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"])

    grouped = df_copy_filtered_for_cm.groupby("FinalCustomerName", as_index=False).apply(lambda x: pd.Series({
        "Revenue": x[x["Group1"].isin(REVENUE_GROUPS)]["Amount in USD"].sum(),
        "Cost": x[x["Group1"].isin(COST_GROUPS)]["Amount in USD"].sum()
    })).reset_index(drop=True)

    grouped["CM_Ratio"] = np.where(
        grouped["Revenue"] != 0,
        (grouped["Revenue"] - grouped["Cost"]) / grouped["Revenue"],
        np.nan
    )
    
    grouped["CM_Ratio"] = grouped["CM_Ratio"].replace([np.inf, -np.inf], np.nan)
    grouped.dropna(subset=['CM_Ratio'], inplace=True)

    if grouped.empty:
        return pd.DataFrame(columns=["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"])

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
    elif filter_type == "equals" and lower_bound is not None:
        filtered = filtered[np.isclose(filtered["CM_Ratio"], lower_bound)]
    
    if filtered.empty:
        return pd.DataFrame(columns=["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"])

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

    return filtered[["S.No", "FinalCustomerName", "Revenue", "Cost", "CM (%)", "CM_Value"]] 

# ---------- TRANSPORTATION COST TREND ANALYSIS ----------
def analyze_transportation_cost_trend(df_pl, query_details):
    """
    Analyzes cost trends for a specified segment
    between two dynamic periods,
    identifying costs that increased.
    Requires 'secondary_start_date',
    'secondary_end_date', 'start_date', 'end_date'
    from query_details.
    """
    df_copy = df_pl.copy()

    # Apply dynamic filters first
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "P&L data")
    if df_copy.empty:
        return {"Message": filter_message} # Return message from filter function

    # Convert dates to pandas Timestamps for robust comparison and ensure they are naive
    prev_period_start = pd.Timestamp(query_details.get("secondary_start_date")).tz_localize(None) if query_details.get("secondary_start_date") else None
    prev_period_end = pd.Timestamp(query_details.get("secondary_end_date")).tz_localize(None) if query_details.get("secondary_end_date") else None
    current_period_start = pd.Timestamp(query_details.get("start_date")).tz_localize(None) if query_details.get("start_date") else None
    current_period_end = pd.Timestamp(query_details.get("end_date")).tz_localize(None) if query_details.get("end_date") else None
    
    if not all([prev_period_start, prev_period_end, current_period_start, current_period_end]):
        return "Could not determine the two consecutive months/quarters for comparison from your query. Please be explicit (e.g., 'May and June 2025' or 'last month vs previous month', or 'last quarter vs this quarter')."
    
    # Filter for 'Cost' type (Segment filter is now handled by apply_filters)
    segment_cost_df = df_copy[
        (df_copy["Type"] == "Cost") # Ensure 'Type' column exists and correctly identifies costs
    ].copy()
    
    if segment_cost_df.empty:
        return f"No cost data found for analysis after applying filters."
    
    # Filter for previous period data
    prev_period_df = segment_cost_df[
        (segment_cost_df["Date"] >= prev_period_start) & 
        (segment_cost_df["Date"] <= prev_period_end)
    ]
    
    # Filter for current period data
    current_period_df = segment_cost_df[
        (df_copy["Date"] >= current_period_start) & 
        (df_copy["Date"] <= current_period_end)
    ]
    
    if prev_period_df.empty and current_period_df.empty:
        prev_period_name = prev_period_start.strftime('%b %Y') if (prev_period_end - prev_period_start).days < 90 else f"Q{((prev_period_start.month - 1) // 3) + 1} {prev_period_start.year}"
        current_period_name = current_period_start.strftime('%b %Y') if (current_period_end - current_period_start).days < 90 else f"Q{((current_period_start.month - 1) // 3) + 1} {current_period_start.year}"
        return f"No specific costs increased from {prev_period_name} to {current_period_name} after applying filters."
    else:
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
            return f"No specific costs increased from {prev_period_name} to {current_period_name} after applying filters."
        else:
            # Format for display
            increased_costs["Previous_Period_Cost"] = increased_costs["Previous_Period_Cost"].apply(lambda x: f"${x:,.2f}")
            increased_costs["Current_Period_Cost"] = increased_costs["Current_Period_Cost"].apply(lambda x: f"${x:,.2f}")
            increased_costs["Cost_Increase"] = increased_costs["Cost_Increase"].apply(lambda x: f"${x:,.2f}")
            
            return increased_costs[['Group Description', 'Previous_Period_Cost', 'Current_Period_Cost', 'Cost_Increase']]

# ---------- C&B COST VARIATION CALCULATION ----------
def calculate_cb_variation(df_pl, query_details):
    """
    Calculates the variation of C&B Cost between two specified periods.
    Expected query_details to contain 'start_date', 'end_date' for the primary period
    and 'secondary_start_date', 'secondary_end_date' for the secondary period.
    Returns a pandas DataFrame with the variation details and a message string.
    """
    df_copy = df_pl.copy()

    # Apply dynamic filters first
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "P&L data")
    if df_copy.empty:
        # Ensure the DataFrame returned is empty but has the expected columns for downstream processing
        return pd.DataFrame(columns=["Metric", "Value"]), filter_message

    # Convert dates to pandas Timestamps for robust comparison and ensure they are naive
    primary_period_start = pd.Timestamp(query_details.get("start_date")).tz_localize(None) if query_details.get("start_date") else None
    primary_period_end = pd.Timestamp(query_details.get("end_date")).tz_localize(None) if query_details.get("end_date") else None
    secondary_period_start = pd.Timestamp(query_details.get("secondary_start_date")).tz_localize(None) if query_details.get("secondary_start_date") else None
    secondary_period_end = pd.Timestamp(query_details.get("secondary_end_date")).tz_localize(None) if query_details.get("secondary_end_date") else None
    
    if not all([primary_period_start, primary_period_end, secondary_period_start, secondary_period_end]):
        message = "Could not determine the two periods for C&B cost comparison from your query. Please be explicit (e.g., 'last month to this month' or 'last quarter to this quarter')."
        return pd.DataFrame(columns=["Metric", "Value"]), message

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
                # Use new helper for fiscal quarter name
                prev_fy_cal_year, prev_fq = get_fiscal_quarter_and_year(secondary_period_start.date())
                secondary_period_desc = get_fiscal_quarter_name_from_fy_and_q(prev_fy_cal_year, prev_fq)

    else: # For relative periods like "last month to this month" or "last quarter to this quarter"
        is_month_comparison = (primary_period_end - primary_period_start).days < 60 # Heuristic for month vs. quarter

        if is_month_comparison:
            primary_period_desc = primary_period_start.strftime('%B %Y')
            secondary_period_desc = secondary_period_start.strftime('%B %Y')
        else:
            # Use new helper for fiscal quarter names
            primary_fy_cal_year, primary_fq = get_fiscal_quarter_and_year(primary_period_start.date())
            secondary_fy_cal_year, secondary_fq = get_fiscal_quarter_and_year(secondary_period_start.date())
            
            primary_period_desc = get_fiscal_quarter_name_from_fy_and_q(primary_fy_cal_year, primary_fq)
            secondary_period_desc = get_fiscal_quarter_name_from_fy_and_q(secondary_fy_cal_year, secondary_fq)

    # Filter for C&B Cost data
    cb_df = df_copy[df_copy["Group Description"].isin(CB_COST_GROUPS)].copy()

    if cb_df.empty:
        message = "No C&B Cost data found for analysis after applying filters."
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=["Metric", "Value"]), message

    # Check if "Amount in USD" column exists before summing
    if "Amount in USD" not in cb_df.columns:
        message = "Error: 'Amount in USD' column not found in the filtered C&B data. Please ensure your P&L dataset has a column named 'Amount in USD' with financial figures."
        return pd.DataFrame(columns=["Metric", "Value"]), message

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
    
    percentage_variation = 0
    if secondary_period_total_cb_cost != 0:
        percentage_variation = (variation / secondary_period_total_cb_cost) * 100

    # Prepare data for the table
    data = {
        "Metric": [
            f"C&B Cost ({primary_period_desc})",
            f"C&B Cost ({secondary_period_desc})",
            "Variation",
            "Percentage Variation"
        ],
        "Value": [
            f"{primary_period_total_cb_cost:,.2f}",
            f"{secondary_period_total_cb_cost:,.2f}",
            f"{variation:,.2f}",
            f"{percentage_variation:,.2f}%"
        ]
    }
    result_df = pd.DataFrame(data)
    
    # Construct the message string
    response_parts = []
    response_parts.append(f"C&B Cost for {primary_period_desc}: **${primary_period_total_cb_cost:,.2f}**")
    response_parts.append(f"C&B Cost for {secondary_period_desc}: **${secondary_period_total_cb_cost:,.2f}**")
    
    if variation > 0:
        response_parts.append(f"The C&B Cost **increased** by **${variation:,.2f}** from {secondary_period_desc} to {primary_period_desc}.")
    elif variation < 0:
        response_parts.append(f"The C&B Cost **decreased** by **${abs(variation):,.2f}** from {secondary_period_desc} to {primary_period_desc}.")
    else:
        response_parts.append(f"The C&B Cost remained **unchanged** from {secondary_period_desc} to {primary_period_desc}.")
    
    message = "\n\n".join(response_parts)

    return result_df, message
# ---------- C&B REVENUE TREND ANALYSIS ----------
def calculate_cb_revenue_trend(df_pl, query_details):
    """
    Calculates the monthly trend of C&B Cost, Total Revenue, their difference,
    and the ratio of (C&B Cost - Total Revenue) to C&B Cost.
    Handles explicit date ranges (FY, year, quarter) or defaults to the last 12 months.
    Returns a DataFrame with relevant monthly data.
    """
    df_copy = df_pl.copy()

    # Apply dynamic filters first
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "P&L data")
    if df_copy.empty:
        return {"Message": filter_message} # Return message from filter function

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

# ---------- NEW: HC TREND CALCULATION ----------
def calculate_hc_trend(df_ut, query_details):
    """
    Calculates the monthly trend of Headcount (HC) by Final Customer Name.
    HC is defined as the distinct count of "PSNo".
    Handles explicit date ranges or defaults to the last 12 months.
    Can filter for a specific customer if 'customer_name_filter' is provided in query_details.
    Returns a DataFrame with 'Month', 'FinalCustomerName', and 'HC'.
    """
    df_copy = df_ut.copy()

    # Apply dynamic filters first
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "UT data")
    if df_copy.empty:
        return {"Message": filter_message} # Return message from filter function

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
        return pd.DataFrame(columns=['Month', 'FinalCustomerName', 'HC'])

    # Extract month and year for grouping
    df_filtered['Month_Year'] = df_filtered['Date'].dt.to_period('M')

    # Calculate HC (distinct count of "PSNo") per month and FinalCustomerName
    # Ensure that if a PSNo appears multiple times for the same customer in the same month,
    # it's only counted once. This is what nunique does.
    hc_data = df_filtered.groupby(['Month_Year', 'FinalCustomerName']).agg(
        HC=('PSNo', 'nunique')
    ).reset_index()

    # Convert Month_Year Period to datetime for sorting and display
    hc_data['Month'] = hc_data['Month_Year'].dt.to_timestamp()
    hc_data = hc_data.sort_values(by=['Month', 'FinalCustomerName']).reset_index(drop=True)

    # Format Month for display
    hc_data['Month'] = hc_data['Month'].dt.strftime('%Y-%m')

    return hc_data[['Month', 'FinalCustomerName', 'HC']]


# ---------- NEW: REVENUE TREND ANALYSIS CALCULATION (Simplified for mainn.py to re-aggregate) ----------
def analyze_revenue_trend(df_pl, query_details):
    """
    Filters the P&L data for revenue items within the specified date range.
    Returns the filtered DataFrame and the initial grouping dimension from the query.
    All further grouping and trend calculations will happen in mainn.py.
    """
    df_revenue_only = df_pl.copy()

    # Apply dynamic filters first
    df_revenue_only, filter_message = apply_filters(df_revenue_only, query_details.get("column_filters", {}), "P&L data")
    if df_revenue_only.empty:
        return {"Message": filter_message} # Return message from filter function

    df_revenue_only = df_revenue_only[df_revenue_only["Group1"].isin(REVENUE_GROUPS)].copy()
    df_revenue_only['Date'] = pd.to_datetime(df_revenue_only['Date'], errors='coerce').dt.tz_localize(None)
    df_revenue_only.dropna(subset=['Date'], inplace=True)

    # Apply date filter if present in query_details
    start_date = query_details.get("start_date")
    end_date = query_details.get("end_date")

    if start_date and end_date:
        start_date_ts = pd.Timestamp(start_date).tz_localize(None)
        end_date_ts = pd.Timestamp(end_date).tz_localize(None)
        df_revenue_only = df_revenue_only[
            df_revenue_only['Date'].between(start_date_ts, end_date_ts, inclusive='both')
        ]
    
    if df_revenue_only.empty:
        return {"Message": "No revenue data found for the specified date range after applying filters."}

    # Determine the initial grouping dimension from the query
    initial_grouping_dimension_from_query = query_details.get("trend_dimension", "DU") # Default to DU if not specified

    return {
        "df_filtered_for_charts": df_revenue_only, # The base filtered dataframe
        "grouping_dimension_from_query": initial_grouping_dimension_from_query
    }



# ---------- NEW: UT TREND ANALYSIS CALCULATION ----------
def analyze_ut_trend(df_ut, query_details):
    """
    Calculates the UT% trend for a specified period and dimension (DU, BU, Account).
    UT% = sum("TotalBillableHours") / sum("NetAvailableHours")
    """
    df_copy = df_ut.copy()

    # Apply dynamic filters first
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "UT data")
    if df_copy.empty:
        return {"Message": filter_message} # Return message from filter function

    df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce').dt.tz_localize(None)
    df_copy.dropna(subset=['Date'], inplace=True)

    start_date = query_details.get("start_date")
    end_date = query_details.get("end_date")
    trend_dimension = query_details.get("trend_dimension", "Account") # Default to Account for UT
    trend_granularity = query_details.get("trend_granularity", "quarterly") # Default to quarterly for UT

    if not start_date or not end_date:
        return {"Message": "Please specify a clear date range for UT trend analysis (e.g., 'last 2 quarters', 'last 3 months', 'FY25Q4 to FY26Q1')."}

    start_date_ts = pd.Timestamp(start_date).tz_localize(None)
    end_date_ts = pd.Timestamp(end_date).tz_localize(None)

    df_filtered = df_copy[
        df_copy['Date'].between(start_date_ts, end_date_ts, inclusive='both')
    ].copy()

    if df_filtered.empty:
        return {"Message": f"No UT data found for the period {query_details.get('description', 'specified date range')} after applying filters."}

    # Mapping for dimension names to actual column names in df_ut
    grouping_col_map = {
        "DU": "Exec DU",
        "BU": "Exec DG",
        "Account": "FinalCustomerName",
        "All": None # For overall UT%
    }
    
    selected_dim_col_name = grouping_col_map.get(trend_dimension)

    # Validate if the selected dimension column exists and has non-null values
    if selected_dim_col_name is None or \
       selected_dim_col_name not in df_filtered.columns or \
       df_filtered[selected_dim_col_name].isnull().all():
        
        selected_dim_col_name = None # Fallback to "All"
        trend_dimension = "All" # Update display name

    # Determine frequency for grouping
    freq_map = {
        "monthly": 'MS',
        "quarterly": 'QS-APR', # Fiscal quarters (April-March)
        "yearly": 'AS-APR' # Fiscal year (April-March)
    }
    freq_str = freq_map.get(trend_granularity, 'QS-APR') # Default to quarterly if not recognized

    if selected_dim_col_name:
        grouped_ut = df_filtered.groupby([pd.Grouper(key='Date', freq=freq_str), selected_dim_col_name]).agg(
            TotalBillableHours=('TotalBillableHours', 'sum'),
            NetAvailableHours=('NetAvailableHours', 'sum')
        ).reset_index()
        grouped_ut.rename(columns={'Date': 'Period'}, inplace=True)
    else: # Group by 'All' (total)
        grouped_ut = df_filtered.groupby(pd.Grouper(key='Date', freq=freq_str)).agg(
            TotalBillableHours=('TotalBillableHours', 'sum'),
            NetAvailableHours=('NetAvailableHours', 'sum')
        ).reset_index()
        grouped_ut.rename(columns={'Date': 'Period'}, inplace=True)
        grouped_ut['Dimension_Name'] = 'Total UT' # Dummy column for coloring

    # Calculate UT%
    grouped_ut['UT_Percent'] = np.where(
        grouped_ut['NetAvailableHours'] != 0,
        (grouped_ut['TotalBillableHours'] / grouped_ut['NetAvailableHours']) * 100,
        np.nan
    )
    grouped_ut.dropna(subset=['UT_Percent'], inplace=True) # Drop rows where UT% is NaN

    # Format Period for display
    if trend_granularity == "monthly":
        grouped_ut['Period_Formatted'] = grouped_ut['Period'].dt.strftime('%Y-%m')
    elif trend_granularity == "quarterly":
        # For fiscal quarters, format as FYXXQY using the new helper
        grouped_ut['Period_Formatted'] = grouped_ut['Period'].apply(
            lambda x: get_fiscal_quarter_name_from_fy_and_q(*get_fiscal_quarter_and_year(x.date()))
        )
    elif trend_granularity == "yearly":
        # For fiscal years, format as FYXX using the new helper
        grouped_ut['Period_Formatted'] = grouped_ut['Period'].apply(
            lambda x: f"FY{get_fiscal_quarter_and_year(x.date())[0] + 1}" # Get fiscal year start cal year, add 1
        )


    # Select and return relevant columns
    if selected_dim_col_name:
        return {
            "df_ut_trend": grouped_ut[['Period_Formatted', selected_dim_col_name, 'TotalBillableHours', 'NetAvailableHours', 'UT_Percent']],
            "trend_dimension_display": trend_dimension,
            "trend_granularity_display": trend_granularity
        }
    else:
        return {
            "df_ut_trend": grouped_ut[['Period_Formatted', 'TotalBillableHours', 'NetAvailableHours', 'UT_Percent']],
            "trend_dimension_display": "All", # Will be "All"
            "trend_granularity_display": trend_granularity
        }

# ---------- NEW: FRESHER UT TREND ANALYSIS CALCULATION ----------
def analyze_fresher_ut_trend(df_ut, query_details):
    """
    Calculates the monthly UT% trend for freshers by Delivery Unit (Exec DU).
    Filters for specific "FresherAgeingCategory" values.
    UT% = sum("TotalBillableHours") / sum("NetAvailableHours")
    """
    df_copy = df_ut.copy()

    # Apply dynamic filters first (including FresherAgeingCategory if specified by user)
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "UT data")
    if df_copy.empty:
        return {"Message": filter_message} # Return message from filter function

    df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce').dt.tz_localize(None)
    df_copy.dropna(subset=['Date'], inplace=True)

    # CRITICAL FIX: Strip whitespace from the column values before filtering
    # This specific filter is still applied here to ensure only relevant fresher categories are included
    # regardless of whether the user explicitly asked for them via column_filters.
    if "FresherAgeingCategory" in df_copy.columns:
        df_copy["FresherAgeingCategory"] = df_copy["FresherAgeingCategory"].str.strip()
    
    # Filter for FresherAgeingCategory (always apply this for Fresher UT trend)
    df_filtered_freshers = df_copy[
        df_copy["FresherAgeingCategory"].isin(FRESHER_AGEING_CATEGORIES)
    ].copy()

    if df_filtered_freshers.empty:
        return {"Message": "No data found for the specified Fresher Ageing Categories after applying filters. Please ensure the 'FresherAgeingCategory' column exists and contains values like 'Freshers ET(0-3 Months)', 'Freshers ET(4-6 Months)', 'Freshers PGET(0-3 Months)', or 'Freshers ETPremium(0-3 Months)' (case and whitespace sensitive)."}

    # Determine the date range (default to last 12 months if not specified)
    start_date = query_details.get("start_date")
    end_date = query_details.get("end_date")

    if not start_date or not end_date:
        today = datetime.now(pytz.timezone("Asia/Kolkata"))
        end_date = datetime(today.year, today.month, calendar.monthrange(today.year, today.month)[1], 23, 59, 59, 999999)
        start_date = (today - pd.DateOffset(months=11)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    start_date_ts = pd.Timestamp(start_date).tz_localize(None)
    end_date_ts = pd.Timestamp(end_date).tz_localize(None)

    df_filtered_freshers_date = df_filtered_freshers[
        df_filtered_freshers['Date'].between(start_date_ts, end_date_ts, inclusive='both')
    ].copy()

    if df_filtered_freshers_date.empty:
        return {"Message": f"No fresher UT data found for the period {query_details.get('description', 'specified date range')} after applying filters."}

    # Group by month and Delivery_Unit (Exec DU) - CORRECTED COLUMN NAME
    if "Exec DU" not in df_filtered_freshers_date.columns or df_filtered_freshers_date["Exec DU"].isnull().all():
        return {"Message": "The 'Exec DU' (Delivery Unit) column is missing or contains no data for fresher UT trend analysis. Cannot group by Delivery Unit."}

    grouped_fresher_ut = df_filtered_freshers_date.groupby([pd.Grouper(key='Date', freq='MS'), 'Exec DU']).agg(
        TotalBillableHours=('TotalBillableHours', 'sum'),
        NetAvailableHours=('NetAvailableHours', 'sum')
    ).reset_index()
    grouped_fresher_ut.rename(columns={'Date': 'Period'}, inplace=True)

    # Calculate UT%
    grouped_fresher_ut['UT_Percent'] = np.where(
        grouped_fresher_ut['NetAvailableHours'] != 0,
        (grouped_fresher_ut['TotalBillableHours'] / grouped_fresher_ut['NetAvailableHours']) * 100,
        np.nan
    )
    grouped_fresher_ut.dropna(subset=['UT_Percent'], inplace=True) # Drop rows where UT% is NaN

    # Format Period for display
    grouped_fresher_ut['Period_Formatted'] = grouped_fresher_ut['Period'].dt.strftime('%Y-%m')

    return {
        "df_fresher_ut_trend": grouped_fresher_ut[['Period_Formatted', 'Exec DU', 'TotalBillableHours', 'NetAvailableHours', 'UT_Percent']],
        "trend_dimension_display": "DU",
        "trend_granularity_display": "monthly"
    }

# ---------- NEW: REVENUE PER PERSON TREND ANALYSIS CALCULATION ----------
def analyze_revenue_per_person_trend(df_merged, query_details):
    """
    Calculates the monthly Revenue per Person trend by Final Customer Name using the merged DataFrame.
    Revenue is from 'Amount in USD' and HC (distinct PSNo) is from 'PSNo'.
    """
    df_copy = df_merged.copy()

    # Apply dynamic filters first
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "merged data")
    if df_copy.empty:
        return {"Message": filter_message} # Return message from filter function

    # Ensure Date column is datetime and timezone-naive
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce').dt.tz_localize(None)
    df_copy.dropna(subset=['Date'], inplace=True)

    # Determine the date range (default to last 12 months if not specified)
    start_date = query_details.get("start_date")
    end_date = query_details.get("end_date")

    if not start_date or not end_date:
        today = datetime.now(pytz.timezone("Asia/Kolkata"))
        end_date = datetime(today.year, today.month, calendar.monthrange(today.year, today.month)[1], 23, 59, 59, 999999)
        start_date = (today - pd.DateOffset(months=11)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    start_date_ts = pd.Timestamp(start_date).tz_localize(None)
    end_date_ts = pd.Timestamp(end_date).tz_localize(None)

    # Filter the merged DataFrame by date
    df_filtered = df_copy[
        df_copy['Date'].between(start_date_ts, end_date_ts, inclusive='both')
    ].copy()

    if df_filtered.empty:
        return {"Message": f"No sufficient data found for Revenue per Person analysis for the period {query_details.get('description', 'specified date range')} after applying filters."}

    # Check for required columns after merge
    required_cols = ['FinalCustomerName', 'Amount in USD', 'PSNo']
    for col in required_cols:
        if col not in df_filtered.columns:
            return {"Message": f"Missing required column '{col}' in the merged data for Revenue per Person analysis. Please ensure data loading and merging are correct."}
    
    # 1. Aggregate Monthly Revenue by FinalCustomerName
    df_filtered['Month_Year'] = df_filtered['Date'].dt.to_period('M')
    
    # Filter for revenue groups from the P&L side of the merged data
    # Assuming 'Group1' is from P&L and remains 'Group1' in merged (not a merge key, so no suffix)
    monthly_revenue = df_filtered[df_filtered["Group1"].isin(REVENUE_GROUPS)].groupby(
        ['Month_Year', 'FinalCustomerName']
    ).agg(
        TotalRevenue=('Amount in USD', 'sum') # Use Amount in USD_pl for revenue
    ).reset_index()

    # 2. Aggregate Monthly Headcount (distinct PSNo) by FinalCustomerName
    # PSNo is from UT data, and since it's not a merge key, it retains its name
    monthly_hc = df_filtered.groupby(
        ['Month_Year', 'FinalCustomerName']
    ).agg(
        Headcount=('PSNo', 'nunique') # Use PSNo for headcount
    ).reset_index()

    # 3. Merge Revenue and Headcount data
    merged_rev_hc = pd.merge(
        monthly_revenue,
        monthly_hc,
        on=['Month_Year', 'FinalCustomerName'],
        how='outer' # Use outer to keep all months/customers even if one is missing
    ).fillna(0) # Fill NaN with 0 for calculations

    if merged_rev_hc.empty:
        return {"Message": "No combined Revenue and Headcount data found for the specified period and customer names after aggregation."}

    # Convert Month_Year Period to datetime for sorting and display
    merged_rev_hc['Month'] = merged_rev_hc['Month_Year'].dt.to_timestamp()
    merged_rev_hc = merged_rev_hc.sort_values(by=['Month', 'FinalCustomerName']).reset_index(drop=True)

    # Calculate Revenue Per Person
    merged_rev_hc['Revenue_Per_Person'] = np.where(
        merged_rev_hc['Headcount'] != 0,
        merged_rev_hc['TotalRevenue'] / merged_rev_hc['Headcount'],
        np.nan # Assign NaN if Headcount is zero
    )
    merged_rev_hc.dropna(subset=['Revenue_Per_Person'], inplace=True) # Drop rows where Revenue per Person is NaN

    # Format Month for display
    merged_rev_hc['Month_Formatted'] = merged_rev_hc['Month'].dt.strftime('%Y-%m')

    return {
        "df_revenue_per_person_trend": merged_rev_hc[['Month_Formatted', 'FinalCustomerName', 'TotalRevenue', 'Headcount', 'Revenue_Per_Person']],
        "trend_dimension_display": "Account",
        "trend_granularity_display": "monthly"
    }

# ---------- NEW: REALIZED RATE DROP ANALYSIS CALCULATION ----------
def analyze_realized_rate_drop(df_merged, query_details):
    """
    Calculates quarterly Realized Rates by Final Customer Name and identifies accounts
    where the realized rate dropped by more than a specified threshold between two consecutive fiscal quarters.
    Realized Rate = Revenue / sum of "Net Available Hours"
    """
    df_copy = df_merged.copy()

    # Apply dynamic filters first
    df_copy, filter_message = apply_filters(df_copy, query_details.get("column_filters", {}), "merged data")
    if df_copy.empty:
        return {"Message": filter_message} # Return message from filter function

    # Ensure Date column is datetime and timezone-naive
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce').dt.tz_localize(None)
    df_copy.dropna(subset=['Date'], inplace=True)

    # Check for required columns after merge
    required_revenue_col = 'Amount in USD'
    required_hours_col = 'NetAvailableHours'

    # --- DEBUGGING: Check columns inside analyze_realized_rate_drop ---
    # print(f"DEBUG (analyze_realized_rate_drop): Columns in df_copy at start: {df_copy.columns.tolist()}")
    # if required_revenue_col in df_copy.columns:
    #     print(f"DEBUG (analyze_realized_rate_drop): Sample of '{required_revenue_col}':\n{df_copy[required_revenue_col].head()}")
    # else:
    #     print(f"DEBUG (analyze_realized_rate_drop): '{required_revenue_col}' NOT found in df_copy.")
    # if required_hours_col in df_copy.columns:
    #     print(f"DEBUG (analyze_realized_rate_drop): Sample of '{required_hours_col}':\n{df_copy[required_hours_col].head()}")
    # else:
    #     print(f"DEBUG (analyze_realized_rate_drop): '{required_hours_col}' NOT found in df_copy.")
    # --- END DEBUGGING ---


    if required_revenue_col not in df_copy.columns:
        return {"Message": f"Missing required column '{required_revenue_col}' (Revenue from P&L) in the merged data. Please ensure 'Amount in USD' is present in 'P&L data.csv' and the merge is successful."}
    if required_hours_col not in df_copy.columns:
        return {"Message": f"Missing required column '{required_hours_col}' (Net Available Hours from UT) in the merged data. Please ensure 'NetAvailableHours' is present in 'UT data.csv' and the merge is successful."}

    drop_threshold = query_details.get("drop_threshold")
    drop_threshold_type = query_details.get("drop_threshold_type") # 'percentage' or 'absolute'

    if drop_threshold is None:
        return {"Message": "Please specify a numeric drop threshold for realized rate (e.g., 'more than $3' or 'more than 60%')."}

    # --- NEW ROBUST LOGIC: Use dates from query_details, which are now more accurately set by parse_date_range_from_query ---
    
    current_q_start_date = query_details.get("start_date")
    current_q_end_date = query_details.get("end_date")
    prev_q_start_date = query_details.get("secondary_start_date")
    prev_q_end_date = query_details.get("secondary_end_date")

    # If for some reason dates are still None, or only one set is present, fallback to today's date logic
    if not all([current_q_start_date, current_q_end_date, prev_q_start_date, prev_q_end_date]):
        india_tz = pytz.timezone("Asia/Kolkata")
        today = datetime.now(india_tz).date()
        
        current_fy_start_cal_year, current_fq = get_fiscal_quarter_and_year(today)
        
        # Calculate start/end dates for current fiscal quarter (primary period)
        if current_fq == 1: # FY-Q1 (Apr-Jun)
            current_q_start_date = datetime(current_fy_start_cal_year, 4, 1)
            current_q_end_date = datetime(current_fy_start_cal_year, 6, 30, 23, 59, 59, 999999)
        elif current_fq == 2: # FY-Q2 (Jul-Sep)
            current_q_start_date = datetime(current_fy_start_cal_year, 7, 1)
            current_q_end_date = datetime(current_fy_start_cal_year, 9, 30, 23, 59, 59, 999999)
        elif current_fq == 3: # FY-Q3 (Oct-Dec)
            current_q_start_date = datetime(current_fy_start_cal_year, 10, 1)
            current_q_end_date = datetime(current_fy_start_cal_year, 12, 31, 23, 59, 59, 999999)
        else: # FY-Q4 (Jan-Mar)
            current_q_start_date = datetime(current_fy_start_cal_year + 1, 1, 1)
            current_q_end_date = datetime(current_fy_start_cal_year + 1, 3, 31, 23, 59, 59, 999999)

        # Calculate start/end dates for previous fiscal quarter (secondary period)
        prev_fy_start_cal_year = current_fy_start_cal_year
        prev_fq = current_fq - 1
        if prev_fq == 0: # If current is Q1, previous was Q4 of prior FY
            prev_fq = 4
            prev_fy_start_cal_year -= 1
        
        if prev_fq == 1: # FY-Q1 (Apr-Jun)
            prev_q_start_date = datetime(prev_fy_start_cal_year, 4, 1)
            prev_q_end_date = datetime(prev_fy_start_cal_year, 6, 30, 23, 59, 59, 999999)
        elif prev_fq == 2: # FY-Q2 (Jul-Sep)
            prev_q_start_date = datetime(prev_fy_start_cal_year, 7, 1)
            prev_q_end_date = datetime(prev_fy_start_cal_year, 9, 30, 23, 59, 59, 999999)
        elif prev_fq == 3: # FY-Q3 (Oct-Dec)
            prev_q_start_date = datetime(prev_fy_start_cal_year, 10, 1)
            prev_q_end_date = datetime(prev_fy_start_cal_year, 12, 31, 23, 59, 59, 999999)
        else: # FY-Q4 (Jan-Mar)
            prev_q_start_date = datetime(prev_fy_start_cal_year + 1, 1, 1)
            prev_q_end_date = datetime(prev_fy_start_cal_year + 1, 3, 31, 23, 59, 59, 999999)

    current_q_name = get_fiscal_quarter_name_from_fy_and_q(*get_fiscal_quarter_and_year(current_q_start_date.date()))
    prev_q_name = get_fiscal_quarter_name_from_fy_and_q(*get_fiscal_quarter_and_year(prev_q_start_date.date()))
    
    # print(f"DEBUG (analyze_realized_rate_drop): Final Quarters for Analysis:")
    # print(f"   Current Quarter: {current_q_name} ({current_q_start_date.strftime('%Y-%m-%d')} to {current_q_end_date.strftime('%Y-%m-%d')})")
    # print(f"   Previous Quarter: {prev_q_name} ({prev_q_start_date.strftime('%Y-%m-%d')} to {prev_q_end_date.strftime('%Y-%m-%d')})")
    # --- END NEW ROBUST LOGIC ---

    # Filter data for the two quarters
    current_q_df = df_copy[
        (df_copy['Date'] >= current_q_start_date) & (df_copy['Date'] <= current_q_end_date)
    ].copy()
    prev_q_df = df_copy[
        (df_copy['Date'] >= prev_q_start_date) & (df_copy['Date'] <= prev_q_end_date)
    ].copy()

    # --- NEW: More granular empty checks ---
    if current_q_df.empty:
        return {"Message": f"No raw data found for {current_q_name} in the input files for realized rate calculation after applying filters."}
    if prev_q_df.empty:
        return {"Message": f"No raw data found for {prev_q_name} in the input files for realized rate calculation after applying filters."}
    # --- END NEW ---

    # Calculate total revenue (from P&L side) and net available hours (from UT side) per account for each quarter
    # Ensure to filter for revenue groups from the P&L side of the merged data
    current_q_agg = current_q_df[current_q_df["Group1"].isin(REVENUE_GROUPS)].groupby('FinalCustomerName').agg(
        TotalRevenue=(required_revenue_col, 'sum'),
        TotalNetAvailableHours=(required_hours_col, 'sum')
    ).reset_index()

    prev_q_agg = prev_q_df[prev_q_df["Group1"].isin(REVENUE_GROUPS)].groupby('FinalCustomerName').agg(
        TotalRevenue=(required_revenue_col, 'sum'),
        TotalNetAvailableHours=(required_hours_col, 'sum')
    ).reset_index()

    # Calculate Realized Rate for each quarter
    current_q_agg['Realized_Rate'] = np.where(
        current_q_agg['TotalNetAvailableHours'] != 0,
        current_q_agg['TotalRevenue'] / current_q_agg['TotalNetAvailableHours'],
        np.nan
    )
    prev_q_agg['Realized_Rate'] = np.where(
        prev_q_agg['TotalNetAvailableHours'] != 0,
        prev_q_agg['TotalRevenue'] / prev_q_agg['TotalNetAvailableHours'],
        np.nan
    )

    # Merge the quarterly data
    merged_rates_initial = pd.merge(
        current_q_agg[['FinalCustomerName', 'Realized_Rate']],
        prev_q_agg[['FinalCustomerName', 'Realized_Rate']],
        on='FinalCustomerName',
        how='inner', # Only consider accounts present in both quarters
        suffixes=(f'_{current_q_name}', f'_{prev_q_name}')
    )

    # --- NEW: Check for common accounts after initial merge ---
    if merged_rates_initial.empty:
        return {"Message": f"No common accounts found between {current_q_name} and {prev_q_name} with revenue and hours data for realized rate calculation after applying filters."}
    # --- END NEW ---

    # Drop rows where realized rate is NaN in either quarter
    merged_rates = merged_rates_initial.dropna(subset=[f'Realized_Rate_{current_q_name}', f'Realized_Rate_{prev_q_name}']).copy()

    # --- NEW: Check if any valid realized rate data remains after dropping NaNs ---
    if merged_rates.empty:
        return {"Message": f"No accounts have complete (non-zero hours) realized rate data for both {current_q_name} and {prev_q_name} after initial filtering and NaN removal."}
    # --- END NEW ---

    # --- NEW: Calculate drop based on drop_threshold_type ---
    if drop_threshold_type == 'percentage':
        # Calculate percentage drop: ((Prev Rate - Current Rate) / Prev Rate) * 100
        merged_rates['Rate_Drop_Value'] = np.where(
            merged_rates[f'Realized_Rate_{prev_q_name}'] != 0,
            ((merged_rates[f'Realized_Rate_{prev_q_name}'] - merged_rates[f'Realized_Rate_{current_q_name}']) / merged_rates[f'Realized_Rate_{prev_q_name}']) * 100,
            np.nan # Handle division by zero
        )
        drop_unit_display = "%"
        drop_col_name = "Rate Drop (%)"
    else: # 'absolute' (or fallback)
        # Calculate absolute drop: Prev Rate - Current Rate
        merged_rates['Rate_Drop_Value'] = merged_rates[f'Realized_Rate_{prev_q_name}'] - merged_rates[f'Realized_Rate_{current_q_name}']
        drop_unit_display = "$"
        drop_col_name = "Rate Drop (USD)"

    # Drop rows where Rate_Drop_Value is NaN (e.g., due to division by zero in percentage calc)
    merged_rates.dropna(subset=['Rate_Drop_Value'], inplace=True)

    # Filter for accounts where the drop is more than the threshold
    accounts_with_drop = merged_rates[merged_rates['Rate_Drop_Value'] > drop_threshold].copy()

    if accounts_with_drop.empty:
        # Format threshold for message
        formatted_threshold = f"${drop_threshold:,.2f}" if drop_threshold_type == 'absolute' else f"{drop_threshold:.2%}"
        return {"Message": f"No accounts experienced a realized rate drop of more than {formatted_threshold} from {prev_q_name} to {current_q_name} after applying all criteria."}
    else:
        accounts_with_drop = accounts_with_drop.sort_values(by='Rate_Drop_Value', ascending=False).reset_index(drop=True)
        accounts_with_drop.insert(0, "S.No", accounts_with_drop.index + 1)
        
        # Rename columns for display
        accounts_with_drop.rename(columns={
            f'Realized_Rate_{current_q_name}': f'Realized Rate ({current_q_name})',
            f'Realized_Rate_{prev_q_name}': f'Realized Rate ({prev_q_name})',
            'Rate_Drop_Value': drop_col_name # Dynamic column name
        }, inplace=True)

        return {
            "df_realized_rate_drop": accounts_with_drop[['S.No', 'FinalCustomerName', f'Realized Rate ({prev_q_name})', f'Realized Rate ({current_q_name})', drop_col_name]],
            "current_quarter_name": current_q_name,
            "previous_quarter_name": prev_q_name,
            "drop_threshold": drop_threshold,
            "drop_threshold_type": drop_threshold_type # Pass this for display message in mainn.py
        } 
