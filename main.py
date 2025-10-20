# main.py (your entry point)
import os
from collections import defaultdict

import streamlit as st
import pandas as pd
import plotly.express as px
import math

from st_aggrid import GridUpdateMode, ColumnsAutoSizeMode
from st_aggrid import AgGrid, GridOptionsBuilder
import json

import plotly.express as px
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import yfinance as yf  # For live FX rates
import requests
from typing import Dict, List, Optional
from globals_manager import set_global, get_global, clear_global
from datetime import datetime, timedelta
import pytz
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe

# Configure page - keep it as wide for desktop
st.set_page_config(
    page_title="JP Empire",
    layout="wide",  # Keep wide for desktop
    initial_sidebar_state="auto"
)

# Add this CSS specifically for iPhone Pro Max and other mobile devices
st.markdown("""
<style>
    

    /* Mobile styles - will override desktop styles on small screens */
    @media (max-width: 430px) {
        .block-container {
            max-width: 400px !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
            padding-bottom: 5rem !important;  /* Added bottom padding */
            margin-left: auto !important;
            margin-right: auto !important;
        }

        .stApp {
            margin-top: 10px !important;
        }

        /* Force centered layout on mobile */
        .reportview-container .main {
            max-width: 100% !important;
        }

        section.main .block-container {
            padding: 1rem 1rem 5rem 1rem !important;  /* Added bottom padding here too */
        }

        /* Fix metric containers to prevent text truncation */
        [data-testid="metric-container"] {
            min-width: unset !important;
            width: 100% !important;
            max-width: 100% !important;
            padding: 8px !important;
            word-wrap: break-word !important;
            white-space: normal !important;
            overflow: visible !important;
        }

        /* Prevent text truncation in metric labels and values */
        [data-testid="metric-container"] * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
        }

        /* Specific targeting for metric text */
        [data-testid="metric-container"] div {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
        }

        /* Make sure the container doesn't restrict content */
        .stMetric {
            min-width: unset !important;
            max-width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)


## Every year change starting_balance =, starting_capital = and base_risk =

def calculate_mae_recommendations(selected_pair, risk_multiplier, min_trades=10):
    """
    Calculate MAE recommendations dynamically from Google Sheets data
    Fetches last 20 records for the specific symbol+strategy combination

    Parameters:
    - selected_pair: The trading pair
    - risk_multiplier: The strategy
    - min_trades: Minimum number of trades required (default: 20)

    Returns:
    - tuple: (suggested_stop_loss, avg_winning_mae) or (None, None)
    """

    try:
        # Load fresh data from Google Sheets
        df = load_data_from_sheets("Trade", "Trade.csv")

        if df is None or df.empty:
            return None, None

        # Filter for current symbol and strategy
        filtered_data = df[
            (df['Symbol'] == selected_pair) &
            (df['Strategy'] == risk_multiplier)
            ].copy()

        # Sort by date to get most recent trades (newest first)
        if 'Date' in filtered_data.columns:
            filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], errors='coerce')
            filtered_data = filtered_data.sort_values('Date', ascending=False)

        # Take the last 20 records (most recent)
        recent_trades = filtered_data.head(min_trades)

        # Check if we have enough trades (20 total trades like tab6)
        trade_count = len(recent_trades)
        if trade_count < min_trades:
            return None, None

        # Ensure we have the required columns
        required_columns = ['Maximum Adverse Excursion', 'PnL', 'Result']
        if not all(col in recent_trades.columns for col in required_columns):
            return None, None

        # Clean the data - remove any rows with invalid MAE values
        recent_trades = recent_trades.dropna(subset=['Maximum Adverse Excursion'])

        # Re-check count after cleaning
        if len(recent_trades) < min_trades:
            return None, None

        # CRITICAL CHANGE: Use only WINNING TRADES for MAE analysis
        winning_trades = recent_trades[recent_trades['Result'] == 'Win']

        # CRITICAL FIX: Check if we have ANY winning trades (like tab6 does)
        if winning_trades.empty:
            return None, None

        # Calculate MAE percentiles from WINNING TRADES only (uncensored data)
        winning_mae_percentiles = np.percentile(winning_trades['Maximum Adverse Excursion'], [25, 50, 75, 90, 95])

        # Use 90th percentile of WINNING trades (not 75th of all trades)
        suggested_stop_loss = winning_mae_percentiles[3]  # 90th percentile of winners

        # Calculate average winning trade MAE
        avg_winning_mae = winning_trades['Maximum Adverse Excursion'].mean()

        return suggested_stop_loss, avg_winning_mae

    except Exception as e:
        st.error(f"Error calculating MAE recommendations: {e}")
        return None, None

def clean_data_for_google_sheets(df):
    """
    Clean data specifically for Google Sheets to avoid JSON serialization errors
    """
    if df is None or df.empty:
        return df

    df_clean = df.copy()

    # Handle Date column specifically - ensure YYYY-MM-DD format
    if 'Date' in df_clean.columns:
        # Convert to string first
        df_clean['Date'] = df_clean['Date'].astype(str)

        # Remove time components if present and format to YYYY-MM-DD
        df_clean['Date'] = df_clean['Date'].str.split().str[0]

        # Ensure consistent format
        try:
            # Try to parse and reformat any dates
            dates_parsed = pd.to_datetime(df_clean['Date'], errors='coerce')
            valid_dates = ~dates_parsed.isna()
            df_clean.loc[valid_dates, 'Date'] = dates_parsed[valid_dates].dt.strftime('%Y-%m-%d')
        except:
            # If parsing fails, keep as is
            pass

        df_clean['Date'] = df_clean['Date'].fillna('')

    # Ensure Variance is treated as string
    if 'Variance' in df_clean.columns:
        df_clean['Variance'] = df_clean['Variance'].astype(str)
        variance_mapping = {
            '50.0': '50',
            '559.0': '559-66',
            '66.0': '66-805',
            '805.0': '>805'
        }
        df_clean['Variance'] = df_clean['Variance'].replace(variance_mapping)

    # Convert other datetime columns to strings
    for col in df_clean.columns:
        if col != 'Date' and (pd.api.types.is_datetime64_any_dtype(df_clean[col]) or hasattr(df_clean[col], 'dt')):
            df_clean[col] = df_clean[col].astype(str)

    # Ensure all other object types are strings
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object' and col != 'Date':
            df_clean[col] = df_clean[col].astype(str)

    # Convert numeric columns (excluding Variance)
    numeric_columns = ['PnL', 'RR', 'PROP_Pct', 'Risk_Percentage', 'Lot_Size',
                       'Starting_Balance', 'Ending_Balance', 'Withdrawal_Deposit']

    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(0.0)
            df_clean[col] = df_clean[col].apply(lambda x: float(x) if pd.notna(x) else 0.0)

    # Handle any remaining NaN values
    for col in df_clean.columns:
        if col not in numeric_columns and col != 'Variance':
            df_clean[col] = df_clean[col].fillna('')
            df_clean[col] = df_clean[col].astype(str)

    return df_clean


def clean_data_for_calculations(df):
    """
    Clean data to match CSV upload format and ensure proper data types
    This fixes the Google Sheets vs CSV data type differences
    """
    if df is None or df.empty:
        return df

    df_clean = df.copy()

    # Define column mappings and expected types
    numeric_columns = [
        'PnL', 'RR', 'PROP_Pct', 'Risk_Percentage', 'Lot_Size',
        'Starting_Balance', 'Ending_Balance', 'Withdrawal_Deposit',
        'MonthNum', 'Cash_Flow', 'Risk_Amount'
    ]

    date_columns = ['Date', 'Entry_Time', 'Exit_Time']

    string_columns = [
        'Symbol', 'Direction', 'Strategy', 'Result', 'Grade',
        'Month', 'Year', 'MonthYear', 'Variance', 'Trend Position'
    ]

    # Clean numeric columns
    for col in numeric_columns:
        if col in df_clean.columns:
            # Convert to string first to handle any Google Sheets formatting
            series_str = df_clean[col].astype(str)

            # Remove common non-numeric characters that might come from Google Sheets
            series_clean = series_str.str.replace(r'[^\d.-]', '', regex=True)

            # Handle empty strings and convert to numeric
            series_clean = series_clean.replace('', '0').replace('nan', '0')

            df_clean[col] = pd.to_numeric(series_clean, errors='coerce')
            df_clean[col] = df_clean[col].fillna(0)

    # Handle Variance as string
    if 'Variance' in df_clean.columns:
        # Convert to string and handle any numeric values that might be there
        df_clean['Variance'] = df_clean['Variance'].astype(str)
        # Map any legacy numeric values to their string equivalents
        variance_mapping = {
            '50.0': '50',
            '559.0': '559-66',
            '66.0': '66-805',
            '805.0': '>805',
            '50': '50',
            '559': '559-66',
            '66': '66-805',
            '805': '>805'
        }
        df_clean['Variance'] = df_clean['Variance'].replace(variance_mapping)

    # Clean date columns - be more careful not to break existing formats
    for col in date_columns:
        if col in df_clean.columns:
            # First, ensure it's string to see what we're working with
            df_clean[col] = df_clean[col].astype(str)

            # Only convert to datetime if it looks like it needs conversion
            # Check if we have datetime strings with time components
            has_time = df_clean[col].str.contains(r'\d{1,2}:\d{2}:\d{2}', na=False, regex=True)

            if has_time.any():
                # Convert only the ones with time components to datetime and format
                df_clean.loc[has_time, col] = pd.to_datetime(
                    df_clean.loc[has_time, col], errors='coerce'
                ).dt.strftime('%Y-%m-%d')

            # Fill any NaT values with empty string
            df_clean[col] = df_clean[col].fillna('')

    # Clean string columns - ensure they're proper strings
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Replace 'nan' strings with empty string
            df_clean[col] = df_clean[col].replace('nan', '')
            df_clean[col] = df_clean[col].fillna('')

    return df_clean


def clean_trading_data(df):
    """Clean and convert data types for trading data"""
    if df is None or df.empty:
        return df

    df_clean = df.copy()

    # Convert numeric columns
    numeric_columns = ['PnL', 'RR', 'PROP_Pct', 'Risk_Percentage', 'Lot_Size', 'Starting_Balance', 'Ending_Balance']

    for col in numeric_columns:
        if col in df_clean.columns:
            # Remove any non-numeric characters and convert to float
            df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace('[^\d.-]', '', regex=True),
                                          errors='coerce')

    # Convert date columns
    date_columns = ['Date', 'Entry_Time', 'Exit_Time']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    # Clean string columns
    string_columns = ['Symbol', 'Direction', 'Strategy', 'Result', 'Grade']
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean


# Calculate metrics
def calculate_metrics_2(data):
    if data.empty:
        return pd.DataFrame()

    metrics = data.groupby(['Year', 'Month', 'MonthYear']).agg(
        Total_Trades=('Result', 'count'),
        Wins=('Result', lambda x: (x == 'Win').sum()),
        Losses=('Result', lambda x: (x == 'Loss').sum()),
        Win_Rate=('Result', lambda x: (x == 'Win').mean() * 100),
        Avg_RR=('RR', 'mean'),
        Total_PnL=('PnL', 'sum'),
        Avg_PnL=('PnL', 'mean')
    ).reset_index()

    metrics['Month_Year_Order'] = pd.to_datetime(metrics['MonthYear'].astype(str))
    metrics = metrics.sort_values('Month_Year_Order')

    return metrics


def calculate_next_risk_percentage(trades, ending_balance):
    base_risk = 0.02  # 2%
    total_gain = trades['PnL'].sum()

    def smoothed_tiered_risk(cumulative_gain_pct):
        """
        Calculate smoothed tiered risk percentage based on cumulative gain.

        Parameters:
        cumulative_gain_percent (float): Cumulative gain in percentage since monthly reset.

        Returns:
        float: Risk percentage (capped at 1.5%).
        """

    if (total_gain > 0):
        cumulative_gain = round(total_gain / ending_balance * 100, 2)
        current_risk = base_risk
        word = str(cumulative_gain) + "%" + " / $" + str(total_gain)
    else:
        current_risk = base_risk
        word = "Need Gains for the Month"
    return current_risk, word, total_gain


def calculate_strategy_grade(winrate, num_trades):
    """Determine the strategy grade based on winrate and number of trades"""
    for grade, criteria in sorted(TRADING_GRADE_SCALE.items(), key=lambda x: x[1]['min_winrate'], reverse=True):
        if winrate >= criteria['min_winrate'] and num_trades >= criteria['min_trades']:
            return grade, criteria['multiplier']
        elif num_trades < criteria['min_trades']:
            return "Under Assessment", 1.0
    return "Winrate Too Low", 0.0


def calculate_strategy_grade_static(strategy_name):
    """
    Returns a static grade and multiplier based on the strategy name.

    Parameters:
    - strategy_name (str): Name of the trading strategy

    Returns:
    - tuple: (grade, multiplier) or (None, None) if strategy not found
    """

    # Define the static mapping of strategy grades and multipliers
    strategy_grades = {
        # 1 Touch - fib>percentage>wick

        '1_BNR': ("A", 0.91),  # Fib + wick/ob
        "1_BNR_TPF": ("A", 1.0),  # wick/ob

        # 2 Touch fib>wick>percentage

        "2_BNR": ("A", 1.1),  # Fib + wick/ob
        "2_BNR_TPF": ("A", 1.2),  # Fake out or wick/ob

        # 3 Touch  wick>percentage>fib

        # "3_BNR_TPF": ("A", 1), # Fib + wick/ob + 50% outside of discount zone

    }

    # Lookup the strategy (case insensitive)
    for strategy, (grade, multiplier) in strategy_grades.items():
        if strategy_name.lower() == strategy.lower():
            return grade, multiplier

    # Return None if strategy not found
    return None, None


def set_sect_grup(pair_name):
    """
    Returns a static grade and multiplier based on the strategy name.

    Parameters:
    - strategy_name (str): Name of the trading strategy

    Returns:
    - tuple: (grade, multiplier) or (None, None) if strategy not found
    """

    # Define the static mapping of strategy grades and multipliers
    xxxaud_group = 2
    yen_group = 2
    trade_group = 2
    Europe_group = 2
    gold = 2


def analyze_strategy(group):
    """Calculate metrics for each strategy group"""

    filtered_group = group[group['Result'] != "BE"]
    wins = (filtered_group['Result'] == "Win").sum()
    total_no_be = len(filtered_group)

    total = len(group)
    if (wins <= 0):
        winrate = 0
    else:
        winrate = (wins / total_no_be) * 100 if total > 0 else 0
    # avg_return = group['PnL'].mean()

    win_group = group[group['Result'] == "Win"]
    if (len(win_group) <= 0):
        avg_win_rr = 0
    else:
        avg_win_rr = win_group['RR'].sum() / len(win_group)

    total_return = group['PnL'].sum()

    # if(len(group)<20):
    grade, multiplier = calculate_strategy_grade_static(group.name)
    # else:
    # grade, multiplier = calculate_strategy_grade(winrate, total_no_be)

    return pd.Series({
        'Win Rate (%)': winrate,
        'Total Trades': total_no_be,
        'Total Return': total_return,
        'Avg Win RR': avg_win_rr,
        'Grade': grade,
        'Multiplier': multiplier
    })


def calculate_be_rate(len_be, len_df):
    """
    Calculate BE rate percentage where PnL is >= 0 and < 3000
    """
    # Filter rows that meet the BE condition

    # Calculate percentage
    if len_be > 0:
        be_rate = f"{(len_be / len_df) * 100:.1f}%"
    else:
        be_rate = 0

    return be_rate


def styled_metric(label, value, delta=None, label_size="20px", value_size="16px"):
    html = f"""
    <div style="
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    ">
        <div style="
            font-size: {label_size};
            font-weight: bold;
        ">{label}</div>
        <div style="
            font-size: {value_size};
            font-weight: 800;
            margin: 5px 0;
        ">{value}</div>
        {f'<div style="font-family: \'{label_font}\', sans-serif; font-size: 12px;">{delta}</div>' if delta else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def styled_metric_value(value, delta=None, label_size="20px", value_size="16px"):
    html = f"""
    <div style="
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    ">
        <div style="
            font-size: {value_size};
            font-weight: 800;
            margin: 5px 0;
        ">{value}</div>
        {f'<div style="font-family: \'{label_font}\', sans-serif; font-size: 12px;">{delta}</div>' if delta else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# === SAFE SESSION PERSISTENCE ===
def initialize_persistent_session():
    """Initialize session state without interfering with widgets"""
    try:
        if os.path.exists("app_session.json"):
            with open("app_session.json", "r") as f:
                saved_data = json.load(f)

            # Only restore specific keys that won't conflict with widgets
            safe_keys = ['uploaded_data_filename', 'current_page', 'saved_records', 'file_processed',
                         'cloud_data_loaded']
            for key in safe_keys:
                if key in saved_data and key not in st.session_state:
                    st.session_state[key] = saved_data[key]

            # Reload uploaded data if filename exists
            if (st.session_state.get('uploaded_data_filename') and
                    os.path.exists(st.session_state.uploaded_data_filename)):
                try:
                    st.session_state.uploaded_data = pd.read_csv(st.session_state.uploaded_data_filename)
                    st.session_state.file_processed = True
                except Exception as e:
                    print(f"Could not reload data file: {e}")

    except Exception as e:
        print(f"Session restore warning: {e}")


def save_persistent_session():
    """Save session state without widget interference"""
    try:
        # Only save specific safe keys
        safe_data = {}
        safe_keys = ['current_page', 'saved_records', 'file_processed', 'uploaded_data_filename', 'cloud_data_loaded']

        for key in safe_keys:
            if key in st.session_state:
                safe_data[key] = st.session_state[key]

        with open("app_session.json", "w") as f:
            json.dump(safe_data, f)

    except Exception as e:
        print(f"Session save error: {e}")


def clear_persistent_session():
    """Clear persisted session data"""
    try:
        if os.path.exists("app_session.json"):
            os.remove("app_session.json")
        if st.session_state.get('uploaded_data_filename') and os.path.exists(st.session_state.uploaded_data_filename):
            os.remove(st.session_state.uploaded_data_filename)
    except:
        pass


# Initialize persistent session
initialize_persistent_session()
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
# Initialize session state variables
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'saved_records' not in st.session_state:
    st.session_state.saved_records = []
if 'file_processed' not in st.session_state:  # Add this flag
    st.session_state.file_processed = False
if 'cloud_data_loaded' not in st.session_state:
    st.session_state.cloud_data_loaded = False

# Load persistent data (do this once at startup)
if 'session_initialized' not in st.session_state:
    initialize_persistent_session()
    st.session_state.session_initialized = True

if not st.session_state.current_page:
    st.session_state.current_page = "Home"

@st.cache_resource
def get_google_sheets_client():
    """Initialize Google Sheets connection - supports both Streamlit secrets and GitHub secrets"""
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']

        creds_dict = None

        # Option 1: Check for GitHub Actions secrets (environment variables)
        if all(key in os.environ for key in ['GCP_TYPE', 'GCP_PRIVATE_KEY', 'GCP_CLIENT_EMAIL']):
            st.sidebar.info(" Using GitHub Actions secrets")
            creds_dict = {
                "type": os.environ['GCP_TYPE'],
                "project_id": os.environ.get('GCP_PROJECT_ID', ''),
                "private_key_id": os.environ.get('GCP_PRIVATE_KEY_ID', ''),
                "private_key": os.environ['GCP_PRIVATE_KEY'].replace('\\n', '\n'),
                "client_email": os.environ['GCP_CLIENT_EMAIL'],
                "client_id": os.environ.get('GCP_CLIENT_ID', ''),
                "auth_uri": os.environ.get('GCP_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                "token_uri": os.environ.get('GCP_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                "auth_provider_x509_cert_url": os.environ.get('GCP_AUTH_PROVIDER_CERT_URL',
                                                              'https://www.googleapis.com/oauth2/v1/certs'),
                "client_x509_cert_url": os.environ.get('GCP_CLIENT_CERT_URL', '')
            }

        # Option 2: Check for Streamlit secrets (secrets.toml)
        elif "gcp_service_account" in st.secrets:
            st.sidebar.info(" Using Streamlit secrets")
            creds_dict = st.secrets["gcp_service_account"]

        else:
            st.sidebar.error(" No Google Sheets credentials found")
            st.sidebar.info("Configure either GitHub Actions secrets or Streamlit secrets")
            return None

        # Validate we have the minimum required fields
        if not creds_dict or not creds_dict.get('private_key') or not creds_dict.get('client_email'):
            st.sidebar.error(" Missing required credentials (private_key or client_email)")
            return None

        #st.sidebar.success(f" Service Account: {creds_dict.get('client_email', 'Unknown')}")

        # Create credentials
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)

        #st.sidebar.success(" Google Sheets connected successfully!")
        return client

    except Exception as e:
        st.sidebar.error(f" Google Sheets connection failed: {str(e)}")
        return None


def load_data_from_sheets(sheet_name="Trade", worksheet_name="Trade.csv"):
    """Load data from Google Sheets and clean it"""
    try:
        client = get_google_sheets_client()
        if client is None:
            return None

        # Try to access the sheet
        try:
            sheet = client.open(sheet_name)
            worksheet = sheet.worksheet(worksheet_name)
            records = worksheet.get_all_records()

            if records:
                df = pd.DataFrame(records)
                #st.sidebar.success(f" Loaded {len(df)} rows from Google Sheets")

                # CLEAN THE DATA - This is the key fix!
                df_clean = clean_data_for_calculations(df)

                return df_clean
            else:
                st.sidebar.warning("Sheet found but no data returned")
                return None

        except gspread.SpreadsheetNotFound:
            available_sheets = [s.title for s in client.openall()]
            st.sidebar.error(f"Sheet '{sheet_name}' not found")
            st.sidebar.info(f"Available sheets: {available_sheets}")
            return None

        except gspread.WorksheetNotFound:
            st.sidebar.error(f"Worksheet '{worksheet_name}' not found")
            return None

    except Exception as e:
        st.sidebar.error(f"Error loading from Google Sheets: {e}")
        return None


def save_data_to_sheets(df, sheet_name="Trade", worksheet_name="Trade.csv"):
    """Save data to Google Sheets with better error handling"""
    try:
        client = get_google_sheets_client()
        if client is None:
            return False

        # Try to open existing sheet or create new one
        try:
            sheet = client.open(sheet_name)
            try:
                worksheet = sheet.worksheet(worksheet_name)
            except gspread.WorksheetNotFound:
                # Create new worksheet
                worksheet = sheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)

        except gspread.SpreadsheetNotFound:
            # Create new spreadsheet
            sheet = client.create(sheet_name)
            worksheet = sheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)

        # Ensure data is clean for JSON serialization
        df_clean = df.copy()

        # Convert all data to basic Python types
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].apply(lambda x: float(x) if pd.notna(x) else 0.0)
            else:
                df_clean[col] = df_clean[col].astype(str).fillna('')

        # Clear and update data in smaller batches if needed
        worksheet.clear()

        # Prepare data for update
        data_to_update = [df_clean.columns.values.tolist()] + df_clean.values.tolist()

        # Update in one go
        worksheet.update(data_to_update, value_input_option='RAW')

        st.sidebar.success(f" Data saved to '{worksheet_name}' in Google Sheets")
        return True

    except Exception as e:
        st.sidebar.error(f"Error saving to Google Sheets: {e}")
        return False


def replace_data_on_sheet(data, sheet_name="Trade", worksheet_name="News"):
    """Replace data in Google Sheets after filtering by Date >= today in Melbourne time"""
    try:
        # Step 1: Convert input to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Step 2: Parse 'Date' column and filter rows
        if "Date" in df.columns:
            # Convert to datetime and normalize to date only (remove time component)
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce').dt.normalize()

            # ✅ Get today's date in Australia/Melbourne timezone
            melbourne_tz = pytz.timezone("Australia/Melbourne")
            today_melbourne = datetime.now(melbourne_tz).date()

            # Convert to datetime64[ns] for comparison with normalized df["Date"]
            today = pd.to_datetime(today_melbourne)

            # Show which dates are considered past vs future
            past_dates = df[df["Date"] < today]["Date"].unique()
            future_dates = df[df["Date"] >= today]["Date"].unique()

            # Keep only rows on or after today
            df = df[df["Date"] >= today]
            df = df.reset_index(drop=True)

        else:
            st.sidebar.warning("No 'Date' column found — skipping filtering")

        # Step 3: Connect to Google Sheets
        client = get_google_sheets_client()
        if client is None:
            return False

        try:
            sheet = client.open(sheet_name)
        except gspread.SpreadsheetNotFound:
            st.sidebar.error(f"Sheet '{sheet_name}' not found")
            return False

        try:
            worksheet = sheet.worksheet(worksheet_name)
            worksheet.clear()
        except gspread.WorksheetNotFound:
            row_count, col_count = df.shape
            worksheet = sheet.add_worksheet(
                title=worksheet_name,
                rows=str(max(row_count, 100)),
                cols=str(max(col_count, 20))
            )

        # Step 4: Write filtered data to the worksheet
        set_with_dataframe(worksheet, df)
        return True

    except Exception as e:
        st.sidebar.error(f"Error in replace_data_on_sheet: {e}")
        return False

def delete_data_from_sheets(sheet_name="Trade.csv"):
    """Delete data from Google Sheets"""
    try:
        client = get_google_sheets_client()
        client.del_spreadsheet(sheet_name)
        return True
    except Exception as e:
        st.error(f"Error deleting data: {e}")
        return False


# Navigation with safe session saving
def create_nav_button(label, page, key):
    if st.sidebar.button(label, key=key):
        st.session_state.current_page = page
        # Use a small delay to avoid conflicts
        import time
        time.sleep(0.1)
        save_persistent_session()


# Core Section
st.sidebar.markdown("**Core**")
create_nav_button("Home Dashboard", "Home", "nav_home")
create_nav_button("Account Overview", "Account Overview", "nav_account")
create_nav_button("Symbol Statistics", "Symbol Stats", "nav_symbol")

st.sidebar.markdown("---")

# Trading Section
st.sidebar.markdown("**Trading**")
create_nav_button("Risk Calculation", "Risk Calculation", "Risk_Calculation")
create_nav_button("Active Opportunities", "Active Opps", "Active_Opps")
create_nav_button("Trade Signal", "Trade Signal", "Trade Signal")

st.sidebar.markdown("---")

# Analysis Section
st.sidebar.markdown("**Analysis**")
create_nav_button("Trading Guidelines", "Guidelines", "Guidelines")
create_nav_button("Performance Stats", "Stats", "Stats")

st.sidebar.markdown("---")

# Tools Section
st.sidebar.markdown("**Tools**")
create_nav_button("Entry Model Check", "Entry Criteria Check", "Entry_Criteria_Check")

# Session Management (Always show in sidebar)
with st.sidebar.expander("⚙Session Management"):
    st.write(f"Current page: `{st.session_state.current_page}`")
    if st.session_state.uploaded_data is not None:
        st.write(f"Data: {len(st.session_state.uploaded_data)} rows")
        if st.session_state.cloud_data_loaded:
            st.write("Source: Cloud Storage")
        else:
            st.write("Source: Local Upload")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Save Session"):
            save_persistent_session()
            st.success("Saved!")
    with col2:
        if st.button(" Clear Session"):
            clear_persistent_session()
            st.session_state.clear()
            st.rerun()

starting_capital = 50000

# Page content
if st.session_state.current_page == "Home":
    # AUTOMATIC CLOUD SYNC ON APP LOAD
    if 'auto_sync_done' not in st.session_state:
        with st.spinner(" Syncing with cloud..."):
            df = load_data_from_sheets()
            if df is not None and not df.empty:
                st.session_state.uploaded_data = df
                st.session_state.cloud_data_loaded = True
                st.session_state.file_processed = True
                st.session_state.auto_sync_done = True
                save_persistent_session()
            else:
                st.session_state.auto_sync_done = True

    # BACKGROUND SYMBOL STATS COMPUTATION
    if (st.session_state.uploaded_data is not None and
            'symbol_stats_computed' not in st.session_state and
            'performance_gap_data' not in st.session_state):

        # Run in background without displaying anything
        try:
            # Define starting_capital (you may need to adjust this based on your data)
            starting_capital = 50000  # Replace with your actual starting capital logic

            # Extract the core computation logic from Symbol Stats page
            df = st.session_state.uploaded_data.copy()

            # Convert date and extract year/month
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
            df = df.dropna(subset=['Date'])

            df['Year'] = df['Date'].dt.year
            selected_year = df['Year'].max()

            # Filter data for selected year
            year_data = df[df['Year'] == selected_year].copy()

            if not year_data.empty:
                # CORE COMPUTATION LOGIC FROM SYMBOL STATS TAB1
                base_risk = 0.02
                base_risk_money = starting_capital * base_risk
                base_risk_money_multi = base_risk * 100

                # Define symbol groups and their yearly percentage targets
                symbol_groups = {
                    'XAUUSD': {'symbols': ['XAUUSD'], 'target': base_risk * 2625},
                    'USDCAD_AUDUSD': {'symbols': ['USDCAD', 'AUDUSD'], 'target': base_risk * 1125},
                    'GBPUSD_EURUSD': {'symbols': ['GBPUSD', 'EURUSD'], 'target': base_risk * 2625},
                    'JPY_Pairs': {'symbols': ['GBPJPY', 'EURJPY', 'AUDJPY', 'USDJPY'], 'target': base_risk * 750},
                    'GBPAUD_EURAUD': {'symbols': ['GBPAUD', 'EURAUD'], 'target': base_risk * 375}
                }

                # Create a mapping from individual symbols to group names
                symbol_to_group = {}
                for group_name, group_info in symbol_groups.items():
                    for symbol in group_info['symbols']:
                        symbol_to_group[symbol] = group_name

                # Add group column to the data
                year_data_with_group = year_data.copy()
                year_data_with_group['Group'] = year_data_with_group['Symbol'].map(symbol_to_group)

                # Calculate performance metrics for each group
                symbol_performance = year_data_with_group.groupby('Group').agg({
                    'PnL': ['sum', 'mean', 'count', 'std'],
                    'RR': ['mean', 'count'],
                    'Result': lambda x: (x == 'Win').sum() / len(x) * 100 if len(x) > 0 else 0
                }).round(2)

                # Flatten column names
                symbol_performance.columns = [
                    'Total_PnL', 'Avg_PnL', 'Trade_Count', 'PnL_StdDev',
                    'Avg_RR', 'RR_Count', 'Win_Rate'
                ]

                # Ensure all groups are included, even with no trades
                all_groups = pd.DataFrame(index=symbol_groups.keys())
                symbol_performance = all_groups.join(symbol_performance, how='left')

                # Fill NaN values with appropriate defaults
                symbol_performance = symbol_performance.fillna({
                    'Total_PnL': 0,
                    'Avg_PnL': 0,
                    'Trade_Count': 0,
                    'PnL_StdDev': 0,
                    'Avg_RR': 0,
                    'RR_Count': 0,
                    'Win_Rate': 0
                })

                symbol_performance = symbol_performance.sort_values('Total_PnL', ascending=False)

                # Add Yearly Percentage Gain column
                symbol_performance['Yearly_Pct_Gain'] = symbol_performance.apply(
                    lambda row: 0 if row['Trade_Count'] == 0 else
                    (row['Total_PnL'] / starting_capital * 100) if starting_capital != 0 else 0,
                    axis=1
                ).round(2)

                # Add Target column with percentage values
                symbol_performance['Target_Pct'] = symbol_performance.index.map(
                    lambda x: symbol_groups[x]['target']
                )

                # CREATE THE DICTIONARY TO STORE GROUP PERFORMANCE GAP DATA
                group_performance_gap = {}

                # Populate the dictionary with group names and their performance gap
                for group_name in symbol_groups.keys():
                    if group_name in symbol_performance.index:
                        actual_pct_gain = symbol_performance.loc[group_name, 'Yearly_Pct_Gain']
                        target_pct = symbol_performance.loc[group_name, 'Target_Pct']
                        # Calculate gap between actual percentage gain and target percentage
                        gap_pct = (target_pct - actual_pct_gain) * 1.07
                        # Convert percentage gap to dollar amount
                        gap_dollar = round((gap_pct / 100) * starting_capital, 0)
                        group_performance_gap[group_name] = gap_dollar

                # Add PROP data to the dictionary
                prop_pct_value = df["PROP_Pct"].iloc[0] if not df.empty and "PROP_Pct" in df.columns else 0
                actual_prop_pct = prop_pct_value
                target_prop_pct = 30  # PROP target percentage
                gap_prop_pct = actual_prop_pct - target_prop_pct
                gap_prop_dollar = (gap_prop_pct / 100) * starting_capital
                group_performance_gap['PROP'] = gap_prop_dollar

                # Store in session state for cross-page access
                st.session_state.performance_gap_data = group_performance_gap
                st.session_state.symbol_stats_computed = True
                st.session_state.performance_gap_timestamp = pd.Timestamp.now()

        except Exception as e:
            # Silent fail - don't show error to user
            print(f"Background symbol stats computation failed: {e}")
            st.session_state.symbol_stats_computed = False

    # Show data management options only on Home page
    with st.sidebar.expander("Cloud Data Management", expanded=True):
        st.subheader("Google Sheets Storage")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Load from Cloud"):
                df = load_data_from_sheets()
                if df is not None and not df.empty:
                    st.session_state.uploaded_data = df
                    st.session_state.cloud_data_loaded = True
                    st.session_state.file_processed = True
                    save_persistent_session()
                    st.success(f"Loaded {len(df)} rows from cloud")
                    st.rerun()
                else:
                    st.info("No data found in cloud storage")

        with col2:
            if st.button(" Save to Cloud"):
                if st.session_state.uploaded_data is not None:
                    # Clean data before saving to fix JSON serialization issues
                    data_to_save = clean_data_for_google_sheets(st.session_state.uploaded_data)
                    success = save_data_to_sheets(data_to_save)
                    if success:
                        st.success("Data saved to cloud!")
                    else:
                        st.error("Failed to save to cloud")
                else:
                    st.warning("No data to save")

        # Manual file upload as backup - ONLY ON HOME PAGE
        st.subheader("Manual Upload (Backup)")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], key="file_uploader")
        if uploaded_file is not None and not st.session_state.file_processed:
            try:
                # Load CSV
                df_raw = pd.read_csv(uploaded_file)

                # CLEAN THE DATA to ensure consistency
                st.session_state.uploaded_data = clean_data_for_calculations(df_raw)
                st.session_state.file_processed = True

                # Auto-save to cloud
                data_to_save = clean_data_for_google_sheets(st.session_state.uploaded_data)
                success = save_data_to_sheets(data_to_save)
                if success:
                    st.success("File uploaded and saved to cloud!")
                st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")

        # Data management
        if st.session_state.uploaded_data is not None:
            st.write(f"**Current data:** {len(st.session_state.uploaded_data)} rows")
            if st.button("🗑️ Clear All Data"):
                if st.session_state.cloud_data_loaded:
                    delete_data_from_sheets()
                st.session_state.uploaded_data = None
                st.session_state.file_processed = False
                st.session_state.cloud_data_loaded = False
                st.rerun()

    # Main Home page content
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data

        # Clean the data first (without debug output)
        data = clean_data_for_calculations(data)

        # Handle Date column - ensure it's in YYYY-MM-DD format but don't break existing dates
        if 'Date' in data.columns:
            # First, ensure it's treated as string to preserve existing formats
            data['Date'] = data['Date'].astype(str)

            # Remove any time components if they exist (handle '2025-09-04 00:00:00' format)
            data['Date'] = data['Date'].str.split().str[0]  # Keep only the date part if time exists

            # For any dates that are still in datetime format, convert to YYYY-MM-DD
            # But don't break already properly formatted dates
            try:
                # Only convert if we detect datetime objects or improperly formatted dates
                mask = data['Date'].str.contains('00:00:00', na=False)
                if mask.any():
                    data.loc[mask, 'Date'] = pd.to_datetime(data.loc[mask, 'Date']).dt.strftime('%Y-%m-%d')
            except:
                # If conversion fails, keep the original values
                pass

        # Track newly added rows
        if 'original_data_count' not in st.session_state:
            # First time loading, all rows are considered "original"
            st.session_state.original_data_count = len(data)

        st.write("Your uploaded raw trading data:")

        # Add New Record Form with exact same fields and specific dropdown values
        with st.expander("➕ Add New Record", expanded=False):
            st.subheader("Add New Trading Record")

            # First row of fields
            col1, col2, col3 = st.columns(3)
            with col1:
                # Date picker with calendar
                new_date = st.date_input("Date", value=datetime.now(), key="new_date")
                # Convert to YYYY-MM-DD format
                new_date_str = new_date.strftime("%Y-%m-%d")

                # Symbol dropdown with specific values
                symbol_options = ["GBPUSD", "EURUSD", "AUDUSD", "USDJPY", "EURJPY", "GBPJPY",
                                  "AUDJPY", "USDCAD", "XAUUSD", "GBPAUD", "EURAUD"]
                new_symbol = st.selectbox("Symbol", options=symbol_options, key="new_symbol")

                # Direction dropdown
                direction_options = ["buy", "sell"]
                new_direction = st.selectbox("Direction", options=direction_options, key="new_direction")

            with col2:
                # POI dropdown
                poi_options = ["Weekly", "2_Daily"]
                new_poi = st.selectbox("POI", options=poi_options, key="new_poi")

                # Strategy dropdown (uses existing values from data)
                strategy_options = data['Strategy'].unique() if 'Strategy' in data.columns else []
                new_strategy = st.selectbox("Strategy", options=strategy_options, key="new_strategy")

                # Variance dropdown - STORE AS STRING
                variance_display = ["50", "559-66", "66-805", ">805"]
                new_variance = st.selectbox("Variance", options=variance_display, key="new_variance")
                # Store as string directly (no numeric conversion)
                new_variance_str = new_variance

            with col3:
                # Result dropdown
                result_options = ["BE", "Win", "Loss"]
                new_result = st.selectbox("Result", options=result_options, key="new_result")

                new_rr = st.number_input("RR", value=0.0, step=0.01, key="new_rr")
                new_pnl = st.number_input("PnL", value=0.0, step=0.01, key="new_pnl")


            # Second row of fields
            col4, col5, col6, col7 = st.columns(4)
            with col4:
                new_withdrawal_deposit = st.number_input("Withdrawal_Deposit", value=0.0, step=0.01,
                                                         key="new_withdrawal_deposit")

            with col5:
                new_prop_pct = st.number_input("PROP_Pct", value=0.0, step=0.01, key="new_prop_pct")

                # Trend Position dropdown
                trend_position_options = ["3%-4.99%", "5%-6.99%", "7%-8.99%", "9%-10.99%", "11%-12.99%", ">=13%"]
                new_trend_position = st.selectbox("Trend Position", options=trend_position_options,
                                                  key="new_trend_position")

            with col6:
                # Optional additional fields that might be in your data
                if 'Risk_Percentage' in data.columns:
                    new_risk_percentage = st.number_input("Risk_Percentage", value=0.0, step=0.01,
                                                          key="new_risk_percentage")
                if 'Lot_Size' in data.columns:
                    new_lot_size = st.number_input("Lot_Size", value=0.0, step=0.01, key="new_lot_size")

            with col7:
                stop_loss_pct = st.number_input("Stop Loss Percentage", value=0.0, step=0.01, key="stop_loss_pct")
                max_adverse_excursion = st.number_input("Maximum Adverse Excursion", value=0.0, step=0.01, key="max_adverse_excursion")

            if st.button("Add Record", type="primary", key="add_record_btn"):
                # Create new record with exact field names
                new_record = {}

                # Exact fields as requested
                if 'Date' in data.columns:
                    new_record['Date'] = new_date_str  # Use the formatted date string (YYYY-MM-DD)
                if 'Symbol' in data.columns:
                    new_record['Symbol'] = new_symbol
                if 'Direction' in data.columns:
                    new_record['Direction'] = new_direction
                if 'POI' in data.columns:
                    new_record['POI'] = new_poi
                if 'Strategy' in data.columns:
                    new_record['Strategy'] = new_strategy
                if 'Variance' in data.columns:
                    new_record['Variance'] = new_variance_str  # Use string value instead of numeric
                if 'Result' in data.columns:
                    new_record['Result'] = new_result
                if 'RR' in data.columns:
                    new_record['RR'] = new_rr
                if 'PnL' in data.columns:
                    new_record['PnL'] = new_pnl
                if 'Stop Loss Percentage' in data.columns:
                    new_record['Stop Loss Percentage'] = stop_loss_pct
                if 'Maximum Adverse Excursion' in data.columns:
                    new_record['Maximum Adverse Excursion'] = max_adverse_excursion
                if 'Withdrawal_Deposit' in data.columns:
                    new_record['Withdrawal_Deposit'] = new_withdrawal_deposit
                if 'PROP_Pct' in data.columns:
                    new_record['PROP_Pct'] = new_prop_pct

                # Trend Position field (existing field with space)
                if 'Trend Position' in data.columns:
                    new_record['Trend Position'] = new_trend_position

                # Optional additional fields
                if 'Risk_Percentage' in data.columns:
                    new_record['Risk_Percentage'] = new_risk_percentage
                if 'Lot_Size' in data.columns:
                    new_record['Lot_Size'] = new_lot_size
                if 'Starting_Balance' in data.columns:
                    new_record['Starting_Balance'] = 0.0
                if 'Ending_Balance' in data.columns:
                    new_record['Ending_Balance'] = 0.0

                # Add the new record to the dataframe
                new_df = pd.DataFrame([new_record])
                updated_data = pd.concat([data, new_df], ignore_index=True)

                # AUTO-RECALCULATE METRICS after adding record
                updated_data = clean_data_for_calculations(updated_data)

                # Update session state
                st.session_state.uploaded_data = updated_data
                save_persistent_session()

                st.success(" New record added successfully! Metrics recalculated.")
                st.rerun()

        # Configure grid for VIEWING ONLY (no editing)
        gb = GridOptionsBuilder.from_dataframe(data)

        # Define columns to hide (they will still be in the data, just not visible)
        columns_to_hide = [
            'Is_Loss', 'Loss_Streak', 'Year', 'Month', 'MonthNum',
            'Drawdown', 'Peak', 'equity', 'Drawdown_Limit', 'Running_Equity', 'Peak_Equity'
        ]

        # Pagination
        gb.configure_pagination(
            paginationAutoPageSize=False,
            paginationPageSize=25,
        )

        # DISABLE editing features
        gb.configure_default_column(
            filterable=True,
            sortable=True,
            resizable=True,
            editable=False,  # DISABLE editing for all columns
            min_column_width=100
        )

        # Build options
        grid_options = gb.build()

        # Manually hide the columns in the grid options
        for column in grid_options['columnDefs']:
            if column['field'] in columns_to_hide:
                column['hide'] = True
                column['editable'] = False  # Don't allow editing hidden columns

        # Display
        st.title("Trading Data Dashboard")
        st.markdown("Use the grid below to explore and filter trading data")

        # Delete records - SIMPLIFIED VERSION
        # st.subheader("Delete Records")

        # Get the current data
        current_data = st.session_state.uploaded_data

        if len(current_data) > 0:
            # Show the most recent record for reference
            last_record = current_data.iloc[-1].copy()
            # st.write(
            # f"**Last record:** Row {len(current_data)} - {last_record['Date']} - {last_record['Symbol']} - {last_record.get('Direction', 'N/A')} - PnL: {last_record.get('PnL', 'N/A')}")

            # Delete Last Record button
            if st.button("🗑️ Delete Last Record", type="secondary"):
                try:
                    # Remove the last row
                    updated_data = current_data.iloc[:-1].reset_index(drop=True)

                    # AUTO-RECALCULATE METRICS after deleting record
                    updated_data = clean_data_for_calculations(updated_data)

                    # Update session state
                    st.session_state.uploaded_data = updated_data
                    save_persistent_session()

                    st.success(" Last record deleted successfully! Metrics recalculated.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error deleting last record: {e}")
        else:
            st.write("No records available to delete.")

        # Row Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Refresh Grid"):
                st.rerun()
        with col2:
            if st.button(" Save All Changes"):
                if st.session_state.uploaded_data is not None:
                    try:
                        # Clean data before saving to fix JSON serialization issues
                        data_to_save = clean_data_for_google_sheets(st.session_state.uploaded_data)

                        if st.session_state.cloud_data_loaded:
                            success = save_data_to_sheets(data_to_save)
                            if success:
                                st.success("All changes saved to cloud!")
                            else:
                                st.error("Failed to save to cloud")
                        else:
                            st.success("Changes saved locally!")
                    except Exception as e:
                        st.error(f"Error saving data: {e}")
                else:
                    st.warning("No data to save")

        # Display the grid
        grid_response = AgGrid(
            data,
            gridOptions=grid_options,
            height=500,
            width='100%',
            theme='streamlit',
            update_mode=GridUpdateMode.NO_UPDATE,  # No updates since editing is disabled
            allow_unsafe_jscode=True,
            key="home_aggrid_main",
            enable_enterprise_modules=False,
            reload_data=True
        )

        # Show data stats
        try:
            current_data = st.session_state.uploaded_data
            st.write(f"**Total records:** {len(current_data)} rows")

            # Quick stats
            if len(grid_response['data']) < len(current_data):
                st.success(f" Filter active: Showing {len(grid_response['data'])} of {len(current_data)} rows")
            else:
                st.info(" Showing complete dataset")

        except Exception as e:
            st.error(f"Error processing grid response: {str(e)}")

    else:
        # Show welcome message when no data is loaded
        st.title("Trading Data Dashboard")
        st.info("👆 Use the **Cloud Data Management** section in the sidebar to load your data")
        st.markdown("""
        ### How to get started:
        1. **Load from Cloud** - Load your existing data from Google Sheets
        2. **Upload CSV** - Upload a new CSV file (will auto-save to cloud)
        3. **Save to Cloud** - Manually save current data to cloud storage

        Your data will be available across all pages once loaded.
        """)



elif st.session_state.current_page == "Account Overview":
    st.title("Account Overview")
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        if 'PnL' in df.columns:
            # Pure trade PnL equity curve starting from 0
            df['equity'] = df['PnL'].cumsum()

            # For display purposes - current balance is just the final equity value
            current_balance = df['equity'].iloc[-1]

        df['Peak'] = df['equity'].cummax()
        df['Drawdown'] = (df['equity'] - df['Peak']) / starting_capital
        total_return = df['equity'].iloc[-1] / abs(df['equity'].iloc[0]) if abs(df['equity'].iloc[0]) > 0 else 0
        max_drawdown = df['Drawdown'].min()
        sharpe_ratio = df['Returns'].mean() / df['Returns'].std() * np.sqrt(252) if 'Returns' in df.columns and df[
            'Returns'].std() != 0 else 0

        # Calculate longest losing streak
        df['Is_Loss'] = df['PnL'] < 0
        df['Loss_Streak'] = df['Is_Loss'].groupby((~df['Is_Loss']).cumsum()).cumsum()
        longest_losing_streak = df['Loss_Streak'].max()

        # Create figure with subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Equity Curve (Pure PnL)
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['equity'],
                       name='Equity (Pure PnL)', line=dict(color='royalblue')),
            row=1, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Drawdown'],
                       fill='tozeroy', fillcolor='rgba(255,0,0,0.2)',
                       line=dict(color='red'), name='Drawdown'),
            row=2, col=1
        )

        # Add zero line instead of starting capital line
        fig.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Zero Baseline", row=1, col=1)

        # Layout customization
        fig.update_layout(
            height=700,
            title_text="Equity Curve (Pure Trade PnL) with Drawdown",
            hovermode='x unified'
        )

        # Y-axis labels
        fig.update_yaxes(title_text="Pure PnL ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Performance Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Trade PnL", f"${current_balance:,.2f}")

        filtered_trades = df[df['Result'] != "BE"]
        filtered_be = df[df['Result'] == "BE"]

        total_trades = len(filtered_trades)

        if total_trades > 0:
            wins = len(filtered_trades[filtered_trades['Result'] == "Win"])
            winrate = (wins / total_trades) * 100
        else:
            winrate = 0
        col2.metric("Win Rate (Without BE Stat)", f"{winrate:.1f}%")

        winner_pnl = filtered_trades[filtered_trades['PnL'] > 0]
        loser_pnl = filtered_trades[filtered_trades['PnL'] < 0]
        winner_pnls = winner_pnl['PnL'].sum()
        loser_pnls = abs(loser_pnl['PnL'].sum())
        profit_factor = round(winner_pnls / loser_pnls if loser_pnls != 0 else float('inf'), 1)

        col3.metric("BE Rate", calculate_be_rate(len(filtered_be), len(df)))
        col4.metric("Profit Factor", profit_factor)
        col5.metric("Total Trades", len(df))

        # Performance Metrics2
        col1, col2, col3, col4, col5 = st.columns(5)
        # col1.metric("Net Trade PNL", f"${df['PnL'].sum():.2f}")
        col1.metric("Longest Losing Streak", f"{longest_losing_streak}")
        col2.metric("Max Drawdown", f"{max_drawdown:.2%}")
        col3.metric("Total R Gain including BE", f"{df['RR'].sum():.2f}")

        win_pnl = filtered_trades[filtered_trades['PnL'] > 0]
        win_rr_sum = win_pnl['RR'].sum()
        average_win_rr = win_rr_sum / len(win_pnl) if len(win_pnl) > 0 else 0
        col4.metric("Avg RR Per Win Trade", str(f"{average_win_rr:.2f}"))
        col5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    else:
        st.warning("Please upload data first")

elif st.session_state.current_page == "Symbol Stats":
    if st.session_state.uploaded_data is not None:
        df2 = st.session_state.uploaded_data.copy()
        df = st.session_state.uploaded_data.copy()

        # Convert date and extract year/month
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        df = df.dropna(subset=['Date'])  # Remove rows with invalid dates

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Month_Name'] = df['Date'].dt.month_name()

        st.sidebar.header("Filters")

        # Add multi-select for symbols
        all_symbols = sorted(df['Symbol'].unique())
        selected_symbols = st.sidebar.multiselect(
            "Select Symbols (All if empty)",
            options=all_symbols,
            default=all_symbols
        )

        # Filter by selected symbols if any are selected
        if selected_symbols:
            df = df[df['Symbol'].isin(selected_symbols)]

        selected_year = st.sidebar.selectbox(
            "Select Year",
            options=sorted(df['Year'].unique()),
            index=len(df['Year'].unique()) - 1
        )

        # Filter data for selected year
        year_data = df[df['Year'] == selected_year].copy()

        if year_data.empty:
            st.warning(f"No data found for {selected_year}")
        else:
            # Create tabs for better organization - ADDED TAB6
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Overall Performance",
                "Monthly Analysis",
                "Direction Analysis",
                "Strategy Analysis",
                "Visualizations",
                "MAE Analysis & Stop Loss Optimization"
            ])

            with tab1:
                base_risk = 0.02
                base_risk_money = starting_capital * base_risk
                base_risk_money_multi = base_risk * 100
                # Overall Symbol Performance
                st.header("Overall Symbol Performance")

                # Define symbol groups and their yearly percentage targets
                symbol_groups = {
                    'XAUUSD': {'symbols': ['XAUUSD'], 'target': base_risk * 2625},
                    'USDCAD_AUDUSD': {'symbols': ['USDCAD', 'AUDUSD'], 'target': base_risk * 1125},
                    'GBPUSD_EURUSD': {'symbols': ['GBPUSD', 'EURUSD'], 'target': base_risk * 2625},
                    'JPY_Pairs': {'symbols': ['GBPJPY', 'EURJPY', 'AUDJPY', 'USDJPY'], 'target': base_risk * 750},
                    'GBPAUD_EURAUD': {'symbols': ['GBPAUD', 'EURAUD'], 'target': base_risk * 375}
                }

                # Create a mapping from individual symbols to group names
                symbol_to_group = {}
                for group_name, group_info in symbol_groups.items():
                    for symbol in group_info['symbols']:
                        symbol_to_group[symbol] = group_name

                # Add group column to the data
                year_data_with_group = year_data.copy()
                year_data_with_group['Group'] = year_data_with_group['Symbol'].map(symbol_to_group)

                # Calculate performance metrics for each group
                symbol_performance = year_data_with_group.groupby('Group').agg({
                    'PnL': ['sum', 'mean', 'count', 'std'],
                    'RR': ['mean', 'count'],
                    'Result': lambda x: (x == 'Win').sum() / len(x) * 100 if len(x) > 0 else 0
                }).round(2)

                # Flatten column names (removed Long_Pct)
                symbol_performance.columns = [
                    'Total_PnL', 'Avg_PnL', 'Trade_Count', 'PnL_StdDev',
                    'Avg_RR', 'RR_Count', 'Win_Rate'
                ]

                # Ensure all groups are included, even with no trades
                all_groups = pd.DataFrame(index=symbol_groups.keys())
                symbol_performance = all_groups.join(symbol_performance, how='left')

                # Fill NaN values with appropriate defaults
                symbol_performance = symbol_performance.fillna({
                    'Total_PnL': 0,
                    'Avg_PnL': 0,
                    'Trade_Count': 0,
                    'PnL_StdDev': 0,
                    'Avg_RR': 0,
                    'RR_Count': 0,
                    'Win_Rate': 0
                })

                symbol_performance = symbol_performance.sort_values('Total_PnL', ascending=False)

                # Add a profit factor column
                symbol_performance['Profit_Factor'] = symbol_performance.apply(
                    lambda row: float('inf') if row['Trade_Count'] == 0 else
                    year_data_with_group[year_data_with_group['Group'] == row.name].groupby('Group').apply(
                        lambda x: x[x['PnL'] > 0]['PnL'].sum() / abs(x[x['PnL'] < 0]['PnL'].sum())
                        if x[x['PnL'] < 0]['PnL'].sum() != 0 else float('inf')
                    ).iloc[0],
                    axis=1
                ).round(2)

                # Add Yearly Percentage Gain column using global starting_capital
                symbol_performance['Yearly_Pct_Gain'] = symbol_performance.apply(
                    lambda row: 0 if row['Trade_Count'] == 0 else
                    (row['Total_PnL'] / starting_capital * 100) if starting_capital != 0 else 0,
                    axis=1
                ).round(2)

                # Add Target column with percentage values
                symbol_performance['Target_Pct'] = symbol_performance.index.map(
                    lambda x: symbol_groups[x]['target']
                )

                # ADD TARGET $ COLUMN - Calculate target in dollars using base_risk_money
                symbol_performance['Target_$'] = symbol_performance['Target_Pct'].apply(
                    lambda target_pct: round((base_risk_money * (target_pct / 100) * 100) / base_risk_money_multi, 2)
                ).round(2)

                # ADD ACTUAL $ COLUMN - Calculate difference between Target_$ and Total_PnL
                symbol_performance['Remain_$'] = symbol_performance.apply(
                    lambda row: row['Target_$'] - row['Total_PnL'],
                    axis=1
                ).round(2)

                # Add Target Status column with three states: Completed, In Progress, Failed
                symbol_performance['Target'] = symbol_performance.apply(
                    lambda row: "Completed" if row['Yearly_Pct_Gain'] >= row['Target_Pct'] else
                    "Failed" if row['Yearly_Pct_Gain'] < -row['Target_Pct'] else
                    "In Progress",
                    axis=1
                )

                # Get PROP_Pct value from global dataframe df2
                prop_pct_value = df2["PROP_Pct"].iloc[0] if not df2.empty and "PROP_Pct" in df2.columns else 0

                # Create PROP row with all zeros except specified columns
                prop_row = pd.DataFrame({
                    'Total_PnL': [0],
                    'Avg_PnL': [0],
                    'Trade_Count': [0],
                    'PnL_StdDev': [0],
                    'Avg_RR': [0],
                    'RR_Count': [0],
                    'Win_Rate': [0],
                    'Profit_Factor': [0],
                    'Yearly_Pct_Gain': [prop_pct_value],  # Get from global df2["PROP_Pct"].iloc[0]
                    'Target_Pct': [30],  # Set to 30 as requested
                    'Target_$': [base_risk_money * 0.30 * 100],  # Add Target $ for PROP (30% target)
                    'Target': ["Completed" if prop_pct_value >= 30 else
                               "Failed" if prop_pct_value < -30 else
                               "In Progress"]  # Apply same three-state logic
                }, index=['PROP'])

                # Add PROP row to the performance dataframe
                # symbol_performance = pd.concat([symbol_performance, prop_row])

                # CREATE THE DICTIONARY TO STORE GROUP PERFORMANCE GAP DATA
                group_performance_gap = {}

                # Populate the dictionary with group names and their performance gap (actual gain vs target)
                for group_name in symbol_groups.keys():
                    if group_name in symbol_performance.index:
                        actual_pct_gain = symbol_performance.loc[group_name, 'Yearly_Pct_Gain']
                        target_pct = symbol_performance.loc[group_name, 'Target_Pct']
                        # Calculate gap between actual percentage gain and target percentage
                        gap_pct = (target_pct - actual_pct_gain) * 1.07
                        # Convert percentage gap to dollar amount
                        gap_dollar = round((gap_pct / 100) * starting_capital, 0)
                        group_performance_gap[group_name] = gap_dollar

                # Add PROP data to the dictionary
                actual_prop_pct = prop_pct_value
                target_prop_pct = 30  # PROP target percentage
                gap_prop_pct = actual_prop_pct - target_prop_pct
                gap_prop_dollar = (gap_prop_pct / 100) * starting_capital
                group_performance_gap['PROP'] = gap_prop_dollar

                # STORE IN SESSION STATE FOR CROSS-PAGE ACCESS
                # Use a unique key to avoid conflicts
                st.session_state.performance_gap_data = group_performance_gap
                # Also store the timestamp to track when it was updated
                st.session_state.performance_gap_timestamp = pd.Timestamp.now()

                # ADD SUMMARY ROW (excluding PROP)
                # Filter out PROP row for summary calculations
                symbol_performance_no_prop = symbol_performance[symbol_performance.index != 'PROP']

                # Create summary row
                summary_row = pd.DataFrame({
                    'Total_PnL': [symbol_performance_no_prop['Total_PnL'].sum()],
                    'Avg_PnL': [symbol_performance_no_prop['Avg_PnL'].mean()],
                    'Trade_Count': [symbol_performance_no_prop['Trade_Count'].sum()],
                    'PnL_StdDev': [symbol_performance_no_prop['PnL_StdDev'].mean()],
                    'Avg_RR': [symbol_performance_no_prop['Avg_RR'].mean()],
                    'RR_Count': [symbol_performance_no_prop['RR_Count'].sum()],
                    'Win_Rate': [symbol_performance_no_prop['Win_Rate'].mean()],
                    'Profit_Factor': [symbol_performance_no_prop['Profit_Factor'].mean()],
                    'Yearly_Pct_Gain': [symbol_performance_no_prop['Yearly_Pct_Gain'].sum()],  # Sum of Yearly_Pct_Gain
                    'Target_Pct': [symbol_performance_no_prop['Target_Pct'].sum()],  # Sum of Target_Pct
                    'Remain_$': [
                        symbol_performance_no_prop['Target_$'].sum() - symbol_performance_no_prop['Total_PnL'].sum()],
                    'Target_$': [symbol_performance_no_prop['Target_$'].sum()],
                    # SUM of Target_$ (changed from mean to sum)
                    'Target': ["Completed" if symbol_performance_no_prop['Yearly_Pct_Gain'].sum() >=
                                              symbol_performance_no_prop['Target_Pct'].sum() else
                               "Failed" if symbol_performance_no_prop['Yearly_Pct_Gain'].sum() < -
                               symbol_performance_no_prop['Target_Pct'].sum() else
                               "In Progress"]  # Apply same three-state logic as other rows
                }, index=['SUMMARY'])

                # Add summary row to the performance dataframe
                symbol_performance = pd.concat([symbol_performance, summary_row])

                # Rename index to 'Symbol' for compatibility with existing plotting code
                symbol_performance = symbol_performance.rename_axis('Symbol').reset_index()

                # Display group performance with conditional formatting
                st.dataframe(
                    symbol_performance.style.format({
                        'Total_PnL': '${:,.2f}',
                        'Avg_PnL': '${:.2f}',
                        'Trade_Count': '{:.0f}',
                        'PnL_StdDev': '${:.2f}',
                        'Avg_RR': '{:.2f}',
                        'RR_Count': '{:.0f}',
                        'Win_Rate': '{:.1f}%',
                        'Profit_Factor': '{:.2f}',
                        'Yearly_Pct_Gain': '{:.2f}%',
                        'Target_Pct': '{:.2f}%',
                        'Remain_$': '${:,.2f}',
                        'Target_$': '${:,.2f}'  # Add formatting for the new Target $ column
                    }).apply(lambda x: ['background-color: dodgerblue; color: white' if x['Symbol'] == 'PROP' else
                                        'background-color: firebrick; color: white' if x['Yearly_Pct_Gain'] < -x['Target_Pct'] else
                                        'background-color: darkred; color: white' if x['Total_PnL'] < 0 else
                                        'background-color: forestgreen; color: white' if x['Total_PnL'] > 0 else
                                        'background-color: seagreen; color: white' if x['Target'] == 'Completed' else
                                        'background-color: darkorange; color: white' if x['Target'] == 'In Progress' else
                                        'background-color: dimgray; color: white' if x['Symbol'] == 'SUMMARY' else
                 '' for _ in x], axis=1),
                    use_container_width=True,
                    height=400
                )

                # Download button for the performance data
                csv = symbol_performance.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Performance Data as CSV",
                    data=csv,
                    file_name=f'group_performance_{selected_year}.csv',
                    mime='text/csv',
                )

            with tab2:
                # Monthly Performance by Symbol
                st.header("Monthly Performance by Symbol")

                monthly_symbol_perf = year_data.groupby(['Month_Name', 'Symbol', 'Month']).agg({
                    'PnL': 'sum',
                    'Result': lambda x: (x == 'Win').sum() / len(x) * 100
                }).reset_index()

                monthly_symbol_perf = monthly_symbol_perf.sort_values('Month')

                # Pivot for better visualization
                monthly_pivot = monthly_symbol_perf.pivot_table(
                    index='Symbol',
                    columns='Month_Name',
                    values='PnL',
                    aggfunc='sum',
                    fill_value=0
                )

                # Reorder columns by month number
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December']
                available_months = [m for m in month_order if m in monthly_pivot.columns]
                monthly_pivot = monthly_pivot.reindex(columns=available_months)

                st.dataframe(
                    monthly_pivot.style.format('${:,.2f}').background_gradient(cmap='RdYlGn', axis=None),
                    use_container_width=True
                )

                # Monthly heatmap visualization
                st.subheader("Monthly Performance Heatmap")

                # Prepare data for heatmap
                heatmap_data = monthly_pivot.fillna(0)

                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Month", y="Symbol", color="PnL"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    aspect="auto",
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(title=f"Monthly PnL Heatmap for {selected_year}")
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                # Direction Performance - Improved with Visualizations
                st.header("Direction Performance Analysis")

                # Calculate direction performance metrics
                direction_stats = year_data.groupby('Direction').agg({
                    'PnL': 'sum',
                    'Result': lambda x: (x == 'Win').sum() / len(x) * 100,
                    'Symbol': 'count'
                }).rename(columns={'Symbol': 'Trade_Count'}).round(2)

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Direction Win Rate Pie Chart
                    win_rate_by_direction = year_data.groupby('Direction')['Result'].apply(
                        lambda x: (x == 'Win').sum() / len(x) * 100
                    ).reset_index(name='Win_Rate')

                    fig = px.pie(
                        win_rate_by_direction,
                        values='Win_Rate',
                        names='Direction',
                        title='Win Rate by Direction',
                        color='Direction',
                        color_discrete_map={'Long': 'lightgreen', 'Short': 'lightblue'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Trade Distribution by Direction
                    trade_count_by_direction = year_data['Direction'].value_counts().reset_index()
                    trade_count_by_direction.columns = ['Direction', 'Count']

                    fig = px.pie(
                        trade_count_by_direction,
                        values='Count',
                        names='Direction',
                        title='Trade Distribution by Direction',
                        color='Direction',
                        color_discrete_map={'Long': 'lightgreen', 'Short': 'lightblue'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    # PnL by Direction
                    pnl_by_direction = year_data.groupby('Direction')['PnL'].sum().reset_index()

                    fig = px.bar(
                        pnl_by_direction,
                        x='Direction',
                        y='PnL',
                        title='Total PnL by Direction',
                        color='Direction',
                        color_discrete_map={'Long': 'lightgreen', 'Short': 'lightblue'}
                    )
                    fig.update_yaxes(tickprefix='$')
                    st.plotly_chart(fig, use_container_width=True)

                # Best Performing Direction by Symbol
                st.subheader("Best Performing Direction by Symbol")

                best_direction_by_symbol = []
                for symbol in year_data['Symbol'].unique():
                    if (symbol != "PROP"):
                        symbol_data = year_data[year_data['Symbol'] == symbol]
                        direction_stats = symbol_data.groupby('Direction').agg({
                            'PnL': 'sum',
                            'Result': lambda x: (x == 'Win').sum() / len(x) * 100
                        }).round(2)

                        if not direction_stats.empty:
                            best_direction = direction_stats.nlargest(1, 'PnL').iloc[0]
                            best_direction_name = direction_stats.nlargest(1, 'PnL').index[0]
                            best_direction_by_symbol.append({
                                'Symbol': symbol,
                                'Best_Direction': best_direction_name,
                                'Direction_PnL': best_direction['PnL'],
                                'Direction_Win_Rate': best_direction['Result']
                            })

                best_direction_df = pd.DataFrame(best_direction_by_symbol).sort_values('Direction_PnL', ascending=False)

                st.dataframe(
                    best_direction_df.style.format({
                        'Direction_PnL': '${:,.2f}',
                        'Direction_Win_Rate': '{:.1f}%'
                    }).apply(lambda x: ['background-color: lightgreen' if x['Direction_PnL'] > 0 else
                                        'background-color: lightcoral' for _ in x], axis=1),
                    use_container_width=True
                )

            with tab4:
                # Strategy Performance by Symbol
                st.header("Strategy Performance by Symbol")

                # Create a selectbox to choose a symbol for strategy analysis
                strategy_symbol = st.selectbox(
                    "Select Symbol to View Strategy Performance",
                    options=sorted(year_data['Symbol'].unique()),
                    key="strategy_symbol_select"
                )

                if strategy_symbol:
                    symbol_strategy_data = year_data[year_data['Symbol'] == strategy_symbol].copy()

                    # Calculate strategy performance for the selected symbol
                    strategy_perf = symbol_strategy_data.groupby('Strategy').agg({
                        'PnL': ['sum', 'mean', 'count'],
                        'RR': 'mean',
                        'Result': lambda x: (x == 'Win').sum() / len(x) * 100
                    }).round(2)

                    # Flatten column names
                    strategy_perf.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count', 'Avg_RR', 'Win_Rate']
                    strategy_perf = strategy_perf.sort_values('Total_PnL', ascending=False)

                    # Display strategy performance for the selected symbol
                    st.subheader(f"Strategy Performance for {strategy_symbol}")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        best_strategy = strategy_perf.index[0] if not strategy_perf.empty else "N/A"
                        best_pnl = strategy_perf['Total_PnL'].iloc[0] if not strategy_perf.empty else 0
                        st.metric("Best Strategy", best_strategy, f"${best_pnl:,.2f}")

                    with col2:
                        total_strategy_pnl = strategy_perf['Total_PnL'].sum()
                        st.metric("Total Strategy PnL", f"${total_strategy_pnl:,.2f}")

                    with col3:
                        avg_strategy_rr = strategy_perf['Avg_RR'].mean()
                        st.metric("Avg Strategy RR", f"{avg_strategy_rr:.2f}")

                    with col4:
                        total_strategy_trades = strategy_perf['Trade_Count'].sum()
                        st.metric("Total Trades", total_strategy_trades)

                    # Display strategy performance table
                    st.dataframe(
                        strategy_perf.style.format({
                            'Total_PnL': '${:,.2f}',
                            'Avg_PnL': '${:.2f}',
                            'Avg_RR': '{:.2f}',
                            'Win_Rate': '{:.1f}%'
                        }).apply(lambda x: ['background-color: lightgreen' if x['Total_PnL'] > 0 else
                                            'background-color: lightcoral' for _ in x], axis=1),
                        use_container_width=True
                    )

                # Summary of Best Strategies Across All Symbols
                st.header("Best Strategies Summary")

                # Get top strategy for each symbol
                best_strategies = []
                for symbol in year_data['Symbol'].unique():
                    if (symbol != "PROP"):
                        symbol_data = year_data[year_data['Symbol'] == symbol]
                        strategy_stats = symbol_data.groupby('Strategy').agg({
                            'PnL': 'sum',
                            'Result': lambda x: (x == 'Win').sum() / len(x) * 100
                        }).round(2)

                        if not strategy_stats.empty:
                            best_strategy = strategy_stats.nlargest(1, 'PnL').iloc[0]
                            best_strategies.append({
                                'Symbol': symbol,
                                'Best_Strategy': strategy_stats.nlargest(1, 'PnL').index[0],
                                'Strategy_PnL': best_strategy['PnL'],
                                'Strategy_Win_Rate': best_strategy['Result']
                            })

                best_strategies_df = pd.DataFrame(best_strategies).sort_values('Strategy_PnL', ascending=False)

                st.dataframe(
                    best_strategies_df.style.format({
                        'Strategy_PnL': '${:,.2f}',
                        'Strategy_Win_Rate': '{:.1f}%'
                    }).apply(lambda x: ['background-color: lightgreen' if x['Strategy_PnL'] > 0 else
                                        'background-color: lightcoral' for _ in x], axis=1),
                    use_container_width=True
                )

                # Strategy Effectiveness Matrix
                st.subheader("Strategy Effectiveness Matrix")

                # Create a matrix showing which strategies work best for which symbols
                strategy_symbol_matrix = year_data.pivot_table(
                    index='Strategy',
                    columns='Symbol',
                    values='PnL',
                    aggfunc='sum',
                    fill_value=0
                )

                # Display the matrix with color coding
                st.dataframe(
                    strategy_symbol_matrix.style.background_gradient(cmap='RdYlGn', axis=None),
                    use_container_width=True
                )

            with tab5:
                # Visualization Section
                st.header("Performance Visualization")

                col1, col2 = st.columns(2)

                with col1:
                    # Top 10 Symbols by PnL
                    top_symbols = symbol_performance.nlargest(10, 'Total_PnL')
                    fig = px.bar(
                        top_symbols.reset_index(),
                        x='Symbol',
                        y='Total_PnL',
                        title='Top 10 Symbols by PnL',
                        color='Win_Rate',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_yaxes(tickprefix='$')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Win Rate vs Avg RR
                    fig = px.scatter(
                        symbol_performance.reset_index(),
                        x='Win_Rate',
                        y='Avg_RR',
                        color='Total_PnL',
                        size='Trade_Count',
                        title='Win Rate vs Risk-Reward Ratio',
                        hover_data=['Symbol', 'Trade_Count'],
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Cumulative PnL Over Time
                st.subheader("Cumulative PnL Over Time")

                # Prepare cumulative data
                cumulative_data = year_data.sort_values('Date').groupby(['Date', 'Symbol'])['PnL'].sum().reset_index()
                cumulative_data['Cumulative_PnL'] = cumulative_data.groupby('Symbol')['PnL'].cumsum()

                fig = px.line(
                    cumulative_data,
                    x='Date',
                    y='Cumulative_PnL',
                    color='Symbol',
                    title='Cumulative PnL Over Time'
                )
                fig.update_yaxes(tickprefix='$')
                st.plotly_chart(fig, use_container_width=True)

            with tab6:
                st.header("MAE Analysis & Stop Loss Optimization")

                # Initialize session state for MAE results if not exists
                if 'mae_analysis_results' not in st.session_state:
                    st.session_state.mae_analysis_results = {}

                # Filters for MAE Analysis
                col1, col2 = st.columns(2)

                with col1:
                    mae_symbol = st.selectbox(
                        "Select Symbol for MAE Analysis",
                        options=sorted(year_data['Symbol'].unique()),
                        key="mae_symbol_select"
                    )

                with col2:
                    # Get available strategies for the selected symbol
                    symbol_strategies = year_data[year_data['Symbol'] == mae_symbol]['Strategy'].unique()
                    mae_strategy = st.selectbox(
                        "Select Strategy",
                        options=sorted(symbol_strategies),
                        key="mae_strategy_select"
                    )

                if mae_symbol and mae_strategy:
                    # Filter data for selected symbol and strategy
                    mae_data = year_data[
                        (year_data['Symbol'] == mae_symbol) &
                        (year_data['Strategy'] == mae_strategy)
                        ].copy()

                    if mae_data.empty:
                        st.warning(f"No data found for {mae_symbol} with strategy {mae_strategy}")
                    else:
                        st.subheader(f"MAE vs PnL Analysis for {mae_symbol} - {mae_strategy}")

                        # Basic statistics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            total_trades = len(mae_data)
                            st.metric("Total Trades", total_trades)

                        with col2:
                            win_rate = (mae_data['Result'] == 'Win').sum() / total_trades * 100
                            st.metric("Win Rate", f"{win_rate:.1f}%")

                        with col3:
                            # Calculate MAE for WINNING TRADES only (uncensored data)
                            winning_trades = mae_data[mae_data['Result'] == 'Win']
                            avg_winning_mae = winning_trades[
                                'Maximum Adverse Excursion'].mean() if not winning_trades.empty else 0
                            st.metric("Avg Winning MAE", f"{avg_winning_mae:.4f}")

                        with col4:
                            avg_pnl = mae_data['PnL'].mean()
                            st.metric("Avg PnL", f"${avg_pnl:.2f}")

                        # MAE vs PnL Scatter Plot
                        fig = px.scatter(
                            mae_data,
                            x='Maximum Adverse Excursion',
                            y='PnL',
                            color='Result',
                            title=f'MAE vs PnL - {mae_symbol} ({mae_strategy})',
                            hover_data=['Date', 'RR', 'Stop Loss Percentage'],
                            color_discrete_map={'Win': 'green', 'Loss': 'red'}
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="black")
                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                        st.plotly_chart(fig, use_container_width=True)

                        # MAE Distribution by Result
                        col1, col2 = st.columns(2)

                        with col1:
                            fig = px.box(
                                mae_data,
                                x='Result',
                                y='Maximum Adverse Excursion',
                                title=f'MAE Distribution by Trade Result',
                                color='Result',
                                color_discrete_map={'Win': 'green', 'Loss': 'red'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            fig = px.histogram(
                                mae_data,
                                x='Maximum Adverse Excursion',
                                color='Result',
                                title='MAE Distribution Histogram',
                                barmode='overlay',
                                color_discrete_map={'Win': 'green', 'Loss': 'red'},
                                opacity=0.7
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Stop Loss Optimization Analysis - CORRECTED VERSION
                        st.subheader("Stop Loss Optimization Analysis")

                        # Calculate optimal stop loss based on MAE of WINNING TRADES only
                        winning_trades = mae_data[mae_data['Result'] == 'Win']
                        losing_trades = mae_data[mae_data['Result'] == 'Loss']

                        if not winning_trades.empty:
                            # Analyze MAE percentiles for WINNING TRADES only (uncensored data)
                            winning_mae_percentiles = np.percentile(winning_trades['Maximum Adverse Excursion'],
                                                                    [25, 50, 75, 90, 95])

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Winners MAE 25th %ile", f"{winning_mae_percentiles[0]:.4f}")
                                st.metric("Winners MAE 50th %ile", f"{winning_mae_percentiles[1]:.4f}")

                            with col2:
                                st.metric("Winners MAE 75th %ile", f"{winning_mae_percentiles[2]:.4f}")
                                st.metric("Winners MAE 90th %ile", f"{winning_mae_percentiles[3]:.4f}")

                            with col3:
                                st.metric("Winners MAE 95th %ile", f"{winning_mae_percentiles[4]:.4f}")
                                avg_winning_mae = winning_trades['Maximum Adverse Excursion'].mean()
                                st.metric("Avg Winning Trade MAE", f"{avg_winning_mae:.4f}")

                            # Current vs Suggested Stop Loss
                            current_avg_sl = mae_data['Stop Loss Percentage'].mean()

                            # Suggest optimal stop loss based on 90th percentile of WINNING trades
                            suggested_sl = winning_mae_percentiles[3]  # 90th percentile of winners

                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("Current Avg Stop Loss", f"{current_avg_sl:.4f}")

                            with col2:
                                st.metric("Suggested Stop Loss (90th %ile Winners)", f"{suggested_sl:.4f}",
                                          help="Based on 90th percentile of winning trades' MAE")

                            # Winner Survival Analysis - Better approach
                            st.subheader("Winner Survival Analysis")

                            # Test various stop loss levels from 5% to 50%
                            stop_levels = np.arange(0.05, 0.51, 0.05)
                            survival_rates = []

                            for stop_level in stop_levels:
                                # What % of winning trades would have survived this stop loss?
                                survivors = len(
                                    winning_trades[winning_trades['Maximum Adverse Excursion'] > -stop_level])
                                survival_rate = (survivors / len(winning_trades)) * 100
                                survival_rates.append({
                                    'Stop_Level': stop_level,
                                    'Survival_Rate': survival_rate,
                                    'Surviving_Trades': survivors,
                                    'Total_Winners': len(winning_trades)
                                })

                            survival_df = pd.DataFrame(survival_rates)

                            # Plot survival curve
                            fig_survival = px.line(
                                survival_df,
                                x='Stop_Level',
                                y='Survival_Rate',
                                title='Percentage of Winning Trades That Survive Various Stop Loss Levels',
                                labels={'Stop_Level': 'Stop Loss Level', 'Survival_Rate': 'Winners Surviving (%)'},
                                markers=True
                            )
                            # Add reference lines
                            fig_survival.add_hline(y=90, line_dash="dash", line_color="red",
                                                   annotation_text="90% Survival Target")
                            fig_survival.add_vline(x=current_avg_sl, line_dash="dot", line_color="orange",
                                                   annotation_text="Current Stop")
                            fig_survival.add_vline(x=suggested_sl, line_dash="dot", line_color="blue",
                                                   annotation_text="Suggested Stop")

                            st.plotly_chart(fig_survival, use_container_width=True)

                            # Display survival table
                            st.dataframe(
                                survival_df.style.format({
                                    'Stop_Level': '{:.2f}',
                                    'Survival_Rate': '{:.1f}%'
                                }).apply(lambda x: ['background-color: lightgreen' if x['Survival_Rate'] >= 90 else
                                                    'background-color: lightyellow' for _ in x], axis=1),
                                use_container_width=True
                            )

                            # Interpretation Guide
                            st.subheader("Interpretation Guide")

                            if suggested_sl > current_avg_sl * 1.1:  # More than 10% larger
                                st.warning("""
                                ** Consider WIDER Stop Loss:** 
                                - Your winning trades typically experience drawdowns up to {:.2f}% before recovering
                                - Current stop loss ({:.2f}%) may be cutting winners too early
                                - Wider stop could improve win rate and average profit
                                """.format(suggested_sl * 100, current_avg_sl * 100))

                            elif suggested_sl < current_avg_sl * 0.9:  # More than 10% smaller
                                st.info("""
                                ** Consider TIGHTER Stop Loss:**
                                - Your winning trades rarely draw down more than {:.2f}%
                                - Current stop loss ({:.2f}%) may be too wide
                                - Tighter stop could reduce risk per trade
                                """.format(suggested_sl * 100, current_avg_sl * 100))

                            else:
                                st.success("""
                                ** Current Stop Loss Appears Reasonable:**
                                - Aligns well with winning trades' behavior
                                - Good balance between letting winners run and controlling risk
                                """)

                            # AUTO-SAVE TO SESSION STATE FOR TRADE COUNT > 20
                            if total_trades > 10:
                                # Create unique key for this symbol+strategy combination
                                key = f"{mae_symbol}_{mae_strategy}"

                                # Save the analysis results
                                st.session_state.mae_analysis_results[key] = {
                                    'symbol': mae_symbol,
                                    'strategy': mae_strategy,
                                    'trade_count': total_trades,
                                    'suggested_stop_loss': suggested_sl,
                                    'avg_winning_mae': avg_winning_mae,
                                    'win_rate': win_rate,
                                    'avg_pnl': avg_pnl,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }

                                st.success(
                                    f"MAE analysis saved for {mae_symbol} - {mae_strategy} (Trade count: {total_trades})")

                        else:
                            st.warning(
                                "No winning trades found for this combination. Cannot perform stop loss optimization.")

                        # Detailed MAE Analysis Table
                        st.subheader("Detailed MAE Analysis by Trade")

                        detailed_mae = mae_data[[
                            'Date', 'Direction', 'PnL', 'Result', 'RR',
                            'Maximum Adverse Excursion', 'Stop Loss Percentage'
                        ]].copy()

                        detailed_mae = detailed_mae.sort_values('Maximum Adverse Excursion', ascending=False)

                        st.dataframe(
                            detailed_mae.style.format({
                                'PnL': '${:.2f}',
                                'RR': '{:.2f}',
                                'Maximum Adverse Excursion': '{:.4f}',
                                'Stop Loss Percentage': '{:.4f}'
                            }).apply(lambda x: ['background-color: lightgreen' if x['Result'] == 'Win' else
                                                'background-color: lightcoral' for _ in x], axis=1),
                            use_container_width=True,
                            height=400
                        )

                        # Download MAE analysis data
                        csv = detailed_mae.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download MAE Analysis Data",
                            data=csv,
                            file_name=f'mae_analysis_{mae_symbol}_{mae_strategy}_{selected_year}.csv',
                            mime='text/csv',
                        )

                # Display saved MAE results
                if st.session_state.mae_analysis_results:
                    st.subheader("Saved MAE Analysis Results")

                    saved_results_list = []
                    for key, result in st.session_state.mae_analysis_results.items():
                        saved_results_list.append({
                            'Symbol': result['symbol'],
                            'Strategy': result['strategy'],
                            'Trade_Count': result['trade_count'],
                            'Suggested_SL': f"{result['suggested_stop_loss']:.4f}",
                            'Avg_Winning_MAE': f"{result['avg_winning_mae']:.4f}",
                            'Win_Rate': f"{result['win_rate']:.1f}%",
                            'Avg_PnL': f"${result['avg_pnl']:.2f}",
                            'Last_Updated': result['timestamp']
                        })

                    saved_results_df = pd.DataFrame(saved_results_list)
                    st.dataframe(
                        saved_results_df,
                        use_container_width=True
                    )

                    # Clear saved results button
                    if st.button("Clear Saved MAE Results"):
                        st.session_state.mae_analysis_results = {}
                        st.rerun()

    else:
        st.warning("Please upload data first to analyze symbol statistics")

elif st.session_state.current_page == "Risk Calculation":

    st.markdown("""
    <style>
        /* Desktop styles (default) */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 0rem;
        }
        .stApp {
            margin-top: 15px;
        }

        /* Mobile styles - will override desktop styles on small screens */
        @media (max-width: 430px) {
            .block-container {
                max-width: 400px !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1rem !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }

            .stApp {
                margin-top: 10px !important;
            }

            /* Force centered layout on mobile */
            .reportview-container .main {
                max-width: 100% !important;
            }

            section.main .block-container {
                padding: 1rem 1rem !important;
            }

            /* Fix metric containers to prevent text truncation */
            [data-testid="metric-container"] {
                min-width: unset !important;
                width: 100% !important;
                max-width: 100% !important;
                padding: 8px !important;
                word-wrap: break-word !important;
                white-space: normal !important;
                overflow: visible !important;
            }

            /* Prevent text truncation in metric labels and values */
            [data-testid="metric-container"] * {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: unset !important;
                word-wrap: break-word !important;
                word-break: break-word !important;
            }

            /* Specific targeting for metric text */
            [data-testid="metric-container"] div {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: unset !important;
            }

            /* Make sure the container doesn't restrict content */
            .stMetric {
                min-width: unset !important;
                max-width: 100% !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for saved selections if not exists
    if 'saved_selections' not in st.session_state:
        st.session_state.saved_selections = {}

    # Initialize saved_records if not exists
    if 'saved_records' not in st.session_state:
        st.session_state.saved_records = []

    # Initialize MAE analysis results if not exists
    if 'mae_analysis_results' not in st.session_state:
        st.session_state.mae_analysis_results = {}

    # st.title("Risk Calculation")
    next_risk = 0
    if st.session_state.uploaded_data is not None:
        # Access the precomputed performance gap data directly
        def get_performance_gap_data():
            """Helper function to safely retrieve precomputed performance gap data"""
            if 'performance_gap_data' in st.session_state:
                return st.session_state.performance_gap_data.copy()
            else:
                # If not precomputed, compute on the fly as fallback
                st.warning("Performance gap data not precomputed. Computing now...")
                try:
                    # You might want to call the background computation function here
                    # For now, return empty dict
                    return {}
                except:
                    return {}


        # Usage - this will now use precomputed data
        performance_gaps = get_performance_gap_data()

        if performance_gaps:
            prop_gap = performance_gaps.get('PROP', 0)
            XAUUSD_gap = performance_gaps.get('XAUUSD', 0)
            USDCAD_AUDUSD_gap = performance_gaps.get('USDCAD_AUDUSD', 0)
            GBPUSD_EURUSD_gap = performance_gaps.get('GBPUSD_EURUSD', 0)
            JPY_Pairs_gap = performance_gaps.get('JPY_Pairs', 0)
            GBPAUD_EURAUD_gap = performance_gaps.get('GBPAUD_EURAUD', 0)
            sum_gap = XAUUSD_gap + USDCAD_AUDUSD_gap + GBPAUD_EURAUD_gap + GBPUSD_EURUSD_gap + JPY_Pairs_gap
        else:
            # Fallback values if computation failed
            XAUUSD_gap = 0
            USDCAD_AUDUSD_gap = 0
            GBPUSD_EURUSD_gap = 0
            JPY_Pairs_gap = 0
            GBPAUD_EURAUD_gap = 0
            sum_gap = 0

        a_momemtum_text = ''
        df = st.session_state.uploaded_data
        current_month_name = datetime.now().strftime("%B")

        df = st.session_state.uploaded_data
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month_name()
        df['MonthNum'] = df['Date'].dt.month
        year_data = df[df['Year'] == datetime.now().year]


        def calculate_monthly_stats_2(year_data):

            monthly_actual_loss = 0
            current_month_stats = []
            starting_balance = 50000  # Initial balance - REPLACE WITH YOUR ACTUAL STARTING BALANCE
            monthly_loss_limit = starting_balance * 0.12
            prev_month_balance = starting_balance
            # Get all months in order
            months = year_data.groupby(['MonthNum', 'Month']).size().reset_index().sort_values('MonthNum')

            for _, (month_num, month_name, _) in months.iterrows():
                month_data = year_data[year_data['MonthNum'] == month_num]
                winners = month_data[month_data['Result'] == 'Win']
                lossers = month_data[month_data['Result'] == 'Loss']

                be_trades = month_data[month_data['Result'] == 'BE']
                month_be_pnl = be_trades['PnL'].sum()

                month_loss_pnl = lossers['PnL'].sum()
                month_win_pnl = winners['PnL'].sum() + month_be_pnl

                withdraw_deposit = month_data[month_data['Withdrawal_Deposit'].notna()]
                cash_flow = withdraw_deposit['Withdrawal_Deposit'].sum()

                month_pnl = month_data['PnL'].sum()
                ending_balance = starting_balance + month_pnl

                # cash_flow = withdraw_deposit.sum()

                # Calculate monthly percentage gain compared to previous month
                if prev_month_balance != 0:
                    monthly_pct_gain = ((ending_balance - prev_month_balance) / prev_month_balance) * 100
                else:
                    monthly_pct_gain = 0

                if (len(month_data[month_data['Result'] != 'BE']) > 0):
                    winrate_month = round(len(winners) / len(month_data[month_data['Result'] != 'BE']) * 100, 1)
                    total_nobe = len(month_data[month_data['Result'] != 'BE'])
                else:
                    winrate_month = 0

                if (month_name == current_month_name):

                    ten_percent_goal = starting_balance + round(starting_balance * 0.10, 2)

                    month_data["Running_Equity"] = starting_balance + month_data["PnL"].cumsum()
                    month_data["Peak_Equity"] = month_data["Running_Equity"].cummax()

                    if (month_data["Peak_Equity"].iloc[-1] >= ten_percent_goal):
                        month_data["Drawdown_Limit"] = round(month_data["Peak_Equity"] * 0.12, 2)
                        monthly_loss_limit = starting_balance * 0.12
                        # monthly_loss_limit = month_data["Drawdown_Limit"].iloc[-1]
                    else:
                        monthly_loss_limit = starting_balance * 0.12

                        # month_data["Drawdown_Limit"] = round(month_data["Peak_Equity"] * 0.12, 2)
                        # monthly_loss_limit = month_data["Drawdown_Limit"].iloc[-1]
                        # monthly_loss_limit = round(starting_balance * 0.08, 2)

                    # monthly_loss_limit = round(starting_balance * 0.08, 2)
                    monthly_loss_limit = starting_balance * 0.12
                    monthly_actual_loss = month_loss_pnl
                    current_month_stats = month_data

                else:
                    # monthly_loss_limit = round(ending_balance * 0.08, 2)
                    monthly_actual_loss = 0
                    monthly_actual_gain = 0
                    monthly_loss_limit = starting_balance * 0.12
                prev_month_balance = ending_balance + cash_flow
                starting_balance = ending_balance + cash_flow

            return current_month_stats, monthly_loss_limit, monthly_actual_loss, ending_balance


        current_month_stats, monthly_loss_limit, monthly_actual_loss, ending_balance = calculate_monthly_stats_2(
            year_data)

        if (df['Withdrawal_Deposit'].sum() != 0):
            cash_flow = df['Withdrawal_Deposit'].cumsum()
        else:
            cash_flow = 0

        df['equity'] = df['PnL'].cumsum().iloc[-1] + starting_capital + cash_flow
        filled_equity = df['equity'].ffill()
        equity = filled_equity.iloc[-1]

        col1, col2, col3 = st.columns([1, 1, 0.01], gap="small")

        with col1:
            st.subheader("Grading Based Risk Multiplier")
            strategy_stats = df.groupby('Strategy').apply(analyze_strategy)
            strategy_stats = strategy_stats.sort_values(by=['Win Rate (%)', 'Total Return'], ascending=False)

            # Format display
            display_df = strategy_stats.copy()
            display_df['Win Rate (%)'] = display_df['Win Rate (%)'].map("{:.1f}%".format)
            display_df['Avg Win RR'] = display_df['Avg Win RR'].map("{:,.2f}".format)
            display_df['Total Return'] = display_df['Total Return'].map("${:,.2f}".format)

            st.dataframe(display_df)

        with col2:
            st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)
            st.session_state.current_risk = 1.0  # Base risk

            # st.subheader("Adaptive Risk Calculation")
            # trades = df['Result'].tolist()
            if (len(current_month_stats) > 0):
                trades = current_month_stats
                base_risk, word, total_gain = calculate_next_risk_percentage(trades, ending_balance)

            else:
                base_risk = 0.02
                word = "Need Data for the Month"
            dollar_risk = round(base_risk * starting_capital, 2)
            result_df = pd.DataFrame({
                "Metric": ["Next Trade Base Risk", "Current Monthly Gain From Peak",
                           "Monthly Drawdown Limit From Peak"],
                "Value": [f"{base_risk:.2%}" + " /$" + str(dollar_risk), str(word), "Balance $" + str(
                    round(round(monthly_loss_limit, 2) + round(monthly_actual_loss, 2), 2)) + " @" + str(
                    round((round(monthly_loss_limit, 2) + round(monthly_actual_loss, 2)) / starting_capital * 100,
                          2)) + "%"]
            }).set_index("Metric")

            st.table(result_df)
            st.markdown("&nbsp;" * 20, unsafe_allow_html=True)
            next_risk = base_risk


        # Google Sheets integration functions
        def save_workflow_to_sheets(data):
            """Save workflow data to Google Sheets"""
            try:
                if isinstance(data, list):
                    data_to_save = pd.DataFrame(data)
                else:
                    data_to_save = data.copy()

                required_columns = ['selected_pair', 'risk_multiplier', 'position_size', 'stop_pips',
                                    'entry_price', 'exit_price', 'target_price', 'status', 'timestamp']
                for col in required_columns:
                    if col not in data_to_save.columns:
                        data_to_save[col] = None

                data_clean = clean_data_for_google_sheets(data_to_save)
                success = save_data_to_sheets(data_clean, sheet_name="Trade", worksheet_name="Workflow")
                return success
            except Exception as e:
                st.error(f"Error saving workflow data: {e}")
                return False


        def load_workflow_from_sheets():
            """Load workflow data from Google Sheets"""
            try:
                df = load_data_from_sheets(sheet_name="Trade", worksheet_name="Workflow")
                if df is not None and not df.empty:
                    if 'timestamp' in df.columns:
                        df['timestamp'] = df['timestamp'].astype(str)
                    numeric_columns = ['entry_price', 'exit_price', 'target_price', 'stop_pips', 'position_size']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = df[col].replace('None', None)
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    return df
                return pd.DataFrame()
            except Exception as e:
                st.error(f"Error loading workflow data: {e}")
                return pd.DataFrame()


        # Load existing records from Google Sheets on page load
        if not st.session_state.saved_records:
            with st.spinner("Loading existing records from cloud..."):
                workflow_data = load_workflow_from_sheets()
                if not workflow_data.empty:
                    st.session_state.saved_records = workflow_data.to_dict('records')
                    # st.success(f"Loaded {len(st.session_state.saved_records)} existing records from cloud")


        # PAIR GROUPING FUNCTIONS
        def get_pair_group(selected_pair):
            """Determine which group a pair belongs to"""
            group_one = ["EURAUD", "GBPAUD"]
            group_two = ["GBPJPY", "EURJPY", "AUDJPY", "USDJPY"]
            group_three = ["AUDUSD", "USDCAD"]
            group_four = ["GBPUSD", "EURUSD"]
            group_five = ["XAUUSD"]

            if selected_pair in group_one:
                return "group_one"
            elif selected_pair in group_two:
                return "group_two"
            elif selected_pair in group_three:
                return "group_three"
            elif selected_pair in group_four:
                return "group_four"
            elif selected_pair in group_five:
                return "group_five"
            else:
                return None


        def get_group_pairs(group_name):
            """Get all pairs in a specific group"""
            groups = {
                "group_one": ["EURAUD", "GBPAUD"],
                "group_two": ["GBPJPY", "EURJPY", "AUDJPY", "USDJPY"],
                "group_three": ["AUDUSD", "USDCAD"],
                "group_four": ["GBPUSD", "EURUSD"],
                "group_five": ["XAUUSD"]
            }
            return groups.get(group_name, [])


        def check_and_handle_group_conflict(new_pair, new_final_risk, saved_records):
            """Check if new pair conflicts with existing pairs in same group and handle accordingly"""
            new_pair_group = get_pair_group(new_pair)

            if not new_pair_group:
                return saved_records, None  # No group, no conflict

            # Find existing records in the same group that are in Speculation stage
            conflicting_records = []
            for i, record in enumerate(saved_records):
                if (record.get('status') == 'Speculation' and
                        get_pair_group(record['selected_pair']) == new_pair_group):
                    conflicting_records.append((i, record, record.get('final_risk', 0)))

            if not conflicting_records:
                return saved_records, None  # No conflicts

            # Find the highest risk among existing records
            max_existing_risk = max(risk for _, _, risk in conflicting_records)
            max_risk_records = [(i, record, risk) for i, record, risk in conflicting_records if
                                risk == max_existing_risk]

            # Compare with new record
            if new_final_risk > max_existing_risk:
                # New record has higher risk - remove all conflicting
                records_to_keep = [record for i, record in enumerate(saved_records)
                                   if not (i in [idx for idx, _, _ in conflicting_records])]
                removed_pairs = [record['selected_pair'] for _, record, _ in conflicting_records]
                return records_to_keep, f"Replaced {len(conflicting_records)} lower risk pair(s): {', '.join(removed_pairs)}"

            elif new_final_risk == max_existing_risk:
                # Same risk - remove all existing and keep new one (latest wins)
                records_to_keep = [record for i, record in enumerate(saved_records)
                                   if not (i in [idx for idx, _, _ in conflicting_records])]
                removed_pairs = [record['selected_pair'] for _, record, _ in conflicting_records]
                return records_to_keep, f"Replaced {len(conflicting_records)} same risk pair(s): {', '.join(removed_pairs)}"

            else:
                # Existing has higher risk - don't add new one
                highest_pair = max_risk_records[0][1]['selected_pair']  # Get first highest risk pair
                return saved_records, f"Higher risk pair ({highest_pair} - ${max_existing_risk}) already exists in same group"


        # st.subheader("Risk Calculation")
        def check_existing_active_records(selected_pair, saved_records):
            """Check if there are existing records for the same pair or group in active stages"""
            active_stages = ['Order Ready', 'Order Placed', 'Order Filled']

            # Check for same pair in active stages
            same_pair_records = [
                record for record in saved_records
                if (record['selected_pair'] == selected_pair and
                    record.get('status') in active_stages)
            ]

            if same_pair_records:
                stages = list(set([record['status'] for record in same_pair_records]))
                return False, f"Same pair ({selected_pair}) already exists in {', '.join(stages)} stage(s). Please delete it first."

            # Check for same group in active stages
            selected_pair_group = get_pair_group(selected_pair)
            if selected_pair_group:
                same_group_records = [
                    record for record in saved_records
                    if (get_pair_group(record['selected_pair']) == selected_pair_group and
                        record.get('status') in active_stages)
                ]

                if same_group_records:
                    group_pairs = list(set([record['selected_pair'] for record in same_group_records]))
                    stages = list(set([record['status'] for record in same_group_records]))
                    return False, f"Same group pair(s) ({', '.join(group_pairs)}) already exist in {', '.join(stages)} stage(s). Please delete them first."

            return True, "No conflicts found"


        def check_cooldown_period(selected_pair, uploaded_data, cooldown_days=1):
            """Check if there's a cooldown period for the selected pair's group"""
            if uploaded_data is None or uploaded_data.empty:
                return True, "No data available"

            # Define symbol groups
            symbol_groups = {
                'XAUUSD': {'symbols': ['XAUUSD'], 'target': 0},
                'USDCAD_AUDUSD': {'symbols': ['USDCAD', 'AUDUSD'], 'target': 0},
                'GBPUSD_EURUSD': {'symbols': ['GBPUSD', 'EURUSD'], 'target': 0},
                'JPY_Pairs': {'symbols': ['GBPJPY', 'EURJPY', 'AUDJPY', 'USDJPY'], 'target': 0},
                'GBPAUD_EURAUD': {'symbols': ['GBPAUD', 'EURAUD'], 'target': 0}
            }

            # Find which group the selected pair belongs to
            selected_group = None
            for group_name, group_info in symbol_groups.items():
                if selected_pair in group_info['symbols']:
                    selected_group = group_name
                    break

            if selected_group is None:
                return True, "Pair not in any group"

            # Get all symbols in the same group
            group_symbols = symbol_groups[selected_group]['symbols']

            try:
                # Ensure Date column is datetime
                df = uploaded_data.copy()
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])

                # Filter for relevant records: same group symbols with BE or Loss results
                recent_trades = df[
                    (df['Symbol'].isin(group_symbols)) &
                    (df['Result'].isin(['BE', 'Loss']))
                    ]

                if recent_trades.empty:
                    return True, "No recent BE/Loss trades in group"

                # Get the most recent trade date in the group
                most_recent_date = recent_trades['Date'].max()

                # Get current datetime in Melbourne timezone
                melbourne_tz = pytz.timezone('Australia/Melbourne')
                current_melbourne = datetime.now(melbourne_tz)

                # Calculate cooldown end datetime (trade date + 1 day at 3 PM)
                cooldown_end_datetime = melbourne_tz.localize(
                    datetime(most_recent_date.year, most_recent_date.month, most_recent_date.day, 15, 0, 0)
                ) + timedelta(days=1)

                # Check if we're still in cooldown period
                if current_melbourne < cooldown_end_datetime:
                    # Calculate time remaining until cooldown ends at 3 PM next day
                    time_remaining = cooldown_end_datetime - current_melbourne
                    total_hours = max(0, int(time_remaining.total_seconds() // 3600))
                    remaining_minutes = max(0, int((time_remaining.total_seconds() % 3600) // 60))

                    error_msg = (
                        f"Group {selected_group} Cooldown period active! "
                        f"({total_hours}h {remaining_minutes}m remaining)"
                    )
                    return False, error_msg  # BLOCK the order
                else:
                    return True, "Cooldown period has passed"

            except Exception as e:
                return True, f"Error checking cooldown: {str(e)}"
        def get_live_rate(pair):
            url = f"https://open.er-api.com/v6/latest/{pair[:3]}"  # Base currency (e.g., "USD")
            response = requests.get(url).json()
            rate = response["rates"][pair[3:]]  # Quote currency (e.g., "JPY")
            return float(rate)  # Convert string to float!


        def get_usd_cad():
            url = "https://open.er-api.com/v6/latest/USD"
            try:
                data = requests.get(url).json()
                return float(data["rates"]["CAD"])
            except:
                st.warning("Failed to fetch live rate. Using fallback: 1.35")
                return 0  # Fallback rate


        def get_usd_chf():
            url = "https://open.er-api.com/v6/latest/USD"
            try:
                data = requests.get(url).json()
                return float(data["rates"]["CHF"])  # USD/CHF rate
            except:
                st.warning("Failed to fetch live rate. Using fallback: 0.9000")
                return 0


        def get_aud_usd():
            url = "https://open.er-api.com/v6/latest/AUD"
            try:
                data = requests.get(url).json()
                return 1 / float(data["rates"]["USD"])  # AUD/USD rate
            except:
                st.warning("Failed to fetch AUD/USD. Using fallback: 0.7000")
                return 0


        def calculate_position_size(risk_amount, stop_pips, pair):
            lot_size = 100000  # Standard lot size (100,000 units)

            if "JPY" in pair:
                current_price = get_live_rate("USDJPY")  # Live rate for accuracy
                pip_value = 1000 / current_price  # Precise JPY pip value
            elif "XAU" in pair:
                # For XAU/USD, 1 pip = $0.01 per ounce for a standard lot

                pip_value = lot_size * 0.001
            elif "CAD" in pair:
                pip_value = 10 / get_usd_cad()  # USD/CAD pip value (USD account)
            elif pair == "EURAUD" or pair == "GBPAUD":
                pip_value = 10 / get_aud_usd()

            elif "CHF" in pair:
                pip_value = 10 / get_usd_chf()
            else:
                # For other pairs, 1 pip = 0.0001
                pip_value = lot_size * 0.0001

            position_size = risk_amount / (stop_pips * pip_value)
            return round(position_size, 2)


        def calculate_position_size_propfirm(risk_amount, stop_pips, pair):
            lot_size = 100000  # Standard lot size (100,000 units)

            if "JPY" in pair:
                current_price = get_live_rate("USDJPY")  # Live rate for accuracy
                pip_value = 1000 / current_price  # Precise JPY pip value
            elif "XAU" in pair:
                # For XAU/USD, 1 pip = $0.01 per ounce for a standard lot

                pip_value = lot_size * 0.001
            elif "CAD" in pair:
                pip_value = 10 / get_usd_cad()  # USD/CAD pip value (USD account)
            elif pair == "EURAUD" or pair == "GBPAUD":
                pip_value = 10 / get_aud_usd()

            elif "CHF" in pair:
                pip_value = 10 / get_usd_chf()
            else:
                # For other pairs, 1 pip = 0.0001
                pip_value = lot_size * 0.0001

            position_size_propfirm = risk_amount / (stop_pips * pip_value)
            return round(position_size_propfirm, 2)


        currency_pairs = [
            "AUDUSD", "EURUSD", "GBPUSD", "GBPAUD", "EURAUD",
            "USDCAD", "GBPJPY", "EURJPY", "AUDJPY",
            "USDJPY", "XAUUSD"
        ]

        xxxaud = [
            "GBPAUD", "EURAUD"
        ]
        yens = [
            "GBPJPY", "EURJPY", "AUDJPY",
            "USDJPY"
        ]
        minor_yens = [
            "GBPJPY", "EURJPY", "AUDJPY"

        ]
        trade_curr = [
            "AUDUSD", "USDCAD"
        ]
        cad_sec = [
            "USDCAD"
        ]
        aud_family1 = [
            "AUDUSD", "AUDJPY", "GBPAUD", "EURAUD"
        ]

        europe_major = [
            "EURUSD", "GBPUSD"
        ]
        gold_comm = [
            "XAUUSD"
        ]

        xxxusd = [
            "AUDUSD", "USDCAD", "EURUSD", "GBPUSD"
        ]
        majors_nocad = ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        majors = ["AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDJPY", "XAUUSD"]
        majors_dollar = ["AUDUSD", "EURUSD", "GBPUSD", "XAUUSD"]
        minors = ["GBPAUD", "EURAUD", "GBPJPY", "EURJPY", "AUDJPY"]

        strategies = ['1_BNR', '1_BNR_TPF', '2_BNR', '2_BNR_TPF']
        time_frame = ['Weekly Structure', 'Two_Daily Structure']
        incompatible_map = {
            "USDCAD": ['1_BNR'],
            "GBPAUD": ["1_BNR"],
            "AUDUSD": [""],
            "EURAUD": ["1_BNR"],
            "GBPJPY": ["1_BNR", "1_BNR_TPF", 'EM_2b', 'EM_3b'],
            "EURJPY": ["1_BNR", "1_BNR_TPF", 'EM_2b', 'EM_3b'],
            "AUDJPY": ["1_BNR", "1_BNR_TPF", 'EM_2b', 'EM_3b'],
            # Add more restrictions as needed
        }

        potential_rr = ["3.41-4.41", "5.41-7.41", "8.41-10.41", ">=11.41"]
        within_61 = ["No", "Yes"]
        incompatible_map_4 = {
            "2_BNR": [],
            "2_BNR_TPF": [],
            "3_BNR_TPF": []
        }
        incompatible_map_2 = {
            "EM_1c": ["8.41-10.41", ">=11.41"],
            "EM_2c": ["8.41-10.41", ">=11.41"],
            "EM_3c": ["8.41-10.41", ">=11.41"],
            "EM_1b": [">=11.41"],
            "EM_2b": [">=11.41"],
            "EM_3b": [">=11.41"], }
        Variance = ["50", "559 - 66", "66 - 805", "> 805"]
        Trend_Positions = ["3%-4.99%", "5%-6.99%", "7%-8.99%", "9%-10.99%", "11%-12.99%", ">=13%"]
        incompatible_map_3 = {
            "1_TPF": ["50", "> 805"],
            "1_BNR": ["50", "559 - 66", "> 805"],
            "1_BNR_TPF": ["50"],
            "2_BNR": [""],
            "2_BNR_TPF": [""],
            "3_BNR_TPF": ["50", "> 805", "66 - 805"], }
        incompatible_map_5 = {
            "GBPAUD": ["No"],
            "EURAUD": ["No"],
            "GBPJPY": ["No"],
            "EURJPY": ["No"],
            "AUDJPY": ["No"]
        }

        incompatible_map_6 = {
            "> 805": ["No"],
        }

        incompatible_map_7 = {
            "GBPAUD": ["50"],
            "EURAUD": ["50"],
            "GBPJPY": ["50"],
            "EURJPY": ["50"],
            "USDJPY": ["50"],
            "AUDJPY": ["50"],
            "USDCAD": ["50"]}

        incompatible_map_8 = {
            "GBPAUD": ["3%-4.99%"],
            "EURAUD": ["3%-4.99%"],
            "GBPJPY": ["3%-4.99%"],
            "EURJPY": ["3%-4.99%"],
            "USDJPY": ["3%-4.99%"],
            "AUDJPY": ["3%-4.99%"],
            "XAUUSD": ["3%-4.99%"]}

        incompatible_map_9 = {
            "Cross Wave": ['2_BNR', '1_BNR', '1_BNR_TPF'],
            "9%-10.99%Wave 1": ['1_BNR'],
            "11%-12.99%Wave 1": ['1_BNR', '1_BNR_TPF'],
            ">=13%Wave 1": ['1_BNR', '1_BNR_TPF'],
            # Add more restrictions as needed
        }

        incompatible_map_10 = {
            "3%-4.99%1_BNR_TPF": ["559 - 66"],
            "9%-10.99%1_BNR_TPF": [],
            "11%-12.99%1_BNR_TPF": [],
            ">=13%1_BNR_TPF": ["559 - 66"],
            "EURAUD1_BNR_TPF": ["559 - 66", "66 - 805"],
            "GBPAUD1_BNR_TPF": ["559 - 66", "66 - 805"]
            # Add more restrictions as needed
        }

        incompatible_map_11 = {
            "AUDJPY": ['Two_Daily Structure', 'Daily BOS', '8H/4H BOS'],
            "GBPJPY": ['Two_Daily Structure', 'Daily BOS', '8H/4H BOS'],
            "EURJPY": ['Two_Daily Structure', 'Daily BOS', '8H/4H BOS'],
            "EURAUD": ['8H/4H BOS'],
            "GBPAUD": ['8H/4H BOS'],
        }

        incompatible_map_12 = {
            "3%-4.99%": ['1_BNR'],
            "XAUUSD5%-6.99%": [],
            "XAUUSD>=13%": [],
            ">=13%": [],

        }

        incompatible_map_13 = {
            "Yes": [],
            "No": ['1_BNR', '1_BNR_TPF']

        }

        incompatible_map_14 = {
            "3%-4.99%": [],
            "5%-6.99%": [],
            "7%-8.99%": [],
            "9%-10.99%": [],
            "11%-12.99%": [],
            ">=13%": []
        }

        incompatible_map_15 = {
            "Two_Daily Structure": [],

        }


        def get_available_timeframe(selected_pair):
            disabled_timeframe = incompatible_map_11.get(selected_pair, [])
            return [s for s in time_frame if s not in disabled_timeframe]


        def get_available_timeframe_2(listone, trend):
            disabled_timeframe_2 = incompatible_map_14.get(trend, [])
            return [s for s in listone if s not in disabled_timeframe_2]


        def get_available_position(pairlist):
            disabled_position = incompatible_map_7.get(pairlist, [])
            return [s for s in pairlist if s not in disabled_position]


        def get_available_trend_position(pair):
            disabled_trend_position = incompatible_map_8.get(pair, [])
            return [s for s in Trend_Positions if s not in disabled_trend_position]


        def get_available_rr(strategy):
            disabled_rr = incompatible_map_2.get(strategy, [])
            return [s for s in potential_rr if s not in disabled_rr]


        def get_available_61(strategy):
            disabled_61 = incompatible_map_4.get(strategy, [])
            return [s for s in within_61 if s not in disabled_61]


        def get_available_61_2(pair, list_available):
            diabled_61 = incompatible_map_5.get(pair, [])
            return [s for s in list_available if s not in diabled_61]


        def get_available_61_3(position, list_available):
            diabled_61 = incompatible_map_6.get(position, [])
            return [s for s in list_available if s not in diabled_61]


        def get_available_strategies(pair):
            disabled_strategies = incompatible_map.get(pair, [])
            return [s for s in strategies if s not in disabled_strategies]


        def get_available_strategies2(cross_fib, strats):
            disabled_strategies_2 = incompatible_map_9.get(cross_fib, [])
            return [s for s in strats if s not in disabled_strategies_2]


        def get_available_strategies3(trend_positions, pair, strats_2):

            if (pair == "XAUUSD"):
                var = pair + trend_positions
            else:
                var = trend_positions
            disabled_strategies_3 = incompatible_map_12.get(var, [])
            return [s for s in strats_2 if s not in disabled_strategies_3]


        def get_available_strategies4(hh_ll, stratslist):
            disabled_strategies_4 = incompatible_map_13.get(hh_ll, [])
            return [s for s in stratslist if s not in disabled_strategies_4]


        def get_available_strategies5(timeframe, stratslist):
            disabled_strategies_5 = incompatible_map_15.get(timeframe, [])
            return [s for s in stratslist if s not in disabled_strategies_5]


        def get_available_variance(entry_model):
            available_variance = incompatible_map_3.get(entry_model, [])
            return [s for s in Variance if s not in available_variance]


        def get_available_variance_2(trend, variancelist):
            available_variances = incompatible_map_10.get(trend, [])
            return [s for s in variancelist if s not in available_variances]


        def get_pair_prior_result(current_month_stats, selected_pair):
            if (len(current_month_stats) > 0):
                pair_trades = current_month_stats[current_month_stats['Symbol'] == selected_pair]
                if (len(pair_trades) < 1):
                    return 'N'
                else:
                    loss_pair_trades = pair_trades[pair_trades["Result"] == "Loss"]
                    be_pair_trades = pair_trades[pair_trades["Result"] == "BE"]
                    if (len(loss_pair_trades) >= 2):
                        return "X"
                    elif (len(be_pair_trades) > 2):
                        return "X"
                    else:
                        non_be_pair_trades = pair_trades[pair_trades["Result"] != "BE"]
                        if (len(non_be_pair_trades) > 0):
                            last_result = str(non_be_pair_trades["Result"].iloc[-1])
                            if (last_result == "Win"):
                                return 'W'
                            else:
                                return "L"
                            return last_result
                        else:
                            return "N"
            else:
                return "N"


        def get_focus_pair(current_month_stats):
            """Determine which pair to focus on based on most recent loss"""
            major_pairs = ["AUDUSD", "GBPUSD", "EURUSD"]
            focus_pair = None

            if len(current_month_stats) > 0:
                # Get all trades for major pairs, sorted by date (most recent first)
                major_trades = current_month_stats[
                    current_month_stats['Symbol'].isin(major_pairs)
                ].sort_values('Date', ascending=False)

                if not major_trades.empty:
                    # Find the most recent loss in these pairs
                    recent_loss = major_trades[major_trades['Result'] == "Loss"].head(1)

                    if not recent_loss.empty:
                        focus_pair = recent_loss.iloc[0]['Symbol']

                        # Check if the focus pair has since had a win (which would reset the rule)
                        # Get ALL trades for the focus pair, sorted by date
                        pair_trades = current_month_stats[
                            current_month_stats['Symbol'] == focus_pair
                            ].sort_values('Date', ascending=False)

                        if not pair_trades.empty:
                            # Get the most recent trade for this pair
                            most_recent_trade = pair_trades.iloc[0]

                            # ONLY reset if the most recent trade is a Win
                            # BE trades do NOT reset the focus rule
                            if most_recent_trade['Result'] == "Win":
                                focus_pair = None  # Rule reset

            return focus_pair


        def get_pair_sect_count(current_month_stats, pair):
            current_month_stats_losses = []
            focus_pair = get_focus_pair(current_month_stats)  # Use full data, not just losses

            if (len(current_month_stats) > 0):
                current_month_stats_losses = current_month_stats[current_month_stats['Result'] == "Loss"]
                current_month_stats_linked = current_month_stats_losses[current_month_stats_losses['Symbol'].isin(xxxusd)]
                current_month_stats_linked2 = current_month_stats_losses[current_month_stats_losses['Symbol'].isin(aud_family1)]



            remain_count = 2
            xxxaud_count = 1
            yen_count = 1
            audjpy_count = 1
            audusd_count = 2
            cad_count = 2
            europe_count = 2
            eurusd_count = 2
            gbpusd_count = 2
            gold_count = 2
            pair_trades = 0

            if (len(current_month_stats_losses) > 0):
                if (pair in xxxaud):
                    pair_trades = len(current_month_stats_losses[current_month_stats_losses['Symbol'].isin(xxxaud)])
                    xxxaud_count = xxxaud_count - pair_trades
                    if (len(current_month_stats_linked2) > 0):
                        pair_trades_aud = len(current_month_stats_linked2)
                        if (pair_trades_aud >= 2):
                            xxxaud_count = xxxaud_count - pair_trades_aud
                    return xxxaud_count

                elif (pair == "AUDJPY"):
                    if (len(current_month_stats_linked2) > 0):
                        pair_trades_aud2 = len(current_month_stats_linked2)
                        if (pair_trades_aud2 >= 2):
                            audjpy_count = audjpy_count - pair_trades_aud2
                    YEN_trades = len(current_month_stats_losses[current_month_stats_losses['Symbol'].isin(yens)])
                    audjpy_count = audjpy_count - YEN_trades
                    return audjpy_count

                elif (pair in yens):
                    pair_trades = len(current_month_stats_losses[current_month_stats_losses['Symbol'].isin(yens)])
                    yen_count = yen_count - pair_trades
                    return yen_count

                elif (pair == "USDCAD"):
                    pair_trades = len(current_month_stats_losses[current_month_stats_losses['Symbol'] == "USDCAD"])
                    if (pair_trades > 0):
                        cad_count = 0
                    if (len(current_month_stats_linked) > 0):
                        any_exist = current_month_stats_linked['Symbol'].isin(europe_major).any()
                        cross_eur = len(current_month_stats_linked['Symbol'].isin(europe_major))
                        if (any_exist):
                            cad_count = cad_count - cross_eur
                    pair_trades_ASIA = len(current_month_stats_losses[current_month_stats_losses['Symbol'].isin(trade_curr)])
                    cad_count = cad_count - pair_trades_ASIA
                    return cad_count

                elif (pair == "AUDUSD"):
                    # Apply focus rule - if focus pair exists and it's not this pair, return 0
                    if focus_pair is not None and focus_pair != "AUDUSD":
                        return 0

                    audusd_count = 2  # Start with default count

                    if (len(current_month_stats_linked2) > 0):
                        pair_trades_aud3 = len(current_month_stats_linked2)
                        if (pair_trades_aud3 >= 2):
                            audusd_count = audusd_count - pair_trades_aud3
                    pair_trades_TRADE = len(current_month_stats_losses[current_month_stats_losses['Symbol'].isin(trade_curr)])
                    audusd_count = audusd_count - pair_trades_TRADE
                    if (len(current_month_stats_linked) > 0):
                        any_exist = current_month_stats_linked['Symbol'].isin(europe_major).any()
                        cross_eur = len(current_month_stats_linked['Symbol'].isin(europe_major))
                        if (any_exist):
                            audusd_count = audusd_count - cross_eur
                    return max(0, audusd_count)  # Ensure non-negative

                elif (pair == "EURUSD"):
                    # Apply focus rule - if focus pair exists and it's not this pair, return 0
                    if focus_pair is not None and focus_pair != "EURUSD":
                        return 0

                    eurusd_count = 2  # Start with default count

                    pair_trades = len(current_month_stats_losses[current_month_stats_losses['Symbol'].isin(europe_major)])
                    eurusd_count = eurusd_count - pair_trades
                    if (len(current_month_stats_linked) > 0):
                        any_exist = current_month_stats_linked['Symbol'].isin(trade_curr).any()
                        cross_trade = len(current_month_stats_linked['Symbol'].isin(trade_curr))
                        if (any_exist):
                            eurusd_count = eurusd_count - cross_trade
                    return max(0, eurusd_count)  # Ensure non-negative

                elif (pair == "GBPUSD"):
                    # Apply focus rule - if focus pair exists and it's not this pair, return 0
                    if focus_pair is not None and focus_pair != "GBPUSD":
                        return 0

                    gbpusd_count = 2  # Start with default count

                    pair_trades = len(current_month_stats_losses[current_month_stats_losses['Symbol'].isin(europe_major)])
                    gbpusd_count = gbpusd_count - pair_trades
                    if (len(current_month_stats_linked) > 0):
                        any_exist = current_month_stats_linked['Symbol'].isin(trade_curr).any()
                        cross_trade = len(current_month_stats_linked['Symbol'].isin(trade_curr))
                        if (any_exist):
                            gbpusd_count = gbpusd_count - cross_trade
                    return max(0, gbpusd_count)  # Ensure non-negative

                elif (pair in gold_comm):
                    pair_trades = len(current_month_stats_losses[current_month_stats_losses['Symbol'].isin(gold_comm)])
                    gold_count = gold_count - pair_trades
                    return gold_count

                return remain_count
            else:
                return 2


        col1, col2 = st.columns(2, gap="small")
        with col1:
            account_balance = equity
            # account_balance = st.number_input("Account balance", equity)
            # account_balance = st.number_input(
            # "Account balance",
            # value=equity,
            # disabled=True  # Disables editing
            # )
            st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)
            selected_pair = st.selectbox("Trading Pair", currency_pairs)
            available_trend_position = get_available_trend_position(selected_pair)
            trend_position = st.selectbox("Trend Position", available_trend_position)

            available_time_frame = get_available_timeframe(selected_pair)
            available_time_frame_2 = get_available_timeframe_2(available_time_frame, trend_position)
            POI = st.selectbox(
                "POI Type (2_Daily is 2nd to Last on Weekly -- Note that the wick on first weekly Candle need to be >=0.50%)",
                available_time_frame_2)
            cross_fib = st.selectbox("Wave Status", ['Wave 2+', 'Wave 1', 'Cross Wave'])
            HH_LL = st.selectbox("FIB drawn on Highest High (Buy)/ Lowest Low (Sell)", ['Yes', 'No'])

            available_strategies = get_available_strategies(selected_pair)
            available_strats = get_available_strategies2(cross_fib, available_strategies)
            varcross = trend_position + cross_fib
            available_strats_crossfib = get_available_strategies2(varcross, available_strats)

            available_strats_2 = get_available_strategies3(trend_position, selected_pair, available_strats_crossfib)
            available_strats_3 = get_available_strategies4(HH_LL, available_strats_2)
            available_strats_4 = get_available_strategies5(POI, available_strats_3)

            risk_multiplier = st.selectbox("Entry Model",
                                           available_strats_4,
                                           index=0,
                                           help="Adjust risk based on trade quality")

            # Adaptive_value = st.number_input("Adaptive risk based on streak",next_risk,format="%.3f")
            # Adaptive_value = st.number_input(
            #	"Adaptive risk based on streak",
            #	value=next_risk,
            #	disabled=True,
            #	format = "%.3f"
            # )
            available_vairances = get_available_variance(risk_multiplier)
            if (selected_pair not in majors_dollar):
                if ('50' in available_vairances):
                    available_vairances.remove("50")
            concat_trend = trend_position + risk_multiplier
            final_variance1 = get_available_variance_2(concat_trend, available_vairances)

            concat_trend2 = selected_pair + risk_multiplier
            final_variance2 = get_available_variance_2(concat_trend2, final_variance1)

            Variances = st.selectbox("Position Variance", final_variance2)

            stop_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=None, step=1.0)
            Adaptive_value = next_risk
            # available_rr = get_available_rr(risk_multiplier)
            available_61 = get_available_61(risk_multiplier)
            # Potential = st.selectbox("Potential RR", available_rr)

            available_61 = get_available_61(risk_multiplier)
            available_61_2 = get_available_61_2(selected_pair, available_61)
            available_61_3 = get_available_61_3(Variances, available_61_2)

            within_61 = st.selectbox("Price Within 64", available_61_3)
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

            pair_result = get_pair_prior_result(current_month_stats, selected_pair)
            # Pair_prior_result = st.selectbox("Pair Prior Result in Month", pair_result, disabled=True)

            sect_count = get_pair_sect_count(current_month_stats, selected_pair)

        with col2:

            strategy_stats = df.groupby('Strategy').apply(analyze_strategy)
            # st.write(strategy_stats.items())
            # strategy_stats_df = pd.DataFrame(strategy_stats.tolist(), index=strategy_stats.index)
            multiplier = 1.0
            POI_multiplier = 1.0
            rr_multiplier = 1.0
            variance_multiplier = 1.0
            prior_result_multiplier = 1.0
            sect_count_multiplier = 1.0
            trend_position_multiplier = 1.0
            cross_fib_multiplier = 1.0

            big_risk_multiplier = 1.0
            for strategy, metrics in strategy_stats.items():
                # st.write(strategy)
                # st.write(metrics)

                if strategy == "Multiplier":
                    # multiplier = metrics["Multiplier"]
                    for Strategy_, Multiplier_ in metrics.items():
                        if Strategy_ == risk_multiplier:
                            multiplier = Multiplier_
                            # st.write(Strategy_,Multiplier_)

                    break

            # multiplier = calculate_strategy_grade_temp(risk_multiplier)
            if (POI == 'Weekly Structure'):
                POI_multiplier = 1.0
            elif (POI == 'Two_Daily Structure'):
                POI_multiplier = 0.91

            if (within_61 == 'Yes'):
                sixone_multiplier = 1.1
            else:
                sixone_multiplier = 1.0
            # if(Potential == '3.41-4.41'):
            # rr_multiplier = 1.0
            # elif(Potential == '5.41-7.41'):
            # rr_multiplier = 1.1
            # elif(Potential == '8.41-10.41'):
            # rr_multiplier = 1.2
            # elif (Potential == '>=11.41'):
            # rr_multiplier = 1.3
            if (trend_position == "3%-4.99%"):
                trend_position_multiplier = 1.0

            elif (trend_position == "5%-6.99%"):
                if (risk_multiplier == "1_BNR_TPF"):
                    if (Variances == "559 - 66"):
                        trend_position_multiplier = 1.0
                    else:
                        trend_position_multiplier = 1.05
                else:
                    trend_position_multiplier = 1.05
            elif (trend_position == "7%-8.99%"):
                if (risk_multiplier == "1_BNR_TPF"):
                    if (Variances == "559 - 66"):
                        trend_position_multiplier = 1.05
                    else:
                        trend_position_multiplier = 1.1
                else:
                    trend_position_multiplier = 1.1
            elif (trend_position == "9%-10.99%"):
                if (risk_multiplier == "1_BNR_TPF"):
                    if (Variances == "559 - 66"):
                        trend_position_multiplier = 0.95
                    else:
                        trend_position_multiplier = 1.0
                else:
                    trend_position_multiplier = 1.0
            elif (trend_position == "11%-12.99%"):
                if (risk_multiplier == "1_BNR" or risk_multiplier == "1_BNR_TPF"):
                    if (Variances == "559 - 66"):
                        trend_position_multiplier = 0.90
                    else:
                        trend_position_multiplier = 0.95
                else:
                    trend_position_multiplier = 0.95
            else:
                if (risk_multiplier == "1_BNR" or risk_multiplier == "1_BNR_TPF"):
                    trend_position_multiplier = 0.85
                else:
                    trend_position_multiplier = 0.9

            if (Variances == "50"):
                variance_multiplier = 0.91
            elif (Variances == "559 - 66"):
                variance_multiplier = 1.0
            elif (Variances == "66 - 805"):
                variance_multiplier = 1.0
            else:
                variance_multiplier = 1.0

            if (pair_result == "N"):
                prior_result_multiplier = 1.0
            elif (pair_result == "W"):
                prior_result_multiplier = 1.15
            elif (pair_result == "L"):
                prior_result_multiplier = 0.91
            elif (pair_result == "X"):
                prior_result_multiplier = 0

            if (sect_count > 0):
                sect_count_multiplier = 1.0
            else:
                sect_count_multiplier = 0.0

            if (cross_fib == "Cross Wave"):
                cross_fib_multiplier = 0.91
            else:
                cross_fib_multiplier = 1.0

            if (selected_pair in europe_major or selected_pair == "XAUUSD"):
                if (pair_result == "W"):
                    if (risk_multiplier == "2_BNR" or risk_multiplier == "2_BNR_TPF"):
                        if (
                                trend_position == "3%-4.99%" or trend_position == "5%-6.99%" or trend_position == "7%-8.99%"):
                            big_risk_multiplier = 1.5
            else:
                big_risk_multiplier = 1.0
            yearly_factor = starting_capital
            final_risk_1 = (
                                   yearly_factor * Adaptive_value) * multiplier * POI_multiplier * trend_position_multiplier * sixone_multiplier * prior_result_multiplier * sect_count_multiplier * big_risk_multiplier * cross_fib_multiplier * variance_multiplier
            final_risk = math.floor(final_risk_1)
            if (final_risk > (yearly_factor * 0.05)):
                final_risk = yearly_factor * 0.05

            if (stop_pips != None):
                position_size = calculate_position_size(final_risk, stop_pips, selected_pair)
                position_size_propfirm = calculate_position_size_propfirm(2000, stop_pips, selected_pair)
                set_global("position_size", position_size)
                set_global("position_size_propfirm", position_size_propfirm)
            else:
                position_size = 0
                position_size_propfirm = 0

            mae_cache_key = f"mae_{selected_pair}_{risk_multiplier}"

            if mae_cache_key not in st.session_state:
                with st.spinner(f"Loading Calc for {selected_pair}..."):
                    st.session_state[mae_cache_key] = calculate_mae_recommendations(selected_pair, risk_multiplier)

            suggested_sl, avg_winning_mae = st.session_state[mae_cache_key]

            if suggested_sl is not None:
                mae_based_stop_loss = suggested_sl
                winning_trade_mae = avg_winning_mae

            target_in = 0
            def getPairEntrySL(pair):
                if (suggested_sl is None or avg_winning_mae is None):
                    #target_in = 0
                    if (pair == "GBPUSD"):
                        base_entry = 12
                        base_sl = 18
                        return str(base_entry),str(base_sl)
                    elif (pair == "EURUSD"):
                        base_entry = 12
                        base_sl = 18
                        return str(base_entry), str(base_sl)
                    elif (pair == "AUDUSD"):
                        base_entry = 12
                        base_sl = 22
                        return str(base_entry), str(base_sl)
                    elif (pair == "XAUUSD"):
                        base_entry = 12
                        base_sl = 30
                        return str(base_entry), str(base_sl)
                    elif (pair == "USDJPY"):
                        base_entry = 12
                        base_sl = 18
                        return str(base_entry), str(base_sl)
                    elif (pair == "USDCAD"):
                        base_entry = 12
                        base_sl = 18
                        return str(base_entry), str(base_sl)
                    else:
                        base_entry = 12
                        base_sl = 30
                        return str(base_entry), str(base_sl)


                else:
                    #target_in = (99 // round(mae_based_stop_loss,0)) + 0.41
                    if (pair == "GBPUSD"):
                        return str((round(12-winning_trade_mae,0)+6)), str(18)
                    elif (pair == "EURUSD"):
                        return str((round(12-winning_trade_mae,0)+6)), str(18)
                    elif (pair == "AUDUSD"):
                        return str((round(12-winning_trade_mae,0)+6)), str(22)
                    elif (pair == "XAUUSD"):
                        return str((round(12-winning_trade_mae,0)+6)), str(30)
                    elif (pair == "USDJPY"):
                        return str((round(12-winning_trade_mae,0)+6)), str(18)
                    elif (pair == "USDCAD"):
                        str((round(12-winning_trade_mae,0)+6)), str(18)
                    else:
                        return "12", "30"


            entry_title = ""
            entry_text = ""
            SL_title = ""
            SL_text = ""
            exit_title = ""
            exit_text = ""
            entry_pip, sl_pip = getPairEntrySL(selected_pair)


            def get_one_target(selected_pair):
                open_target = 0
                if (selected_pair in yens):
                    open_target = round(JPY_Pairs_gap / final_risk, 2)
                elif (selected_pair in gold_comm):
                    open_target = round(XAUUSD_gap / final_risk, 2)
                elif (selected_pair in xxxaud):
                    open_target = round(GBPAUD_EURAUD_gap / final_risk, 2)
                elif (selected_pair in europe_major):
                    open_target = round(GBPUSD_EURUSD_gap / final_risk, 2)
                elif (selected_pair in trade_curr):
                    open_target = round(USDCAD_AUDUSD_gap / final_risk, 2)
                return open_target


            def compare_target(open_target, desire_target):
                if (open_target < desire_target):
                    if (open_target > 3):
                        return str(open_target)
                    else:
                        return "3"
                elif (open_target > desire_target):
                    return str(desire_target)

            def get_pair_volatile(pair, within_61):
                if (pair == "GBPUSD"):
                    if (within_61 == 'Yes'):
                        return str(18)
                    else:
                        return str(24)
                elif (pair == "EURUSD"):
                    if (within_61 == 'Yes'):
                        return str(18)
                    else:
                        return str(24)
                elif (pair == "AUDUSD"):
                    if (within_61 == 'Yes'):
                        return str(22)
                    else:
                        return str(24)
                elif (pair == "XAUUSD"):
                    if (within_61 == 'Yes'):
                        return str(30)
                    else:
                        return str(33)
                elif (pair == "USDJPY"):
                    if (within_61 == 'Yes'):
                        return str(30)
                    else:
                        return str(33)
                elif (pair == "USDCAD"):
                    if (within_61 == 'Yes'):
                        return str(18)
                    else:
                        return str(24)
                else:
                    return "30"


            if (len(a_momemtum_text) < 1):
                a_momemtum_text = "To be Filled"
            if (risk_multiplier == "1_BNR"):
                if (within_61 == "Yes"):
                    if (selected_pair == "XAUUSD"):
                        entry_title = "Entry Guide:"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "% Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + "%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 7.41)
                    elif (selected_pair == "AUDUSD"):
                        entry_title = "Entry Guide:"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "% Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " %"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 5.41)
                    elif (selected_pair == "USDJPY"):
                        entry_title = "Entry Guide:"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "% Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " %"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 6.41)
                    else:
                        entry_title = "Entry Guide:"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "% Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " %"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 6.41)

                else:
                    if (selected_pair == "XAUUSD"):
                        entry_title = "Entry Guide:"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "% Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + "%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 6.41)
                    elif (selected_pair == "AUDUSD"):
                        entry_title = "Entry Guide:"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "% Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " %"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 4.41)
                    elif (selected_pair == "USDJPY"):
                        entry_title = "Entry Guide:"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "% Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " %"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 5.41)
                    else:
                        entry_title = "Entry Guide:"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "% Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " %"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 5.41)
            elif (risk_multiplier == "1_BNR_TPF"):
                if (within_61 == "Yes"):
                    if (selected_pair == "XAUUSD"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 30%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 7.41)

                        if (Variances == "> 805"):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: 30%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                            exit_text = compare_target(get_one_target(selected_pair), 8.41)
                    elif (selected_pair in minors or selected_pair == "AUDUSD"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 22%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 5.41)

                        if (Variances == "> 805"):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: 22%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                            exit_text = compare_target(get_one_target(selected_pair), 6.41)
                    elif (selected_pair == "USDJPY"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 18%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 20)

                        if (Variances == "> 805"):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: 18%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                            exit_text = compare_target(get_one_target(selected_pair), 20)
                    else:
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 18%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 6.41)

                        if (Variances == "> 805"):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: 18%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                            exit_text = compare_target(get_one_target(selected_pair), 7.41)


                else:
                    if (selected_pair == "XAUUSD"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 30%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 6.41)
                    elif (selected_pair in minors or selected_pair == "AUDUSD"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 22%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 4.41)
                    elif (selected_pair == "USDJPY"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 24%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 5.41)
                    else:
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 18%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR): Adjust TP, SL and Entry if target < 559 Fib"
                        exit_text = compare_target(get_one_target(selected_pair), 5.41)



            elif (risk_multiplier == "2_BNR" or risk_multiplier == "2_BNR_TPF"):
                length_title = "Leg One Length Requirement:"
                length_text = ""


                def get_sum_target():
                    sum_target = round(sum_gap / final_risk, 2)
                    if (sum_target > 3):
                        return sum_target
                    else:
                        return 3


                def get_open_target(selected_pair):
                    open_target = 0
                    total_target = 0
                    if (selected_pair in yens):
                        open_target = round(JPY_Pairs_gap / final_risk, 2)
                        if (open_target > 3):
                            return open_target
                        else:
                            return 3
                    elif (selected_pair in gold_comm):
                        open_target = round(XAUUSD_gap / final_risk, 2)
                        total_target = round(get_sum_target(),2)
                        if (open_target >= 5.41):
                            return open_target
                        elif(total_target >= 5.41):
                            return total_target
                        else:
                            return 5.41
                    elif (selected_pair in xxxaud):
                        open_target = round(GBPAUD_EURAUD_gap / final_risk, 2)
                        if (open_target > 3):
                            return open_target
                        else:
                            return 3
                    elif (selected_pair in europe_major):
                        open_target = round(GBPUSD_EURUSD_gap / final_risk, 2)
                        total_target = round(get_sum_target(), 2)
                        if (open_target >= 5.41):
                            return open_target
                        elif (total_target >= 5.41):
                            return total_target
                        else:
                            return 5.41
                    elif (selected_pair in trade_curr):
                        open_target = round(USDCAD_AUDUSD_gap / final_risk, 2)
                        total_target = round(get_sum_target(), 2)
                        if (open_target >= 4.41):
                            return open_target
                        elif (total_target >= 4.41):
                            return total_target
                        else:
                            return 4.41



                if (selected_pair in minor_yens):
                    if (Variances == "> 805"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 33%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR):"
                        exit_text = "6.41"

                    elif (Variances == "559 - 66"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 33%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR):"
                        exit_text = "5.41"

                    elif (Variances == "66 - 805"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: 33%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR):"
                        exit_text = "5.41"


                else:
                    if (trend_position == "9%-10.99%" or trend_position == "11%-12.99%" or trend_position == ">=13%"):
                        if(trend_position == "9%-10.99%" and within_61 == "Yes"):
                            targeting = get_open_target(selected_pair)
                        elif (trend_position == "11%-12.99%" and within_61 == "Yes" and selected_pair == "XAUUSD"):
                            targeting = get_open_target(selected_pair)
                        elif (trend_position == ">=13%" and within_61 == "Yes" and selected_pair == "XAUUSD"):
                            targeting = get_open_target(selected_pair)
                        else:
                            targeting = 5.41
                    elif (selected_pair not in minors and selected_pair != "USDJPY" and selected_pair not in trade_curr):
                        if (trend_position == "3%-4.99%" or trend_position == "5%-6.99%"):
                            if (big_risk_multiplier > 1):
                                total_target = get_sum_target()
                                compare_target = get_open_target(selected_pair) * 1.725
                                if (compare_target < total_target):
                                    targeting = round(compare_target, 2)
                                else:
                                    targeting = total_target
                            else:
                                targeting = get_open_target(selected_pair)
                        else:
                            targeting = get_open_target(selected_pair)
                    else:
                        targeting = get_open_target(selected_pair)
                    if (Variances == "> 805"):
                        entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                        entry_text = "TOP or MIDDLE of TPF Zone"
                        SL_title = "SL Guide:"
                        SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                        exit_title = "Target Guide One (RR):"
                        exit_text = targeting

                    elif (Variances == "559 - 66"):
                        if (risk_multiplier == "2_BNR_TPF" and selected_pair in majors_dollar):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting

                        elif(risk_multiplier == "2_BNR_TPF" and selected_pair not in majors_dollar):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                        else:
                            entry_title = "Entry Guide: One Zone below the First Leg Zone"
                            entry_text = "TOP or Adjust by SL Cover END of zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting

                    elif (Variances == "66 - 805"):
                        if (risk_multiplier == "2_BNR_TPF" and selected_pair in majors_dollar):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                        elif (risk_multiplier == "2_BNR_TPF" and selected_pair not in majors_dollar):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                        else:
                            entry_title = "Entry Guide: One Zone below the First Leg Zone"
                            entry_text = "TOP or Adjust by SL Cover END of zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                    elif (Variances == "50"):
                        if (risk_multiplier == "2_BNR_TPF" and selected_pair in majors_dollar):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide: NOTE THAT 50 NEEDS 618 ZONE TO BE TAPPED"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                        elif (risk_multiplier == "2_BNR_TPF" and selected_pair not in majors_dollar):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide: NOTE THAT 50 NEEDS 618 ZONE TO BE TAPPED"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                        else:
                            entry_title = "Entry Guide: One Zone below the First Leg Zone"
                            entry_text = "TOP or Adjust by SL Cover END of zone"
                            SL_title = "SL Guide: NOTE THAT 50 NEEDS 618 ZONE TO BE TAPPED"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting

                    if (selected_pair == "XAUUSD"):
                        if (trend_position == "9%-10.99%" or trend_position == "11%-12.99%" or trend_position == ">=13%"):
                            if (trend_position == "9%-10.99%" and within_61 == "Yes"):
                                targeting = get_open_target(selected_pair)
                            elif (trend_position == "11%-12.99%" and within_61 == "Yes" and selected_pair == "XAUUSD"):
                                targeting = get_open_target(selected_pair)
                            elif (trend_position == ">=13%" and within_61 == "Yes" and selected_pair == "XAUUSD"):
                                targeting = get_open_target(selected_pair)
                            else:
                                targeting = 5.41
                        elif (trend_position == "7%-8.99%" or trend_position == "5%-6.99%"):
                            if (big_risk_multiplier > 1):
                                total_target = get_sum_target()
                                compare_target = get_open_target(selected_pair) * 1.725
                                if (compare_target < total_target):
                                    targeting = round(compare_target, 2)
                                else:
                                    targeting = total_target
                            else:
                                targeting = get_open_target(selected_pair)
                        else:
                            targeting = get_open_target(selected_pair)
                        if (Variances == "> 805"):
                            entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                            entry_text = "TOP or MIDDLE of TPF Zone"
                            SL_title = "SL Guide:"
                            SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                        elif (Variances == "559 - 66"):
                            if (risk_multiplier == "2_BNR_TPF"):
                                entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                                entry_text = "TOP or MIDDLE of TPF Zone"
                                SL_title = "SL Guide:"
                                SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting
                            else:
                                entry_title = "Entry Guide: One Zone below the First Leg Zone"
                                entry_text = "TOP or Adjust by SL Cover END of zone"
                                SL_title = "SL Guide:"
                                SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting

                        elif (Variances == "66 - 805"):
                            if (risk_multiplier == "2_BNR_TPF"):
                                entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                                entry_text = "TOP or MIDDLE of TPF Zone"
                                SL_title = "SL Guide:"
                                SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting
                            else:
                                entry_title = "Entry Guide: One Zone below the First Leg Zone"
                                entry_text = "TOP or Adjust by SL Cover END of zone"
                                SL_title = "SL Guide:"
                                SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting
                        elif (Variances == "50"):
                            if (risk_multiplier == "2_BNR_TPF"):
                                entry_title = "Entry Guide: Draw Zone Based on where TPF/OMSS is"
                                entry_text = "TOP or MIDDLE of TPF Zone"
                                SL_title = "SL Guide: NOTE THAT 50 NEEDS 618 ZONE TO BE TAPPED"
                                SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting
                            else:
                                entry_title = "Entry Guide: One Zone below the First Leg Zone"
                                entry_text = "TOP or Adjust by SL Cover END of zone"
                                SL_title = "SL Guide: NOTE THAT 50 NEEDS 618 ZONE TO BE TAPPED"
                                SL_text = "Middle of: "+get_pair_volatile(selected_pair,within_61)+"%, Entry zone +6%, 33%"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting


            col1, col2, col3 = st.columns([0.03, 2, 0.01], gap="small")

            with col2:
                Risk_percentage = round(final_risk / account_balance * 100, 2)
                container = st.container()
                container.markdown("<div style='height: 70px; padding: 0px; margin-left: 2000px;'></div>",
                                   unsafe_allow_html=True)
                if (get_global('entry_model') != None):
                    container.markdown(f"""
                            <div style="color:deepskyblue; font-weight:600;">
                                --Entry must not exceed 33%<br>
                                --Enter Middle of Zone if TPF last 25th Percentile<br>
                                --Only Draw Double Zone (Max 2) if <= 27% Combined<br>
                                --SL MUST at least cover end of zone<br>
                                --Entry: {get_global('entry_model')}<br>
                            </div>
                            """, unsafe_allow_html=True)
                elif (get_global('entry_model') == None):
                    entry_percent, base_percent = getPairEntrySL(selected_pair)
                    container.markdown(f"""
                            <div style="color:deepskyblue; font-weight:600;">
                                --Entry must not exceed 33%<br>
                                --Enter Middle of Zone if TPF last 25th Percentile<br>
                                --Only Draw Double Zone (Max 2) if <= 27% Combined<br>
                                --SL MUST at least cover end of zone<br>
                                --Min Length for {selected_pair} is {base_percent}%<br>
                            </div>
                            """, unsafe_allow_html=True)


                if (monthly_loss_limit + monthly_actual_loss - final_risk < 0):
                    container.markdown("<br>", unsafe_allow_html=True)
                    # container.metric("Risk amount exceeded your monthly limit", "$"+ str(round(final_risk + round(monthly_loss_limit+monthly_actual_loss,2),2)))
                    container.metric("Risk amount exceeded your monthly limit", "$0 (0.0% of Account)")
                else:
                    container.markdown("<br>", unsafe_allow_html=True)
                    container.metric("Your Next risk risk should be:", f"${final_risk} ({Risk_percentage}% of Account)")
                    set_global("final_risk", Risk_percentage)
                    if (position_size > 0):
                        if (position_size_propfirm > 0):
                            if (risk_multiplier == "2_BNR" or risk_multiplier == "2_BNR_TPF"):
                                if (pair_result == "W" or pair_result == "N"):
                                    if (selected_pair in europe_major or selected_pair in gold_comm):
                                        container.metric("Your Calculated lot size should be:",
                                                         f"Personal {position_size} lots /" f"Propfirm {position_size_propfirm} lots")
                                    else:
                                        container.metric("Your Calculated lot size should be:", f"{position_size} lots")
                                else:
                                    container.metric("Your Calculated lot size should be:", f"{position_size} lots")
                            else:
                                container.metric("Your Calculated lot size should be:", f"{position_size} lots")
                        else:
                            container.metric("Your Calculated lot size should be:", f"{position_size} lots")
                    else:
                        container.metric("Your Calculated lot size should be:", "Please Enter stop pips")

                    if (risk_multiplier == "1_BNR_TPF" or risk_multiplier == "1_BNR" or risk_multiplier == "3_BNR_TPF"):
                        container.metric(entry_title, entry_text)
                        container.metric(SL_title, SL_text)
                        container.metric(exit_title, exit_text)

                        # Determine if button should be disabled
                        add_order_disabled = False

                        if st.button(" Add Order", type="secondary", disabled=add_order_disabled):
                            # NEW: Check cooldown period first
                            cooldown_ok, cooldown_message = check_cooldown_period(
                                selected_pair,
                                st.session_state.uploaded_data,
                                cooldown_days=1  # 1 day cooldown as requested
                            )

                            if not cooldown_ok:
                                st.error(cooldown_message)
                            else:
                                # Check for existing active records first
                                can_proceed, error_message = check_existing_active_records(selected_pair,
                                                                                           st.session_state.saved_records)

                                if not can_proceed:
                                    st.error(error_message)
                                else:
                                    # Check if Stop Loss is 0 or missing
                                    if stop_pips is None or stop_pips == 0:
                                        st.error("Cannot add order: Stop Loss is required and cannot be 0!")
                                    elif (position_size <= 0 or final_risk <=0):
                                        st.error("Position size is required to be greater than 0")
                                    else:
                                        # Check if maximum records reached (only for new records, not updates)
                                        existing_index = None
                                        for i, existing_record in enumerate(st.session_state.saved_records):
                                            if existing_record['selected_pair'] == selected_pair:
                                                existing_index = i
                                                break

                                        # If it's a new record (not updating existing) and we're at max capacity
                                        if existing_index is None and len(st.session_state.saved_records) >= 6:
                                            st.error(
                                                "Maximum of 6 records reached! Please delete a record from the Records page before adding a new one.")
                                        else:
                                            # Create a record with timestamp and all selections
                                            record = {
                                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                # More detailed timestamp
                                                'selected_pair': selected_pair,
                                                'trend_position': trend_position,
                                                'POI': POI,
                                                'cross_fib': cross_fib,
                                                'HH_LL': HH_LL,
                                                'risk_multiplier': risk_multiplier,
                                                'Variances': Variances,
                                                'stop_pips': stop_pips,
                                                'within_61': within_61,
                                                'final_risk': final_risk,  # This is important for comparison
                                                'position_size': position_size,
                                                'position_size_propfirm': position_size_propfirm,
                                                'entry_price': 0.0,  # Default values
                                                'exit_price': 0.0,
                                                'target_price': 0.0,
                                                'status': 'Speculation',  # Default status
                                                'exit_rr': exit_text
                                            }

                                            # Handle group conflicts for new records
                                            if existing_index is None:
                                                updated_records, conflict_message = check_and_handle_group_conflict(
                                                    selected_pair, final_risk, st.session_state.saved_records
                                                )

                                                if conflict_message and "Higher risk pair" in conflict_message:
                                                    st.warning(conflict_message)
                                                    # Don't add the new record - use a flag to skip the rest
                                                    should_add_record = False
                                                else:
                                                    st.session_state.saved_records = updated_records
                                                    if conflict_message and "Replaced" in conflict_message:
                                                        st.info(conflict_message)
                                                    # Add new record
                                                    st.session_state.saved_records.append(record)
                                                    should_add_record = True

                                            else:
                                                # Update existing record (no group conflict check for updates)
                                                st.session_state.saved_records[existing_index] = record
                                                should_add_record = True

                                            # Only save to Google Sheets if we should add/update the record
                                            if should_add_record:
                                                # SAVE TO GOOGLE SHEETS
                                                if save_workflow_to_sheets(st.session_state.saved_records):
                                                    st.success("Successfully Processed Order and Saved to Cloud!")
                                                else:
                                                    st.error("Order processed locally but failed to save to cloud!")

                        # Add View Active Opps button below the Add Order button (same size)
                        if st.button("View Active Opportunities"):
                            st.session_state.current_page = "Active Opps"
                            st.rerun()

                        st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)
                    elif (risk_multiplier == "2_BNR" or risk_multiplier == "2_BNR_TPF"):

                        container.metric(SL_title, SL_text)

                        container.metric(entry_title, entry_text)

                        container.metric(exit_title, exit_text)

                        # Determine if button should be disabled

                        add_order_disabled = False

                        if st.button(" Add Order", type="secondary", disabled=add_order_disabled):

                            # NEW: Check cooldown period first

                            cooldown_ok, cooldown_message = check_cooldown_period(

                                selected_pair,

                                st.session_state.uploaded_data,

                                cooldown_days=1  # 1 day cooldown as requested

                            )

                            if not cooldown_ok:

                                st.error(cooldown_message)

                            else:

                                # Check for existing active records first

                                can_proceed, error_message = check_existing_active_records(selected_pair,

                                                                                           st.session_state.saved_records)

                                if not can_proceed:

                                    st.error(error_message)

                                else:

                                    # Check if Stop Loss is 0 or missing

                                    if stop_pips is None or stop_pips == 0:

                                        st.error("Cannot add order: Stop Loss is required and cannot be 0!")
                                    elif (position_size <= 0 or final_risk <= 0):
                                        st.error("Position size is required to be greater than 0")

                                    else:

                                        # Check if maximum records reached (only for new records, not updates)

                                        existing_index = None

                                        for i, existing_record in enumerate(st.session_state.saved_records):

                                            if existing_record['selected_pair'] == selected_pair:
                                                existing_index = i

                                                break

                                        # If it's a new record (not updating existing) and we're at max capacity

                                        if existing_index is None and len(st.session_state.saved_records) >= 6:

                                            st.error(

                                                "Maximum of 6 records reached! Please delete a record from the Records page before adding a new one.")

                                        else:

                                            # Create a record with timestamp and all selections

                                            record = {

                                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                                                'selected_pair': selected_pair,

                                                'trend_position': trend_position,

                                                'POI': POI,

                                                'cross_fib': cross_fib,

                                                'HH_LL': HH_LL,

                                                'risk_multiplier': risk_multiplier,

                                                'Variances': Variances,

                                                'stop_pips': stop_pips,

                                                'within_61': within_61,

                                                'final_risk': final_risk,  # This is important for comparison

                                                'position_size': position_size,

                                                'position_size_propfirm': position_size_propfirm,

                                                'entry_price': 0.0,  # Default values

                                                'exit_price': 0.0,

                                                'target_price': 0.0,

                                                'status': 'Speculation',  # Default status

                                                'exit_rr': exit_text

                                            }

                                            # Handle group conflicts for new records

                                            if existing_index is None:

                                                updated_records, conflict_message = check_and_handle_group_conflict(

                                                    selected_pair, final_risk, st.session_state.saved_records

                                                )

                                                if conflict_message and "Higher risk pair" in conflict_message:

                                                    st.warning(conflict_message)

                                                    # Don't add the new record - use a flag to skip the rest

                                                    should_add_record = False

                                                else:

                                                    st.session_state.saved_records = updated_records

                                                    if conflict_message and "Replaced" in conflict_message:
                                                        st.info(conflict_message)

                                                    # Add new record

                                                    st.session_state.saved_records.append(record)

                                                    should_add_record = True

                                            else:

                                                # Update existing record (no group conflict check for updates)

                                                st.session_state.saved_records[existing_index] = record

                                                should_add_record = True

                                            # Only save to Google Sheets if we should add/update the record

                                            if should_add_record:

                                                # SAVE TO GOOGLE SHEETS

                                                if save_workflow_to_sheets(st.session_state.saved_records):

                                                    st.success("Successfully Processed Order and Saved to Cloud!")

                                                else:

                                                    st.error("Order processed locally but failed to save to cloud!")

                        # Add View Active Opps button below the Add Order button (same size)

                        if st.button("View Active Opportunities"):
                            st.session_state.current_page = "Active Opps"

                            st.rerun()

                        st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)


elif st.session_state.current_page == "Active Opps":
    import streamlit as st
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime, timedelta, timezone
    import time
    import xml.etree.ElementTree as ET
    import pytz
    from dateutil import parser as dp
    from zoneinfo import ZoneInfo  # Python 3.9+
    from dateutil.parser import isoparse
    from collections import defaultdict


    # ==================== CROSS GROUP VALIDATION FUNCTION ====================
    def check_cross_group_conflict(selected_pair, current_stage_records):
        """Check if there's already a record from the same cross group in active stages"""

        # Define cross groups
        cross_groups = [
            ["AUDUSD", "AUDJPY", "GBPAUD", "EURAUD"],
            ["AUDUSD", "USDCAD", "EURUSD", "GBPUSD"]
        ]

        # Find ALL groups the selected pair belongs to
        target_groups = []
        for group in cross_groups:
            if selected_pair in group:
                target_groups.append(group)

        if not target_groups:
            return False, None, []  # No cross group restriction for this pair

        # Check if any record in active stages belongs to ANY of the target groups
        conflicting_pairs = []
        conflicting_groups = []

        for record in current_stage_records:
            for target_group in target_groups:
                if record['selected_pair'] in target_group and record['selected_pair'] != selected_pair:
                    conflicting_pairs.append(record['selected_pair'])
                    conflicting_groups.append(target_group)
                    break  # No need to check other groups for this record

        if conflicting_pairs:
            # Remove duplicates but preserve order
            conflicting_pairs = list(dict.fromkeys(conflicting_pairs))
            conflicting_groups = list(dict.fromkeys(str(g) for g in conflicting_groups))
            return True, target_groups, conflicting_pairs

        return False, target_groups, []


    # ==================== AUTO-DELETE CROSS GROUP SPECULATION FUNCTION ====================
    def delete_cross_group_speculation(filled_pair, saved_records):
        """Delete speculation records that belong to the same cross group as a filled order"""

        # Define cross groups
        cross_groups = [
            ["AUDUSD", "AUDJPY", "GBPAUD", "EURAUD"],
            ["AUDUSD", "USDCAD", "EURUSD", "GBPUSD"]
        ]

        # Find ALL groups the filled pair belongs to
        target_groups = []
        for group in cross_groups:
            if filled_pair in group:
                target_groups.append(group)

        if not target_groups:
            return saved_records, []  # No cross group restriction for this pair

        # Find speculation records in ANY of the target groups
        records_to_delete = []
        updated_records = []

        for record in saved_records:
            if record.get('status') == 'Speculation':
                # Check if this speculation record belongs to ANY of the target groups
                should_delete = False
                for target_group in target_groups:
                    if record['selected_pair'] in target_group:
                        should_delete = True
                        records_to_delete.append(record['selected_pair'])
                        break  # No need to check other groups

                if not should_delete:
                    updated_records.append(record)
            else:
                # Keep all non-speculation records
                updated_records.append(record)

        return updated_records, records_to_delete


    # ==================== ROBUST RED NEWS FUNCTION ====================
    def get_red_news_from_json_with_rate_limit():
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print("Request error:", e)
            return []

        text = resp.text
        if text.lower().startswith("<!doctype html") or "request denied" in text.lower():
            print("Received HTML or Denial page instead of JSON — likely rate limit triggered")
            return []

        try:
            data = resp.json()
        except ValueError as e:
            print("JSON parse error:", e)
            return []

        red_news = []

        # Timezones
        melbourne_tz = pytz.timezone("Australia/Melbourne")
        now_melb = datetime.now(melbourne_tz)
        today_melb = now_melb.date()

        # ✅ Define start and end of the current week (Monday to Sunday)
        start_of_week = today_melb - timedelta(days=today_melb.weekday())  # Monday
        end_of_week = start_of_week + timedelta(days=6)  # Sunday

        for ev in data:
            date_str = ev.get('date')
            impact = ev.get('impact', '').strip().lower()

            # Include both high impact and holiday events
            if impact not in ['high', 'holiday'] or not date_str:
                continue

            try:
                event_dt_utc = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                event_dt_melb = event_dt_utc.astimezone(melbourne_tz)
                event_date_melb = event_dt_melb.date()
            except Exception as e:
                print(f"Error parsing date: {date_str} — {e}")
                continue

            # ✅ Include all events within current week (not just future ones)
            if start_of_week <= event_date_melb <= end_of_week:
                red_news.append({
                    'Date': event_date_melb.strftime('%Y-%m-%d'),
                    'TimeMelbourne': event_dt_melb.isoformat(),
                    'Currency': ev.get('country', 'N/A'),
                    'Event': ev.get('title', 'N/A'),
                    'Forecast': ev.get('forecast', 'N/A'),
                    'Previous': ev.get('previous', 'N/A'),
                    'Actual': ev.get('actual', 'N/A') if 'actual' in ev else 'N/A',
                    'Impact': impact.upper()  # Use actual impact (e.g., HIGH or HOLIDAY)
                })

        return red_news


    # Initialize session states
    if 'saved_records' not in st.session_state:
        st.session_state.saved_records = []
    if 'ready_to_order' not in st.session_state:
        st.session_state.ready_to_order = []
    if 'order_placed' not in st.session_state:
        st.session_state.order_placed = []
    if 'in_trade' not in st.session_state:
        st.session_state.in_trade = []
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'Speculation'
    if 'last_action' not in st.session_state:
        st.session_state.last_action = None
    if 'last_sync_time' not in st.session_state:
        st.session_state.last_sync_time = datetime.now()
    # Initialize session state
    if 'red_events' not in st.session_state:
        st.session_state.red_events = []
    if 'last_news_fetch' not in st.session_state:
        st.session_state.last_news_fetch = None
    if 'last_deleted_pairs' not in st.session_state:
        st.session_state.last_deleted_pairs = []

    # Inject CSS to color the expander header text blue
    st.markdown("""
        <style>
        /* Target expander label text */
        [data-testid="stExpander"] > details > summary {
            color: #007BFF !important;  /* Bootstrap blue */
            font-weight: bold !important;
            font-size: 1.1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Set Melbourne timezone
    melbourne_tz = ZoneInfo('Australia/Melbourne')


    def save_events_to_sheets(events):
        """Save events to Google Sheets - COMPLETE REPLACEMENT"""
        if not events:
            st.warning("No events to save")
            return False

        try:
            # Convert events to DataFrame
            df = pd.DataFrame(events)

            # Ensure we have the required columns
            required_columns = ['Date', 'TimeMelbourne', 'Currency', 'Event', 'Forecast', 'Previous', 'Actual',
                                'Impact']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ""

            df = df[required_columns]

            # Use the new replace function instead of save
            success = replace_data_on_sheet(df, sheet_name="Trade", worksheet_name="News")

            if not success:
                st.error("Failed to replace news data in Google Sheets")

            return success

        except Exception as e:
            st.error(f"Failed to save to Google Sheets: {e}")
            return False


    def load_events_from_sheets():
        """Load events from Google Sheets using your existing function"""
        try:
            df = load_data_from_sheets(sheet_name="Trade", worksheet_name="News")

            if df is None or df.empty:
                return None

            # Convert DataFrame back to event format
            events = df.to_dict('records')
            return events

        except Exception as e:
            st.error(f"Failed to load from Google Sheets: {e}")
            return None


    # Initialize session state for tracking data source
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None  # 'cloud' or 'forex_factory'

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Refresh News", key="refresh_red_news_json", use_container_width=True):
            with st.spinner("Checking for high impact news..."):
                # Fetch new data
                new_events = get_red_news_from_json_with_rate_limit()


                if new_events:
                    # Save to Google Sheets USING REPLACEMENT

                    if save_events_to_sheets(new_events):  # This now uses replace_data_on_sheet internally
                        st.session_state.red_events = new_events
                        st.session_state.last_news_fetch = datetime.now()
                        st.session_state.data_source = 'forex_factory'
                        #st.success("News data REPLACED and updated successfully!")
                    else:
                        st.error("Failed to REPLACE news in Google Sheets")


    with col2:
        if st.session_state.last_news_fetch:
            st.write(f"Last: {st.session_state.last_news_fetch.strftime('%H:%M:%S')}")
        else:
            st.write("Click refresh")

    # Initial load - try session state, then Google Sheets, then fetch new
    if 'red_events' not in st.session_state or not st.session_state.red_events:
        with st.spinner("Loading high impact news..."):
            # First try to load from Google Sheets

            sheet_events = load_events_from_sheets()

            if sheet_events:
                st.session_state.red_events = sheet_events
                st.session_state.last_news_fetch = datetime.now()
                st.session_state.data_source = 'cloud'
                #st.success("Loaded news data from Google Sheets")
            else:
                # If no data in sheets, fetch new data

                new_events = get_red_news_from_json_with_rate_limit()
                if new_events:
                    st.session_state.red_events = new_events
                    st.session_state.last_news_fetch = datetime.now()
                    st.session_state.data_source = 'forex_factory'
                    # Save to Google Sheets for next time USING REPLACEMENT

                    if not save_events_to_sheets(new_events):  # This now uses replace
                        st.error("Fetched new data but failed to REPLACE Google Sheets")
                else:
                    st.session_state.red_events = []
                    st.session_state.last_news_fetch = None
                    st.session_state.data_source = None
                    st.warning("No news data available")

    red_events = st.session_state.red_events

    # Determine expander title based on data source
    if st.session_state.data_source == 'cloud':
        expander_title = "Upcoming News - Weekly View - Last Loaded from Cloud"
    elif st.session_state.data_source == 'forex_factory':
        expander_title = "Upcoming News - Weekly View - Last Loaded from Forex Factory"
    else:
        expander_title = "Upcoming News - Weekly View"

    if red_events:
        now_melb = datetime.now(melbourne_tz)

        filtered = []
        start_of_week = now_melb.date() - timedelta(days=now_melb.weekday())  # Monday
        end_of_week = start_of_week + timedelta(days=6)  # Sunday

        for e in red_events:
            try:
                dt_local = isoparse(e['TimeMelbourne'])
                event_date = dt_local.date()
                if start_of_week <= event_date <= end_of_week:
                    filtered.append((dt_local, e))
            except Exception as ex:
                st.write(f"Error parsing event time: {ex}")
                continue

        filtered.sort(key=lambda x: x[0])  # Sort by time

        events_by_day = defaultdict(list)
        for dt_local, e in filtered:
            date_key = dt_local.date()
            events_by_day[date_key].append((dt_local, e))

        today = now_melb.date()
        week_dates = [today + timedelta(days=i) for i in range(7)]

        # Use dynamic expander title
        with st.expander(expander_title, expanded=False):
            cols = st.columns(7)
            for i, day in enumerate(week_dates):
                with cols[i]:
                    day_events = events_by_day.get(day, [])

                    day_name = day.strftime('%a')
                    day_num = day.strftime('%d %b')

                    # Header styling
                    if day == today:
                        # Blue for today
                        st.markdown(f"""
                            <a href="https://www.forexfactory.com/calendar?day=today" target="_blank" style="text-decoration: none;">
                                <div style="
                                    background-color: #007bff;
                                    color: white;
                                    padding: 10px;
                                    border-radius: 8px;
                                    text-align: center;
                                    font-weight: bold;
                                    margin-bottom: 15px;
                                ">
                                    {day_name}<br>{day_num}
                                </div>
                            </a>
                        """, unsafe_allow_html=True)
                    else:
                        # Black for other days
                        st.markdown(f"""
                            <div style="
                                background-color: black;
                                color: white;
                                padding: 10px;
                                border-radius: 8px;
                                text-align: center;
                                font-weight: bold;
                                margin-bottom: 15px;
                            ">
                                {day_name}<br>{day_num}
                            </div>
                        """, unsafe_allow_html=True)

                    if not day_events:
                        st.caption("No events")
                    else:
                        for dt_local, e in sorted(day_events, key=lambda x: x[0]):
                            time_str = dt_local.strftime('%I:%M %p')
                            event_name = e['Event']
                            currency = e['Currency']
                            impact = e.get('Impact', '').upper()

                            is_holiday = (impact == 'HOLIDAY')
                            should_highlight = (
                                    impact == 'HIGH' and any(
                                keyword.lower() in event_name.lower()
                                for keyword in [
                                    "FOMC", "Cash Rate", "Interest Rate", "Unemployment Rate",
                                    "GDP", "Non-Farm", "CPI", "election", "non farm",
                                    "PMI", "Unemployment claims"
                                ]
                            )
                            )

                            if is_holiday:
                                st.markdown(f"""
                                    <div style="
                                        background-color: #d3d3d3;
                                        color: black;
                                        padding: 8px;
                                        border-radius: 6px;
                                        margin-bottom: 6px;
                                    ">
                                    <b>{time_str}</b> [{currency}] {event_name}
                                    </div>
                                """, unsafe_allow_html=True)
                            elif should_highlight:
                                st.markdown(f"""
                                    <div style="
                                        background-color: #ff4444;
                                        color: white;
                                        padding: 8px;
                                        border-radius: 6px;
                                        margin-bottom: 6px;
                                    ">
                                    <b>{time_str}</b> [{currency}] {event_name}
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.write(f"**{time_str}** [{currency}] {event_name}")

                    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
    else:
        st.write("No high-impact events found.")


    # Use your existing Google Sheets functions
    def load_workflow_from_sheets():
        """Load workflow data from Google Sheets using existing functions"""
        try:
            df = load_data_from_sheets(sheet_name="Trade", worksheet_name="Workflow")
            if df is not None and not df.empty:
                if 'timestamp' in df.columns:
                    df['timestamp'] = df['timestamp'].astype(str)
                numeric_columns = ['entry_price', 'exit_price', 'target_price', 'stop_pips', 'position_size',
                                   'stop_loss_pct', 'max_adverse_excursion']  # ADDED NEW FIELDS
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = df[col].replace('None', None)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                return df
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading workflow data: {e}")
            return pd.DataFrame()


    def save_workflow_to_sheets(data):
        """Save workflow data to Google Sheets using existing functions"""
        try:
            if isinstance(data, list):
                data_to_save = pd.DataFrame(data)
            else:
                data_to_save = data.copy()

            required_columns = ['selected_pair', 'risk_multiplier', 'position_size', 'stop_pips',
                                'entry_price', 'exit_price', 'target_price', 'status', 'timestamp',
                                'stop_loss_pct', 'max_adverse_excursion']  # ADDED NEW FIELDS
            for col in required_columns:
                if col not in data_to_save.columns:
                    data_to_save[col] = None

            data_clean = clean_data_for_google_sheets(data_to_save)
            success = save_data_to_sheets(data_clean, sheet_name="Trade", worksheet_name="Workflow")
            return success
        except Exception as e:
            st.error(f"Error saving workflow data: {e}")
            return False


    def sync_with_trade_signals():
        """Two-way sync between Active Opps and Trade Signals - FIXED VERSION"""
        try:
            # Get ALL records from Active Opps (not just by status)
            all_active_opps_records = {r['timestamp']: r for r in st.session_state.saved_records}

            # SYNC: Ready to Order - Include ALL Order Ready records
            order_ready_records = [r for r in st.session_state.saved_records if r.get('status') == 'Order Ready']

            # Clear and rebuild ready_to_order to ensure it matches Active Opps exactly
            st.session_state.ready_to_order = []
            for record in order_ready_records:
                st.session_state.ready_to_order.append({
                    'timestamp': record['timestamp'],
                    'selected_pair': record['selected_pair'],
                    'risk_multiplier': record['risk_multiplier'],
                    'position_size': record['position_size'],
                    'stop_pips': record['stop_pips'],
                    'entry_price': record['entry_price'],
                    'exit_price': record['exit_price'],
                    'target_price': record['target_price'],
                    'trend_position': record.get('trend_position', 'Not set'),
                    'variances': record.get('Variances', 'Not set'),
                    'status': 'Order Ready'
                })

            # SYNC: Order Placed - Include ALL Order Placed records
            order_placed_records = [r for r in st.session_state.saved_records if r.get('status') == 'Order Placed']

            # Clear and rebuild order_placed to ensure it matches Active Opps exactly
            st.session_state.order_placed = []
            for record in order_placed_records:
                st.session_state.order_placed.append({
                    'timestamp': record['timestamp'],
                    'selected_pair': record['selected_pair'],
                    'risk_multiplier': record['risk_multiplier'],
                    'position_size': record['position_size'],
                    'stop_pips': record['stop_pips'],
                    'entry_price': record['entry_price'],
                    'exit_price': record['exit_price'],
                    'target_price': record['target_price'],
                    'order_status': 'PENDING',
                    'status': 'Order Placed'
                })

            # SYNC: In Trade - Include ALL Order Filled records
            order_filled_records = [r for r in st.session_state.saved_records if r.get('status') == 'Order Filled']

            # Clear and rebuild in_trade to ensure it matches Active Opps exactly
            st.session_state.in_trade = []
            for record in order_filled_records:
                st.session_state.in_trade.append({
                    'timestamp': record['timestamp'],
                    'selected_pair': record['selected_pair'],
                    'risk_multiplier': record['risk_multiplier'],
                    'position_size': record['position_size'],
                    'stop_pips': record['stop_pips'],
                    'entry_price': record['entry_price'],
                    'exit_price': record['exit_price'],
                    'target_price': record['target_price'],
                    'order_status': 'FILLED',
                    'status': 'Order Filled'
                })

            return True, "Sync completed successfully"

        except Exception as e:
            return False, f"Sync error: {str(e)}"


    # SIMPLE ACTION FUNCTIONS
    def handle_update_record(record_index, entry_price, exit_price, target_price):
        """Handle record update"""
        try:
            st.session_state.saved_records[record_index]['entry_price'] = entry_price
            st.session_state.saved_records[record_index]['exit_price'] = exit_price
            st.session_state.saved_records[record_index]['target_price'] = target_price

            success = save_workflow_to_sheets(st.session_state.saved_records)
            if success:
                st.session_state.last_action = f"updated_record_{record_index}"
                return True
            return False
        except Exception as e:
            st.error(f"Update error: {e}")
            return False


    def handle_move_record(record_index, new_status):
        """Handle moving record to new status with cross-group auto-deletion"""
        try:
            # Check if moving to Order Ready and limit is reached
            if new_status == 'Order Ready':
                current_order_ready_count = sum(
                    1 for r in st.session_state.saved_records if r.get('status') == 'Order Ready')
                if current_order_ready_count >= 2:
                    st.error("Maximum limit of 2 Order Ready orders reached! Cannot move this record to Order Ready.")
                    return False

            # Store the pair for cross-group checking
            moving_pair = st.session_state.saved_records[record_index]['selected_pair']
            old_status = st.session_state.saved_records[record_index].get('status')

            # Update the record status
            st.session_state.saved_records[record_index]['status'] = new_status

            # AUTO-DELETE: If moving to Order Filled, delete same-group speculation records
            deleted_pairs = []
            if new_status == 'Order Filled':
                updated_records, deleted_pairs = delete_cross_group_speculation(
                    moving_pair, st.session_state.saved_records
                )
                st.session_state.saved_records = updated_records

                # Show deletion notification
                if deleted_pairs:
                    st.warning(f"Auto-deleted speculation records in same cross group: {', '.join(deleted_pairs)}")

            success = save_workflow_to_sheets(st.session_state.saved_records)
            if success:
                sync_with_trade_signals()  # Sync after status change
                st.session_state.last_action = f"moved_record_{record_index}_to_{new_status}"

                # Store deleted pairs in session state for display
                if deleted_pairs:
                    st.session_state.last_deleted_pairs = deleted_pairs

                return True
            return False
        except Exception as e:
            st.error(f"Move error: {e}")
            return False


    def handle_delete_record(record_index):
        """Handle record deletion"""
        try:
            st.session_state.saved_records.pop(record_index)
            success = save_workflow_to_sheets(st.session_state.saved_records)
            if success:
                sync_with_trade_signals()  # Sync after deletion
                st.session_state.last_action = f"deleted_record_{record_index}"
                return True
            return False
        except Exception as e:
            st.error(f"Delete error: {e}")
            return False


    # Helper function
    def safe_float(value, default=0.0):
        if value is None or value == 'None' or value == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


    # Generate unique key for each record
    def generate_unique_key(record_index, record, field_name):
        """Generate truly unique key using index, pair, and timestamp"""
        pair = record['selected_pair'].replace('/', '_').replace(' ', '_')
        timestamp = record['timestamp'].replace(':', '_').replace(' ', '_').replace('-', '_').replace('.', '_')
        return f"{field_name}_{record_index}_{pair}_{timestamp}"


    # AUTO-RELOAD FUNCTION - CRITICAL FOR TWO-WAY SYNC
    def check_and_reload_from_sheets():
        """Check if Google Sheets has been updated by Trade Signals and reload if needed"""
        try:
            # Load current data from Google Sheets
            current_sheets_data = load_workflow_from_sheets()
            if current_sheets_data.empty:
                return False

            # Convert to comparable format
            current_timestamps = {str(rec.get('timestamp')) for rec in st.session_state.saved_records}
            sheets_timestamps = {str(rec.get('timestamp')) for rec in current_sheets_data.to_dict('records')}

            # Check if data has changed
            if current_timestamps != sheets_timestamps:
                return True

            # Also check if any statuses have changed
            current_data_df = pd.DataFrame(st.session_state.saved_records)
            if not current_data_df.empty and not current_sheets_data.empty:
                merged = current_data_df.merge(current_sheets_data, on='timestamp', suffixes=('_current', '_sheets'))
                if not merged.empty and 'status_current' in merged.columns and 'status_sheets' in merged.columns:
                    status_changed = any(merged['status_current'] != merged['status_sheets'])
                    if status_changed:
                        return True

            return False
        except Exception as e:
            print(f"Check reload error: {e}")
            return False


    # AUTO-RELOAD ON EVERY PAGE LOAD
    should_reload = check_and_reload_from_sheets()
    if should_reload:
        with st.spinner(" Reloading updated data from cloud..."):
            workflow_data = load_workflow_from_sheets()
            if not workflow_data.empty:
                st.session_state.saved_records = workflow_data.to_dict('records')
                sync_with_trade_signals()
                st.session_state.last_sync_time = datetime.now()
                # st.success(" Data updated from cloud!")

    # INITIAL LOAD
    if not st.session_state.saved_records:
        with st.spinner(" Loading data from cloud..."):
            workflow_data = load_workflow_from_sheets()
            if not workflow_data.empty:
                st.session_state.saved_records = workflow_data.to_dict('records')
                sync_with_trade_signals()
                st.success("Workflow data loaded and synced!")

    # Clean up speculation records for existing Order Filled records
    if st.session_state.saved_records:
        # Get all Order Filled pairs
        filled_pairs = [r['selected_pair'] for r in st.session_state.saved_records if r.get('status') == 'Order Filled']

        # Delete speculation records for each filled pair's cross group
        for filled_pair in filled_pairs:
            updated_records, deleted_pairs = delete_cross_group_speculation(
                filled_pair, st.session_state.saved_records
            )
            st.session_state.saved_records = updated_records

            if deleted_pairs:
                st.info(f"Cleaned up speculation records for filled order {filled_pair}: {', '.join(deleted_pairs)}")

    # Handle action results and force rerun
    if st.session_state.last_action:
        action = st.session_state.last_action
        st.session_state.last_action = None

        if "updated_record" in action:
            st.success(" Record updated successfully!")
        elif "moved_record" in action:
            st.success(" Record moved successfully!")
            # Show auto-deletion message if any pairs were deleted
            if st.session_state.last_deleted_pairs:
                deleted_pairs = st.session_state.last_deleted_pairs
                st.warning(f" Auto-deleted speculation records in same cross group: {', '.join(deleted_pairs)}")
                st.session_state.last_deleted_pairs = []  # Clear after showing
        elif "deleted_record" in action:
            st.success(" Record deleted successfully!")
        elif "loaded_data" in action:
            st.success(" Data loaded successfully!")
        elif "synced_signals" in action:
            st.success(" Trade signals synced successfully!")
        elif "cleared_all" in action:
            st.success(" All records cleared successfully!")

        st.rerun()

    # SYNC STATUS SECTION - ENHANCED FOR TWO-WAY SYNC
    st.write("---")
    col_sync1, col_sync2, col_sync3, col_sync4 = st.columns(4)  # Changed to 4 columns

    with col_sync1:
        if st.button(" Check for Updates", key="check_updates", use_container_width=True):
            with st.spinner("Checking for updates..."):
                should_reload = check_and_reload_from_sheets()
                if should_reload:
                    workflow_data = load_workflow_from_sheets()
                    if not workflow_data.empty:
                        st.session_state.saved_records = workflow_data.to_dict('records')
                        sync_with_trade_signals()
                        st.session_state.last_sync_time = datetime.now()
                        # st.success(" Data updated from cloud!")
                        st.rerun()
                    else:
                        st.info("No data found in cloud")

    with col_sync2:
        if st.button(" Add New Speculation", key="add_new_spec", use_container_width=True):
            st.session_state.current_page = "Risk Calculation"
            st.rerun()

    with col_sync3:
        if st.button(" View Trade Signals", key="view_signals", use_container_width=True):
            st.session_state.current_page = "Trade Signal"
            st.rerun()

    with col_sync4:
        st.write(f"**Last sync:** {st.session_state.last_sync_time.strftime('%H:%M:%S')}")

    # CSV Upload as fallback
    st.sidebar.markdown("---")
    with st.sidebar.expander("CSV Backup", expanded=False):
        st.subheader("Upload Workflow Backup")
        uploaded_workflow_file = st.file_uploader("Upload Workflow CSV", type=['csv'], key="workflow_uploader")

        if uploaded_workflow_file is not None:
            try:
                # Load CSV with expected columns
                workflow_csv_data = pd.read_csv(uploaded_workflow_file)

                # Validate required columns
                required_columns = ['selected_pair', 'risk_multiplier', 'position_size', 'stop_pips',
                                    'entry_price', 'exit_price', 'target_price', 'status', 'timestamp',
                                    'stop_loss_pct', 'max_adverse_excursion']  # ADDED NEW FIELDS

                missing_columns = [col for col in required_columns if col not in workflow_csv_data.columns]

                if missing_columns:
                    st.error(f"Missing required columns in CSV: {', '.join(missing_columns)}")
                    st.info(
                        "Expected columns: selected_pair, risk_multiplier, position_size, stop_pips, entry_price, exit_price, target_price, status, timestamp")
                else:
                    # Clean the data - handle None values and convert numeric columns
                    for col in ['position_size', 'stop_pips', 'entry_price', 'exit_price', 'target_price',
                                'stop_loss_pct', 'max_adverse_excursion']:  # ADDED NEW FIELDS
                        if col in workflow_csv_data.columns:
                            workflow_csv_data[col] = workflow_csv_data[col].replace('None', None)
                            workflow_csv_data[col] = pd.to_numeric(workflow_csv_data[col], errors='coerce').fillna(0.0)

                    # Ensure status values are valid (UPDATED STATUSES)
                    valid_statuses = ['Speculation', 'Order Ready', 'Order Placed', 'Order Filled']
                    workflow_csv_data['status'] = workflow_csv_data['status'].apply(
                        lambda x: x if x in valid_statuses else 'Speculation'
                    )

                    # Convert DataFrame to list of dictionaries
                    csv_records = workflow_csv_data.to_dict('records')

                    # Show preview and options
                    st.success(f" Valid CSV format! Found {len(csv_records)} records")

                    # Show summary statistics
                    status_counts = workflow_csv_data['status'].value_counts()
                    st.write("**Records by status:**")
                    for status, count in status_counts.items():
                        st.write(f"- {status}: {count}")

                    col_replace, col_merge = st.columns(2)
                    with col_replace:
                        if st.button("Replace Current Data", key="replace_csv", type="primary"):
                            st.session_state.saved_records = csv_records
                            # Sync trade signals after replacing data
                            sync_with_trade_signals()
                            # Save to Google Sheets
                            save_workflow_to_sheets(st.session_state.saved_records)
                            st.session_state.last_action = "loaded_data"
                            st.rerun()

                    with col_merge:
                        if st.button("Merge with Current", key="merge_csv"):
                            # Merge without duplicates based on timestamp
                            current_timestamps = {r.get('timestamp') for r in st.session_state.saved_records}
                            new_records = [r for r in csv_records if r.get('timestamp') not in current_timestamps]

                            if new_records:
                                st.session_state.saved_records.extend(new_records)
                                # Sync trade signals after merging data
                                sync_with_trade_signals()
                                # Save to Google Sheets
                                save_workflow_to_sheets(st.session_state.saved_records)
                                st.success(f" Added {len(new_records)} new records from CSV!")
                                st.rerun()
                            else:
                                st.info("No new records to add (all timestamps already exist)")

                    # Show preview of CSV data
                    with st.expander(" Preview CSV Data (first 5 rows)"):
                        st.dataframe(workflow_csv_data.head())

                    # Show data quality info
                    with st.expander(" Data Quality Check"):
                        st.write(f"**Total records:** {len(csv_records)}")
                        st.write(f"**Unique pairs:** {workflow_csv_data['selected_pair'].nunique()}")
                        st.write(f"**Records with entry price > 0:** {(workflow_csv_data['entry_price'] > 0).sum()}")
                        st.write(f"**Records with target price > 0:** {(workflow_csv_data['target_price'] > 0).sum()}")

            except Exception as e:
                st.error(f"Error reading workflow CSV: {e}")
                st.info("Make sure your CSV has the correct format with these columns:")
                st.code("selected_pair, risk_multiplier, position_size, stop_pips, entry_price, exit_price, target_price, status, timestamp, stop_loss_pct, max_adverse_excursion")

    # Count records by status
    speculation_count = sum(1 for r in st.session_state.saved_records if r.get('status') == 'Speculation')
    order_ready_count = sum(1 for r in st.session_state.saved_records if r.get('status') == 'Order Ready')
    order_placed_count = sum(1 for r in st.session_state.saved_records if r.get('status') == 'Order Placed')
    order_filled_count = sum(1 for r in st.session_state.saved_records if r.get('status') == 'Order Filled')
    # Only count Order Placed and Order Filled as active records (exclude Order Ready)
    total_active_count = order_ready_count + order_placed_count + order_filled_count

    # Display counts with Order Ready limit warning
    # st.write(f"**Order Ready Records:** {order_ready_count}/3")
    # if order_ready_count >= 3:
    # st.warning("⚠️ Maximum limit of 3 Order Ready orders reached!")
    # st.write(f"**Total Records:** {len(st.session_state.saved_records)}/5")
    # st.write(f"**Active Records (Order Placed + Order Filled):** {total_active_count}/2")
    # st.write(
    # f"Speculation: {speculation_count}, Order Ready: {order_ready_count}, Order Placed: {order_placed_count}, Order Filled: {order_filled_count}")

    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(" Load from Cloud", key="load_cloud"):
            with st.spinner("Loading..."):
                workflow_data = load_workflow_from_sheets()
                if not workflow_data.empty:
                    st.session_state.saved_records = workflow_data.to_dict('records')
                    sync_with_trade_signals()
                    st.session_state.last_action = "loaded_data"
                    st.rerun()
                else:
                    st.error("No data found")

    with col2:
        if st.button(" Sync Trade Signals", key="sync_signals"):
            success, message = sync_with_trade_signals()
            if success:
                st.session_state.last_action = "synced_signals"
                st.rerun()
            else:
                st.error(f"Sync failed: {message}")

    with col3:
        if st.session_state.saved_records:
            csv_data = pd.DataFrame(st.session_state.saved_records)
            column_order = ['selected_pair', 'risk_multiplier', 'position_size', 'stop_pips',
                            'entry_price', 'exit_price', 'target_price', 'status', 'timestamp',
                            'stop_loss_pct', 'max_adverse_excursion']  # ADDED NEW FIELDS
            for col in column_order:
                if col not in csv_data.columns:
                    csv_data[col] = None
            csv_data = csv_data[column_order]
            csv = csv_data.to_csv(index=False)
            st.download_button(
                " Export CSV",
                csv,
                f"workflow_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                key="export_csv"
            )
        else:
            st.button(" Export CSV", disabled=True, key="export_disabled")

    if not st.session_state.saved_records:
        st.info(
            "No records saved yet. Go to Risk Calculation page and save some records, or upload a CSV backup above.")
    else:
        # Display all records
        records_df = pd.DataFrame(st.session_state.saved_records)
        st.dataframe(records_df, use_container_width=True)

        # WORKFLOW VISUALIZATION
        st.markdown("---")
        st.subheader("Workflow Stages")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button(f"Speculation ({speculation_count})", use_container_width=True,
                         type="primary" if st.session_state.current_stage == 'Speculation' else "secondary",
                         key="stage_speculation"):
                st.session_state.current_stage = 'Speculation'
                st.rerun()
        with col2:
            if st.button(f"Order Ready ({order_ready_count})", use_container_width=True,
                         type="primary" if st.session_state.current_stage == 'Order Ready' else "secondary",
                         key="stage_order_ready"):
                st.session_state.current_stage = 'Order Ready'
                st.rerun()
        with col3:
            if st.button(f"Order Placed ({order_placed_count})", use_container_width=True,
                         type="primary" if st.session_state.current_stage == 'Order Placed' else "secondary",
                         key="stage_order_placed"):
                st.session_state.current_stage = 'Order Placed'
                st.rerun()
        with col4:
            if st.button(f"Order Filled ({order_filled_count})", use_container_width=True,
                         type="primary" if st.session_state.current_stage == 'Order Filled' else "secondary",
                         key="stage_order_filled"):
                st.session_state.current_stage = 'Order Filled'
                st.rerun()

        # Progress bars
        st.markdown("<br>", unsafe_allow_html=True)
        col_prog1, col_prog2, col_prog3, col_prog4 = st.columns(4)
        with col_prog1:
            st.progress(speculation_count / max(len(st.session_state.saved_records), 1))
        with col_prog2:
            st.progress(order_ready_count / max(len(st.session_state.saved_records), 1))
        with col_prog3:
            st.progress(order_placed_count / max(len(st.session_state.saved_records), 1))
        with col_prog4:
            st.progress(order_filled_count / max(len(st.session_state.saved_records), 1))

        st.markdown("---")
        st.subheader(f"{st.session_state.current_stage} Stage")

        # Filter records for current stage
        stage_records = [
            (i, record) for i, record in enumerate(st.session_state.saved_records)
            if record.get('status') == st.session_state.current_stage
        ]

        if not stage_records:
            st.info(f"No records in {st.session_state.current_stage.lower()} stage.")
        else:
            for record_index, record in stage_records:
                # Generate unique keys for this record
                unique_key_base = generate_unique_key(record_index, record, "")

                with st.expander(f"Record {record_index + 1}: {record['selected_pair']} - {record['timestamp']}",
                                 expanded=False):
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                    with col1:
                        st.write(f"**Pair:** {record['selected_pair']}")
                        st.write(f"**Strategy:** {record['risk_multiplier']}")
                        st.write(f"**Position Size:** {record['position_size']}")
                        st.write(f"**Stop Pips:** {record.get('stop_pips', 'N/A')}")
                        st.write(f"**Target RR:** {record.get('exit_rr', 'N/A')}")
                        # Show final risk if available
                        if 'final_risk' in record:
                            st.write(f"**Final Risk:** ${record['final_risk']}")

                    with col2:
                        entry_price = st.number_input(
                            "Entry Price",
                            value=safe_float(record.get('entry_price'), 0.0),
                            format="%.5f",
                            key=f"entry_{unique_key_base}"
                        )

                    with col3:
                        exit_price = st.number_input(
                            "Exit Price",
                            value=safe_float(record.get('exit_price'), 0.0),
                            format="%.5f",
                            key=f"exit_{unique_key_base}"
                        )

                    with col4:
                        target_price = st.number_input(
                            "Target Price",
                            value=safe_float(record.get('target_price'), 0.0),
                            format="%.5f",
                            key=f"target_{unique_key_base}"
                        )

                    # SPECULATION STAGE ACTIONS
                    if st.session_state.current_stage == 'Speculation':
                        # Calculate expected stop pips
                        expected_stop_pips = None
                        if entry_price != 0 and exit_price != 0:
                            if record['selected_pair'] == 'XAUUSD':
                                expected_stop_pips = abs(exit_price - entry_price)
                            else:
                                price_diff = abs(exit_price - entry_price)
                                if any(currency in record['selected_pair'] for currency in ['JPY', 'XAU']):
                                    pip_size = 0.01
                                else:
                                    pip_size = 0.0001
                                expected_stop_pips = price_diff / pip_size

                        # Show validation if stop pips don't match
                        current_stop_pips = record.get('stop_pips')
                        if expected_stop_pips is not None and current_stop_pips is not None:
                            if abs(expected_stop_pips - current_stop_pips) > 0.1:
                                st.error(
                                    f"Stop pips mismatch! Current: {current_stop_pips:.2f}, Expected: {expected_stop_pips:.2f}")

                        # Check required fields
                        entry_price_valid = entry_price > 0
                        exit_price_valid = exit_price > 0
                        target_price_valid = target_price > 0
                        all_required_fields_valid = entry_price_valid and exit_price_valid and target_price_valid

                        if not entry_price_valid:
                            st.error("Entry price must be > 0 to move to Order Ready")
                        if not exit_price_valid:
                            st.error("Exit price must be > 0 to move to Order Ready")
                        if not target_price_valid:
                            st.error("Target price must be > 0 to move to Order Ready")

                        # NEW: Check cross group conflicts
                        active_stage_records = [
                            r for r in st.session_state.saved_records
                            if r.get('status') in ['Order Ready', 'Order Placed', 'Order Filled']
                        ]

                        has_cross_group_conflict, conflict_groups, conflicting_pairs = check_cross_group_conflict(
                            record['selected_pair'], active_stage_records
                        )


                        col_update, col_move, col_delete = st.columns(3)

                        with col_update:
                            if st.button("Update Record", key=f"update_{unique_key_base}"):
                                if handle_update_record(record_index, entry_price, exit_price, target_price):
                                    st.rerun()

                        with col_move:
                            # Check Order Ready limit
                            current_order_ready_count = sum(
                                1 for r in st.session_state.saved_records if r.get('status') == 'Order Ready')

                            # Check all conditions for moving to Order Ready
                            can_move = (
                                    all_required_fields_valid and
                                    total_active_count < 2 and
                                    current_order_ready_count < 2 and
                                    not has_cross_group_conflict
                            )

                            if st.button("Move to Order Ready",
                                         key=f"move_{unique_key_base}",
                                         disabled=not can_move):
                                if handle_move_record(record_index, 'Order Ready'):
                                    st.rerun()

                            # Show appropriate error messages
                            if has_cross_group_conflict:
                                st.error(
                                    f"Cross-group conflict! {record['selected_pair']} shares groups with: {', '.join(conflicting_pairs)}")
                            if total_active_count >= 2:
                                st.error("Max 2 active records reached (Order Ready + Order Placed + Order Filled)")
                            elif current_order_ready_count >= 2:
                                st.error("Maximum limit of 2 'Order Ready' orders reached!")

                        with col_delete:
                            if st.button("Delete", key=f"delete_{unique_key_base}"):
                                if handle_delete_record(record_index):
                                    st.rerun()


                    # ORDER READY STAGE ACTIONS
                    elif st.session_state.current_stage == 'Order Ready':
                        col_update, col_spec, col_move, col_delete = st.columns(4)  # Changed to 4 columns

                        with col_update:
                            if st.button("Update Record", key=f"update_{unique_key_base}"):
                                if handle_update_record(record_index, entry_price, exit_price, target_price):
                                    st.rerun()

                        with col_spec:  # NEW: Move to Speculation button
                            if st.button("Move to Speculation", key=f"move_spec_{unique_key_base}"):
                                if handle_move_record(record_index, 'Speculation'):
                                    st.rerun()

                        with col_move:
                            if st.button("Move to Order Placed", key=f"move_{unique_key_base}"):
                                if handle_move_record(record_index, 'Order Placed'):
                                    st.rerun()

                        with col_delete:
                            if st.button("Delete", key=f"delete_{unique_key_base}"):
                                if handle_delete_record(record_index):
                                    st.rerun()

                    # ORDER PLACED STAGE ACTIONS
                    elif st.session_state.current_stage == 'Order Placed':
                        col_update, col_back, col_move, col_delete = st.columns(4)

                        with col_update:
                            if st.button("Update Record", key=f"update_{unique_key_base}"):
                                if handle_update_record(record_index, entry_price, exit_price, target_price):
                                    st.rerun()

                        with col_move:
                            if st.button("Move to Order Filled", key=f"move_{unique_key_base}"):
                                if handle_move_record(record_index, 'Order Filled'):
                                    st.rerun()

                        with col_back:
                            if st.button("Back to Order Ready", key=f"back_{unique_key_base}"):
                                if handle_move_record(record_index, 'Order Ready'):
                                    st.rerun()

                        with col_delete:  # NEW: Delete button
                            if st.button("Delete", key=f"delete_{unique_key_base}"):
                                if handle_delete_record(record_index):
                                    st.rerun()

                    # ORDER FILLED STAGE ACTIONS
                    else:
                        # Additional fields for Order Filled stage
                        col5, col6, col7, col8 = st.columns(4)

                        with col5:
                            result_options = ["BE", "Loss", "Win"]
                            new_result = st.selectbox(
                                "Result",
                                options=result_options,
                                index=0,
                                key=f"result_{unique_key_base}"
                            )

                        with col6:
                            direction_options = ["buy", "sell"]
                            new_direction = st.selectbox(
                                "Direction",
                                options=direction_options,
                                index=0,
                                key=f"direction_{unique_key_base}"
                            )

                        with col7:
                            new_rr = st.number_input(
                                "RR",
                                value=0.0,
                                step=0.01,
                                format="%.2f",
                                key=f"rr_{unique_key_base}"
                            )

                        with col8:
                            new_pnl = st.number_input(
                                "PnL",
                                value=0.0,
                                step=0.01,
                                format="%.2f",
                                key=f"pnl_{unique_key_base}"
                            )

                        # NEW: Add the two additional numeric fields
                        col9, col10 = st.columns(2)
                        with col9:
                            stop_loss_pct = st.number_input(
                                "Stop Loss Percentage",
                                value=0.0,
                                step=0.01,
                                format="%.2f",
                                key=f"stop_loss_pct_{unique_key_base}",
                                help="Stop loss as percentage (e.g., 1.5 for 1.5%)"
                            )

                        with col10:
                            max_adverse_excursion = st.number_input(
                                "Maximum Adverse Excursion",
                                value=0.0,
                                step=0.01,
                                format="%.2f",
                                key=f"max_adverse_excursion_{unique_key_base}",
                                help="Maximum adverse excursion as percentage"
                            )

                        # Additional required fields
                        col11, col12 = st.columns(2)
                        with col11:
                            poi_options = ["Weekly", "2_Daily"]

                            # Get existing POI value from record and map to dropdown options
                            existing_poi = record.get('poi', '')
                            poi_display = "Weekly"  # Default

                            if existing_poi == "Weekly Structure" or existing_poi == "Weekly":
                                poi_display = "Weekly"
                            elif existing_poi == "Two_Daily Structure" or existing_poi == "2_Daily":
                                poi_display = "2_Daily"
                            elif existing_poi in poi_options:
                                poi_display = existing_poi

                            st.text_input(
                                "POI",
                                value=poi_display,
                                key=f"poi_{unique_key_base}",
                                disabled=True,
                                help="POI is set from the original record"
                            )
                            # Store the value for the record
                            new_poi = poi_display

                        with col12:
                            st.text_input(
                                "Strategy",
                                value=record['risk_multiplier'],
                                key=f"strategy_{unique_key_base}",
                                disabled=True
                            )

                        # Display existing Trend Position and Variance (read-only)
                        st.write("---")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            existing_trend_position = record.get('trend_position', 'Not set')
                            st.write(f"**Trend Position:** {existing_trend_position}")
                        with col_info2:
                            existing_variance = record.get('Variances', 'Not set')
                            st.write(f"**Variance:** {existing_variance}")

                        # Action buttons for Order Filled stage
                        col_update_only, col_back, col_close = st.columns([1, 1, 1])

                        with col_update_only:
                            if st.button("Update Record", key=f"filled_update_{unique_key_base}"):
                                # Update all fields including the new ones
                                st.session_state.saved_records[record_index]['entry_price'] = entry_price
                                st.session_state.saved_records[record_index]['exit_price'] = exit_price
                                st.session_state.saved_records[record_index]['target_price'] = target_price
                                st.session_state.saved_records[record_index]['result'] = new_result
                                st.session_state.saved_records[record_index]['direction'] = new_direction
                                st.session_state.saved_records[record_index]['rr'] = new_rr
                                st.session_state.saved_records[record_index]['pnl'] = new_pnl
                                st.session_state.saved_records[record_index]['poi'] = new_poi
                                st.session_state.saved_records[record_index]['stop_loss_pct'] = stop_loss_pct  # NEW
                                st.session_state.saved_records[record_index][
                                    'max_adverse_excursion'] = max_adverse_excursion  # NEW

                                success = save_workflow_to_sheets(st.session_state.saved_records)
                                if success:
                                    st.session_state.last_action = f"updated_record_{record_index}"
                                    st.rerun()

                        with col_close:
                            if st.button("Finalize & Close Trade", key=f"filled_close_{unique_key_base}",
                                         type="primary"):
                                # Validate required fields
                                if (new_result and new_direction and new_poi and
                                        new_rr is not None and new_pnl is not None):

                                    # Get existing Trend Position and Variance from the record
                                    existing_trend_position = record.get('trend_position')
                                    existing_variance = record.get('Variances')

                                    # Validate that we have the required existing values
                                    if not existing_trend_position or not existing_variance:
                                        st.error(
                                            "Missing Trend Position or Variance data from original record. Cannot close trade.")
                                    else:
                                        # Create the completed trade record with specified fields
                                        completed_trade = {
                                            'Date': datetime.now().strftime("%Y-%m-%d"),
                                            'Symbol': record['selected_pair'],
                                            'Direction': new_direction,
                                            'Trend Position': existing_trend_position,
                                            'POI': new_poi,
                                            'Strategy': record['risk_multiplier'],
                                            'Variance': existing_variance,
                                            'Result': new_result,
                                            'RR': new_rr,
                                            'PnL': new_pnl,
                                            'Stop Loss Percentage': stop_loss_pct,  # NEW FIELD
                                            'Maximum Adverse Excursion': max_adverse_excursion,  # NEW FIELD
                                            'Withdrawal_Deposit': 0.0,
                                            'PROP_Pct': 0.0
                                        }

                                        try:
                                            # Load current Trade.csv data
                                            current_trade_data = load_data_from_sheets(sheet_name="Trade",
                                                                                       worksheet_name="Trade.csv")

                                            if current_trade_data is not None and not current_trade_data.empty:
                                                # Ensure only the specified columns exist in current data
                                                required_columns = ['Date', 'Symbol', 'Direction', 'Trend Position',
                                                                    'POI', 'Strategy', 'Variance', 'Result', 'RR',
                                                                    'PnL', 'Stop Loss Percentage',
                                                                    'Maximum Adverse Excursion',  # ADDED NEW COLUMNS
                                                                    'Withdrawal_Deposit', 'PROP_Pct']

                                                # Add missing columns if they don't exist
                                                for col in required_columns:
                                                    if col not in current_trade_data.columns:
                                                        current_trade_data[col] = None

                                                # Keep only the required columns
                                                current_trade_data = current_trade_data[required_columns]

                                                # Append the new completed trade
                                                new_trade_df = pd.DataFrame([completed_trade])
                                                updated_trade_data = pd.concat([current_trade_data, new_trade_df],
                                                                               ignore_index=True)

                                            else:
                                                # If no existing trade data, create new with only specified columns
                                                updated_trade_data = pd.DataFrame([completed_trade])

                                            # Save to Trade.csv worksheet
                                            success = save_data_to_sheets(updated_trade_data, sheet_name="Trade",
                                                                          worksheet_name="Trade.csv")

                                            if success:
                                                # Remove from active opps
                                                st.session_state.saved_records.pop(record_index)
                                                # Save the updated workflow
                                                save_workflow_to_sheets(st.session_state.saved_records)
                                                # Sync trade signals after closing trade
                                                sync_with_trade_signals()
                                                st.success(" Trade finalized and saved to trade history!")
                                                st.rerun()
                                            else:
                                                st.error("Failed to save to trade history - Google Sheets error")

                                        except Exception as e:
                                            st.error(f"Error saving trade history: {str(e)}")
                                            st.info("Please check your Google Sheets connection and try again.")
                                else:
                                    st.error("Please fill in all required fields")

                        with col_back:
                            if st.button("Back to Order Placed", key=f"filled_back_{unique_key_base}"):
                                if handle_move_record(record_index, 'Order Placed'):
                                    st.rerun()

        # Clear all records button
        st.markdown("---")
        if st.button("Clear All Records", type="secondary", key="clear_all"):
            st.session_state.saved_records = []
            st.session_state.ready_to_order = []
            st.session_state.order_placed = []
            st.session_state.in_trade = []
            save_workflow_to_sheets([])
            st.session_state.last_action = "cleared_all"
            st.rerun()

elif st.session_state.current_page == "Trade Signal":
    # Install MetaApi SDK if not available
    try:
        from metaapi_cloud_sdk import MetaApi

        metaapi_available = True
    except ImportError:
        metaapi_available = False
        st.error("MetaApi SDK not installed. Please add 'metaapi-cloud-sdk' to requirements.txt")
        st.stop()

    st.title("Trade Signals")

    # Add CSS for responsive UI
    st.markdown("""
    <style>
        .stButton button {
            transition: all 0.2s ease-in-out;
        }
        .stButton button:active {
            transform: scale(0.98);
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session states
    if 'trade_signals' not in st.session_state:
        st.session_state.trade_signals = []
    if 'metaapi_connected' not in st.session_state:
        st.session_state.metaapi_connected = False
    if 'metaapi_account_id' not in st.session_state:
        st.session_state.metaapi_account_id = "a8776e6a-6bcd-46a5-9e57-9f1d50c6499b"
    if 'metaapi_connection' not in st.session_state:
        st.session_state.metaapi_connection = None
    if 'ready_to_order' not in st.session_state:
        st.session_state.ready_to_order = []
    if 'order_placed' not in st.session_state:
        st.session_state.order_placed = []
    if 'in_trade' not in st.session_state:
        st.session_state.in_trade = []
    if 'last_trade_result' not in st.session_state:
        st.session_state.last_trade_result = None
    if 'open_positions' not in st.session_state:
        st.session_state.open_positions = []
    if 'auto_move_checked' not in st.session_state:
        st.session_state.auto_move_checked = False
    if 'last_action' not in st.session_state:
        st.session_state.last_action = None

    # RESET ALL CONFIRMATION STATES WHEN PAGE LOADS
    if 'confirmation_states_reset' not in st.session_state:
        # Clear any existing confirmation states
        keys_to_remove = []
        for key in st.session_state.keys():
            if (key.startswith('confirm_execute_') or
                    key.startswith('confirm_back_ready_') or
                    key.startswith('confirm_update_sl_') or
                    key.startswith('confirm_be_')):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del st.session_state[key]

        st.session_state.confirmation_states_reset = True


    # Add helper function first
    def safe_float(value, default=0.0):
        """Safely convert value to float, handling None and string 'None'"""
        if value is None or value == 'None' or value == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


    def calculate_direction(entry_price, exit_price):
        """Calculate direction based on entry and exit prices"""
        if entry_price is None or exit_price is None:
            return "Unknown"
        try:
            entry = safe_float(entry_price, 0.0)
            exit = safe_float(exit_price, 0.0)
            if entry > exit:
                return "BUY"
            elif entry < exit:
                return "SELL"
            else:
                return "Unknown"
        except:
            return "Unknown"


    def calculate_be_price(entry_price, exit_price, direction, strategy):
        """Calculate Break-Even Price based on strategy and direction."""
        try:
            entry = safe_float(entry_price, 0.0)
            exit_val = safe_float(exit_price, 0.0)

            if entry == 0 or exit_val == 0:
                return 0.0

            risk_distance = abs(entry - exit_val)

            # Strategy → Multiplier mapping
            strategy_multipliers = {
                '1_BNR': 2.5,
                '1_BNR_TPF': 2.5,
                '2_BNR': 2.0,
                '2_BNR_TPF': 2.0
            }
            multiplier = strategy_multipliers.get(strategy, 2.0)

            direction = direction.upper()
            if direction == 'BUY':
                return entry + (risk_distance * multiplier)
            elif direction == 'SELL':
                return entry - (risk_distance * multiplier)
            else:
                return 0.0  # Invalid direction
        except:
            return 0.0

    # Add this new function to calculate first trail price
    def calculate_first_trail_price(entry_price, stop_loss, direction, strategy):
        """Calculate first trail price based on strategy and trade direction ('BUY' or 'SELL')"""
        try:
            entry = safe_float(entry_price, 0.0)
            stop = safe_float(stop_loss, 0.0)

            if entry == 0 or stop == 0:
                return 0.0

            risk_distance = abs(entry - stop)

            # Strategy → Multiplier mapping
            strategy_multipliers = {
                '1_BNR': 4.0,
                '1_BNR_TPF': 4.0,
                '2_BNR': 4.0,
                '2_BNR_TPF': 4.0
            }
            multiplier = strategy_multipliers.get(strategy, 4.0)

            direction = direction.upper()
            if direction == 'BUY':
                return entry + (risk_distance * multiplier)
            elif direction == 'SELL':
                return entry - (risk_distance * multiplier)
            else:
                return 0.0  # Invalid direction
        except:
            return 0.0


    def format_symbol_for_pepperstone(symbol):
        """Add .a suffix to symbols for Pepperstone broker"""
        pepperstone_symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
            'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
            'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
            'CADJPY', 'CHFJPY', 'NZDJPY',
            'XAUUSD', 'XAGUSD',
            'USOIL', 'UKOIL', 'NAS100', 'US30', 'SPX500', 'GER30', 'UK100'
        ]
        if symbol in pepperstone_symbols:
            return f"{symbol}.a"
        else:
            return symbol


    # Generate unique key for each record
    def generate_unique_key(record_index, record, field_name):
        """Generate truly unique key using index, pair, and timestamp"""
        pair = record['selected_pair'].replace('/', '_').replace(' ', '_')
        timestamp = record['timestamp'].replace(':', '_').replace(' ', '_').replace('-', '_').replace('.', '_')
        return f"{field_name}_{record_index}_{pair}_{timestamp}"


    # METAAPI SDK Functions
    def get_metaapi_config():
        """Get MetaApi configuration from secrets.toml"""
        try:
            metaapi_config = st.secrets.get("metaapi", {})
            return metaapi_config
        except:
            return {}


    async def get_metaapi_account():
        """Get MetaApi account instance"""
        try:
            from metaapi_cloud_sdk import MetaApi
            config = get_metaapi_config()
            token = config.get("token", "")
            account_id = st.session_state.metaapi_account_id

            if not token or not account_id:
                return None, "Token or account ID not configured"

            api = MetaApi(token)
            account = await api.metatrader_account_api.get_account(account_id)
            return account, None
        except Exception as e:
            return None, f"Error getting account: {str(e)}"


    async def test_metaapi_connection():
        """Test connection to MetaAPI - SIMPLIFIED"""
        try:
            from metaapi_cloud_sdk import MetaApi
            config = get_metaapi_config()
            token = config.get("token", "")

            if not token:
                return False, "MetaApi token not configured"

            api = MetaApi(token)
            return True, "Connected to MetaApi successfully (token is valid)"
        except Exception as e:
            return False, f" MetaApi connection error: {str(e)}"


    async def place_trade(symbol: str, volume: float, order_type: str, entry_price: float, sl: float, tp: float):
        """Place a LIMIT trade with MetaApi - SL and TP are MANDATORY"""
        try:
            account, error = await get_metaapi_account()
            if error:
                return False, error

            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            if sl is None or tp is None:
                await connection.close()
                return False, " Stop loss and take profit are mandatory"

            formatted_symbol = format_symbol_for_pepperstone(symbol)

            if order_type.upper() == "BUY":
                result = await connection.create_limit_buy_order(
                    formatted_symbol,
                    volume,
                    entry_price,
                    stop_loss=sl,
                    take_profit=tp
                )
            else:
                result = await connection.create_limit_sell_order(
                    formatted_symbol,
                    volume,
                    entry_price,
                    stop_loss=sl,
                    take_profit=tp
                )

            await connection.close()
            return True, f" Order placed successfully"

        except Exception as e:
            try:
                await connection.close()
            except:
                pass
            return False, f" Trade error: {str(e)}"


    async def cancel_pending_order(symbol: str, entry_price: float):
        """Cancel a pending limit order in MT5"""
        try:
            from metaapi_cloud_sdk import MetaApi

            config = get_metaapi_config()
            token = config.get("token", "")
            account_id = st.session_state.metaapi_account_id

            if not token or not account_id:
                return False, "Token or account ID not configured"

            api = MetaApi(token)
            account = await api.metatrader_account_api.get_account(account_id)

            if account.state != 'DEPLOYED':
                await account.deploy()
            await account.wait_connected()

            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            formatted_symbol = format_symbol_for_pepperstone(symbol)

            # Get all pending orders
            orders = await connection.get_orders()

            # Find the specific limit order to cancel
            order_to_cancel = None
            for order in orders:
                if (order['symbol'] == formatted_symbol and
                        abs(float(order['openPrice']) - float(
                            entry_price)) < 0.0001):  # Small tolerance for price matching
                    order_to_cancel = order
                    break

            if order_to_cancel:
                # Cancel the order
                result = await connection.cancel_order(order_to_cancel['id'])
                await connection.close()
                return True, f"Order cancelled successfully: {symbol} at {entry_price}"
            else:
                await connection.close()
                return False, f"No pending order found for {symbol} at {entry_price}"

        except Exception as e:
            try:
                await connection.close()
            except:
                pass
            return False, f"Error cancelling order: {str(e)}"


    async def quick_get_positions():
        """Quickly get positions without extensive synchronization"""
        try:
            from metaapi_cloud_sdk import MetaApi

            config = get_metaapi_config()
            token = config.get("token", "")
            account_id = st.session_state.metaapi_account_id

            if not token or not account_id:
                return None, "Token or account ID not configured"

            api = MetaApi(token)
            account = await api.metatrader_account_api.get_account(account_id)

            # Quick check - don't deploy if not ready
            if account.state != 'DEPLOYED':
                return [], None

            connection = account.get_rpc_connection()
            await connection.connect()

            # Quick sync with short timeout
            try:
                await asyncio.wait_for(connection.wait_synchronized(), timeout=5)
            except asyncio.TimeoutError:
                await connection.close()
                return [], "Connection timeout"

            positions = await connection.get_positions()
            await connection.close()

            # Format positions data quickly - USE CAMELCASE DIRECTLY FROM METAAPI
            formatted_positions = []
            for position in positions:
                formatted_positions.append({
                    'id': position['id'],
                    'symbol': position['symbol'],
                    'type': position['type'],
                    'volume': position['volume'],
                    'openPrice': position['openPrice'],
                    'currentPrice': position['currentPrice'],
                    'stopLoss': position.get('stopLoss'),
                    'takeProfit': position.get('takeProfit'),
                    'profit': position.get('profit', 0)
                })

            return formatted_positions, None

        except Exception as e:
            try:
                await connection.close()
            except:
                pass
            return [], f"Quick position error: {str(e)}"


    def quick_auto_move_filled_orders(positions):
        """Quick auto-move - only match instrument and position size, update all position values"""
        try:
            if not st.session_state.order_placed or not positions:
                return 0

            moved_count = 0
            moved_details = []

            for order in st.session_state.order_placed[:]:
                order_symbol = order['selected_pair']
                order_volume = safe_float(order.get('position_size'), 0.1)

                # Format the order symbol for Pepperstone comparison
                formatted_order_symbol = format_symbol_for_pepperstone(order_symbol)

                # Look for matching position
                for position in positions:
                    position_symbol = position['symbol']
                    position_volume = safe_float(position.get('volume'), 0.0)

                    # SIMPLIFIED SYMBOL MATCHING - ONLY INSTRUMENT
                    symbol_match = False

                    # Case 1: Direct match (AUDUSD.a vs AUDUSD.a)
                    if position_symbol == formatted_order_symbol:
                        symbol_match = True

                    # Case 2: Position has .a but order doesn't (AUDUSD.a vs AUDUSD)
                    elif position_symbol.endswith('.a') and position_symbol.replace('.a', '') == order_symbol:
                        symbol_match = True

                    # Case 3: Order has .a but position doesn't (AUDUSD vs AUDUSD.a) - less common but possible
                    elif order_symbol.endswith('.a') and order_symbol.replace('.a', '') == position_symbol:
                        symbol_match = True

                    # Case 4: Both without .a but match (AUDUSD vs AUDUSD)
                    elif position_symbol == order_symbol:
                        symbol_match = True

                    # Volume matching ONLY (5.01 lots vs 5.01 lots)
                    volume_match = abs(position_volume - order_volume) < 0.01

                    # ONLY CHECK THESE TWO FIELDS - IGNORE DIRECTION AND PRICE
                    if symbol_match and volume_match:
                        # FIX: ALWAYS use the actual position type from MetaApi, NEVER use calculated direction
                        position_type = position.get('type', '')
                        if position_type == 'POSITION_TYPE_SELL':
                            actual_direction = "sell"
                        elif position_type == 'POSITION_TYPE_BUY':
                            actual_direction = "buy"
                        else:
                            actual_direction = "Unknown"

                        print(
                            f"DEBUG: Position {position_symbol} has type: {position_type}, setting direction to: {actual_direction}")

                        # Create trade record - USE ACTUAL POSITION DATA, NOT ORDER DATA
                        trade_record = {
                            'timestamp': order['timestamp'],
                            'selected_pair': order['selected_pair'],
                            'risk_multiplier': order['risk_multiplier'],
                            'position_size': order['position_size'],
                            'stop_pips': order['stop_pips'],
                            # USE ACTUAL POSITION PRICES, NOT ORDER PRICES
                            'entry_price': safe_float(position.get('openPrice'), 0.0),  # Actual filled price
                            'exit_price': safe_float(position.get('stopLoss'), 0.0),  # Actual stop loss
                            'target_price': safe_float(position.get('takeProfit'), 0.0),  # Actual take profit
                            'fill_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'order_status': 'FILLED',
                            'position_id': position['id'],
                            'current_price': safe_float(position.get('currentPrice'), 0.0),
                            'current_sl': safe_float(position.get('stopLoss'), 0.0),
                            'current_tp': safe_float(position.get('takeProfit'), 0.0),
                            'profit': safe_float(position.get('profit', 0), 0.0),
                            'direction': actual_direction,  # USE ACTUAL DIRECTION FROM POSITION TYPE
                            'status': 'Order Filled'
                        }

                        # Move from Order Placed to In Trade
                        st.session_state.order_placed = [o for o in st.session_state.order_placed if
                                                         o['timestamp'] != order['timestamp']]
                        st.session_state.in_trade.append(trade_record)
                        moved_count += 1
                        moved_details.append(f"{order_symbol} ({order_volume} lots)")
                        print(
                            f"DEBUG: Updated prices for {order_symbol}: Entry={trade_record['entry_price']}, SL={trade_record['exit_price']}, TP={trade_record['target_price']}")
                        break

            return moved_count

        except Exception as e:
            print(f" Quick auto-move error: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0


    def update_existing_in_trade_directions(positions):
        """Update direction AND PRICES for existing In Trade records based on actual position types"""
        try:
            updated_count = 0
            print(
                f"DEBUG: Checking {len(positions)} positions against {len(st.session_state.in_trade)} in-trade records")

            for trade in st.session_state.in_trade:
                trade_symbol = trade['selected_pair']
                formatted_trade_symbol = format_symbol_for_pepperstone(trade_symbol)

                print(f"DEBUG: Looking for match for trade: {trade_symbol} (formatted: {formatted_trade_symbol})")

                # Find matching position
                match_found = False
                for position in positions:
                    position_symbol = position['symbol']
                    position_type = position.get('type', 'UNKNOWN_TYPE')

                    # Check symbol match with multiple strategies
                    symbol_match = False

                    # Strategy 1: Direct match
                    if position_symbol == formatted_trade_symbol:
                        symbol_match = True
                    # Strategy 2: Position has .a but trade doesn't
                    elif position_symbol.endswith('.a') and position_symbol.replace('.a', '') == trade_symbol:
                        symbol_match = True
                    # Strategy 3: Trade has .a but position doesn't
                    elif trade_symbol.endswith('.a') and trade_symbol.replace('.a', '') == position_symbol:
                        symbol_match = True
                    # Strategy 4: Both without .a but match
                    elif position_symbol == trade_symbol:
                        symbol_match = True
                    # Strategy 5: Case-insensitive match
                    elif position_symbol.upper() == trade_symbol.upper():
                        symbol_match = True
                    elif position_symbol.upper() == formatted_trade_symbol.upper():
                        symbol_match = True

                    if symbol_match:
                        match_found = True
                        print(f"DEBUG: MATCH FOUND! {trade_symbol} <-> {position_symbol}")

                        # Determine direction from position type
                        if position_type == 'POSITION_TYPE_SELL':
                            new_direction = "sell"
                        elif position_type == 'POSITION_TYPE_BUY':
                            new_direction = "buy"
                        else:
                            new_direction = "Unknown"

                        # Get actual prices from position
                        actual_entry_price = safe_float(position.get('openPrice'), 0.0)
                        actual_sl_price = safe_float(position.get('stopLoss'), 0.0)
                        actual_tp_price = safe_float(position.get('takeProfit'), 0.0)

                        # Check if any updates are needed
                        needs_update = False

                        # Check direction
                        old_direction = trade.get('direction', 'None')
                        if old_direction != new_direction:
                            trade['direction'] = new_direction
                            needs_update = True
                            print(
                                f"DEBUG: Updated {trade_symbol} direction from '{old_direction}' to '{new_direction}'")

                        # Check entry price
                        old_entry = safe_float(trade.get('entry_price'), 0.0)
                        if abs(old_entry - actual_entry_price) > 0.00001:  # Small tolerance for float comparison
                            trade['entry_price'] = actual_entry_price
                            needs_update = True
                            print(f"DEBUG: Updated {trade_symbol} entry price from {old_entry} to {actual_entry_price}")

                        # Check stop loss (exit_price)
                        old_sl = safe_float(trade.get('exit_price'), 0.0)
                        if abs(old_sl - actual_sl_price) > 0.00001:
                            trade['exit_price'] = actual_sl_price
                            needs_update = True
                            print(f"DEBUG: Updated {trade_symbol} stop loss from {old_sl} to {actual_sl_price}")

                        # Check take profit (target_price)
                        old_tp = safe_float(trade.get('target_price'), 0.0)
                        if abs(old_tp - actual_tp_price) > 0.00001:
                            trade['target_price'] = actual_tp_price
                            needs_update = True
                            print(f"DEBUG: Updated {trade_symbol} take profit from {old_tp} to {actual_tp_price}")

                        # Also update current prices
                        trade['current_price'] = safe_float(position.get('currentPrice'), 0.0)
                        trade['current_sl'] = actual_sl_price
                        trade['current_tp'] = actual_tp_price
                        trade['profit'] = safe_float(position.get('profit', 0), 0.0)

                        if needs_update:
                            updated_count += 1
                        break

                if not match_found:
                    print(f"DEBUG: NO MATCH found for trade: {trade_symbol}")

            print(f"DEBUG: Updated {updated_count} trade records (direction and prices)")
            return updated_count

        except Exception as e:
            print(f"Update existing in-trade directions error: {e}")
            import traceback
            traceback.print_exc()
            return 0
    async def fast_order_check(symbol: str, entry_price: float, volume: float):
        """Fast order check with minimal overhead"""
        try:
            from metaapi_cloud_sdk import MetaApi

            config = get_metaapi_config()
            token = config.get("token", "")
            account_id = st.session_state.metaapi_account_id

            if not token:
                return None, "No token configured"

            api = MetaApi(token)
            account = await api.metatrader_account_api.get_account(account_id)

            if account.state != 'DEPLOYED':
                return 'NOT_CONNECTED', None

            connection = account.get_rpc_connection()
            await connection.connect()

            try:
                await asyncio.wait_for(connection.wait_synchronized(), timeout=5)
            except asyncio.TimeoutError:
                await connection.close()
                return 'TIMEOUT', None

            formatted_symbol = format_symbol_for_pepperstone(symbol)

            # Quick orders check
            orders = await connection.get_orders()
            for order in orders:
                if (order['symbol'] == formatted_symbol and
                        abs(float(order['openPrice']) - float(entry_price)) < 0.02):
                    await connection.close()
                    return 'PENDING', order['id']

            # Quick positions check
            positions = await connection.get_positions()
            for position in positions:
                if (position['symbol'] == formatted_symbol and
                        abs(float(position['openPrice']) - float(entry_price)) < 0.02 and
                        abs(float(position['volume']) - float(volume)) < 0.01):
                    await connection.close()
                    return 'FILLED', position['id']

            await connection.close()
            return 'NOT_FOUND', None

        except Exception as e:
            try:
                await connection.close()
            except:
                pass
            return None, f"Quick check error: {str(e)}"


    async def modify_position_sl(position_id: str, new_sl: float):
        """Modify stop loss using correct MetaApi syntax with keyword arguments"""
        try:
            from metaapi_cloud_sdk import MetaApi

            config = get_metaapi_config()
            token = config.get("token", "")
            account_id = st.session_state.metaapi_account_id

            if not token or not account_id:
                return False, "Token or account ID not configured"

            api = MetaApi(token)
            account = await api.metatrader_account_api.get_account(account_id)

            if account.state != 'DEPLOYED':
                await account.deploy()
            await account.wait_connected()

            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            # Get current position details
            positions = await connection.get_positions()
            current_position = None
            for pos in positions:
                if pos['id'] == position_id:
                    current_position = pos
                    break

            if not current_position:
                await connection.close()
                return False, " Position not found"

            symbol = current_position['symbol']
            current_sl = current_position.get('stopLoss')
            current_tp = current_position.get('takeProfit')
            current_price = current_position.get('currentPrice')
            position_type = current_position.get('type')

            # Validate the new SL value
            if new_sl <= 0:
                await connection.close()
                return False, " Stop loss must be greater than 0"

            # For BUY positions: SL should be below current price
            if position_type == 'POSITION_TYPE_BUY' and new_sl >= current_price:
                await connection.close()
                return False, f" For BUY positions, stop loss ({new_sl:.5f}) must be below current price ({current_price:.5f})"

            # For SELL positions: SL should be above current price
            if position_type == 'POSITION_TYPE_SELL' and new_sl <= current_price:
                await connection.close()
                return False, f" For SELL positions, stop loss ({new_sl:.5f}) must be above current price ({current_price:.5f})"

            # CORRECT USAGE: Keyword arguments with snake_case
            if current_tp and current_tp > 0:
                # Modify both SL and TP
                await connection.modify_position(
                    position_id,
                    stop_loss=new_sl,
                    take_profit=current_tp
                )
            else:
                # Modify only SL
                await connection.modify_position(
                    position_id,
                    stop_loss=new_sl
                )

            await connection.close()
            return True, f" {symbol} - Stop loss updated to {new_sl:.5f}"

        except Exception as e:
            try:
                await connection.close()
            except:
                pass
            error_msg = str(e)
            print(f" MetaApi error: {error_msg}")
            return False, f" Failed to modify position: {error_msg}"


    async def close_position(position_id: str):
        """Close an open position"""
        try:
            from metaapi_cloud_sdk import MetaApi

            config = get_metaapi_config()
            token = config.get("token", "")
            account_id = st.session_state.metaapi_account_id

            if not token or not account_id:
                return False, "Token or account ID not configured"

            api = MetaApi(token)
            account = await api.metatrader_account_api.get_account(account_id)

            if account.state != 'DEPLOYED':
                await account.deploy()
            await account.wait_connected()

            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            result = await connection.close_position(position_id)
            await connection.close()

            if result:
                return True, f" Position {position_id} closed successfully"
            else:
                return False, " Failed to close position"

        except Exception as e:
            try:
                await connection.close()
            except:
                pass
            return False, f" Error closing position: {str(e)}"


    # FIXED SYNC FUNCTIONS - SIMPLIFIED AND MORE RELIABLE
    def sync_from_active_opps():
        """Sync from Active Opps to Trade Signals - SIMPLIFIED"""
        try:
            # Load current workflow data directly
            workflow_df = load_data_from_sheets(sheet_name="Trade", worksheet_name="Workflow")

            if workflow_df is not None and not workflow_df.empty:
                print(f" Loaded {len(workflow_df)} records from Workflow sheet")

                # Convert timestamp to string for consistent comparison
                if 'timestamp' in workflow_df.columns:
                    workflow_df['timestamp'] = workflow_df['timestamp'].astype(str)

                # Clear and rebuild all lists
                st.session_state.ready_to_order = []
                st.session_state.order_placed = []
                st.session_state.in_trade = []

                # Convert to list of dictionaries for processing
                all_records = workflow_df.to_dict('records')

                for record in all_records:
                    status = record.get('status')
                    timestamp = str(record.get('timestamp', ''))

                    if status == 'Order Ready':
                        st.session_state.ready_to_order.append({
                            'timestamp': timestamp,
                            'selected_pair': record.get('selected_pair'),
                            'risk_multiplier': record.get('risk_multiplier'),
                            'position_size': record.get('position_size'),
                            'stop_pips': record.get('stop_pips'),
                            'entry_price': record.get('entry_price'),
                            'exit_price': record.get('exit_price'),
                            'target_price': record.get('target_price'),
                            'status': 'Order Ready'
                        })
                    elif status == 'Order Placed':
                        st.session_state.order_placed.append({
                            'timestamp': timestamp,
                            'selected_pair': record.get('selected_pair'),
                            'risk_multiplier': record.get('risk_multiplier'),
                            'position_size': record.get('position_size'),
                            'stop_pips': record.get('stop_pips'),
                            'entry_price': record.get('entry_price'),
                            'exit_price': record.get('exit_price'),
                            'target_price': record.get('target_price'),
                            'order_status': 'PENDING',
                            'status': 'Order Placed'
                        })
                    elif status == 'Order Filled':
                        st.session_state.in_trade.append({
                            'timestamp': timestamp,
                            'selected_pair': record.get('selected_pair'),
                            'risk_multiplier': record.get('risk_multiplier'),
                            'position_size': record.get('position_size'),
                            'stop_pips': record.get('stop_pips'),
                            'entry_price': record.get('entry_price'),
                            'exit_price': record.get('exit_price'),
                            'target_price': record.get('target_price'),
                            'order_status': 'FILLED',
                            'status': 'Order Filled'
                        })

                print(
                    f" Synced: {len(st.session_state.ready_to_order)} ready, {len(st.session_state.order_placed)} placed, {len(st.session_state.in_trade)} filled")
                return True, f"Synced {len(all_records)} records from Active Opps"

            return True, "No Active Opps data found"

        except Exception as e:
            print(f" Sync from Active Opps error: {str(e)}")
            return False, f"Sync error: {str(e)}"


    def update_workflow_status_in_sheets(timestamp, new_status, instrument_name=None):
        """Update a specific record's status in Google Sheets workflow using instrument name + timestamp for precise matching"""
        try:
            print(f" Updating status for {instrument_name} (timestamp: {timestamp}) to {new_status}")

            # Load current workflow data
            workflow_df = load_data_from_sheets(sheet_name="Trade", worksheet_name="Workflow")

            if workflow_df is None or workflow_df.empty:
                print(" No workflow data found to update")
                return False, "No workflow data found"

            # Convert timestamp to string for comparison
            workflow_df['timestamp'] = workflow_df['timestamp'].astype(str)
            timestamp_str = str(timestamp)

            # Find and update the record - PRECISE MATCHING
            record_found = False
            for idx, record in workflow_df.iterrows():
                record_timestamp = str(record['timestamp'])
                record_instrument = record.get('selected_pair', '')

                # Match by both timestamp AND instrument name for precise identification
                if record_timestamp == timestamp_str and record_instrument == instrument_name:
                    workflow_df.at[idx, 'status'] = new_status
                    record_found = True
                    print(f" Found exact match: {record_instrument} - updating status to {new_status}")
                    break

            if not record_found:
                print(f" Record with timestamp {timestamp} and instrument {instrument_name} not found in workflow")
                # Fallback: try to find by instrument only if timestamp fails
                for idx, record in workflow_df.iterrows():
                    record_instrument = record.get('selected_pair', '')
                    if record_instrument == instrument_name and record.get('status') in ['Order Ready', 'Order Placed']:
                        workflow_df.at[idx, 'status'] = new_status
                        record_found = True
                        print(f" Found by instrument fallback: {record_instrument} - updating status to {new_status}")
                        break

            if not record_found:
                return False, f"Record not found: {instrument_name} at {timestamp}"

            # Save updated data back to sheets
            print(" Saving updated workflow to Google Sheets...")
            success = save_data_to_sheets(workflow_df, sheet_name="Trade", worksheet_name="Workflow")

            if success:
                print(" Successfully updated workflow in Google Sheets")
                return True, f"Updated {instrument_name} status to {new_status}"
            else:
                print(" Failed to save to Google Sheets")
                return False, "Failed to save to cloud"

        except Exception as e:
            print(f" Update workflow status error: {str(e)}")
            return False, f"Update error: {str(e)}"


    def handle_move_to_order_placed(signal_index):
        """Move signal from Ready to Order to Order Placed - WITH DIRECT SHEETS UPDATE"""
        try:
            if signal_index >= len(st.session_state.ready_to_order):
                st.error("Invalid signal index")
                return False

            signal = st.session_state.ready_to_order[signal_index]
            timestamp = signal['timestamp']
            symbol = signal['selected_pair']

            print(f" Moving {symbol} to Order Placed (timestamp: {timestamp})")

            # First update the status in Google Sheets WITH INSTRUMENT NAME
            success, message = update_workflow_status_in_sheets(timestamp, 'Order Placed', symbol)

            if success:
                # Only update local state after successful sheets update
                st.session_state.ready_to_order.pop(signal_index)

                # Add to order_placed
                st.session_state.order_placed.append({
                    'timestamp': timestamp,
                    'selected_pair': symbol,
                    'risk_multiplier': signal['risk_multiplier'],
                    'position_size': signal['position_size'],
                    'stop_pips': signal['stop_pips'],
                    'entry_price': signal['entry_price'],
                    'exit_price': signal['exit_price'],
                    'target_price': signal['target_price'],
                    'order_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'order_status': 'PENDING',
                    'direction': calculate_direction(signal.get('entry_price'), signal.get('exit_price')),
                    'status': 'Order Placed'
                })

                st.session_state.last_action = f"moved_to_placed_{signal_index}"
                print(f" Successfully moved {symbol} to Order Placed")
                return True
            else:
                st.error(f" Failed to update Google Sheets: {message}")
                return False

        except Exception as e:
            st.error(f" Move error: {e}")
            import traceback
            traceback.print_exc()
            return False


    def handle_move_to_in_trade(order_index):
        """Move order from Order Placed to In Trade - WITH DIRECT SHEETS UPDATE"""
        try:
            if order_index >= len(st.session_state.order_placed):
                st.error("Invalid order index")
                return False

            order = st.session_state.order_placed[order_index]
            timestamp = order['timestamp']
            symbol = order['selected_pair']

            print(f" Moving {symbol} to In Trade (timestamp: {timestamp})")

            # First update the status in Google Sheets WITH INSTRUMENT NAME
            success, message = update_workflow_status_in_sheets(timestamp, 'Order Filled', symbol)

            if success:
                # Only update local state after successful sheets update
                st.session_state.order_placed.pop(order_index)

                # Add to in_trade
                st.session_state.in_trade.append({
                    'timestamp': timestamp,
                    'selected_pair': symbol,
                    'risk_multiplier': order['risk_multiplier'],
                    'position_size': order['position_size'],
                    'stop_pips': order['stop_pips'],
                    'entry_price': order['entry_price'],
                    'exit_price': order['exit_price'],
                    'target_price': order['target_price'],
                    'fill_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'order_status': 'FILLED',
                    'direction': order.get('direction', 'Unknown'),
                    'status': 'Order Filled'
                })

                st.session_state.last_action = f"moved_to_trade_{order_index}"
                print(f" Successfully moved {symbol} to In Trade")
                return True
            else:
                st.error(f" Failed to update Google Sheets: {message}")
                return False

        except Exception as e:
            st.error(f" Move error: {e}")
            return False


    def handle_move_back_to_ready(order_index):
        """Move order from Order Placed back to Ready to Order - WITH ORDER CANCELLATION"""
        try:
            if order_index >= len(st.session_state.order_placed):
                st.error("Invalid order index")
                return False

            order = st.session_state.order_placed[order_index]
            timestamp = order['timestamp']
            symbol = order['selected_pair']
            entry_price = safe_float(order.get('entry_price'), 0.0)

            print(f" Moving {symbol} back to Ready and cancelling order (timestamp: {timestamp})")

            # First cancel the order in MT5
            import asyncio
            cancellation_success, cancellation_message = asyncio.run(cancel_pending_order(symbol, entry_price))

            if not cancellation_success:
                st.warning(f"Order cancellation warning: {cancellation_message}. Continuing with status update...")

            # Then update the status in Google Sheets WITH INSTRUMENT NAME
            success, message = update_workflow_status_in_sheets(timestamp, 'Order Ready', symbol)

            if success:
                # Only update local state after successful sheets update
                st.session_state.order_placed.pop(order_index)

                # Create a new order object with updated status (CRITICAL FIX)
                updated_order = order.copy()
                updated_order['status'] = 'Order Ready'  # Update the status field
                # Remove any order-specific fields that shouldn't be in ready status
                if 'order_status' in updated_order:
                    del updated_order['order_status']
                if 'order_time' in updated_order:
                    del updated_order['order_time']

                st.session_state.ready_to_order.append(updated_order)

                st.session_state.last_action = f"moved_back_to_ready_{order_index}"

                if cancellation_success:
                    st.success(f"Order cancelled in MT5 and moved back to Ready: {symbol}")
                else:
                    st.success(f"Moved back to Ready (order may still be pending in MT5): {symbol}")

                print(f" Successfully moved {symbol} back to Ready")
                return True
            else:
                st.error(f" Failed to update Google Sheets: {message}")
                return False

        except Exception as e:
            st.error(f"Move error: {e}")
            return False


    def handle_delete_signal(signal_index, list_name):
        """Delete signal from any list"""
        try:
            signal = None
            if list_name == 'ready_to_order':
                signal = st.session_state.ready_to_order[signal_index]
                st.session_state.ready_to_order.pop(signal_index)
            elif list_name == 'order_placed':
                signal = st.session_state.order_placed[signal_index]
                st.session_state.order_placed.pop(signal_index)
            elif list_name == 'in_trade':
                signal = st.session_state.in_trade[signal_index]
                st.session_state.in_trade.pop(signal_index)

            # Update status in Google Sheets to 'Speculation' when deleted WITH INSTRUMENT NAME
            if signal:
                update_workflow_status_in_sheets(signal['timestamp'], 'Speculation', signal['selected_pair'])

            st.session_state.last_action = f"deleted_{list_name}_{signal_index}"
            return True
        except Exception as e:
            st.error(f"Delete error: {e}")
            return False


    # OPTIMIZED TRADE EXECUTION FUNCTION - UPDATED FOR DIRECT SYNC
    async def execute_trade_and_update(signal_index):
        """Execute trade and update state immediately - WITH DIRECT SHEETS UPDATE"""
        try:
            signal = st.session_state.ready_to_order[signal_index]

            # Execute trade
            trade_success, trade_message = await place_trade(
                symbol=signal['selected_pair'],
                volume=float(signal.get('position_size', 0.1)),
                order_type=calculate_direction(signal.get('entry_price'), signal.get('exit_price')),
                entry_price=safe_float(signal.get('entry_price'), 0.0),
                sl=safe_float(signal.get('exit_price'), 0.0),
                tp=safe_float(signal.get('target_price'), 0.0)
            )

            if trade_success:
                # Move to Order Placed with direct sheets update
                if handle_move_to_order_placed(signal_index):
                    st.session_state.last_trade_result = {
                        'success': True,
                        'message': trade_message + " and synced to Active Opps",
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    }
                else:
                    st.session_state.last_trade_result = {
                        'success': False,
                        'message': "Trade executed but failed to sync with Active Opps",
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    }
            else:
                st.session_state.last_trade_result = {
                    'success': False,
                    'message': trade_message,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }

            return trade_success, trade_message

        except Exception as e:
            st.session_state.last_trade_result = {
                'success': False,
                'message': f"Trade execution error: {str(e)}",
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            return False, f"Trade execution error: {str(e)}"


    def execute_trade_immediate_ui(signal_index):
        """Handle trade execution with immediate UI update"""
        import asyncio
        asyncio.run(execute_trade_and_update(signal_index))
        st.rerun()


    # Handle action results
    if st.session_state.last_action:
        action = st.session_state.last_action
        st.session_state.last_action = None

        if "moved_to_placed" in action:
            st.success(" Moved to Order Placed and updated in Active Opps!")
        elif "moved_to_trade" in action:
            st.success(" Moved to In Trade and updated in Active Opps!")
        elif "moved_back_to_ready" in action:
            st.success(" Moved back to Ready and updated in Active Opps!")
        elif "deleted_" in action:
            st.success(" Record deleted and updated in Active Opps!")

        st.rerun()

    # Quick sync on page load
    if not st.session_state.ready_to_order and not st.session_state.order_placed and not st.session_state.in_trade:
        with st.spinner(" Quick syncing from Active Opps..."):
            success, message = sync_from_active_opps()
            if not success:
                st.error(f"{message}")



    # Auto-connect to MetaApi account
    if not st.session_state.metaapi_connected:
        import asyncio

        try:
            success, message = asyncio.run(test_metaapi_connection())
            if success:
                st.session_state.metaapi_connected = True
        except:
            st.session_state.metaapi_connected = False

    # QUICK AUTO-MOVE ON PAGE LOAD (NON-BLOCKING)
    if st.session_state.metaapi_connected and not st.session_state.auto_move_checked:
        import threading


        def background_auto_move():
            try:
                positions, error = asyncio.run(quick_get_positions())
                if positions:
                    moved_count = quick_auto_move_filled_orders(positions)
                    if moved_count > 0:
                        print(f" Auto-moved {moved_count} orders to In Trade")
                        # Update Google Sheets for each moved order WITH INSTRUMENT NAME
                        for order in st.session_state.in_trade[-moved_count:]:
                            update_workflow_status_in_sheets(order['timestamp'], 'Order Filled', order['selected_pair'])
                    st.session_state.open_positions = positions
                st.session_state.auto_move_checked = True
            except Exception as e:
                print(f" Background auto-move error: {str(e)}")
                st.session_state.auto_move_checked = True


        # Run in background thread to avoid blocking
        thread = threading.Thread(target=background_auto_move, daemon=True)
        thread.start()

    # Show last trade result if available
    if st.session_state.last_trade_result:
        result = st.session_state.last_trade_result
        if result['success']:
            st.success(f" {result['message']} (at {result['timestamp']})")
        else:
            st.error(f" {result['message']} (at {result['timestamp']})")
        if st.button("Clear Message", key="clear_trade_msg"):
            st.session_state.last_trade_result = None
            st.rerun()

    # UPDATED Connection Management
    st.subheader("Connection Management")
    col_conn1, col_conn2, col_conn3, col_conn4 = st.columns(4)

    with col_conn1:
        if st.button(" Sync from Opps", key="sync_from_opps", use_container_width=True):
            with st.spinner("Syncing from Active Opps..."):
                success, message = sync_from_active_opps()
                if success:
                    st.success(f" {message}")
                else:
                    st.error(f" {message}")
                st.rerun()

    with col_conn2:
        if st.button(" View Active Opportunities", key="view_opps", use_container_width=True):
            st.session_state.current_page = "Active Opps"
            st.rerun()

    with col_conn3:
        if st.button(" Refresh Positions", key="refresh_positions", use_container_width=True):
            import asyncio


            async def refresh_positions_async():
                positions, error = await quick_get_positions()
                if error:
                    st.error(f" {error}")
                else:
                    # AUTO-MOVE: Check for matching orders and move them automatically
                    moved_count = quick_auto_move_filled_orders(positions)

                    # UPDATE: Also update directions for existing In Trade records
                    updated_count = update_existing_in_trade_directions(positions)

                    st.session_state.open_positions = positions
                    if moved_count > 0:
                        st.success(
                            f" Found {len(positions)} open positions, auto-moved {moved_count} orders to In Trade")
                        # Update Google Sheets for each moved order WITH INSTRUMENT NAME
                        for order in st.session_state.in_trade[-moved_count:]:
                            update_workflow_status_in_sheets(order['timestamp'], 'Order Filled', order['selected_pair'])
                    elif positions:
                        st.info(f" Found {len(positions)} open positions")
                    else:
                        st.info(" No open positions found")


            # Run the async function
            asyncio.run(refresh_positions_async())
            st.rerun()

    with col_conn4:
        if st.button(" Refresh Data", key="refresh_data", use_container_width=True):
            st.rerun()

    # Show connection status
    if not st.session_state.metaapi_connected:
        st.warning("Not Connected to Trading Account - Trades will not execute")



    #st.info(" Status changes are automatically saved to Active Opps workflow")

    st.markdown("---")

    # UPDATED Trade Signals Display Section
    st.subheader("Active Trade Signals")

    if not st.session_state.ready_to_order and not st.session_state.order_placed and not st.session_state.in_trade and not st.session_state.open_positions:
        st.info("""
        ## No Active Trade Signals

        **How to get started:**
        1. **Add signals** in the Active Opportunities page by moving orders to **'Order Ready'**
        2. **Signals will automatically appear** here when synced
        3. **Execute trades** directly via MetaApi
        4. **Status changes are automatically saved** to Active Opps workflow
        """)
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            f" Ready to Order ({len(st.session_state.ready_to_order)})",
            f" Order Placed ({len(st.session_state.order_placed)})",
            f" In Trade ({len(st.session_state.in_trade)})",
            f" Open Positions ({len(st.session_state.open_positions)})"
        ])

        with tab1:
            st.subheader(" Ready to Order")
            #st.info("Signals from Active Opps with 'Order Ready' status. Click 'Execute Order' to place trade.")

            if not st.session_state.ready_to_order:
                st.info("No signals ready for ordering.")
            else:
                for i, signal in enumerate(st.session_state.ready_to_order):
                    unique_key = generate_unique_key(i, signal, "ready")

                    with st.expander(f" {signal['selected_pair']} | {signal.get('timestamp', 'N/A')}", expanded=True):
                        col_instrument, col_execute = st.columns([2, 1])

                        with col_instrument:
                            st.write(f"**Instrument:** {signal['selected_pair']}")
                            st.write(f"**Strategy:** {signal.get('risk_multiplier', 'N/A')}")
                            st.write(f"**Position Size:** {signal.get('position_size', 'N/A')} lots")
                            st.write(f"**Stop Pips:** {signal.get('stop_pips', 'N/A')}")

                        with col_execute:
                            if st.session_state.metaapi_connected:
                                entry_price = safe_float(signal.get('entry_price'), 0.0)
                                stop_val = safe_float(signal.get('exit_price'), 0.0)
                                target_val = safe_float(signal.get('target_price'), 0.0)
                                validation_ok = entry_price > 0 and stop_val > 0 and target_val > 0

                                if validation_ok:
                                    direction = calculate_direction(signal.get('entry_price'), signal.get('exit_price'))
                                    button_text = f" Execute {direction} Order"

                                    # Generate unique confirmation key
                                    confirm_key = f"confirm_execute_{unique_key}"

                                    if st.session_state.get(confirm_key, False):
                                        # Show confirmation button
                                        col_confirm1, col_confirm2 = st.columns([2, 1])
                                        with col_confirm1:
                                            if st.button(
                                                    f"CONFIRM {direction} ORDER",
                                                    key=f"confirm_{unique_key}",
                                                    type="primary",
                                                    use_container_width=True,
                                            ):
                                                execute_trade_immediate_ui(i)
                                        with col_confirm2:
                                            if st.button(
                                                    "✕ Cancel",
                                                    key=f"cancel_{unique_key}",
                                                    type="secondary",
                                                    use_container_width=True,
                                            ):
                                                st.session_state[confirm_key] = False
                                                st.rerun()
                                    else:
                                        # Show initial execute button
                                        if st.button(
                                                button_text,
                                                key=f"execute_{unique_key}",
                                                type="secondary",
                                                use_container_width=True,
                                        ):
                                            # Set confirmation state
                                            st.session_state[confirm_key] = True
                                            st.rerun()
                                else:
                                    st.warning(" Cannot execute - missing required price parameters")
                            else:
                                st.warning(" Not connected to trading account")

                        col2, col3, col4 = st.columns([1, 1, 1])

                        with col2:
                            entry_price = safe_float(signal.get('entry_price'), 0.0)
                            st.write(f"**Entry Price:** {entry_price:.5f}")

                        with col3:
                            sl_price = safe_float(signal.get('exit_price'), 0.0)
                            st.write(f"**Stop Loss:** {sl_price:.5f}")

                        with col4:
                            tp_price = safe_float(signal.get('target_price'), 0.0)
                            st.write(f"**Take Profit:** {tp_price:.5f}")

                            if stop_val > 0 and target_val > 0:
                                stop_distance = abs(entry_price - stop_val)
                                target_distance = abs(entry_price - target_val)
                                if stop_distance > 0:
                                    reward_ratio = target_distance / stop_distance
                                    st.metric("R:R Ratio", f"1:{reward_ratio:.2f}")

                        direction = calculate_direction(signal.get('entry_price'), signal.get('exit_price'))
                        direction_color = "" if direction == "BUY" else "" if direction == "SELL" else ""
                        st.write(f"**Direction:** {direction_color} {direction}")

                        formatted_symbol = format_symbol_for_pepperstone(signal['selected_pair'])
                        if formatted_symbol != signal['selected_pair']:
                            st.info(f"**Trading Symbol:** {formatted_symbol} (Pepperstone format)")

        with tab2:
            st.subheader(" Order Placed")
            st.info("Orders that have been placed but not yet filled.")

            if not st.session_state.order_placed:
                st.info("No orders placed yet.")
            else:
                for i, order in enumerate(st.session_state.order_placed):
                    unique_key = generate_unique_key(i, order, "placed")

                    with st.expander(f" {order['selected_pair']} | {order.get('order_time', 'N/A')}", expanded=False):
                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.write(f"**Instrument:** {order['selected_pair']}")
                            st.write(f"**Direction:** {order.get('direction', 'Unknown')}")
                            st.write(f"**Position Size:** {order.get('position_size', 'N/A')} lots")
                            st.write(f"**Status:** {order.get('order_status', 'PENDING')}")

                        with col2:
                            entry_price = safe_float(order.get('entry_price'), 0.0)
                            st.write(f"**Entry Price:** {entry_price:.5f}")
                            sl_price = safe_float(order.get('exit_price'), 0.0)
                            st.write(f"**Stop Loss:** {sl_price:.5f}")

                        with col3:
                            tp_price = safe_float(order.get('target_price'), 0.0)
                            st.write(f"**Take Profit:** {tp_price:.5f}")
                            st.write(f"**Order Time:** {order.get('order_time', 'N/A')}")

                        col_check, col_back = st.columns(2)  # REMOVED col_move - now only 2 columns

                        with col_check:
                            if st.button(" Quick Check", key=f"check_{unique_key}", use_container_width=True):
                                import asyncio

                                with st.spinner("Quick checking..."):
                                    status, order_id = asyncio.run(fast_order_check(
                                        order['selected_pair'],
                                        safe_float(order.get('entry_price'), 0.0),
                                        safe_float(order.get('position_size'), 0.1)
                                    ))
                                    if status == 'PENDING':
                                        st.success(f" Order pending in MT5 (ID: {order_id})")
                                    elif status == 'FILLED':
                                        st.success(" Order filled! Auto-moved to In Trade tab.")
                                        # Trigger quick auto-move
                                        positions, _ = asyncio.run(quick_get_positions())
                                        if positions:
                                            quick_auto_move_filled_orders(positions)
                                            # Update Google Sheets WITH INSTRUMENT NAME
                                            for trade in st.session_state.in_trade[-1:]:
                                                update_workflow_status_in_sheets(trade['timestamp'], 'Order Filled',
                                                                                 trade['selected_pair'])
                                        st.rerun()
                                    elif status == 'NOT_FOUND':
                                        st.warning(" Order not found in MT5")
                                    else:
                                        st.error(f" Check failed: {order_id}")

                        # REMOVED: col_move section with "Move to In Trade" button

                        with col_back:
                            # Generate unique confirmation key for back to ready
                            back_confirm_key = f"confirm_back_ready_{unique_key}"

                            if st.session_state.get(back_confirm_key, False):
                                # Show confirmation buttons
                                col_confirm_back1, col_confirm_back2 = st.columns(2)
                                with col_confirm_back1:
                                    if st.button(
                                            "CONFIRM Cancel Order",
                                            key=f"confirm_back_{unique_key}",
                                            type="primary",
                                            use_container_width=True,
                                    ):
                                        if handle_move_back_to_ready(i):
                                            st.session_state[back_confirm_key] = False
                                            st.rerun()
                                with col_confirm_back2:
                                    if st.button(
                                            "✕ Keep Order",
                                            key=f"keep_order_{unique_key}",
                                            type="secondary",
                                            use_container_width=True,
                                    ):
                                        st.session_state[back_confirm_key] = False
                                        st.rerun()
                            else:
                                if st.button("Back to Ready", key=f"back_ready_{unique_key}", use_container_width=True):
                                    st.session_state[back_confirm_key] = True
                                    st.rerun()

        with tab3:
            st.subheader(" In Trade")
            st.info("Active trades that have been filled.")

            if not st.session_state.in_trade:
                st.info("No active trades.")
            else:
                for i, trade in enumerate(st.session_state.in_trade):
                    unique_key = generate_unique_key(i, trade, "trade")

                    with st.expander(f" {trade['selected_pair']} | Filled: {trade.get('fill_time', 'N/A')}",
                                     expanded=True):
                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.write(f"**Instrument:** {trade['selected_pair']}")
                            st.write(f"**Direction:** {trade.get('direction', 'Unknown')}")
                            st.write(f"**Position Size:** {trade.get('position_size', 'N/A')} lots")
                            st.write(f"**Status:** {trade.get('order_status', 'FILLED')}")

                        with col2:
                            entry_price = safe_float(trade.get('entry_price'), 0.0)
                            st.write(f"**Entry Price:** {entry_price:.5f}")
                            sl_price = safe_float(trade.get('exit_price'), 0.0)
                            st.write(f"**Stop Loss:** {sl_price:.5f}")

                        with col3:
                            tp_price = safe_float(trade.get('target_price'), 0.0)
                            st.write(f"**Take Profit:** {tp_price:.5f}")
                            st.write(f"**Fill Time:** {trade.get('fill_time', 'N/A')}")

                        st.success(" This trade is now in Active Opps as 'Order Filled'")

        with tab4:
            st.subheader(" Open Positions")
            st.info("Live positions from MT5 account.")

            if not st.session_state.open_positions:
                st.info("No open positions found in MT5 account.")
            else:
                for i, position in enumerate(st.session_state.open_positions):
                    unique_key = generate_unique_key(i,
                                                     {'selected_pair': position['symbol'], 'timestamp': position['id']},
                                                     "position")

                    with st.expander(f" {position['symbol']} | {position['type']} | {position['volume']} lots",
                                     expanded=True):
                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.write(f"**Instrument:** {position['symbol']}")
                            st.write(f"**Type:** {position['type']}")
                            st.write(f"**Volume:** {position['volume']} lots")
                            st.write(f"**Position ID:** {position['id']}")

                            profit = safe_float(position.get('profit', 0), 0.0)
                            pnl_color = "green" if profit >= 0 else "red"
                            st.write(f"**P&L:** :{pnl_color}[${profit:.2f}]")

                        with col2:
                            open_price = safe_float(position.get('openPrice'), 0.0)
                            st.write(f"**Open Price:** {open_price:.5f}")

                            current_price = safe_float(position.get('currentPrice'), 0.0)
                            st.write(f"**Current Price:** {current_price:.5f}")

                        with col3:
                            current_sl_price = safe_float(position.get('stopLoss'), 0.0)  # Current SL (may change)
                            tp_price = safe_float(position.get('takeProfit'), 0.0)

                            # Try to find the ORIGINAL values from in_trade records
                            symbol = position['symbol']
                            direction = 'BUY' if position.get('type') == 'POSITION_TYPE_BUY' else 'SELL'

                            # Initialize storage if not exists
                            if 'be_prices' not in st.session_state:
                                st.session_state.be_prices = {}
                            if 'first_trail_prices' not in st.session_state:
                                st.session_state.first_trail_prices = {}
                            if 'original_sl_prices' not in st.session_state:
                                st.session_state.original_sl_prices = {}

                            # Find original values from when trade was filled
                            original_open_price = open_price  # Fallback to current
                            original_sl_price = current_sl_price  # Fallback to current
                            strategy = "Unknown"

                            for trade in st.session_state.in_trade:
                                if trade['selected_pair'] == symbol.replace('.a', ''):
                                    strategy = trade.get('risk_multiplier', 'Unknown')
                                    # Use ORIGINAL values from when trade was filled
                                    original_open_price = safe_float(trade.get('entry_price'), open_price)
                                    original_sl_price = safe_float(trade.get('exit_price'), current_sl_price)
                                    break

                            # Store original SL for reference
                            if symbol not in st.session_state.original_sl_prices:
                                st.session_state.original_sl_prices[symbol] = original_sl_price

                            # Calculate BE Price once based on ORIGINAL values
                            if symbol not in st.session_state.be_prices:
                                st.session_state.be_prices[symbol] = calculate_be_price(original_open_price,
                                                                                        original_sl_price, direction, strategy)
                            be_price = st.session_state.be_prices[symbol]

                            # Calculate First Trail Price once based on ORIGINAL values
                            if symbol not in st.session_state.first_trail_prices:
                                st.session_state.first_trail_prices[symbol] = calculate_first_trail_price(
                                    original_open_price, original_sl_price, direction, strategy
                                )
                            first_trail_price = st.session_state.first_trail_prices[symbol]

                            # Display current SL (which may change) but fixed BE and First Trail prices
                            st.write(f"**Stop Loss:** {current_sl_price:.5f}")
                            st.write(f"**Take Profit:** {tp_price:.5f}")
                            st.write(f"**BE Price:** {be_price:.5f}")
                            st.write(f"**First Trail Price:** {first_trail_price:.5f}")

                        # MODIFY SL ONLY SECTION
                        st.markdown("---")
                        st.subheader("Modify Stop Loss Only")

                        col_sl, col_action = st.columns([1, 1])

                        with col_sl:
                            new_sl = st.number_input(
                                "New Stop Loss",
                                value=sl_price if sl_price > 0 else open_price - 0.0010,
                                step=0.0001,
                                format="%.5f",
                                key=f"sl_{unique_key}"
                            )

                        with col_action:
                            # Generate unique confirmation key for update SL
                            update_sl_key = f"confirm_update_sl_{unique_key}"

                            if st.session_state.get(update_sl_key, False):
                                # Show confirmation buttons
                                col_confirm_sl1, col_confirm_sl2 = st.columns(2)
                                with col_confirm_sl1:
                                    if st.button(
                                            "CONFIRM Update SL",
                                            key=f"confirm_update_{unique_key}",
                                            type="primary",
                                            use_container_width=True,
                                    ):
                                        import asyncio

                                        with st.spinner("Updating stop loss..."):
                                            success, message = asyncio.run(modify_position_sl(
                                                position['id'],
                                                new_sl
                                            ))
                                            if success:
                                                st.success(message)
                                                # Refresh positions
                                                positions, error = asyncio.run(quick_get_positions())
                                                if positions is not None:
                                                    st.session_state.open_positions = positions
                                                st.session_state[update_sl_key] = False
                                                st.rerun()
                                            else:
                                                st.error(message)
                                with col_confirm_sl2:
                                    if st.button(
                                            "✕ Cancel",
                                            key=f"cancel_update_{unique_key}",
                                            type="secondary",
                                            use_container_width=True,
                                    ):
                                        st.session_state[update_sl_key] = False
                                        st.rerun()
                            else:
                                # Align the Update Stop Loss button with the input field
                                st.write("")  # Add some vertical spacing for alignment
                                st.write("")  # Add some vertical spacing for alignment
                                if st.button(" Update Stop Loss", key=f"update_{unique_key}", type="primary",
                                             use_container_width=True):
                                    st.session_state[update_sl_key] = True
                                    st.rerun()

                        # SET TO BREAK-EVEN BUTTON
                        col_be, col_be_action = st.columns([1, 1])

                        with col_be:
                            # Determine the break-even price based on instrument type and position type
                            position_type = position.get('type', '')
                            symbol = position['symbol']

                            # Check if it's a gold instrument
                            is_gold = any(gold_symbol in symbol.upper() for gold_symbol in ['XAU', 'GOLD'])

                            if position_type == 'POSITION_TYPE_BUY':
                                if is_gold:
                                    be_price = open_price + 2.0  # $2 for gold buy
                                    be_label = f"Break-even (Entry + $2): {be_price:.2f}"
                                else:
                                    be_price = open_price + 0.0005  # 5 pips for forex buy
                                    be_label = f"Break-even (Entry + 5 pips): {be_price:.5f}"
                            elif position_type == 'POSITION_TYPE_SELL':
                                if is_gold:
                                    be_price = open_price - 2.0  # $2 for gold sell
                                    be_label = f"Break-even (Entry - $2): {be_price:.2f}"
                                else:
                                    be_price = open_price - 0.0005  # 5 pips for forex sell
                                    be_label = f"Break-even (Entry - 5 pips): {be_price:.5f}"
                            else:
                                be_price = open_price
                                be_label = "Break-even: Unknown position type"

                            st.info(be_label)

                        with col_be_action:
                            # Generate a simpler unique key for BE confirmation
                            be_confirm_key = f"be_confirm_{i}"  # Use index instead of complex key

                            if st.session_state.get(be_confirm_key, False):
                                # Show confirmation buttons
                                col_confirm_be1, col_confirm_be2 = st.columns(2)
                                with col_confirm_be1:
                                    confirm_clicked = st.button(
                                        "CONFIRM Set to BE",
                                        key=f"confirm_be_btn_{i}",
                                        type="primary",
                                        use_container_width=True,
                                    )
                                    if confirm_clicked:
                                        import asyncio

                                        with st.spinner("Setting to break-even..."):
                                            success, message = asyncio.run(modify_position_sl(
                                                position['id'],
                                                be_price
                                            ))
                                            if success:
                                                st.success(message)
                                                # Refresh positions
                                                positions, error = asyncio.run(quick_get_positions())
                                                if positions is not None:
                                                    st.session_state.open_positions = positions
                                                # Remove the confirmation state
                                                if be_confirm_key in st.session_state:
                                                    del st.session_state[be_confirm_key]
                                                st.rerun()
                                            else:
                                                st.error(message)
                                with col_confirm_be2:
                                    cancel_clicked = st.button(
                                        "✕ Cancel",
                                        key=f"cancel_be_btn_{i}",
                                        type="secondary",
                                        use_container_width=True,
                                    )
                                    if cancel_clicked:
                                        # Remove the confirmation state
                                        if be_confirm_key in st.session_state:
                                            del st.session_state[be_confirm_key]
                                        st.rerun()
                            else:
                                be_clicked = st.button(" Set to BE", key=f"be_btn_{i}", use_container_width=True,
                                                       help="Set stop loss to break-even price (entry ± 5 pips for forex, entry ± $2 for gold)")
                                if be_clicked:
                                    # Set confirmation state
                                    st.session_state[be_confirm_key] = True
                                    st.rerun()

elif st.session_state.current_page == "Stats":

    if st.session_state.uploaded_data is not None:
        def highlight_summary(row):
            return ['background-color: ; font-weight: bold' if row['Month'] == 'Total' else '' for _ in row]


        df = st.session_state.uploaded_data
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month_name()
        df['MonthNum'] = df['Date'].dt.month

        st.sidebar.header("Filters")
        selected_year = st.sidebar.selectbox(
            "Select Year",
            options=sorted(df['Year'].unique()),
            index=len(df['Year'].unique()) - 1  # Default to most recent year
        )

        # Filter data
        year_data = df[df['Year'] == selected_year]


        # Calculate monthly stats with chained balances
        def calculate_monthly_stats(year_data):
            monthly_stats = []
            starting_balance = 45000  # Initial balance - REPLACE WITH YOUR ACTUAL STARTING BALANCE
            prev_month_balance = starting_balance
            # Get all months in order
            months = year_data.groupby(['MonthNum', 'Month']).size().reset_index().sort_values('MonthNum')

            for _, (month_num, month_name, _) in months.iterrows():
                month_data = year_data[year_data['MonthNum'] == month_num]
                winners = month_data[month_data['Result'] == 'Win']

                be_trades = month_data[month_data['Result'] == 'BE']
                month_be_pnl = be_trades['PnL'].sum()

                withdraw_deposit = month_data[month_data['Withdrawal_Deposit'].notna()]
                cash_flow = withdraw_deposit['Withdrawal_Deposit'].sum()

                month_pnl = month_data['PnL'].sum()
                ending_balance = starting_balance + month_pnl

                # cash_flow = withdraw_deposit.sum()

                # Calculate monthly percentage gain compared to previous month
                if prev_month_balance != 0:
                    monthly_pct_gain = ((ending_balance - prev_month_balance) / prev_month_balance) * 100
                    set_global("monthly_limit_left", starting_balance * 0.08)
                else:
                    monthly_pct_gain = 0

                if (len(month_data[month_data['Result'] != 'BE']) > 0):
                    winrate_month = round(len(winners) / len(month_data[month_data['Result'] != 'BE']) * 100, 1)
                    total_nobe = len(month_data[month_data['Result'] != 'BE'])
                else:
                    winrate_month = 0
                set_global("current_month", month_name)
                monthly_stats.append({
                    'MonthNum': month_num,
                    'Month': month_name,
                    'Total_Trades': len(month_data[month_data['Result'] != 'NA']),
                    'BE_Trades': len(month_data[month_data['Result'] == 'BE']),
                    'Wins': len(winners),
                    'Win_Rate': winrate_month,
                    'Total_RR': month_data['RR'].sum(),
                    'Avg_Winner_RR': winners['RR'].mean(),
                    'Starting_Balance': starting_balance,
                    'Ending_Trade_Balance': ending_balance,
                    'Total_PnL': month_pnl,
                    # 'Percentage_Gain': (month_pnl / starting_balance) * 100,
                    'Monthly_Pct_Gain': monthly_pct_gain,
                    "Cash_Flow": cash_flow

                })

                # Update balances for next month
                prev_month_balance = ending_balance + cash_flow
                starting_balance = ending_balance + cash_flow
                df = pd.DataFrame(monthly_stats)
                if len(year_data) >= 1:
                    summary_row = {
                        'MonthNum': None,
                        'Month': 'Total',
                        'Total_Trades': df['Total_Trades'].sum(),
                        'BE_Trades': df['BE_Trades'].sum(),
                        'Wins': df['Wins'].sum(),
                        'Win_Rate': round((df['Wins'].sum() / (df['Total_Trades'].sum() - df['BE_Trades'].sum())) * 100,
                                          1) if df['Total_Trades'].sum() > 0 else 0,
                        'Total_RR': df['Total_RR'].sum(),
                        'Avg_Winner_RR': df['Avg_Winner_RR'].mean(),  # Could also use weighted avg
                        'Starting_Balance': df.iloc[0]['Starting_Balance'],
                        'Ending_Trade_Balance': df.iloc[-1]['Ending_Trade_Balance'],
                        'Total_PnL': df['Total_PnL'].sum(),
                        # 'Monthly_Pct_Gain': round(abs((df.iloc[0]['Starting_Balance']-df.iloc[-1]['Ending_Trade_Balance'])/df.iloc[0]['Starting_Balance'])*100,1),  # Not meaningful as a total
                        'Monthly_Pct_Gain': (df['Total_PnL'].sum()) / (df.iloc[0]['Starting_Balance']) * 100,
                        'Cash_Flow': df['Cash_Flow'].sum()
                    }

                df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

            return df
            # return df
            # return pd.DataFrame(monthly_stats)
            # return pd.concat([df, pd.DataFrame([monthly_stats])], ignore_index=True)


        monthly_stats = calculate_monthly_stats(year_data)

        # Format the table
        formatted_stats = monthly_stats[[
            'Month', 'Total_Trades', 'Wins', 'Win_Rate',
            'Total_RR', 'Avg_Winner_RR', 'Starting_Balance', 'Ending_Trade_Balance',
            'Total_PnL', 'Monthly_Pct_Gain', "Cash_Flow"

        ]]

        formatted_stats.columns = [
            'Month', 'Total Trades', 'Wins', 'Win Rate %',
            'Total RR', 'Avg Winner RR', 'Starting Balance', 'Ending Trade Balance', 'Total Trade PnL',
            'Monthly % Gain', "Monthly Withdraw/Deposit"
        ]

        # Display results
        st.header(f"Monthly Performance for {selected_year}")

        # Main table
        st.dataframe(
            formatted_stats.style.format({
                'Total Trades': '{:.0f}',
                'Wins': '{:.0f}',
                'Win Rate %': '{:.1f}%',
                'Total RR': '{:.2f}',
                'Avg RR': '{:.2f}',
                'Avg Winner RR': '{:.2f}',
                'Starting Balance': '${:,.2f}',
                'Ending Trade Balance': '${:,.2f}',
                'Total Trade PnL': '${:,.2f}',
                'Monthly % Gain': '{:.2f}%',
                "Monthly Withdraw/Deposit": '${:.2f}'

            })
            .apply(highlight_summary, axis=1),
            use_container_width=True
        )

        # Breakeven trades info
        breakeven_count = len(year_data[year_data['Result'] == 'BE'])
        st.caption(f"*Note: Excluded {breakeven_count} breakeven trades from calculations")


elif st.session_state.current_page == "Guidelines":
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem; /* Reduce top padding */
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Guidelines to Follow")

    st.subheader("Length Requirement for First/Middle Leg (Price squeeze out of 559 zone)")
    st.subheader("2_BNR and 2_BNR_TPF")

    table_data = {
        '': ['2_BNR', '2_BNR_TPF'],
        'XAUUSD 50 - 66': ['1.79%', '1.19%'],
        'XAUUSD 66 - 786': ['1.19%', '1.19%'],
        'GBPUSD/EURUSD 50 - 66': ['1.49%', '0.99%'],
        'GBPUSD/EURUSD 66 - 786': ['0.99%', '0.99%'],
        'OTHER 50 - 66': ['0.99%', '0.99%'],
        'OTHER 66 - 786': ['0.99%', '0.99%']
    }

    st.table(table_data)

    st.subheader("Length Requirement for First/Middle Leg (Price squeeze out of Shoulder next fib zone)")
    st.subheader("2_BNR_TPF")

    table_data2 = {
        '': ['2_BNR_TPF'],
        'XAUUSD 50 - 66': ['2.4%'],
        'XAUUSD 66 - 786': ['2.4%'],
        'GBPUSD/EURUSD 50 - 66': ['2%'],
        'GBPUSD/EURUSD 66 - 786': ['2%'],
        'OTHER 50 - 66': ['2%'],
        'OTHER 66 - 786': ['2%']
    }

    st.table(table_data2)

    st.subheader("BE Rules")
    table_data2 = {
        '': ['1_BNR', '1_BNR_TPF', '2_BNR', '2_BNR_TPF'],
        'Trigger Condition': ["Trigger at 2.5 R",
                              'Trigger at 2.5 R',
                              'Trigger at 2.0 R', 'Trigger at 2.0 R'],
        'Action': ["Trail 5 Pips Below/Above Entry", 'Trail 5 Pips Below/Above Entry', "Trail 5 Pips Below/Above Entry",
                   "Trail 5 Pips Below/Above Entry"]
    }
    st.table(table_data2)

    st.subheader("First Trail Rules")
    table_data3 = {
        '': ['1_BNR', '1_BNR_TPF', '2_BNR', '2_BNR_TPF'],
        'Trigger Condition': ["Trigger at 4R", 'Trigger at 4R', 'Trigger at 4R', 'Trigger at 4R'],
        'Action': ["Behind first 8H Structure or huge candle, then no further trailing",
                   'Behind first 8H Structure or huge candle, then no further trailing',
                   "Behind first 8H Structure or huge candle", "Behind first 8H Structure or huge candle"]
    }
    st.table(table_data3)

    st.subheader("Second+ Trail Rules")
    table_data4 = {
        '': ['2_BNR', '2_BNR_TPF'],
        'Trigger Condition': ["After every 2 Daily Candle formation", 'After every 2 Daily Candle formation'],
        'Action': ["After 2 Daily huge candle or structure",
                   'After 2 Daily huge candle or structure']
    }
    st.table(table_data4)

    st.subheader("Retake Rules")
    st.write("1_BNR -> 2_BNR || 1_BNR -> 2_BNR_TPF")
    st.write("1_BNR_TPF -> 2_BNR || 1_BNR_TPF -> 2_BNR_TPF")
    st.write("2_BNR -> 2_BNR_TPF || 2_BNR_TPF -> NO RETAKE")

if st.session_state.current_page == "Entry Criteria Check":
    st.title("Entry Model Identification")
    # Define the question flow and animal database
    ENTRY_DATABASE = {
        "1_BNR": {
            "attempt": 0,
            "impulse_leg_length": True,
            "impulse_leg_BOS": True,
            "TPF_Pattern": False,
        },
        "1_BNR_TPF": {
            "attempt": 0,
            "impulse_leg_length": True,
            "impulse_leg_BOS": True,
            "TPF_Pattern": True,
        },
        "2_BNR": {
            "attempt": 1,
            "impulse_leg_length_2": True,
            "impulse_leg_BOS_2": True,
            "first_attempt_leg_length": True,
            "TPF_Pattern_2": False
        },
        "2_BNR_TPF": {
            "attempt": 1,
            "impulse_leg_length_2": True,
            "impulse_leg_BOS_2": True,
            "first_attempt_leg_length": True,
            "TPF_Pattern_2": True
        },
        "3_BNR_TPF": {
            "attempt": 2,
            "impulse_leg_length_3": True,
            "impulse_leg_BOS_3": True,
            "mid_attempt_leg_length": True,
            "left_leg_length": True,
            "TPF_Pattern_3": True
        },

    }

    QUESTIONS_FLOW = [
        {
            "id": "attempt",
            "text": "How many times have price tested the 0.559 Zone?",
            "options": [0, 1],
            "follow_up": {
                "0": {
                    "id": "impulse_leg_length",
                    "text": "Is the impulse leg used to draw Fib: > 2% two_daily, > 2.5% weekly?",
                    "options": ["Yes", "No"],
                    "follow_up": {
                        "Yes": {
                            "id": "impulse_leg_BOS",
                            "text": "Is the impulse leg used to draw Fib on the same timeframe?",
                            "options": ["Yes", "No"],
                            "follow_up": {
                                "Yes": {
                                    "id": "TPF_Pattern",
                                    "text": "Is there a TPF pattern being formed, obvious TPF on the left or a trigger?",
                                    "options": ["Yes", "No"],
                                    "follow_up": None  # Will complete survey after this
                                },
                            }
                        },
                        "No": None  # Will complete survey after this
                    }
                },
                "1": {
                    "id": "impulse_leg_length_2",
                    "text": "Is the impulse leg used to draw Fib: > 2% two_daily, > 2.5% weekly?",
                    "options": ["Yes", "No"],
                    "follow_up": {
                        "Yes": {
                            "id": "impulse_leg_BOS_2",
                            "text": "Is the impulse leg used to draw Fib on the same timeframe?",
                            "options": ["Yes", "No"],
                            "follow_up": {
                                "Yes": {
                                    "id": "first_attempt_leg_position",
                                    "text": "Whats the price position of the first attempt leg?",
                                    "options": ["<66", "\\>66"],
                                    "follow_up": {
                                        "<66": {
                                            "id": "first_attempt_leg_length",
                                            "text": "Is the first attempt leg >= 0.99% for other pairs and >=1.49% for major pairs (EURUSD, GBPUSD, USDJPY) and >= 1.79% for Gold?",
                                            "options": ["Yes", "No"],
                                            "follow_up": {
                                                "Yes": {
                                                    "id": "TPF_Pattern_2",
                                                    "text": "Is there a TPF pattern being formed, obvious TPF on the left or a trigger?",
                                                    "options": ["Yes", "No"],
                                                    "follow_up": None
                                                },
                                            }
                                        },
                                        "\\>66": {
                                            "id": "first_attempt_leg_length_2",
                                            "text": "Is the first attempt leg squeezed out of 559 zone and >= 0.99%, for Gold it has to be >= 1.19%",
                                            "options": ["Yes", "No"],
                                            "follow_up": {
                                                "Yes": {
                                                    "id": "TPF_Pattern_2",
                                                    "text": "Is there a TPF pattern being formed, obvious TPF on the left or a trigger?",
                                                    "options": ["Yes", "No"],
                                                    "follow_up": None
                                                },
                                            }  # Will complete survey after this
                                        },
                                    }
                                    # Will complete survey after this
                                },

                            }
                        },
                        "No": None  # Will complete survey after this
                    }
                },
                "2": {
                    "id": "impulse_leg_length_3",
                    "text": "Is the impulse leg used to draw Fib: > 2% two_daily, > 2.5% weekly?",
                    "options": ["Yes", "No"],
                    "follow_up": {
                        "Yes": {
                            "id": "impulse_leg_BOS_3",
                            "text": "Is the impulse leg used to draw Fib on the same timeframe?",
                            "options": ["Yes", "No"],
                            "follow_up": {
                                "Yes": {
                                    "id": "mid_attempt_leg_position",
                                    "text": "Whats the price position of the second attempt leg?",
                                    "options": ["<66", "\\>66"],
                                    "follow_up": {
                                        "<66": {
                                            "id": "mid_attempt_leg_length",
                                            "text": "Is the second attempt leg >= 0.99% for other pairs and >= 1.5% for major pairs (EURUSD, GBPUSD, USDJPY) and >= 1.79% for Gold?",
                                            "options": ["Yes", "No"],
                                            "follow_up": {
                                                "Yes": {
                                                    "id": "TPF_Pattern_3",
                                                    "text": "Is there a TPF pattern being formed, obvious TPF on the left or a trigger?",
                                                    "options": ["Yes", "No"],
                                                    "follow_up": None
                                                },
                                            }
                                        },

                                        "\\>66": {
                                            "id": "mid_attempt_leg_length_2",
                                            "text": "Is the second attempt leg squeezed out of 559 zone and >= 0.99%, for Gold it has to be >= 1.19%",
                                            "options": ["Yes", "No"],
                                            "follow_up": {
                                                "Yes": {
                                                    "id": "TPF_Pattern_3",
                                                    "text": "Is there a TPF pattern being formed, obvious TPF on the left or a trigger?",
                                                    "options": ["Yes", "No"],
                                                    "follow_up": None
                                                },
                                                # Will complete survey after this

                                            }
                                        }
                                        # Will complete survey after this
                                    },

                                }
                            },
                            "No": None  # Will complete survey after this
                        }
                    }
                },
            }
        }
    ]


    def init_session():
        if 'question_path' not in st.session_state:
            st.session_state.question_path = []
        if 'answers' not in st.session_state:
            st.session_state.answers = {}
        if 'current_question' not in st.session_state:
            st.session_state.current_question = QUESTIONS_FLOW[0]


    def reset_survey():
        st.session_state.question_path = []
        st.session_state.answers = {}
        st.session_state.current_question = QUESTIONS_FLOW[0]


    def identify_entry_model(answers):
        perfect_matches = []
        for model, traits in ENTRY_DATABASE.items():
            match = True
            for key, db_value in traits.items():
                user_answer = answers.get(key)
                if isinstance(db_value, bool):
                    user_value = True if user_answer == "Yes" else False if user_answer == "No" else None
                else:
                    user_value = user_answer
                if user_value is not None and db_value != user_value:
                    match = False
                    break
            if match:
                perfect_matches.append(model)
        return perfect_matches


    def show_results():
        matched_models = identify_entry_model(st.session_state.answers)
        if matched_models:

            st.success("### Matching Entry Models Found")
            for model in matched_models:
                set_global("entry_model", model)
                st.write(f"**{model}**:")
                st.json(ENTRY_DATABASE[model])
        else:
            st.warning("No matching entry models found")

        st.write("### Your Answers:")
        st.json(st.session_state.answers)

        if st.button(" Start New Analysis"):
            reset_survey()
            st.rerun()


    def get_next_question(current_q, answer):
        """Determine the next question based on current answer"""
        # If current question has no follow-up, return None to complete survey
        if current_q.get('follow_up') is None:
            return None

        # Check for specific follow-up question
        if current_q.get('follow_up') and str(answer) in current_q['follow_up']:
            next_q = current_q['follow_up'][str(answer)]
            if next_q:
                return next_q

        # If no follow-up and we have a path to return to
        if st.session_state.question_path:
            return st.session_state.question_path.pop()

        # No more questions
        return None


    def main():
        init_session()

        # Check if we should show results (when current question is None)
        if st.session_state.current_question is None:
            show_results()
            return

        # Display current question
        current_q = st.session_state.current_question
        st.subheader(current_q['text'])

        if isinstance(current_q['options'][0], int):
            answer = st.selectbox("Your answer:", options=current_q['options'], key=f"select_{current_q['id']}")
        else:
            answer = st.radio("Your answer:", options=current_q['options'], key=f"radio_{current_q['id']}")

        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                if st.session_state.question_path:
                    st.session_state.current_question = st.session_state.question_path.pop()
                st.rerun()

        with col2:
            if st.button("Next →"):
                # Store answer
                st.session_state.answers[current_q['id']] = answer

                # Get next question
                next_q = get_next_question(current_q, answer)

                if next_q:
                    # Push current question to path if we're going into a follow-up
                    if (current_q.get('follow_up') and
                            str(answer) in current_q['follow_up'] and
                            current_q['follow_up'][str(answer)] == next_q):
                        st.session_state.question_path.append(current_q)
                    st.session_state.current_question = next_q
                else:
                    # Survey complete - set current question to None
                    st.session_state.current_question = None

                st.rerun()


    if __name__ == "__main__":
        main()
