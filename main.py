# main.py (your entry point)
import os

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
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

# Configure page
st.set_page_config(layout="wide")
## Every year change starting_balance =, starting_capital = and base_risk =

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



    if(total_gain>0):
        cumulative_gain = round(total_gain/ending_balance*100,2)
        current_risk = base_risk
        word = str(cumulative_gain) + "%" + " / $"+str(total_gain)
    else:
        current_risk = base_risk
        word = "Need Gains for the Month"
    return current_risk , word, total_gain

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

        '1_BNR': ("A", 0.91), # Fib + wick/ob
        "1_BNR_TPF": ("A", 1.0), #wick/ob

        # 2 Touch fib>wick>percentage

        "2_BNR": ("A", 1.1), # Fib + wick/ob
        "2_BNR_TPF": ("A", 1.2), #Fake out or wick/ob

        # 3 Touch  wick>percentage>fib

        #"3_BNR_TPF": ("A", 1), # Fib + wick/ob + 50% outside of discount zone


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

    filtered_group = group[group['Result']!="BE"]
    wins = (filtered_group['Result'] == "Win").sum()
    total_no_be = len(filtered_group)

    total = len(group)
    if(wins<=0):
        winrate = 0
    else:
        winrate = (wins / total_no_be) * 100 if total > 0 else 0
    #avg_return = group['PnL'].mean()

    win_group = group[group['Result']=="Win"]
    if(len(win_group)<=0):
        avg_win_rr = 0
    else:
        avg_win_rr = win_group['RR'].sum()/len(win_group)

    total_return = group['PnL'].sum()

    #if(len(group)<20):
    grade, multiplier = calculate_strategy_grade_static(group.name)
    #else:
    #grade, multiplier = calculate_strategy_grade(winrate, total_no_be)

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
# Initialize session state variables
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
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


@st.cache_resource
def get_google_sheets_client():
    """Initialize Google Sheets connection - supports both Streamlit secrets and GitHub secrets"""
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']

        creds_dict = None

        # Option 1: Check for GitHub Actions secrets (environment variables)
        if all(key in os.environ for key in ['GCP_TYPE', 'GCP_PRIVATE_KEY', 'GCP_CLIENT_EMAIL']):
            st.sidebar.info("🔧 Using GitHub Actions secrets")
            creds_dict = {
                "type": os.environ['GCP_TYPE'],
                "project_id": os.environ.get('GCP_PROJECT_ID', ''),
                "private_key_id": os.environ.get('GCP_PRIVATE_KEY_ID', ''),
                "private_key": os.environ['GCP_PRIVATE_KEY'].replace('\\n', '\n'),
                "client_email": os.environ['GCP_CLIENT_EMAIL'],
                "client_id": os.environ.get('GCP_CLIENT_ID', ''),
                "auth_uri": os.environ.get('GCP_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                "token_uri": os.environ.get('GCP_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                "auth_provider_x509_cert_url": os.environ.get('GCP_AUTH_PROVIDER_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
                "client_x509_cert_url": os.environ.get('GCP_CLIENT_CERT_URL', '')
            }

        # Option 2: Check for Streamlit secrets (secrets.toml)
        elif "gcp_service_account" in st.secrets:
            st.sidebar.info("🔧 Using Streamlit secrets")
            creds_dict = st.secrets["gcp_service_account"]

        else:
            st.sidebar.error("❌ No Google Sheets credentials found")
            st.sidebar.info("Configure either GitHub Actions secrets or Streamlit secrets")
            return None

        # Validate we have the minimum required fields
        if not creds_dict or not creds_dict.get('private_key') or not creds_dict.get('client_email'):
            st.sidebar.error("❌ Missing required credentials (private_key or client_email)")
            return None

        st.sidebar.success(f"✅ Service Account: {creds_dict.get('client_email', 'Unknown')}")

        # Create credentials
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)

        st.sidebar.success("✅ Google Sheets connected successfully!")
        return client

    except Exception as e:
        st.sidebar.error(f"❌ Google Sheets connection failed: {str(e)}")
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
                st.sidebar.success(f"✅ Loaded {len(df)} rows from Google Sheets")

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

        st.sidebar.success(f"✅ Data saved to '{worksheet_name}' in Google Sheets")
        return True

    except Exception as e:
        st.sidebar.error(f"Error saving to Google Sheets: {e}")
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
create_nav_button("✅ Entry Model Check", "Entry Criteria Check", "Entry_Criteria_Check")

# Session Management (Always show in sidebar)
with st.sidebar.expander("⚙️ Session Management"):
    st.write(f"Current page: `{st.session_state.current_page}`")
    if st.session_state.uploaded_data is not None:
        st.write(f"Data: {len(st.session_state.uploaded_data)} rows")
        if st.session_state.cloud_data_loaded:
            st.write("📍 Source: Cloud Storage")
        else:
            st.write("📍 Source: Local Upload")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Session"):
            save_persistent_session()
            st.success("Saved!")
    with col2:
        if st.button("🔄 Clear Session"):
            clear_persistent_session()
            st.session_state.clear()
            st.rerun()

starting_capital = 50000

# Page content
if st.session_state.current_page == "Home":
    # AUTOMATIC CLOUD SYNC ON APP LOAD
    if 'auto_sync_done' not in st.session_state:
        with st.spinner("🔄 Syncing with cloud..."):
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
    with st.sidebar.expander("📁 Cloud Data Management", expanded=True):
        st.subheader("Google Sheets Storage")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load from Cloud"):
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
            if st.button("💾 Save to Cloud"):
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
            col4, col5, col6 = st.columns(3)
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

                st.success("✅ New record added successfully! Metrics recalculated.")
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

                    st.success("✅ Last record deleted successfully! Metrics recalculated.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error deleting last record: {e}")
        else:
            st.write("No records available to delete.")

        # Row Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Grid"):
                st.rerun()
        with col2:
            if st.button("💾 Save All Changes"):
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
                st.success(f"🔍 Filter active: Showing {len(grid_response['data'])} of {len(current_data)} rows")
            else:
                st.info("📊 Showing complete dataset")

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
        df['Drawdown'] = (df['equity'] - df['Peak'])/starting_capital
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
        #col1.metric("Net Trade PNL", f"${df['PnL'].sum():.2f}")
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
            # Create tabs for better organization
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Overall Performance",
                "Monthly Analysis",
                "Direction Analysis",
                "Strategy Analysis",
                "Visualizations"
            ])

            with tab1:
                base_risk = 0.02
                base_risk_money = starting_capital*base_risk
                base_risk_money_multi = base_risk*100
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
                    lambda target_pct: round((base_risk_money * (target_pct / 100) * 100)/base_risk_money_multi,2)
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
                #symbol_performance = pd.concat([symbol_performance, prop_row])

                # CREATE THE DICTIONARY TO STORE GROUP PERFORMANCE GAP DATA
                group_performance_gap = {}

                # Populate the dictionary with group names and their performance gap (actual gain vs target)
                for group_name in symbol_groups.keys():
                    if group_name in symbol_performance.index:
                        actual_pct_gain = symbol_performance.loc[group_name, 'Yearly_Pct_Gain']
                        target_pct = symbol_performance.loc[group_name, 'Target_Pct']
                        # Calculate gap between actual percentage gain and target percentage
                        gap_pct = (target_pct - actual_pct_gain)*1.07
                        # Convert percentage gap to dollar amount
                        gap_dollar = round((gap_pct / 100) * starting_capital,0)
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
                    'Remain_$': [symbol_performance_no_prop['Target_$'].sum() - symbol_performance_no_prop['Total_PnL'].sum()],
                    'Target_$': [symbol_performance_no_prop['Target_$'].sum()],
                    # SUM of Target_$ (changed from mean to sum)
                    'Target': ["Completed" if symbol_performance_no_prop['Yearly_Pct_Gain'].sum() >= symbol_performance_no_prop['Target_Pct'].sum() else
                    "Failed" if symbol_performance_no_prop['Yearly_Pct_Gain'].sum() < -symbol_performance_no_prop['Target_Pct'].sum() else
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
                    }).apply(lambda x: ['background-color: lightblue' if x['Symbol'] == 'PROP' else
                            'background-color: lightcoral' if x['Yearly_Pct_Gain'] < -x['Target_Pct'] else
                            'background-color: lightpink' if x['Total_PnL'] < 0 else
                            'background-color: lightblue' if x['Total_PnL'] > 0 else
                            'background-color: lightgreen' if x['Target'] == 'Completed' else
                            'background-color: lightyellow' if x['Target'] == 'In Progress' else
                            'background-color: lightgray' if x['Symbol'] == 'SUMMARY' else
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
                    if(symbol!="PROP"):
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
                    if(symbol!="PROP"):
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

    else:
        st.warning("Please upload data first to analyze symbol statistics")

elif st.session_state.current_page == "Risk Calculation":
    # Remove all top padding
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 0rem;
                padding-left: 5rem;
                padding-right: 0rem;
            }
            .stApp {
                margin-top: 15px;  # Adjust this negative value as needed
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state for saved selections if not exists
    if 'saved_selections' not in st.session_state:
        st.session_state.saved_selections = {}

    #st.title("Risk Calculation")
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
            sum_gap = XAUUSD_gap+USDCAD_AUDUSD_gap+GBPAUD_EURAUD_gap+GBPUSD_EURUSD_gap+JPY_Pairs_gap
        else:
            # Fallback values if computation failed
            XAUUSD_gap = 0
            USDCAD_AUDUSD_gap = 0
            GBPUSD_EURUSD_gap = 0
            JPY_Pairs_gap = 0
            GBPAUD_EURAUD_gap = 0
            sum_gap = 0


        a_momemtum_text= ''
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
                month_win_pnl = winners['PnL'].sum()+month_be_pnl

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

                    ten_percent_goal = starting_balance + round(starting_balance*0.10,2)

                    month_data["Running_Equity"] = starting_balance + month_data["PnL"].cumsum()
                    month_data["Peak_Equity"] = month_data["Running_Equity"].cummax()

                    if (month_data["Peak_Equity"].iloc[-1] >= ten_percent_goal):
                        month_data["Drawdown_Limit"] = round(month_data["Peak_Equity"] * 0.12, 2)
                        monthly_loss_limit = starting_balance * 0.12
                        #monthly_loss_limit = month_data["Drawdown_Limit"].iloc[-1]
                    else:
                        monthly_loss_limit = starting_balance * 0.12

                        #month_data["Drawdown_Limit"] = round(month_data["Peak_Equity"] * 0.12, 2)
                        #monthly_loss_limit = month_data["Drawdown_Limit"].iloc[-1]
                        #monthly_loss_limit = round(starting_balance * 0.08, 2)

                    #monthly_loss_limit = round(starting_balance * 0.08, 2)
                    monthly_loss_limit = starting_balance * 0.12
                    monthly_actual_loss = month_loss_pnl
                    current_month_stats = month_data

                else:
                    #monthly_loss_limit = round(ending_balance * 0.08, 2)
                    monthly_actual_loss = 0
                    monthly_actual_gain = 0
                    monthly_loss_limit = starting_balance * 0.12
                prev_month_balance = ending_balance + cash_flow
                starting_balance = ending_balance + cash_flow



            return current_month_stats, monthly_loss_limit, monthly_actual_loss, ending_balance


        current_month_stats, monthly_loss_limit, monthly_actual_loss, ending_balance = calculate_monthly_stats_2(year_data)

        if(df['Withdrawal_Deposit'].sum()!=0):
            cash_flow = df['Withdrawal_Deposit'].cumsum()
        else:
            cash_flow = 0

        df['equity'] = df['PnL'].cumsum().iloc[-1] + starting_capital + cash_flow
        filled_equity = df['equity'].ffill()
        equity = filled_equity.iloc[-1]

        #TRADING_GRADE_SCALE = {
            #"A+": {"min_winrate": 71, "multiplier": 1.3, "min_trades": 20},
            #"A": {"min_winrate": 61, "multiplier": 1.2, "min_trades": 20},
            #"B": {"min_winrate": 51, "multiplier": 1.1, "min_trades": 20},
            #"C": {"min_winrate": 41, "multiplier": 1.0, "min_trades": 20},
        #}

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Grading Based Risk Multiplier")
            strategy_stats = df.groupby('Strategy').apply(analyze_strategy)
            strategy_stats = strategy_stats.sort_values(by=['Win Rate (%)','Total Return'], ascending=False)

            # Format display
            display_df = strategy_stats.copy()
            display_df['Win Rate (%)'] = display_df['Win Rate (%)'].map("{:.1f}%".format)
            display_df['Avg Win RR'] = display_df['Avg Win RR'].map("{:,.2f}".format)
            display_df['Total Return'] = display_df['Total Return'].map("${:,.2f}".format)

            st.dataframe(display_df)

        with col2:
            st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)
            st.session_state.current_risk = 1.0  # Base risk

            #st.subheader("Adaptive Risk Calculation")
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
                    round((round(monthly_loss_limit, 2)+ round(monthly_actual_loss, 2)) / starting_capital * 100,
                          2)) + "%"]
            }).set_index("Metric")

            st.table(result_df)
            next_risk = base_risk









        #st.subheader("Risk Calculation")

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
            elif"CAD" in pair:
                pip_value = 10/get_usd_cad()  # USD/CAD pip value (USD account)
            elif pair=="EURAUD" or pair=="GBPAUD":
                pip_value = 10/get_aud_usd()

            elif"CHF" in pair:
                pip_value = 10/get_usd_chf()
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
            elif"CAD" in pair:
                pip_value = 10/get_usd_cad()  # USD/CAD pip value (USD account)
            elif pair=="EURAUD" or pair=="GBPAUD":
                pip_value = 10/get_aud_usd()

            elif"CHF" in pair:
                pip_value = 10/get_usd_chf()
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
            "AUDUSD","USDCAD"
        ]
        cad_sec = [
            "USDCAD"
        ]
        aud_family1 =[
            "AUDUSD","AUDJPY","GBPAUD", "EURAUD"
        ]

        europe_major = [
            "EURUSD", "GBPUSD"
        ]
        gold_comm = [
            "XAUUSD"
        ]

        xxxusd = [
            "AUDUSD", "USDCAD","EURUSD", "GBPUSD"
        ]
        majors_nocad = ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        majors = ["AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDJPY","XAUUSD"]
        majors_dollar = ["AUDUSD", "EURUSD", "GBPUSD", "XAUUSD"]
        minors = ["GBPAUD", "EURAUD","GBPJPY", "EURJPY", "AUDJPY"]

        strategies = ['1_BNR','1_BNR_TPF', '2_BNR','2_BNR_TPF']
        time_frame = ['Weekly Structure','Two_Daily Structure']
        incompatible_map = {
            "USDCAD": ['1_BNR'],
            "GBPAUD": ["1_BNR"],
            "AUDUSD": [""],
            "EURAUD": ["1_BNR"],
            "GBPJPY": ["1_BNR",  "1_BNR_TPF",'EM_2b', 'EM_3b'],
            "EURJPY": ["1_BNR",  "1_BNR_TPF",'EM_2b', 'EM_3b'],
            "AUDJPY": ["1_BNR",  "1_BNR_TPF", 'EM_2b', 'EM_3b'],
            # Add more restrictions as needed
        }

        potential_rr = ["3.41-4.41", "5.41-7.41", "8.41-10.41", ">=11.41"]
        within_61 = ["No","Yes"]
        incompatible_map_4 = {
            "2_BNR": ["No"],
            "2_BNR_TPF": ["No"],
            "3_BNR_TPF":[]
        }
        incompatible_map_2 = {
                        "EM_1c": ["8.41-10.41", ">=11.41"],
                        "EM_2c": ["8.41-10.41", ">=11.41"],
                        "EM_3c": ["8.41-10.41", ">=11.41"],
                        "EM_1b": [">=11.41"],
                        "EM_2b": [">=11.41"],
                        "EM_3b": [">=11.41"],}
        Variance = ["50", "559 - 66", "66 - 805","> 805"]
        Trend_Positions = ["3%-4.99%", "5%-6.99%", "7%-8.99%","9%-10.99%","11%-12.99%",">=13%"]
        incompatible_map_3 = {
            "1_TPF": ["50", "> 805"],
            "1_BNR": ["50","559 - 66", "> 805"],
            "1_BNR_TPF": ["50"],
            "2_BNR": [""],
            "2_BNR_TPF": [""],
            "3_BNR_TPF": ["50","> 805","66 - 805"],}
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
            "Cross Wave": ['2_BNR','1_BNR','1_BNR_TPF'],
            "9%-10.99%Wave 1": ['1_BNR'],
            "11%-12.99%Wave 1": ['1_BNR','1_BNR_TPF'],
            # Add more restrictions as needed
        }

        incompatible_map_10 = {
            "3%-4.99%1_BNR_TPF": ["559 - 66"],
            "9%-10.99%1_BNR_TPF": [],
            "11%-12.99%1_BNR_TPF": [],
            ">=13%": ["559 - 66"],
            "EURAUD1_BNR_TPF": ["559 - 66", "66 - 805"],
            "GBPAUD1_BNR_TPF": ["559 - 66", "66 - 805"]
            # Add more restrictions as needed
        }

        incompatible_map_11 = {
            "AUDJPY": ['Two_Daily Structure', 'Daily BOS', '8H/4H BOS'],
            "GBPJPY":['Two_Daily Structure','Daily BOS', '8H/4H BOS'],
            "EURJPY": ['Two_Daily Structure', 'Daily BOS', '8H/4H BOS'],
            "EURAUD":['8H/4H BOS'],
            "GBPAUD": ['8H/4H BOS'],
        }

        incompatible_map_12 = {
            "3%-4.99%": ['1_BNR'],
            "XAUUSD5%-6.99%": ['1_BNR'],
            "XAUUSD>=13%": ['1_BNR',"1_BNR_TPF"],
            ">=13%": ['1_BNR',"1_BNR_TPF"],


        }

        incompatible_map_13 = {
            "Yes": [],
            "No": ['1_BNR','1_BNR_TPF']

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

        def get_available_timeframe_2(listone,trend):
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
                        diabled_61 = incompatible_map_5.get(pair,[])
                        return [s for s in list_available if s not in diabled_61]
        def get_available_61_3(position, list_available):
                        diabled_61 = incompatible_map_6.get(position,[])
                        return [s for s in list_available if s not in diabled_61]
        def get_available_strategies(pair):
            disabled_strategies = incompatible_map.get(pair, [])
            return [s for s in strategies if s not in disabled_strategies]

        def get_available_strategies2(cross_fib,strats):
            disabled_strategies_2 = incompatible_map_9.get(cross_fib, [])
            return [s for s in strats if s not in disabled_strategies_2]

        def get_available_strategies3(trend_positions,pair,strats_2):

            if(pair=="XAUUSD"):
                var = pair+trend_positions
            else:
                var = trend_positions
            disabled_strategies_3 = incompatible_map_12.get(var, [])
            return [s for s in strats_2 if s not in disabled_strategies_3]

        def get_available_strategies4(hh_ll,stratslist):
            disabled_strategies_4 = incompatible_map_13.get(hh_ll, [])
            return [s for s in stratslist if s not in disabled_strategies_4]

        def get_available_strategies5(timeframe,stratslist):
            disabled_strategies_5 = incompatible_map_15.get(timeframe, [])
            return [s for s in stratslist if s not in disabled_strategies_5]
        def get_available_variance(entry_model):
            available_variance = incompatible_map_3.get(entry_model, [])
            return [s for s in Variance if s not in available_variance]
        def get_available_variance_2(trend,variancelist):
            available_variances = incompatible_map_10.get(trend, [])
            return [s for s in variancelist if s not in available_variances]

        def get_pair_prior_result(current_month_stats,selected_pair):
            if(len(current_month_stats)>0):
                pair_trades = current_month_stats[current_month_stats['Symbol'] == selected_pair]
                if(len(pair_trades)<1):
                    return 'N'
                else:
                    loss_pair_trades = pair_trades[pair_trades["Result"]=="Loss"]
                    be_pair_trades = pair_trades[pair_trades["Result"]=="BE"]
                    if(len(loss_pair_trades)>=2):
                        return "X"
                    elif(len(be_pair_trades)>2):
                        return "X"
                    else:
                        non_be_pair_trades = pair_trades[pair_trades["Result"]!="BE"]
                        if(len(non_be_pair_trades)>0):
                            last_result = str(non_be_pair_trades["Result"].iloc[-1])
                            if(last_result=="Win"):
                                return 'W'
                            else:
                                return "L"
                            return last_result
                        else:
                            return "N"
            else:
                return "N"
        def get_pair_sect_count(current_month_stats,pair):
            if (len(current_month_stats) > 0):
                current_month_stats = current_month_stats[current_month_stats['Result'] == "Loss"]
                current_month_stats_linked = current_month_stats[current_month_stats['Symbol'].isin(xxxusd)]
                current_month_stats_linked2 = current_month_stats[current_month_stats['Symbol'].isin(aud_family1)]

            remain_count = 2
            xxxaud_count = 1
            yen_count = 1
            audjpy_count = 1
            audusd_count = 2
            cad_count = 2
            europe_count = 2
            gold_count = 2
            pair_trades = 0
            if (len(current_month_stats) > 0):
                if(pair in xxxaud):
                    pair_trades = len(current_month_stats[current_month_stats['Symbol'].isin(xxxaud)])

                    xxxaud_count = xxxaud_count - pair_trades
                    if (len(current_month_stats_linked2) > 0):
                        pair_trades_aud = len(current_month_stats_linked2)
                        if(pair_trades_aud>=2):
                            xxxaud_count = xxxaud_count - pair_trades_aud
                    return  xxxaud_count
                elif (pair == "AUDJPY"):
                    if (len(current_month_stats_linked2) > 0):
                        pair_trades_aud2 = len(current_month_stats_linked2)
                        if (pair_trades_aud2 >= 2):
                            audjpy_count = audjpy_count - pair_trades_aud2
                    YEN_trades = len(current_month_stats[current_month_stats['Symbol'].isin(yens)])
                    audjpy_count = audjpy_count - YEN_trades
                    return audjpy_count
                elif(pair in yens):
                    pair_trades = len(current_month_stats[current_month_stats['Symbol'].isin(yens)])
                    yen_count = yen_count - pair_trades

                    return yen_count
                elif(pair == "USDCAD"):
                    pair_trades = len(current_month_stats[current_month_stats['Symbol'] == "USDCAD"])
                    if(pair_trades>0):
                        cad_count = 0
                    if (len(current_month_stats_linked) > 0):
                        any_exist = current_month_stats_linked['Symbol'].isin(europe_major).any()

                        if (any_exist):
                            cad_count = 0
                    pair_trades_ASIA = len(current_month_stats[current_month_stats['Symbol'].isin(trade_curr)])
                    cad_count = cad_count - pair_trades_ASIA
                    return cad_count
                elif (pair == "AUDUSD"):
                    if (len(current_month_stats_linked2) > 0):
                        pair_trades_aud3 = len(current_month_stats_linked2)
                        if (pair_trades_aud3 >= 2):
                            audusd_count = audusd_count - pair_trades_aud3
                    pair_trades_TRADE = len(current_month_stats[current_month_stats['Symbol'].isin(trade_curr)])
                    audusd_count = audusd_count - pair_trades_TRADE
                    if (len(current_month_stats_linked) > 0):
                        any_exist = current_month_stats_linked['Symbol'].isin(europe_major).any()

                        if (any_exist):
                            audusd_count = 0
                    return audusd_count

                elif(pair in europe_major):
                    pair_trades = len(current_month_stats[current_month_stats['Symbol'].isin(europe_major)])
                    europe_count = europe_count - pair_trades
                    if (len(current_month_stats_linked) > 0):
                        any_exist = current_month_stats_linked['Symbol'].isin(trade_curr).any()

                        if (any_exist):
                            europe_count = 0
                    return europe_count
                elif (pair in gold_comm):
                    pair_trades = len(current_month_stats[current_month_stats['Symbol'].isin(gold_comm)])
                    gold_count = gold_count - pair_trades
                    return gold_count
                return remain_count
            else:
                return 2



        col1, col2 = st.columns(2, gap = "small")
        with col1:
            account_balance = equity
            #account_balance = st.number_input("Account balance", equity)
            #account_balance = st.number_input(
                #"Account balance",
                #value=equity,
                #disabled=True  # Disables editing
            #)
            st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)
            selected_pair = st.selectbox("Trading Pair", currency_pairs)
            available_trend_position = get_available_trend_position(selected_pair)
            trend_position = st.selectbox("Trend Position", available_trend_position)

            available_time_frame = get_available_timeframe(selected_pair)
            available_time_frame_2 = get_available_timeframe_2(available_time_frame,trend_position)
            POI = st.selectbox("POI Type (2_Daily is 2nd to Last on Weekly -- Note that the wick on first weekly Candle need to be >=0.50%)", available_time_frame_2)
            cross_fib = st.selectbox("Wave Status", ['Wave 2+', 'Wave 1', 'Cross Wave'])
            HH_LL = st.selectbox("FIB drawn on Highest High (Buy)/ Lowest Low (Sell)",['Yes','No'])




            available_strategies = get_available_strategies(selected_pair)
            available_strats = get_available_strategies2(cross_fib,available_strategies)
            varcross = trend_position+cross_fib
            available_strats_crossfib = get_available_strategies2(varcross,available_strats)

            available_strats_2 = get_available_strategies3(trend_position,selected_pair,available_strats_crossfib)
            available_strats_3 = get_available_strategies4(HH_LL,available_strats_2)
            available_strats_4 = get_available_strategies5(POI,available_strats_3)

            risk_multiplier = st.selectbox("Entry Model",
                                           available_strats_4,
                                           index=0,
                                           help="Adjust risk based on trade quality")

            #Adaptive_value = st.number_input("Adaptive risk based on streak",next_risk,format="%.3f")
            #Adaptive_value = st.number_input(
            #	"Adaptive risk based on streak",
            #	value=next_risk,
            #	disabled=True,
            #	format = "%.3f"
            #)
            available_vairances = get_available_variance(risk_multiplier)
            if(selected_pair not in majors_dollar):
                if('50' in available_vairances):
                    available_vairances.remove("50")
            concat_trend = trend_position+risk_multiplier
            final_variance1 = get_available_variance_2(concat_trend,available_vairances)

            concat_trend2 = selected_pair+risk_multiplier
            final_variance2 = get_available_variance_2(concat_trend2,final_variance1)

            Variances = st.selectbox("Position Variance", final_variance2)




            stop_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=None, step=1.0)
            Adaptive_value = next_risk
            #available_rr = get_available_rr(risk_multiplier)
            available_61 = get_available_61(risk_multiplier)
            #Potential = st.selectbox("Potential RR", available_rr)

            available_61 = get_available_61(risk_multiplier)
            available_61_2 = get_available_61_2(selected_pair,available_61)
            available_61_3 = get_available_61_3(Variances,available_61_2)

            within_61 = st.selectbox("Price Within 64", available_61_3)
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)


            pair_result = get_pair_prior_result(current_month_stats,selected_pair)
            #Pair_prior_result = st.selectbox("Pair Prior Result in Month", pair_result, disabled=True)

            sect_count = get_pair_sect_count(current_month_stats, selected_pair)


        with col2:


            strategy_stats = df.groupby('Strategy').apply(analyze_strategy)
            #st.write(strategy_stats.items())
            #strategy_stats_df = pd.DataFrame(strategy_stats.tolist(), index=strategy_stats.index)
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
                #st.write(strategy)
                #st.write(metrics)

                if strategy == "Multiplier":
                    #multiplier = metrics["Multiplier"]
                    for Strategy_, Multiplier_ in metrics.items():
                        if Strategy_ == risk_multiplier:
                            multiplier = Multiplier_
                            #st.write(Strategy_,Multiplier_)

                    break

            #multiplier = calculate_strategy_grade_temp(risk_multiplier)
            if(POI == 'Weekly Structure'):
                POI_multiplier = 1.0
            elif(POI == 'Two_Daily Structure'):
                POI_multiplier = 0.91


            if (within_61 == 'Yes'):
                sixone_multiplier = 1.1
            else:
                sixone_multiplier = 1.0
            #if(Potential == '3.41-4.41'):
                #rr_multiplier = 1.0
            #elif(Potential == '5.41-7.41'):
                #rr_multiplier = 1.1
            #elif(Potential == '8.41-10.41'):
                #rr_multiplier = 1.2
            #elif (Potential == '>=11.41'):
                #rr_multiplier = 1.3
            if (trend_position == "3%-4.99%"):
                trend_position_multiplier = 1.0

            elif (trend_position == "5%-6.99%"):
                if (risk_multiplier == "1_BNR_TPF"):
                    if(Variances =="559 - 66"):
                        trend_position_multiplier = 0.91
                else:
                    trend_position_multiplier = 1.05
            elif (trend_position == "7%-8.99%"):
                if (risk_multiplier == "1_BNR_TPF"):
                    if(Variances =="559 - 66"):
                        trend_position_multiplier = 0.91
                else:
                    trend_position_multiplier = 1.1
            elif(trend_position == "9%-10.99%"):
                if (risk_multiplier == "1_BNR_TPF"):
                    if(Variances =="559 - 66"):
                        trend_position_multiplier = 0.91
                else:
                    trend_position_multiplier = 1.0
            elif (trend_position == "11%-12.99%"):
                if(risk_multiplier=="1_BNR" or risk_multiplier=="1_BNR_TPF"):
                    if (Variances == "559 - 66"):
                        trend_position_multiplier = 0.81
                    else:
                        trend_position_multiplier = 0.91
                else:
                    trend_position_multiplier = 1.0
            else:
                if (risk_multiplier == "1_BNR" or risk_multiplier == "1_BNR_TPF"):
                    trend_position_multiplier = 0.81
                else:
                    trend_position_multiplier = 0.91

            if (Variances == "50"):
                variance_multiplier = 0.91
            elif (Variances == "559 - 66"):
                variance_multiplier = 1.0
            elif (Variances == "66 - 805"):
                variance_multiplier = 1.0
            else:
                variance_multiplier = 1.0

            if(pair_result == "N"):
                prior_result_multiplier = 1.0
            elif(pair_result == "W"):
                prior_result_multiplier = 1.15
            elif(pair_result == "L"):
                prior_result_multiplier = 0.91
            elif(pair_result=="X"):
                prior_result_multiplier = 0


            if(sect_count>0):
                sect_count_multiplier = 1.0
            else:
                sect_count_multiplier = 0.0

            if(cross_fib == "Cross Wave"):
                cross_fib_multiplier = 0.91
            else:
                cross_fib_multiplier = 1.0

            if(selected_pair in europe_major or selected_pair == "XAUUSD"):
                if(pair_result == "W"):
                    if(risk_multiplier == "2_BNR" or risk_multiplier == "2_BNR_TPF"):
                        if(trend_position == "3%-4.99%" or trend_position == "5%-6.99%" or trend_position == "7%-8.99%"):
                            big_risk_multiplier = 1.5
            else:
                big_risk_multiplier = 1.0
            yearly_factor = starting_capital
            final_risk_1 = (yearly_factor*Adaptive_value)*multiplier*POI_multiplier*trend_position_multiplier*sixone_multiplier*prior_result_multiplier*sect_count_multiplier*big_risk_multiplier*cross_fib_multiplier*variance_multiplier
            final_risk = math.floor(final_risk_1)
            if(final_risk>(yearly_factor*0.05)):
                final_risk = yearly_factor*0.05

            if(stop_pips != None):
                position_size = calculate_position_size(final_risk, stop_pips, selected_pair)
                position_size_propfirm = calculate_position_size_propfirm(2000, stop_pips, selected_pair)
                set_global("position_size",position_size)
                set_global("position_size_propfirm",position_size_propfirm)
            else:
                position_size = 0
                position_size_propfirm = 0
            def getPairEntrySL(pair):
                if(pair == "GBPUSD"):
                    return "223 ", "243"
                elif (pair == "EURUSD"):
                    return "203 ", "223"
                elif(pair == "AUDUSD"):
                    return "153 ", "193"
                elif (pair == "XAUUSD"):
                    return "5 ", "9.3"
                elif (pair == "USDJPY"):
                    return "223 ", "253"
                elif (pair == "USDCAD"):
                    return "203 ", "233"
                else:
                    return "253 ", "283"
            entry_title = ""
            entry_text = ""
            SL_title=""
            SL_text=""
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

            def compare_target(open_target,desire_target):
                if(open_target<desire_target):
                    if(open_target>3):
                        return str(open_target)
                    else:
                        return "3"
                elif(open_target>desire_target):
                    return str(desire_target)

            if(len(a_momemtum_text)<1):
                                a_momemtum_text="To be Filled"
            if(risk_multiplier == "1_BNR"):
                if(within_61=="Yes"):
                    if(selected_pair == "XAUUSD"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "$ Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + "$"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),7.41)
                    elif(selected_pair == "AUDUSD"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair), 5.41)
                    elif(selected_pair == "USDJPY"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair), 6.41)
                    else:
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),6.41)

                else:
                    if (selected_pair == "XAUUSD"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "$ Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + "$"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),6.41)
                    elif (selected_pair == "AUDUSD"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair), 4.41)
                    elif (selected_pair == "USDJPY"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair), 5.41)
                    else:
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From 786 to ON 744 Fib, Max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Behind 786 Fib, Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),5.41)
            elif(risk_multiplier == "1_BNR_TPF"):
                if (within_61 == "Yes"):
                    if (selected_pair == "XAUUSD"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From ON TPF fib to max " + entry_pip + "$ Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Entry set to " + sl_pip + "$"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),7.41)

                        if (Variances == "> 805"):
                            entry_title = "Entry Guide (Within 64% Mandatory):"
                            entry_text = "From ON TPF fib to max " + entry_pip + "$ Distance"
                            SL_title = "SL Guide:"
                            SL_text = "Entry set to " + sl_pip + "$"
                            exit_title = "Target Guide One (RR):"
                            exit_text = compare_target(get_one_target(selected_pair),8.41)
                    elif (selected_pair in minors or selected_pair=="AUDUSD"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From ON TPF fib to max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),5.41)

                        if (Variances == "> 805"):
                            entry_title = "Entry Guide (Within 64% Mandatory):"
                            entry_text = "From ON TPF fib to max " + entry_pip + "Pips Distance"
                            SL_title = "SL Guide:"
                            SL_text = "Entry set to " + sl_pip + " Pips"
                            exit_title = "Target Guide One (RR):"
                            exit_text = compare_target(get_one_target(selected_pair),6.41)
                    elif (selected_pair =="USDJPY"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From ON TPF fib to max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),20)

                        if (Variances == "> 805"):
                            entry_title = "Entry Guide (Within 64% Mandatory):"
                            entry_text = "From ON TPF fib to max " + entry_pip + "Pips Distance"
                            SL_title = "SL Guide:"
                            SL_text = "Entry set to " + sl_pip + " Pips"
                            exit_title = "Target Guide One (RR):"
                            exit_text = compare_target(get_one_target(selected_pair),20)
                    else:
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From ON TPF fib to max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),6.41)

                        if (Variances == "> 805"):
                            entry_title = "Entry Guide (Within 64% Mandatory):"
                            entry_text = "From ON TPF fib to max " + entry_pip + "Pips Distance"
                            SL_title = "SL Guide:"
                            SL_text = "Entry set to " + sl_pip + " Pips"
                            exit_title = "Target Guide One (RR):"
                            exit_text = compare_target(get_one_target(selected_pair),7.41)


                else:
                    if (selected_pair == "XAUUSD"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From ON TPF fib to max " + entry_pip + "$ Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Entry set to " + sl_pip + "$"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),6.41)
                    elif (selected_pair in minors or selected_pair=="AUDUSD"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From ON TPF fib to max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),4.41)
                    elif (selected_pair =="USDJPY"):
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From ON TPF fib to max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),5.41)
                    else:
                        entry_title = "Entry Guide (Within 64% Optional):"
                        entry_text = "From ON TPF fib to max " + entry_pip + "Pips Distance"
                        SL_title = "SL Guide:"
                        SL_text = "Entry set to " + sl_pip + " Pips"
                        exit_title = "Target Guide One (RR):"
                        exit_text = compare_target(get_one_target(selected_pair),5.41)



            elif (risk_multiplier == "2_BNR" or risk_multiplier == "2_BNR_TPF"):
                length_title = "Leg One Length Requirement:"
                length_text = ""

                def get_sum_target():
                    sum_target = round(sum_gap/final_risk,2)
                    if (sum_target > 3):
                        return sum_target
                    else:
                        return 3
                def get_open_target(selected_pair):
                    open_target = 0
                    if(selected_pair in yens):
                        open_target = round(JPY_Pairs_gap/final_risk,2)
                    elif(selected_pair in gold_comm):
                        open_target = round(XAUUSD_gap / final_risk,2)
                    elif (selected_pair in xxxaud):
                        open_target = round(GBPAUD_EURAUD_gap / final_risk,2)
                    elif(selected_pair in europe_major):
                        open_target = round(GBPUSD_EURUSD_gap / final_risk,2)
                    elif(selected_pair in trade_curr):
                        open_target = round(USDCAD_AUDUSD_gap / final_risk,2)
                    if(open_target>3):
                        return open_target
                    else:
                        return 3
                if(selected_pair in minor_yens):
                    if (Variances == "> 805"):
                        entry_title = "Entry Guide (SL__Entry Length):"
                        entry_text = "33% from 91 Fib"
                        SL_title = "SL Guide: Head Fib must be within 31% of wick"
                        SL_text = "2 Pips behind 91 Fib"
                        exit_title = "Target Guide One (RR):"
                        exit_text = "6.41"

                    elif (Variances == "559 - 66"):
                        entry_title = "Entry Guide (SL__Entry Length):"
                        entry_text = "33% from HEAD(1) Fib"
                        SL_title = "SL Guide: Head Fib must be within 31% of wick"
                        SL_text = "On HEAD Fib"
                        exit_title = "Target Guide One (RR):"
                        exit_text = "5.41"

                    elif (Variances == "66 - 805"):
                        entry_title = "Entry Guide (SL__Entry Length):"
                        entry_text = "33% from HEAD(1) Fib"
                        SL_title = "SL Guide: Head Fib must be within 31% of wick"
                        SL_text = "On HEAD Fib"
                        exit_title = "Target Guide One (RR):"
                        exit_text = "5.41"


                else:
                    if(selected_pair not in minors and selected_pair != "USDJPY" and selected_pair not in trade_curr):
                        if(trend_position=="3%-4.99%" or trend_position=="5%-6.99%"):
                            if(big_risk_multiplier>1):
                                total_target = get_sum_target()
                                compare_target = get_open_target(selected_pair) * 1.725
                                if(compare_target<total_target):
                                    targeting = round(compare_target,2)
                                else:
                                    targeting = total_target
                            else:
                                targeting = get_open_target(selected_pair)
                        else:
                            targeting = get_open_target(selected_pair)
                    else:
                        targeting = get_open_target(selected_pair)
                    if (Variances == "> 805"):
                        entry_title = "Entry Guide (SL__Entry Length):"
                        entry_text = "33% from 91 Fib"
                        SL_title = "SL Guide:"
                        SL_text = "2 Pips behind 91 Fib"
                        exit_title = "Target Guide One (RR):"
                        exit_text = targeting

                    elif (Variances == "559 - 66"):
                        if(risk_multiplier == "2_BNR_TPF" and selected_pair in majors_dollar):
                            entry_title = "Entry Guide (SL__Entry Length):"
                            entry_text = "5 Pips before 559 or 618, 33% Max"
                            SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                            SL_text = "Middle of FL and NF (62% Max)"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting

                        else:
                            entry_title = "Entry Guide (SL__Entry Length):"
                            entry_text = "5 Pips before 618 or ON 559, 33% Max"
                            SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                            SL_text = "Middle of FL and NF (62% Max)"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting

                    elif (Variances == "66 - 805"):
                        if (risk_multiplier == "2_BNR_TPF" and selected_pair in majors_dollar):
                            entry_title = "Entry Guide (SL__Entry Length):"
                            entry_text = "5 Pips before 702, 33% Max"
                            SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                            SL_text = "Middle of FL and NF (62% Max)"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                        else:
                            entry_title = "Entry Guide (SL__Entry Length):"
                            entry_text = "5 Pips before 702, 33% Max"
                            SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                            SL_text = "Middle of FL and NF (62% Max)"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                    elif(Variances == "50"):
                        if (risk_multiplier == "2_BNR_TPF" and selected_pair in majors_dollar):
                            entry_title = "Entry Guide (SL__Entry Length): WARNING CAN ONLY ENTER WHEN 618 IS TAPPED"
                            entry_text = "ON 50, 33% Max"
                            SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                            SL_text = "Middle of FL and NF (62% Max)"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                        else:
                            entry_title = "Entry Guide (SL__Entry Length): WARNING CAN ONLY ENTER WHEN 618 IS TAPPED"
                            entry_text = "ON 50, 33% Max"
                            SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                            SL_text = "Middle of FL and NF (62% Max)"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting

                    if (selected_pair == "XAUUSD"):
                        if(trend_position=="7%-8.99%" or trend_position=="5%-6.99%"):
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
                            entry_title = "Entry Guide (SL__Entry Length):"
                            entry_text = "33% from 91 Fib"
                            SL_title = "SL Guide:"
                            SL_text = "2$ behind 91 Fib"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting
                        elif (Variances == "559 - 66"):
                            if (risk_multiplier == "2_BNR_TPF"):
                                entry_title = "Entry Guide (SL__Entry Length):"
                                entry_text = "1 $ before 559 or 618, 33% Max"
                                SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                                SL_text = "Middle of FL and NF (62% Max)"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting
                            else:
                                entry_title = "Entry Guide (SL__Entry Length):"
                                entry_text = "1 $ before 618 or ON 559, 33% Max"
                                SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                                SL_text = "Middle of FL and NF (62% Max)"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting

                        elif (Variances == "66 - 805"):
                            if (risk_multiplier == "2_BNR_TPF"):
                                entry_title = "Entry Guide (SL__Entry Length):"
                                entry_text = "1 $ before 702, 33% Max"
                                SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                                SL_text = "Middle of FL and NF (62% Max)"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting
                            else:
                                entry_title = "Entry Guide (SL__Entry Length):"
                                entry_text = "1 $ before 702, 33% Max"
                                SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                                SL_text = "Middle of FL and NF (62% Max)"
                                exit_title = "Target Guide One (RR):"
                                exit_text = targeting
                        elif (Variances == "50"):
                            entry_title = "Entry Guide (SL__Entry Length): WARNING CAN ONLY ENTER WHEN 618 IS TAPPED"
                            entry_text = "ON 50, 33% Max"
                            SL_title = "SL Guide: Measure From Top of First Leg to Next Fib"
                            SL_text = "Middle of FL and NF (62% Max)"
                            exit_title = "Target Guide One (RR):"
                            exit_text = targeting



            col1, col2, col3 = st.columns([0.03, 1, 0.5], gap="small")

            with col2:
                Risk_percentage = round(final_risk/account_balance*100,2)
                container = st.container()
                container.markdown("<div style='height: 70px; padding: 0px; margin-left: 2000px;'></div>", unsafe_allow_html=True)
                if(get_global('entry_model')!=None):
                    container.metric("--Note that all SL must not exceed 33%","Entry: " + get_global('entry_model')+" ")
                elif(get_global('entry_model')==None):
                    container.metric("--Note that all SL must not exceed 33%", "Entry Criteria Pending")

                if(monthly_loss_limit+monthly_actual_loss-final_risk<0):
                    #container.metric("Risk amount exceeded your monthly limit", "$"+ str(round(final_risk + round(monthly_loss_limit+monthly_actual_loss,2),2)))
                    container.metric("Risk amount exceeded your monthly limit", "$0 (0.0% of Account)")
                else:
                    container.metric("Your Next risk risk should be:", f"${final_risk} ({Risk_percentage}% of Account)")
                    set_global("final_risk", Risk_percentage)
                    if(position_size>0):
                                            if(position_size_propfirm>0):
                                                if(risk_multiplier == "2_BNR" or risk_multiplier == "2_BNR_TPF"):
                                                    if(pair_result == "W" or pair_result == "N"):
                                                        if(selected_pair in europe_major or selected_pair in gold_comm):
                                                            container.metric("Your Calculated lot size should be:",f"Personal {position_size} lots /" f"Propfirm {position_size_propfirm} lots")
                                                        else:
                                                            container.metric("Your Calculated lot size should be:",f"{position_size} lots")
                                                    else:
                                                        container.metric("Your Calculated lot size should be:", f"{position_size} lots")
                                                else:
                                                    container.metric("Your Calculated lot size should be:",f"{position_size} lots")
                                            else:
                                                container.metric("Your Calculated lot size should be:",f"{position_size} lots")
                    else:
                                            container.metric("Your Calculated lot size should be:", "Please Enter stop pips")

                    if(risk_multiplier == "1_BNR_TPF" or risk_multiplier == "1_BNR" or risk_multiplier == "3_BNR_TPF"):
                        container.metric(entry_title, entry_text)
                        container.metric(SL_title, SL_text)
                        container.metric(exit_title, exit_text)
                        #container.markdown("<div style='height:-2000px;'></div>", unsafe_allow_html=True)

                        #entry_price = st.number_input("Entry Price", value=0.0, format="%.5f")
                        #exit_price = st.number_input("Exit Price", value=0.0, format="%.5f")
                        #target_price = st.number_input("Target Price", value=0.0, format="%.5f")




                        # Determine if button should be disabled
                        add_order_disabled = False


                        if st.button("💾 Add Order", type="secondary", use_container_width=False,
                                     disabled=add_order_disabled):
                            # Check if Stop Loss is 0 or missing
                            if stop_pips is None or stop_pips == 0:
                                st.error("Cannot add order: Stop Loss is required and cannot be 0!")
                            else:
                                # Check if maximum records reached (only for new records, not updates)
                                existing_index = None
                                for i, existing_record in enumerate(st.session_state.saved_records):
                                    if existing_record['selected_pair'] == selected_pair:
                                        existing_index = i
                                        break

                                # If it's a new record (not updating existing) and we're at max capacity
                                if existing_index is None and len(st.session_state.saved_records) >= 5:
                                    st.error(
                                        "Maximum of 5 records reached! Please delete a record from the Records page before adding a new one.")
                                else:
                                    # Create a record with timestamp and all selections
                                    record = {
                                        'timestamp': datetime.now().strftime("%Y-%m-%d"),
                                        'selected_pair': selected_pair,
                                        'trend_position': trend_position,
                                        'POI': POI,
                                        'cross_fib': cross_fib,
                                        'HH_LL': HH_LL,
                                        'risk_multiplier': risk_multiplier,
                                        'Variances': Variances,
                                        'stop_pips': stop_pips,
                                        'within_61': within_61,
                                        'final_risk': final_risk,
                                        'position_size': position_size,
                                        'position_size_propfirm': position_size_propfirm,
                                        #'entry_price': entry_price,
                                        #'exit_price': exit_price,

                                        'status': 'Speculation'  # Default status
                                    }

                                    if existing_index is not None:
                                        # Replace existing record (this doesn't count against the limit)
                                        st.session_state.saved_records[existing_index] = record
                                        st.success("Successfully Updated Order!")
                                    else:
                                        # Add new record
                                        st.session_state.saved_records.append(record)
                                        st.success("Successfully Added Order!")

                        #container.markdown("<div style='height:220px;'></div>", unsafe_allow_html=True)
                    elif(risk_multiplier == "2_BNR" or risk_multiplier =="2_BNR_TPF"):
                        #with container:



                            #def small_metric(title, value):
                                #return st.markdown(f"""
                                       # <div style="text-align: left; padding: 0px;">
                                            #<div style="font-size: 13px; color: black; margin-bottom: 2px;">{title}</div>
                                           # <div style="font-size: 35px; color: black; font-weight: light;">{value}</div>
                                       # </div>
                                        #""", unsafe_allow_html=True)

                        container.metric(SL_title, SL_text)
                        container.metric(entry_title, entry_text)



                        container.metric(exit_title, exit_text)
                        #container.metric("Test","Test")
                        #st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

                        #entry_price = st.number_input("Entry Price", value=0.0, format="%.5f")
                        #exit_price = st.number_input("Exit Price", value=0.0, format="%.5f")
                        #target_price = st.number_input("Target Price", value=0.0, format="%.5f")




                        # Determine if button should be disabled
                        add_order_disabled = False


                        if st.button("💾 Add Order", type="secondary", use_container_width=False,
                                     disabled=add_order_disabled):
                            # Check if Stop Loss is 0 or missing
                            if stop_pips is None or stop_pips == 0:
                                st.error("Cannot add order: Stop Loss is required and cannot be 0!")
                            else:
                                # Check if maximum records reached (only for new records, not updates)
                                existing_index = None
                                for i, existing_record in enumerate(st.session_state.saved_records):
                                    if existing_record['selected_pair'] == selected_pair:
                                        existing_index = i
                                        break

                                # If it's a new record (not updating existing) and we're at max capacity
                                if existing_index is None and len(st.session_state.saved_records) >= 5:
                                    st.error(
                                        "Maximum of 5 records reached! Please delete a record from the Records page before adding a new one.")
                                else:
                                    # Create a record with timestamp and all selections
                                    record = {
                                        'timestamp': datetime.now().strftime("%Y-%m-%d"),
                                        'selected_pair': selected_pair,
                                        'trend_position': trend_position,
                                        'POI': POI,
                                        'cross_fib': cross_fib,
                                        'HH_LL': HH_LL,
                                        'risk_multiplier': risk_multiplier,
                                        'Variances': Variances,
                                        'stop_pips': stop_pips,
                                        'within_61': within_61,
                                        'final_risk': final_risk,
                                        'position_size': position_size,
                                        'position_size_propfirm': position_size_propfirm,
                                        #'entry_price': entry_price,
                                        #'exit_price': exit_price,

                                        'status': 'Speculation'  # Default status
                                    }

                                    if existing_index is not None:
                                        # Replace existing record (this doesn't count against the limit)
                                        st.session_state.saved_records[existing_index] = record
                                        st.success("Successfully Updated Order!")
                                    else:
                                        # Add new record
                                        st.session_state.saved_records.append(record)
                                        st.success("Successfully Added Order!")
                        st.markdown("<div style='height:220px;'></div>", unsafe_allow_html=True)





            #container.metric("test", get_live_rate("USDJPY"))


elif st.session_state.current_page == "Active Opps":
    st.title("Saved Records")


    # Use your existing Google Sheets functions
    def load_workflow_from_sheets():
        """Load workflow data from Google Sheets using existing functions"""
        try:
            # Use your existing load_data_from_sheets function but specify the workflow worksheet
            df = load_data_from_sheets(sheet_name="Trade", worksheet_name="Workflow")
            if df is not None and not df.empty:
                # Convert timestamp to string for consistency
                if 'timestamp' in df.columns:
                    df['timestamp'] = df['timestamp'].astype(str)

                # Clean numeric columns - convert string 'None' to actual None/0.0
                numeric_columns = ['entry_price', 'exit_price', 'target_price', 'stop_pips', 'position_size']
                for col in numeric_columns:
                    if col in df.columns:
                        # Replace string 'None' with actual None, then convert to float
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
            # Convert records to DataFrame if it's a list
            if isinstance(data, list):
                data_to_save = pd.DataFrame(data)
            else:
                data_to_save = data.copy()

            # Ensure all required columns exist
            required_columns = ['selected_pair', 'risk_multiplier', 'position_size', 'stop_pips',
                                'entry_price', 'exit_price', 'target_price', 'status', 'timestamp']

            for col in required_columns:
                if col not in data_to_save.columns:
                    data_to_save[col] = None

            # Clean data using your existing function
            data_clean = clean_data_for_google_sheets(data_to_save)

            # Save using your existing function but specify the workflow worksheet
            success = save_data_to_sheets(data_clean, sheet_name="Trade", worksheet_name="Workflow")
            return success
        except Exception as e:
            st.error(f"Error saving workflow data: {e}")
            return False


    # Trade Signal Functions - UPDATED FOR SYNC
    def sync_trade_signals_with_active_opps():
        """Sync trade signals with current Active Opps status"""
        try:
            # Initialize trade_signals in session state if not exists
            if 'trade_signals' not in st.session_state:
                st.session_state.trade_signals = []

            # Get all Order Ready records from Active Opps
            order_ready_records = [record for record in st.session_state.saved_records
                                   if record.get('status') == 'Order Ready']

            # Get timestamps of current Order Ready records
            current_order_ready_timestamps = {record.get('timestamp') for record in order_ready_records}

            # Remove trade signals that are no longer Order Ready
            st.session_state.trade_signals = [
                signal for signal in st.session_state.trade_signals
                if signal.get('timestamp') in current_order_ready_timestamps
            ]

            # Add or update trade signals for current Order Ready records
            for record in order_ready_records:
                timestamp = record.get('timestamp')
                existing_signal = next((sig for sig in st.session_state.trade_signals
                                        if sig.get('timestamp') == timestamp), None)

                if existing_signal:
                    # Update existing signal
                    existing_signal.update({
                        'selected_pair': record.get('selected_pair'),
                        'risk_multiplier': record.get('risk_multiplier'),
                        'position_size': record.get('position_size'),
                        'stop_pips': record.get('stop_pips'),
                        'entry_price': record.get('entry_price'),
                        'exit_price': record.get('exit_price'),
                        'target_price': record.get('target_price'),
                        'trend_position': record.get('trend_position', 'Not set'),
                        'variances': record.get('Variances', 'Not set'),
                        'status': 'Active'
                    })
                else:
                    # Create new trade signal
                    trade_signal = {
                        'timestamp': timestamp,
                        'selected_pair': record.get('selected_pair'),
                        'risk_multiplier': record.get('risk_multiplier'),
                        'position_size': record.get('position_size'),
                        'stop_pips': record.get('stop_pips'),
                        'entry_price': record.get('entry_price'),
                        'exit_price': record.get('exit_price'),
                        'target_price': record.get('target_price'),
                        'trend_position': record.get('trend_position', 'Not set'),
                        'variances': record.get('Variances', 'Not set'),
                        'status': 'Active',
                        'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.trade_signals.append(trade_signal)

            # Save to Google Sheets for persistence
            save_all_trade_signals_to_sheets()
            return True

        except Exception as e:
            st.error(f"Error syncing trade signals: {e}")
            return False


    def save_all_trade_signals_to_sheets():
        """Save all trade signals to Google Sheets"""
        try:
            if st.session_state.trade_signals:
                df = pd.DataFrame(st.session_state.trade_signals)
                success = save_data_to_sheets(df, sheet_name="Trade", worksheet_name="TradeSignals")
                return success
            else:
                # If no signals, save empty dataframe to clear the sheet
                empty_df = pd.DataFrame()
                success = save_data_to_sheets(empty_df, sheet_name="Trade", worksheet_name="TradeSignals")
                return success
        except Exception as e:
            print(f"Error saving trade signals to sheets: {e}")
            return False


    def load_trade_signals_from_sheets():
        """Load trade signals from Google Sheets"""
        try:
            df = load_data_from_sheets(sheet_name="Trade", worksheet_name="TradeSignals")
            if df is not None and not df.empty:
                return df.to_dict('records')
            return []
        except Exception as e:
            print(f"Error loading trade signals: {e}")
            return []


    # Helper function to safely convert record values to float
    def safe_float(value, default=0.0):
        """Safely convert value to float, handling None and string 'None'"""
        if value is None or value == 'None' or value == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


    # Load data from Google Sheets on initial load or when syncing
    if 'saved_records' not in st.session_state or st.button("🔄 Sync with Cloud"):
        workflow_data = load_workflow_from_sheets()
        if not workflow_data.empty:
            # Convert DataFrame back to list of dictionaries
            st.session_state.saved_records = workflow_data.to_dict('records')
            st.success("Workflow data loaded from cloud!")

            # Sync trade signals after loading data
            sync_trade_signals_with_active_opps()
        else:
            # Initialize empty if no data exists in cloud
            st.session_state.saved_records = []
            st.info("No data found in cloud. You can upload a CSV backup below.")

    # Initialize trade signals if not exists
    if 'trade_signals' not in st.session_state:
        st.session_state.trade_signals = load_trade_signals_from_sheets()

    # CSV Upload as fallback
    st.sidebar.markdown("---")
    with st.sidebar.expander("📁 CSV Backup", expanded=False):
        st.subheader("Upload Workflow Backup")
        uploaded_workflow_file = st.file_uploader("Upload Workflow CSV", type=['csv'], key="workflow_uploader")

        if uploaded_workflow_file is not None:
            try:
                # Load CSV with expected columns
                workflow_csv_data = pd.read_csv(uploaded_workflow_file)

                # Validate required columns
                required_columns = ['selected_pair', 'risk_multiplier', 'position_size', 'stop_pips',
                                    'entry_price', 'exit_price', 'target_price', 'status', 'timestamp']

                missing_columns = [col for col in required_columns if col not in workflow_csv_data.columns]

                if missing_columns:
                    st.error(f"Missing required columns in CSV: {', '.join(missing_columns)}")
                    st.info(
                        "Expected columns: selected_pair, risk_multiplier, position_size, stop_pips, entry_price, exit_price, target_price, status, timestamp")
                else:
                    # Clean the data - handle None values and convert numeric columns
                    for col in ['position_size', 'stop_pips', 'entry_price', 'exit_price', 'target_price']:
                        if col in workflow_csv_data.columns:
                            workflow_csv_data[col] = workflow_csv_data[col].replace('None', None)
                            workflow_csv_data[col] = pd.to_numeric(workflow_csv_data[col], errors='coerce').fillna(0.0)

                    # Ensure status values are valid
                    valid_statuses = ['Speculation', 'Order Ready', 'Order Completed']
                    workflow_csv_data['status'] = workflow_csv_data['status'].apply(
                        lambda x: x if x in valid_statuses else 'Speculation'
                    )

                    # Convert DataFrame to list of dictionaries
                    csv_records = workflow_csv_data.to_dict('records')

                    # Show preview and options
                    st.success(f"✅ Valid CSV format! Found {len(csv_records)} records")

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
                            sync_trade_signals_with_active_opps()
                            st.success(f"✅ Replaced current data with {len(csv_records)} records from CSV!")
                            st.rerun()

                    with col_merge:
                        if st.button("Merge with Current", key="merge_csv"):
                            # Merge without duplicates based on timestamp
                            current_timestamps = {r.get('timestamp') for r in st.session_state.saved_records}
                            new_records = [r for r in csv_records if r.get('timestamp') not in current_timestamps]

                            if new_records:
                                st.session_state.saved_records.extend(new_records)
                                # Sync trade signals after merging data
                                sync_trade_signals_with_active_opps()
                                st.success(f"✅ Added {len(new_records)} new records from CSV!")
                            else:
                                st.info("No new records to add (all timestamps already exist)")
                            st.rerun()

                    # Show preview of CSV data
                    with st.expander("📋 Preview CSV Data (first 5 rows)"):
                        st.dataframe(workflow_csv_data.head())

                    # Show data quality info
                    with st.expander("🔍 Data Quality Check"):
                        st.write(f"**Total records:** {len(csv_records)}")
                        st.write(f"**Unique pairs:** {workflow_csv_data['selected_pair'].nunique()}")
                        st.write(f"**Records with entry price > 0:** {(workflow_csv_data['entry_price'] > 0).sum()}")
                        st.write(f"**Records with target price > 0:** {(workflow_csv_data['target_price'] > 0).sum()}")

            except Exception as e:
                st.error(f"Error reading workflow CSV: {e}")
                st.info("Make sure your CSV has the correct format with these columns:")
                st.code(
                    "selected_pair, risk_multiplier, position_size, stop_pips, entry_price, exit_price, target_price, status, timestamp")

    # Count current records by status
    speculation_count = sum(1 for record in st.session_state.saved_records if record.get('status') == 'Speculation')
    order_ready_count = sum(1 for record in st.session_state.saved_records if record.get('status') == 'Order Ready')
    order_completed_count = sum(
        1 for record in st.session_state.saved_records if record.get('status') == 'Order Completed')
    total_active_count = order_ready_count + order_completed_count

    # Show current record counts
    st.write(f"**Total Records:** {len(st.session_state.saved_records)}/5")
    st.write(f"**Active Records (Order Ready + Order Completed):** {total_active_count}/2")
    st.write(
        f"Speculation: {speculation_count}, Order Ready: {order_ready_count}, Order Completed: {order_completed_count}")

    # Manual sync buttons with enhanced error handling
    col_sync1, col_sync2, col_export = st.columns(3)
    with col_sync1:
        if st.button("💾 Save to Cloud"):
            if st.session_state.saved_records:
                success = save_workflow_to_sheets(st.session_state.saved_records)
                if success:
                    # Also sync trade signals
                    sync_trade_signals_with_active_opps()
                    st.success("Workflow data saved to cloud!")
                else:
                    st.error("Failed to save to cloud. Use CSV export below to backup your data.")
            else:
                st.warning("No records to save")

    with col_sync2:
        if st.button("📥 Load from Cloud"):
            workflow_data = load_workflow_from_sheets()
            if not workflow_data.empty:
                st.session_state.saved_records = workflow_data.to_dict('records')
                # Sync trade signals after loading data
                sync_trade_signals_with_active_opps()
                st.success("Workflow data loaded from cloud!")
                st.rerun()
            else:
                st.error("No workflow data found in cloud. Try uploading a CSV backup.")

    with col_export:
        # Export current data to CSV
        if st.session_state.saved_records:
            csv_data = pd.DataFrame(st.session_state.saved_records)

            # Ensure we export with the correct column order
            column_order = ['selected_pair', 'risk_multiplier', 'position_size', 'stop_pips',
                            'entry_price', 'exit_price', 'target_price', 'status', 'timestamp']

            # Reorder columns and fill any missing ones
            for col in column_order:
                if col not in csv_data.columns:
                    csv_data[col] = None

            csv_data = csv_data[column_order]

            csv = csv_data.to_csv(index=False)
            st.download_button(
                label="📤 Export to CSV",
                data=csv,
                file_name=f"workflow_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download current workflow data as CSV backup",
                key="export_csv"
            )
        else:
            st.button("📤 Export to CSV", disabled=True, help="No data to export", key="export_disabled")

    if not st.session_state.saved_records:
        st.info(
            "No records saved yet. Go to Risk Calculation page and save some records, or upload a CSV backup above.")
    else:
        # Convert records to DataFrame for nice display
        records_df = pd.DataFrame(st.session_state.saved_records)
        st.dataframe(records_df, use_container_width=True)

        # WORKFLOW VISUALIZATION - Clickable stage headers
        st.markdown("---")
        st.subheader("Workflow Stages")

        # Create clickable stage headers using columns
        col1, col2, col3 = st.columns(3)

        with col1:
            # Initialize session state for current stage if not exists
            if 'current_stage' not in st.session_state:
                st.session_state.current_stage = 'Speculation'

            # Make Speculation clickable
            if st.button(f"Speculation ({speculation_count})",
                         use_container_width=True,
                         type="primary" if st.session_state.current_stage == 'Speculation' else "secondary"):
                st.session_state.current_stage = 'Speculation'
                st.rerun()

        with col2:
            # Make Order Ready clickable
            if st.button(f"Order Ready ({order_ready_count})",
                         use_container_width=True,
                         type="primary" if st.session_state.current_stage == 'Order Ready' else "secondary"):
                st.session_state.current_stage = 'Order Ready'
                st.rerun()

        with col3:
            # Make Order Completed clickable
            if st.button(f"Order Completed ({order_completed_count})",
                         use_container_width=True,
                         type="primary" if st.session_state.current_stage == 'Order Completed' else "secondary"):
                st.session_state.current_stage = 'Order Completed'
                st.rerun()

        # Progress bars below the clickable headers
        st.markdown("<br>", unsafe_allow_html=True)
        col_prog1, col_prog2, col_prog3 = st.columns(3)
        with col_prog1:
            st.progress(speculation_count / max(len(st.session_state.saved_records), 1))
        with col_prog2:
            st.progress(order_ready_count / max(len(st.session_state.saved_records), 1))
        with col_prog3:
            st.progress(order_completed_count / max(len(st.session_state.saved_records), 1))

        st.markdown("---")


        # Helper function to update record and sync to cloud
        def update_record_and_sync(original_index, updates):
            """Update record and sync to Google Sheets"""
            for key, value in updates.items():
                st.session_state.saved_records[original_index][key] = value

            # Auto-save to cloud
            success = save_workflow_to_sheets(st.session_state.saved_records)
            if success:
                # Sync trade signals after updating record
                sync_trade_signals_with_active_opps()
                st.success(f"Record updated and saved to cloud!")
            else:
                st.warning("Record updated locally but failed to save to cloud. Use CSV export to backup.")


        # Helper function to delete record and sync to cloud
        def delete_record_and_sync(original_index):
            """Delete record and sync to Google Sheets"""
            st.session_state.saved_records.pop(original_index)

            # Auto-save to cloud
            success = save_workflow_to_sheets(st.session_state.saved_records)
            if success:
                # Sync trade signals after deleting record
                sync_trade_signals_with_active_opps()
                st.success(f"Record deleted and changes saved to cloud!")
            else:
                st.warning("Record deleted locally but failed to save to cloud. Use CSV export to backup.")


        # Display records based on selected stage
        current_stage = st.session_state.current_stage
        st.subheader(f"{current_stage} Stage")

        # Get records for the current stage
        if current_stage == 'Speculation':
            stage_records = [record for record in st.session_state.saved_records if
                             record.get('status') == 'Speculation']
        elif current_stage == 'Order Ready':
            stage_records = [record for record in st.session_state.saved_records if
                             record.get('status') == 'Order Ready']
        else:  # Order Completed
            stage_records = [record for record in st.session_state.saved_records if
                             record.get('status') == 'Order Completed']

        if not stage_records:
            st.info(f"No records in {current_stage.lower()} stage.")
        else:
            for i, record in enumerate(stage_records):
                # Find the original index in the main records list
                original_index = next((idx for idx, r in enumerate(st.session_state.saved_records) if
                                       r['timestamp'] == record['timestamp']), None)

                if original_index is not None:
                    with st.expander(
                            f"Record {original_index + 1}: {record['selected_pair']} - {record['timestamp']}",
                            expanded=False):

                        # Common fields for all stages
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                        with col1:
                            st.write(f"**Pair:** {record['selected_pair']}")
                            st.write(f"**Strategy:** {record['risk_multiplier']}")
                            st.write(f"**Position Size:** {record['position_size']}")
                            st.write(f"**Current Stop Pips:** {record.get('stop_pips', 'None')}")

                        with col2:
                            new_entry_price = st.number_input(
                                "Entry Price",
                                value=safe_float(record.get('entry_price'), 0.0),
                                format="%.5f",
                                key=f"{current_stage.lower()}_entry_{original_index}"
                            )

                        with col3:
                            new_exit_price = st.number_input(
                                "Exit Price",
                                value=safe_float(record.get('exit_price'), 0.0),
                                format="%.5f",
                                key=f"{current_stage.lower()}_exit_{original_index}"
                            )

                        with col4:
                            new_target_price = st.number_input(
                                "Target Price",
                                value=safe_float(record.get('target_price'), 0.0),
                                format="%.5f",
                                key=f"{current_stage.lower()}_target_{original_index}"
                            )

                        # Stage-specific logic
                        if current_stage == 'Speculation':
                            # Calculate expected stop pips based on current entry/exit prices
                            expected_stop_pips = None
                            if new_entry_price != 0 and new_exit_price != 0:
                                if record['selected_pair'] == 'XAUUSD':
                                    expected_stop_pips = abs(new_exit_price - new_entry_price)
                                else:
                                    expected_stop_pips = abs(new_exit_price - new_entry_price) / 10

                            # Show validation message if stop pips don't match
                            current_stop_pips = record.get('stop_pips')
                            if expected_stop_pips is not None and current_stop_pips is not None:
                                # Allow small rounding differences (0.01 tolerance)
                                if abs(expected_stop_pips - current_stop_pips) > 0.01:
                                    st.error(
                                        f"Stop pips mismatch! Current: {current_stop_pips:.2f}, Expected from prices: {expected_stop_pips:.2f}")

                            # Check if all required fields are filled for moving to Order Ready
                            entry_price_valid = new_entry_price > 0
                            exit_price_valid = new_exit_price > 0
                            target_price_valid = new_target_price > 0
                            all_required_fields_valid = entry_price_valid and exit_price_valid and target_price_valid

                            # Show validation messages
                            if not entry_price_valid:
                                st.error("Entry price must be greater than 0 to move to Order Ready")
                            if not exit_price_valid:
                                st.error("Exit price must be greater than 0 to move to Order Ready")
                            if not target_price_valid:
                                st.error("Target price must be greater than 0 to move to Order Ready")

                            # Action buttons for Speculation stage
                            col_update_only, col_move, col_delete = st.columns([1, 1, 1])

                            with col_update_only:
                                if st.button(f"Update Record", key=f"spec_update_only_{original_index}"):
                                    updates = {
                                        'entry_price': new_entry_price,
                                        'exit_price': new_exit_price,
                                        'target_price': new_target_price
                                    }
                                    update_record_and_sync(original_index, updates)
                                    st.rerun()

                            with col_move:
                                update_disabled = False
                                if expected_stop_pips is not None and current_stop_pips is not None:
                                    if abs(expected_stop_pips - current_stop_pips) > 0.01:
                                        update_disabled = True

                                # Disable if required fields not filled
                                if not all_required_fields_valid:
                                    update_disabled = True

                                if st.button(f"Move to Order Ready", key=f"spec_move_{original_index}",
                                             disabled=update_disabled):
                                    # Check if updating to active status would exceed the limit
                                    if total_active_count >= 2:
                                        st.error(
                                            "Maximum of 2 active records (Order Ready + Order Completed) reached! You cannot move this record to Order Ready.")
                                    else:
                                        updates = {
                                            'entry_price': new_entry_price,
                                            'exit_price': new_exit_price,
                                            'target_price': new_target_price,
                                            'status': 'Order Ready'  # Automatically move to next stage
                                        }
                                        update_record_and_sync(original_index, updates)
                                        st.rerun()

                            with col_delete:
                                if st.button(f"🗑️ Delete", key=f"spec_delete_{original_index}"):
                                    delete_record_and_sync(original_index)
                                    st.rerun()

                        elif current_stage == 'Order Ready':
                            # Action buttons for Order Ready stage
                            col_update_only, col_move, col_delete = st.columns([1, 1, 1])

                            with col_update_only:
                                if st.button(f"Update Record", key=f"ready_update_only_{original_index}"):
                                    updates = {
                                        'entry_price': new_entry_price,
                                        'exit_price': new_exit_price,
                                        'target_price': new_target_price
                                    }
                                    update_record_and_sync(original_index, updates)
                                    st.rerun()

                            with col_move:
                                if st.button(f"Move to Order Completed", key=f"ready_move_{original_index}"):
                                    updates = {
                                        'entry_price': new_entry_price,
                                        'exit_price': new_exit_price,
                                        'target_price': new_target_price,
                                        'status': 'Order Completed'  # Automatically move to next stage
                                    }
                                    update_record_and_sync(original_index, updates)
                                    st.rerun()

                            with col_delete:
                                if st.button(f"🗑️ Delete", key=f"ready_delete_{original_index}"):
                                    delete_record_and_sync(original_index)
                                    st.rerun()

                        else:  # Order Completed
                            # Additional fields for Order Completed stage
                            col5, col6, col7, col8 = st.columns(4)

                            with col5:
                                # Result dropdown
                                result_options = ["BE", "Loss", "Win"]
                                new_result = st.selectbox(
                                    "Result",
                                    options=result_options,
                                    index=0,
                                    key=f"completed_result_{original_index}"
                                )

                            with col6:
                                # Direction dropdown
                                direction_options = ["buy", "sell"]
                                new_direction = st.selectbox(
                                    "Direction",
                                    options=direction_options,
                                    index=0,
                                    key=f"completed_direction_{original_index}"
                                )

                            with col7:
                                # RR input
                                new_rr = st.number_input(
                                    "RR",
                                    value=0.0,
                                    step=0.01,
                                    format="%.2f",
                                    key=f"completed_rr_{original_index}"
                                )

                            with col8:
                                # PnL input
                                new_pnl = st.number_input(
                                    "PnL",
                                    value=0.0,
                                    step=0.01,
                                    format="%.2f",
                                    key=f"completed_pnl_{original_index}"
                                )

                            # Additional required fields
                            col9, col10 = st.columns(2)
                            with col9:
                                # POI dropdown
                                poi_options = ["Weekly", "2_Daily"]
                                new_poi = st.selectbox(
                                    "POI",
                                    options=poi_options,
                                    index=0,
                                    key=f"completed_poi_{original_index}"
                                )

                            with col10:
                                # Strategy (pre-filled from record)
                                st.text_input(
                                    "Strategy",
                                    value=record['risk_multiplier'],
                                    key=f"completed_strategy_{original_index}",
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

                            # Action buttons for Order Completed stage
                            col_update_only, col_close, col_delete = st.columns([1, 1, 1])

                            with col_update_only:
                                if st.button(f"Update Record", key=f"completed_update_only_{original_index}"):
                                    updates = {
                                        'entry_price': new_entry_price,
                                        'exit_price': new_exit_price,
                                        'target_price': new_target_price,
                                        'result': new_result,
                                        'direction': new_direction,
                                        'rr': new_rr,
                                        'pnl': new_pnl,
                                        'poi': new_poi
                                    }
                                    update_record_and_sync(original_index, updates)
                                    st.rerun()

                            with col_close:
                                if st.button(f"Finalize & Close Trade", key=f"completed_close_{original_index}",
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
                                                'Date': datetime.now().strftime("%Y-%m-%d"),  # YYYY-MM-DD format
                                                'Symbol': record['selected_pair'],
                                                'Direction': new_direction,
                                                'Trend Position': existing_trend_position,  # Use existing value
                                                'POI': new_poi,
                                                'Strategy': record['risk_multiplier'],
                                                'Variance': existing_variance,  # Use existing value
                                                'Result': new_result,
                                                'RR': new_rr,
                                                'PnL': new_pnl,
                                                'Withdrawal_Deposit': 0.0,  # Default 0
                                                'PROP_Pct': 0.0  # Default 0
                                            }

                                            # Load current Trade.csv data
                                            current_trade_data = load_data_from_sheets(sheet_name="Trade",
                                                                                       worksheet_name="Trade.csv")

                                            if current_trade_data is not None:
                                                # Ensure only the specified columns exist in current data
                                                required_columns = ['Date', 'Symbol', 'Direction', 'Trend Position',
                                                                    'POI', 'Strategy',
                                                                    'Variance', 'Result', 'RR', 'PnL',
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

                                                # Save to Trade.csv worksheet
                                                success = save_data_to_sheets(updated_trade_data, sheet_name="Trade",
                                                                              worksheet_name="Trade.csv")

                                                if success:
                                                    # Remove from active opps
                                                    st.session_state.saved_records.pop(original_index)
                                                    # Save the updated workflow
                                                    save_workflow_to_sheets(st.session_state.saved_records)
                                                    # Sync trade signals after closing trade
                                                    sync_trade_signals_with_active_opps()
                                                    st.success("✅ Trade finalized and saved to trade history!")
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to save to trade history")
                                            else:
                                                # If no existing trade data, create new with only specified columns
                                                new_trade_df = pd.DataFrame([completed_trade])
                                                success = save_data_to_sheets(new_trade_df, sheet_name="Trade",
                                                                              worksheet_name="Trade.csv")

                                                if success:
                                                    # Remove from active opps
                                                    st.session_state.saved_records.pop(original_index)
                                                    # Save the updated workflow
                                                    save_workflow_to_sheets(st.session_state.saved_records)
                                                    # Sync trade signals after closing trade
                                                    sync_trade_signals_with_active_opps()
                                                    st.success("✅ Trade finalized and saved to trade history!")
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to save to trade history")
                                    else:
                                        st.error("Please fill in all required fields")

                            with col_delete:
                                if st.button(f"🗑️ Delete", key=f"completed_delete_{original_index}"):
                                    delete_record_and_sync(original_index)
                                    st.rerun()

        st.markdown("---")

        # Option to clear all records
        if st.button("Clear All Records", type="secondary"):
            st.session_state.saved_records = []
            success = save_workflow_to_sheets([])
            if success:
                # Also clear trade signals
                st.session_state.trade_signals = []
                save_all_trade_signals_to_sheets()
                st.success("All records cleared and saved to cloud!")
            else:
                st.warning("Records cleared locally but failed to save to cloud. Use CSV export to backup.")
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

    st.title("📡 Trade Signals")


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


    def format_symbol_for_pepperstone(symbol):
        """Add .a suffix to symbols for Pepperstone broker"""
        # Complete list of symbols that need .a suffix for Pepperstone
        pepperstone_symbols = [
            # Forex Majors
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            # Forex Minors
            'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
            'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
            'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
            'CADJPY', 'CHFJPY', 'NZDJPY',
            # Commodities
            'XAUUSD', 'XAGUSD',
            # Indices
            'USOIL', 'UKOIL', 'NAS100', 'US30', 'SPX500', 'GER30', 'UK100'
        ]

        # Check if the symbol is in our list and add .a suffix
        if symbol in pepperstone_symbols:
            return f"{symbol}.a"
        else:
            return symbol


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
                return False, "❌ MetaApi token not configured"

            # Just create the API instance - if this works, the token is valid
            api = MetaApi(token)

            return True, "✅ Connected to MetaApi successfully (token is valid)"

        except Exception as e:
            return False, f"❌ MetaApi connection error: {str(e)}"


    async def connect_metaapi_account():
        """Connect to MetaApi account"""
        try:
            account, error = await get_metaapi_account()
            if error:
                return False, error

            initial_state = account.state
            deployed_states = ['DEPLOYING', 'DEPLOYED']

            if initial_state not in deployed_states:
                # Deploy account if not deployed
                await account.deploy()

            # Wait for connection to broker
            await account.wait_connected()

            # Get RPC connection
            connection = account.get_rpc_connection()
            await connection.connect()

            # Wait for synchronization
            await connection.wait_synchronized()

            return True, f"✅ Successfully connected to account: {account.name}"

        except Exception as e:
            return False, f"❌ Connection error: {str(e)}"


    async def place_trade(symbol: str, volume: float, order_type: str, entry_price: float, sl: float, tp: float):
        """Place a LIMIT trade with MetaApi - SL and TP are MANDATORY"""
        try:
            account, error = await get_metaapi_account()
            if error:
                return False, error

            # Create a fresh connection for trading
            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            # Validate that SL and TP are provided
            if sl is None or tp is None:
                await connection.close()
                return False, "❌ Stop loss and take profit are mandatory"

            # Format symbol for Pepperstone broker (add .a suffix)
            formatted_symbol = format_symbol_for_pepperstone(symbol)

            # Place LIMIT order with mandatory SL and TP - USING SNAKE_CASE
            if order_type.upper() == "BUY":
                result = await connection.create_limit_buy_order(
                    formatted_symbol,  # Use formatted symbol with .a suffix
                    volume,
                    entry_price,  # Limit price for buy
                    stop_loss=sl,  # CORRECT: snake_case
                    take_profit=tp  # CORRECT: snake_case
                )
            else:  # SELL
                result = await connection.create_limit_sell_order(
                    formatted_symbol,  # Use formatted symbol with .a suffix
                    volume,
                    entry_price,  # Limit price for sell
                    stop_loss=sl,  # CORRECT: snake_case
                    take_profit=tp  # CORRECT: snake_case
                )

            # Close connection after trade
            await connection.close()

            return True, f"✅ Limit order placed successfully for {formatted_symbol} with SL and TP (Code: {result.get('stringCode', 'N/A')})"

        except Exception as e:
            # Try to close connection if it exists
            try:
                await connection.close()
            except:
                pass
            return False, f"❌ Trade error: {str(e)}"


    async def check_order_status(order_id: str):
        """Check if an order has been filled"""
        try:
            account, error = await get_metaapi_account()
            if error:
                return None, error

            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            # Get order information
            orders = await connection.get_orders()
            order = next((o for o in orders if o.id == order_id), None)

            await connection.close()

            if order:
                return order.state, None
            else:
                return None, "Order not found"

        except Exception as e:
            return None, f"Error checking order status: {str(e)}"


    # Add trade signal functions next - UPDATED TO SYNC WITH ACTIVE OPPS
    def load_trade_signals_from_sheets():
        """Load trade signals from Google Sheets"""
        try:
            df = load_data_from_sheets(sheet_name="Trade", worksheet_name="TradeSignals")
            if df is not None and not df.empty:
                # Clean numeric columns
                numeric_columns = ['entry_price', 'exit_price', 'target_price', 'stop_pips', 'position_size']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = df[col].replace('None', None)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

                # Calculate direction for each signal
                if 'entry_price' in df.columns and 'exit_price' in df.columns:
                    df['direction'] = df.apply(
                        lambda row: calculate_direction(row['entry_price'], row['exit_price']),
                        axis=1
                    )
                else:
                    df['direction'] = "Unknown"

                return df.to_dict('records')
            return []
        except Exception as e:
            st.error(f"Error loading trade signals: {e}")
            return []


    def sync_with_active_opps():
        """Sync trade signals with current Active Opps Order Ready records"""
        try:
            # Load current workflow data to get Order Ready records
            workflow_df = load_data_from_sheets(sheet_name="Trade", worksheet_name="Workflow")

            if workflow_df is not None and not workflow_df.empty:
                # Get Order Ready records
                order_ready_records = workflow_df[workflow_df['status'] == 'Order Ready'].to_dict('records')

                # Update session state
                st.session_state.ready_to_order = []

                for record in order_ready_records:
                    trade_signal = {
                        'timestamp': record.get('timestamp'),
                        'selected_pair': record.get('selected_pair'),
                        'risk_multiplier': record.get('risk_multiplier'),
                        'position_size': record.get('position_size'),
                        'stop_pips': record.get('stop_pips'),
                        'entry_price': record.get('entry_price'),
                        'exit_price': record.get('exit_price'),
                        'target_price': record.get('target_price'),
                        'trend_position': record.get('trend_position', 'Not set'),
                        'variances': record.get('Variances', 'Not set'),
                        'status': 'Order Ready'
                    }
                    st.session_state.ready_to_order.append(trade_signal)

                return True
            return False

        except Exception as e:
            st.error(f"Error syncing with Active Opps: {e}")
            return False


    # Initialize session states
    if 'trade_signals' not in st.session_state:
        st.session_state.trade_signals = []

    if 'metaapi_connected' not in st.session_state:
        st.session_state.metaapi_connected = False

    if 'metaapi_account_id' not in st.session_state:
        st.session_state.metaapi_account_id = "da2226bc-34b9-4304-b294-7c542551e4d3"

    if 'metaapi_connection' not in st.session_state:
        st.session_state.metaapi_connection = None

    # Initialize order tracking states
    if 'ready_to_order' not in st.session_state:
        st.session_state.ready_to_order = []

    if 'order_placed' not in st.session_state:
        st.session_state.order_placed = []

    if 'in_trade' not in st.session_state:
        st.session_state.in_trade = []

    if 'order_history' not in st.session_state:
        st.session_state.order_history = {}

    # Sync with Active Opps on page load
    if not st.session_state.ready_to_order:
        with st.spinner("🔄 Syncing with Active Opportunities..."):
            sync_with_active_opps()

    # Auto-connect to MetaApi account in background
    if not st.session_state.metaapi_connected:
        import asyncio

        try:
            # First test basic connection
            success, message = asyncio.run(test_metaapi_connection())
            if success:
                # If basic connection works, try to connect to trading account
                with st.spinner("🔗 Connecting to trading account..."):
                    success, message = asyncio.run(connect_metaapi_account())
                    if success:
                        st.session_state.metaapi_connected = True
                        st.success("✅ Automatically connected to trading account")
                    else:
                        st.session_state.metaapi_connected = False
                        st.error(f"❌ Auto-connection failed: {message}")
            else:
                st.session_state.metaapi_connected = False
                st.error(f"❌ MetaApi connection failed: {message}")
        except Exception as e:
            st.session_state.metaapi_connected = False
            st.error(f"❌ Auto-connection error: {str(e)}")

    # Connection Management
    st.subheader("🔧 Connection Management")
    col_conn1, col_conn2, col_conn3 = st.columns(3)

    with col_conn1:
        if st.button("🔄 Reconnect to Account", type="primary", use_container_width=True):
            import asyncio

            with st.spinner("Reconnecting to trading account..."):
                success, message = asyncio.run(connect_metaapi_account())
                if success:
                    st.session_state.metaapi_connected = True
                    st.success(message)
                else:
                    st.session_state.metaapi_connected = False
                    st.error(message)

    with col_conn2:
        if st.button("🔄 Refresh Signals", type="secondary", use_container_width=True):
            with st.spinner("Syncing with Active Opps..."):
                sync_with_active_opps()
            st.success(f"🔄 Synced {len(st.session_state.ready_to_order)} Order Ready signals")
            st.rerun()

    with col_conn3:
        if st.button("📊 View Active Opps", type="secondary", use_container_width=True):
            st.session_state.current_page = "Active Opps"
            st.rerun()

    # Show connection status
    if st.session_state.metaapi_connected:
        st.success("✅ Connected to trading account - Ready for trading")
    else:
        st.warning("⚠️ Not connected to trading account - Trades will not execute")

    st.markdown("---")

    # Trade Signals Display Section
    st.subheader("🎯 Active Trade Signals")

    # Export functionality
    if st.session_state.ready_to_order or st.session_state.order_placed or st.session_state.in_trade:
        all_signals = st.session_state.ready_to_order + st.session_state.order_placed + st.session_state.in_trade
        csv_data = pd.DataFrame(all_signals)
        csv = csv_data.to_csv(index=False)

        col_export1, col_export2 = st.columns(2)

        with col_export1:
            st.download_button(
                label="📤 Export Signals CSV",
                data=csv,
                file_name=f"metaapi_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_export2:
            if st.button("🔄 Sync with Google Sheets", use_container_width=True):
                with st.spinner("Syncing with Active Opps..."):
                    sync_with_active_opps()
                st.success(f"Synced {len(st.session_state.ready_to_order)} signals from Active Opps")
                st.rerun()
    else:
        st.button("📤 Export CSV", disabled=True)

    # Display trade signals in tabs
    if not st.session_state.ready_to_order and not st.session_state.order_placed and not st.session_state.in_trade:
        st.info("""
        ## 📭 No Active Trade Signals

        **How to get started:**

        1. **Add signals** in the Active Opportunities page by moving orders to **'Order Ready'**
        2. **Signals will automatically appear** here when synced
        3. **Execute trades** directly via MetaApi
        4. **Manage positions** in the MetaApi section above

        When orders are moved to **'Order Completed'**, **'Speculation'**, or **deleted** in Active Opportunities, 
        they will be automatically removed from here.
        """)
    else:
        # Create tabs for workflow
        tab1, tab2, tab3 = st.tabs([
            f"📋 Ready to Order ({len(st.session_state.ready_to_order)})",
            f"⏳ Order Placed ({len(st.session_state.order_placed)})",
            f"✅ In Trade ({len(st.session_state.in_trade)})"
        ])

        with tab1:
            st.subheader("📋 Ready to Order")
            st.info("Signals from Active Opps with 'Order Ready' status. Click 'Execute Order' to place trade.")

            if not st.session_state.ready_to_order:
                st.info("No signals ready for ordering. Check Active Opps page for 'Order Ready' records.")
            else:
                for i, signal in enumerate(st.session_state.ready_to_order):
                    with st.expander(f"🎯 {signal['selected_pair']} | {signal.get('timestamp', 'N/A')}", expanded=True):
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                        with col1:
                            st.write(f"**Instrument:** {signal['selected_pair']}")
                            st.write(f"**Strategy:** {signal.get('risk_multiplier', 'N/A')}")
                            st.write(f"**Position Size:** {signal.get('position_size', 'N/A')} lots")
                            st.write(f"**Stop Pips:** {signal.get('stop_pips', 'N/A')}")
                            st.write(f"**Trend:** {signal.get('trend_position', 'N/A')}")

                        with col2:
                            entry_price = safe_float(signal.get('entry_price'), 0.0)
                            st.write(f"**Entry Price:** {entry_price:.5f}")

                        with col3:
                            sl_price = safe_float(signal.get('exit_price'), 0.0)
                            st.write(f"**Stop Loss:** {sl_price:.5f}")

                        with col4:
                            tp_price = safe_float(signal.get('target_price'), 0.0)
                            st.write(f"**Take Profit:** {tp_price:.5f}")

                            # R:R Ratio
                            entry_val = safe_float(signal.get('entry_price'), 0.0)
                            stop_val = safe_float(signal.get('exit_price'), 0.0)
                            target_val = safe_float(signal.get('target_price'), 0.0)

                            if stop_val > 0 and target_val > 0:
                                stop_distance = abs(entry_val - stop_val)
                                target_distance = abs(entry_val - target_val)
                                if stop_distance > 0:
                                    reward_ratio = target_distance / stop_distance
                                    st.metric("R:R Ratio", f"1:{reward_ratio:.2f}")

                        # Calculate and display Direction
                        direction = calculate_direction(signal.get('entry_price'), signal.get('exit_price'))
                        direction_color = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
                        st.write(f"**Direction:** {direction_color} {direction}")

                        # Show formatted symbol for trading
                        formatted_symbol = format_symbol_for_pepperstone(signal['selected_pair'])
                        if formatted_symbol != signal['selected_pair']:
                            st.info(f"**Trading Symbol:** {formatted_symbol} (Pepperstone format)")

                        # Execution button
                        if st.session_state.metaapi_connected:
                            validation_ok = entry_price > 0 and stop_val > 0 and target_val > 0
                            if validation_ok:
                                if st.button(
                                        f"🎯 Execute {direction} Order",
                                        key=f"execute_ready_{i}",
                                        type="primary",
                                        use_container_width=True):
                                    import asyncio

                                    with st.spinner(f"Placing {direction} limit order for {formatted_symbol}..."):
                                        success, message = asyncio.run(place_trade(
                                            symbol=signal['selected_pair'],
                                            volume=float(signal.get('position_size', 0.1)),
                                            order_type=direction,
                                            entry_price=entry_price,
                                            sl=stop_val,
                                            tp=target_val
                                        ))
                                        if success:
                                            # Move from ready_to_order to order_placed
                                            st.session_state.ready_to_order = [s for s in
                                                                               st.session_state.ready_to_order if
                                                                               s['timestamp'] != signal['timestamp']]
                                            st.session_state.order_placed.append({
                                                **signal,
                                                'order_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                'order_status': 'PENDING',
                                                'direction': direction
                                            })
                                            st.success(f"✅ Order placed: {message}")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Order failed: {message}")
                            else:
                                st.warning("⚠️ Cannot execute - missing required price parameters")
                        else:
                            st.warning("🔒 Not connected to trading account")

                        # Notes section
                        if signal.get('notes'):
                            st.write("---")
                            st.write(f"**Notes:** {signal.get('notes')}")

        with tab2:
            st.subheader("⏳ Order Placed")
            st.info("Orders that have been placed but not yet filled. Check status and mark as filled when executed.")

            if not st.session_state.order_placed:
                st.info("No orders placed yet. Orders will appear here after execution from Ready to Order tab.")
            else:
                for i, order in enumerate(st.session_state.order_placed):
                    with st.expander(f"⏳ {order['selected_pair']} | Placed: {order.get('order_time', 'N/A')}",
                                     expanded=True):
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

                        # Check order status button
                        col_check, col_move = st.columns(2)
                        with col_check:
                            if st.button("🔄 Check Status", key=f"check_{i}", use_container_width=True):
                                # Simulate order fill for demo (replace with actual MetaApi status check)
                                st.success("✅ Order filled! Moving to In Trade tab.")
                                # Move to in_trade
                                st.session_state.order_placed = [o for o in st.session_state.order_placed if
                                                                 o['timestamp'] != order['timestamp']]
                                st.session_state.in_trade.append({
                                    **order,
                                    'fill_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'order_status': 'FILLED'
                                })
                                st.rerun()

                        with col_move:
                            if st.button("✅ Mark as Filled", key=f"fill_{i}", type="primary", use_container_width=True):
                                # Move to in_trade
                                st.session_state.order_placed = [o for o in st.session_state.order_placed if
                                                                 o['timestamp'] != order['timestamp']]
                                st.session_state.in_trade.append({
                                    **order,
                                    'fill_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'order_status': 'FILLED'
                                })
                                st.success("Order moved to In Trade")
                                st.rerun()

        with tab3:
            st.subheader("✅ In Trade")
            st.info("Active trades that have been filled. Monitor these positions and close when trade is completed.")

            if not st.session_state.in_trade:
                st.info("No active trades. Filled orders will appear here.")
            else:
                for i, trade in enumerate(st.session_state.in_trade):
                    with st.expander(f"✅ {trade['selected_pair']} | Filled: {trade.get('fill_time', 'N/A')}",
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

                        # Trade management buttons
                        col_close, col_cancel = st.columns(2)
                        with col_close:
                            if st.button("📊 Close Trade", key=f"close_{i}", type="primary", use_container_width=True):
                                st.session_state.in_trade = [t for t in st.session_state.in_trade if
                                                             t['timestamp'] != trade['timestamp']]
                                st.success("Trade closed and removed from active trades")
                                st.rerun()

                        with col_cancel:
                            if st.button("↩️ Move Back", key=f"back_{i}", use_container_width=True):
                                st.session_state.in_trade = [t for t in st.session_state.in_trade if
                                                             t['timestamp'] != trade['timestamp']]
                                st.session_state.order_placed.append(trade)
                                st.success("Trade moved back to Order Placed")
                                st.rerun()

elif st.session_state.current_page == "Stats":

    if st.session_state.uploaded_data is not None:
        def highlight_summary(row):
            return ['background-color: lightyellow; font-weight: bold' if row['Month'] == 'Total' else '' for _ in row]

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


                #cash_flow = withdraw_deposit.sum()

                # Calculate monthly percentage gain compared to previous month
                if prev_month_balance != 0:
                    monthly_pct_gain = ((ending_balance - prev_month_balance) / prev_month_balance) * 100
                    set_global("monthly_limit_left", starting_balance*0.08)
                else:
                    monthly_pct_gain = 0

                if(len(month_data[month_data['Result'] != 'BE'])>0):
                    winrate_month = round(len(winners) / len(month_data[month_data['Result'] != 'BE']) * 100, 1)
                    total_nobe = len(month_data[month_data['Result'] != 'BE'])
                else:
                    winrate_month = 0
                set_global("current_month",month_name)
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
                    #'Percentage_Gain': (month_pnl / starting_balance) * 100,
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
                        'Win_Rate': round((df['Wins'].sum() / (df['Total_Trades'].sum()-df['BE_Trades'].sum())) * 100, 1) if df['Total_Trades'].sum() > 0 else 0,
                        'Total_RR': df['Total_RR'].sum(),
                        'Avg_Winner_RR': df['Avg_Winner_RR'].mean(),  # Could also use weighted avg
                        'Starting_Balance': df.iloc[0]['Starting_Balance'],
                        'Ending_Trade_Balance': df.iloc[-1]['Ending_Trade_Balance'],
                        'Total_PnL': df['Total_PnL'].sum(),
                        #'Monthly_Pct_Gain': round(abs((df.iloc[0]['Starting_Balance']-df.iloc[-1]['Ending_Trade_Balance'])/df.iloc[0]['Starting_Balance'])*100,1),  # Not meaningful as a total
                        'Monthly_Pct_Gain': (df['Total_PnL'].sum())/(df.iloc[0]['Starting_Balance'])*100,
                        'Cash_Flow': df['Cash_Flow'].sum()
                    }

                df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)


            return df
            #return df
            #return pd.DataFrame(monthly_stats)
            #return pd.concat([df, pd.DataFrame([monthly_stats])], ignore_index=True)





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
    st.title("Guidelines to Follow")

    st.subheader("Length Requirement for Two Entries (Price squeeze out of 559 zone)")

    table_data = {
        '': ['2_BNR', '2_BNR_TPF'],
        'XAUUSD <66': ['1.79%', '1.79%'],
        'XAUUSD >66': ['1.19%', '1.19%'],
        'GBPUSD/EURUSD <66': ['1.49%', '1.49%'],
        'GBPUSD/EURUSD >66': ['0.99%', '0.5%'],
        'OTHER <66': ['0.99%', '0.5%'],
        'OTHER >66': ['0.99%', '0.5%']
    }

    st.table(table_data)

    st.subheader("BE Rules")
    table_data2 = {
        '': ['1_BNR', '1_BNR_TPF', '2_BNR', '2_BNR_TPF'],
        'Trigger Condition': ["For Buys 2 8H Green stick from entry, For Sells 2 8H Red Stick from entry.",
                              'For Buys 2 8H Green stick from entry, For Sells 2 8H Red Stick from entry.',
                              'Trigger at 2.5 R', 'Trigger at 2.5 R'],
        'Action': ["Trail 5 Pips Below/Above Entry", 'Trail 5 Pips Below/Above Entry', "Trail 5 Pips Below/Above Entry",
                   "Trail 5 Pips Below/Above Entry"]
    }
    st.table(table_data2)

    st.subheader("First Trail Rules")
    table_data3 = {
        '': ['1_BNR', '1_BNR_TPF', '2_BNR', '2_BNR_TPF'],
        'Trigger Condition': ["Trigger at 3R", 'Trigger at 3R', 'Trigger at 5R', 'Trigger at 5R'],
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
    st.write("2_BNR -> NO RETAKE || 2_BNR_TPF -> NO RETAKE")


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
            "impulse_leg_BOS_3":True,
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
                                        "<66":{
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

        if st.button("🔄 Start New Analysis"):
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

