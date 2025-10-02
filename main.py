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

# Configure page
st.set_page_config(layout="wide")
## Every year change starting_balance =, starting_capital = and base_risk =

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
            safe_keys = ['uploaded_data_filename', 'current_page', 'saved_records', 'file_processed']
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
        safe_keys = ['current_page', 'saved_records', 'file_processed', 'uploaded_data_filename']

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

# Load persistent data (do this once at startup)
if 'session_initialized' not in st.session_state:
    initialize_persistent_session()
    st.session_state.session_initialized = True


# Navigation with safe session saving
def create_nav_button(label, page, key):
    if st.sidebar.button(label, key=key):
        st.session_state.current_page = page
        # Use a small delay to avoid conflicts
        import time
        time.sleep(0.1)
        save_persistent_session()

# Navigation buttons
col1, col2, col3 = st.sidebar.columns([1,1,1])
create_nav_button("🏠 Home", "Home", "nav_home")
create_nav_button("📊 Account", "Account Overview", "nav_account")
create_nav_button("📊 Symbol", "Symbol Stats", "nav_symbol")
col1, col2 = st.sidebar.columns([1,1])
create_nav_button("📊 Grading", "Risk Calculation", "Risk_Calculation")
create_nav_button("📊 Active Opps", "Active Opps", "Active_Opps")
col1, col2 = st.sidebar.columns([1,1])
create_nav_button("📊 Guidelines", "Guidelines", "Guidelines")
create_nav_button("📊 Stats", "Stats", "Stats")
col1, col2 = st.sidebar.columns([1,0.5])
create_nav_button("📊 Entry Model Check", "Entry Criteria Check", "Entry_Criteria_Check")

# Add session management to your sidebar
with st.sidebar.expander("Session Management"):
    st.write(f"Current page: `{st.session_state.current_page}`")
    if st.session_state.uploaded_data is not None:
        st.write(f"Data: {len(st.session_state.uploaded_data)} rows")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Session"):
            save_persistent_session()
            st.success("Saved!")
    with col2:
        if st.button("🗑️ Clear Session"):
            clear_persistent_session()
            st.session_state.clear()
            st.rerun()


# Safe file uploader
def handle_file_upload():
    uploaded_file = st.sidebar.file_uploader("Upload data", type=['csv'], key="file_uploader")

    if uploaded_file is not None:
        try:
            # Only process if it's a new file
            if not st.session_state.file_processed:
                st.session_state.uploaded_data = pd.read_csv(uploaded_file)
                st.session_state.file_processed = True

                # Save to disk for persistence
                filename = f"uploaded_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.session_state.uploaded_data.to_csv(filename, index=False)
                st.session_state.uploaded_data_filename = filename

                # Save session
                save_persistent_session()

                st.sidebar.success("✅ File uploaded successfully!")

        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    elif st.session_state.file_processed and uploaded_file is None:
        # File was removed
        st.session_state.uploaded_data = None
        st.session_state.file_processed = False
        save_persistent_session()



# Call the file upload handler
handle_file_upload()

starting_capital = 50000



# Page content
if st.session_state.current_page == "Home":
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
        st.write("Your uploaded raw trading data:")
        # Configure grid
        gb = GridOptionsBuilder.from_dataframe(data)

        # Pagination
        gb.configure_pagination(
            paginationAutoPageSize=False,
                paginationPageSize=25,
                #paginationPageSizeSelector=[10, 25, 50, 100]
        )

        # Enable features
        gb.configure_default_column(
                filterable=True,
                sortable=True,
                resizable=True,
                editable=False
        )


        # Build options
        grid_options = gb.build()

        # Display
        st.title("Trading Data Dashboard")
        st.markdown("Use the grid below to explore and filter trading data")

        grid_response = AgGrid(
            data,
            gridOptions=grid_options,
            height=700,
            width='100%',
            theme='streamlit',  # or 'alpine', 'balham', 'material'
            update_mode=GridUpdateMode.FILTERING_CHANGED,
            allow_unsafe_jscode=True
        )

        # Show filtered data stats
        try:
            current_data = pd.DataFrame(grid_response['data'])
            st.write(f"Displaying {len(current_data)} rows")
            if len(current_data) < len(data):
                    st.success(f"Filter active: Showing {len(current_data)} of {len(data)} rows")
            else:
                    st.info("Showing complete dataset (no filters)")
        
        except Exception as e:
                st.error(f"Error processing grid response: {str(e)}")


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
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d', errors='coerce')
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
        # Access the global dictionary directly
        def get_performance_gap_data():
            """Helper function to safely retrieve performance gap data"""
            if 'performance_gap_data' in st.session_state:
                return st.session_state.performance_gap_data.copy()
            else:
                # Return empty dict if not available yet
                st.warning("Performance gap data not available yet. Please run the analysis on the main page first.")
                return {}


        # Usage example on your other page
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
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d')
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

    # Count current Limit Placed and Order Filled records
    limit_placed_count = sum(1 for record in st.session_state.saved_records if record.get('status') == 'Limit Placed')
    order_filled_count = sum(1 for record in st.session_state.saved_records if record.get('status') == 'Order Filled')
    total_active_count = limit_placed_count + order_filled_count
    speculation_count = sum(1 for record in st.session_state.saved_records if record.get('status') == 'Speculation')

    # Show current record counts
    st.write(f"**Records:** {len(st.session_state.saved_records)}/5")
    st.write(f"**Active Records (Limit Placed + Order Filled):** {total_active_count}/2")
    st.write(f"Limit Placed: {limit_placed_count}, Order Filled: {order_filled_count}")

    if not st.session_state.saved_records:
        st.info("No records saved yet. Go to Risk Calculation page and save some records.")
    else:
        # Convert records to DataFrame for nice display
        records_df = pd.DataFrame(st.session_state.saved_records)
        st.dataframe(records_df, use_container_width=True)

        # Create tabs for different status groups with counts
        tab1, tab2, tab3 = st.tabs([
            f"Speculation ({speculation_count})",
            f"Limit Placed ({limit_placed_count})",
            f"Order Filled ({order_filled_count})"
        ])

        # Speculation Tab
        with tab1:
            speculation_records = [record for record in st.session_state.saved_records if
                                   record.get('status') == 'Speculation']
            if not speculation_records:
                st.info("No speculation records.")
            else:
                st.subheader(f"Speculation Records ({len(speculation_records)})")
                for i, record in enumerate(speculation_records):
                    # Find the original index in the main records list
                    original_index = next((idx for idx, r in enumerate(st.session_state.saved_records) if
                                           r['timestamp'] == record['timestamp']), None)

                    if original_index is not None:
                        with st.expander(
                                f"Record {original_index + 1}: {record['selected_pair']} - {record['timestamp']}",expanded = True):
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                            with col1:
                                st.write(f"**Pair:** {record['selected_pair']}")
                                st.write(f"**Strategy:** {record['risk_multiplier']}")
                                st.write(f"**Position Size:** {record['position_size']}")
                                st.write(f"**Current Stop Pips:** {record.get('stop_pips', 'None')}")

                            with col2:
                                new_entry_price = st.number_input(
                                    "Entry Price",
                                    value=0.0,
                                    format="%.5f",
                                    key=f"spec_entry_{original_index}"
                                )

                            with col3:
                                new_exit_price = st.number_input(
                                    "Exit Price",
                                    value=0.0,
                                    format="%.5f",
                                    key=f"spec_exit_{original_index}"
                                )

                            with col4:
                                new_target_price = st.number_input(
                                    "Target Price",
                                    value=record.get('target_price', 0.0),
                                    format="%.5f",
                                    key=f"spec_target_{original_index}"
                                )

                            # Status dropdown in a new row
                            status_options = ["Speculation", "Limit Placed", "Order Filled"]
                            current_status = record.get('status', 'Speculation')

                            # Check if target price is valid for active status
                            target_price_valid = new_target_price > 0

                            # Disable active status options if conditions not met
                            disable_active_status = False
                            warning_message = ""

                            if total_active_count >= 2 and current_status == 'Speculation':
                                disable_active_status = True
                                warning_message = "Maximum active records reached. Cannot change to Limit Placed or Order Filled."
                            elif not target_price_valid:
                                disable_active_status = True
                                warning_message = "Target price must be greater than 0 to change to Limit Placed or Order Filled."

                            if disable_active_status:
                                new_status = st.selectbox(
                                    "Status",
                                    status_options,
                                    index=status_options.index(
                                        current_status) if current_status in status_options else 0,
                                    disabled=True,
                                    key=f"spec_status_{original_index}"
                                )
                                st.warning(warning_message)
                            else:
                                new_status = st.selectbox(
                                    "Status",
                                    status_options,
                                    index=status_options.index(
                                        current_status) if current_status in status_options else 0,
                                    key=f"spec_status_{original_index}"
                                )

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

                            # Update button for this record
                            col_update, col_delete = st.columns(2)
                            with col_update:
                                update_disabled = False
                                if expected_stop_pips is not None and current_stop_pips is not None:
                                    if abs(expected_stop_pips - current_stop_pips) > 0.01:
                                        update_disabled = True

                                if st.button(f"Update Record", key=f"spec_update_{original_index}",
                                             disabled=update_disabled):
                                    # Additional validation for active status
                                    if new_status in ['Limit Placed', 'Order Filled'] and new_target_price <= 0:
                                        st.error("Cannot change to active status: Target price must be greater than 0!")
                                    else:
                                        # Check if updating to active status would exceed the limit
                                        current_status = record.get('status', 'Speculation')
                                        if new_status in ['Limit Placed',
                                                          'Order Filled'] and current_status == 'Speculation':
                                            if total_active_count >= 2:
                                                st.error(
                                                    "Maximum of 2 active records (Limit Placed + Order Filled) reached! You cannot change this record to active status.")
                                            else:
                                                st.session_state.saved_records[original_index][
                                                    'entry_price'] = new_entry_price
                                                st.session_state.saved_records[original_index][
                                                    'exit_price'] = new_exit_price
                                                st.session_state.saved_records[original_index][
                                                    'target_price'] = new_target_price
                                                st.session_state.saved_records[original_index]['status'] = new_status
                                                st.success(f"Record updated for {record['selected_pair']}!")
                                                st.rerun()
                                        else:
                                            st.session_state.saved_records[original_index][
                                                'entry_price'] = new_entry_price
                                            st.session_state.saved_records[original_index][
                                                'exit_price'] = new_exit_price
                                            st.session_state.saved_records[original_index][
                                                'target_price'] = new_target_price
                                            st.session_state.saved_records[original_index]['status'] = new_status
                                            st.success(f"Record updated for {record['selected_pair']}!")
                                            st.rerun()

                            with col_delete:
                                if st.button(f"Delete Record", key=f"spec_delete_{original_index}"):
                                    st.session_state.saved_records.pop(original_index)
                                    st.success(f"Record {original_index + 1} deleted successfully!")
                                    st.rerun()

        # Limit Placed Tab
        with tab2:
            limit_records = [record for record in st.session_state.saved_records if
                             record.get('status') == 'Limit Placed']
            if not limit_records:
                st.info("No limit placed records.")
            else:
                st.subheader(f"Limit Placed Records ({len(limit_records)})")
                for i, record in enumerate(limit_records):
                    original_index = next((idx for idx, r in enumerate(st.session_state.saved_records) if
                                           r['timestamp'] == record['timestamp']), None)

                    if original_index is not None:
                        with st.expander(
                                f"Record {original_index + 1}: {record['selected_pair']} - {record['timestamp']}",expanded = True):
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                            with col1:
                                st.write(f"**Pair:** {record['selected_pair']}")
                                st.write(f"**Strategy:** {record['risk_multiplier']}")
                                st.write(f"**Position Size:** {record['position_size']}")
                                st.write(f"**Current Stop Pips:** {record.get('stop_pips', 'None')}")

                            with col2:
                                new_entry_price = st.number_input(
                                    "Entry Price",
                                    value=0.0,
                                    format="%.5f",
                                    key=f"limit_entry_{original_index}"
                                )

                            with col3:
                                new_exit_price = st.number_input(
                                    "Exit Price",
                                    value=record['exit_price'],
                                    format="%.5f",
                                    key=f"limit_exit_{original_index}"
                                )

                            with col4:
                                new_target_price = st.number_input(
                                    "Target Price",
                                    value=record.get('target_price', 0.0),
                                    format="%.5f",
                                    key=f"limit_target_{original_index}"
                                )

                            # Status dropdown in a new row
                            status_options = ["Speculation", "Limit Placed", "Order Filled"]
                            current_status = record.get('status', 'Limit Placed')
                            new_status = st.selectbox(
                                "Status",
                                status_options,
                                index=status_options.index(current_status) if current_status in status_options else 1,
                                key=f"limit_status_{original_index}"
                            )

                            # Update button for this record
                            col_update, col_delete = st.columns(2)
                            with col_update:
                                if st.button(f"Update Record", key=f"limit_update_{original_index}"):
                                    st.session_state.saved_records[original_index]['entry_price'] = new_entry_price
                                    st.session_state.saved_records[original_index]['exit_price'] = new_exit_price
                                    st.session_state.saved_records[original_index]['target_price'] = new_target_price
                                    st.session_state.saved_records[original_index]['status'] = new_status
                                    st.success(f"Record updated for {record['selected_pair']}!")
                                    st.rerun()

                            with col_delete:
                                if st.button(f"Delete Record", key=f"limit_delete_{original_index}"):
                                    st.session_state.saved_records.pop(original_index)
                                    st.success(f"Record {original_index + 1} deleted successfully!")
                                    st.rerun()

        # Order Filled Tab
        with tab3:
            filled_records = [record for record in st.session_state.saved_records if
                              record.get('status') == 'Order Filled']
            if not filled_records:
                st.info("No order filled records.")
            else:
                st.subheader(f"Order Filled Records ({len(filled_records)})")
                for i, record in enumerate(filled_records):
                    original_index = next((idx for idx, r in enumerate(st.session_state.saved_records) if
                                           r['timestamp'] == record['timestamp']), None)

                    if original_index is not None:
                        with st.expander(
                                f"Record {original_index + 1}: {record['selected_pair']} - {record['timestamp']}",expanded = True):
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                            with col1:
                                st.write(f"**Pair:** {record['selected_pair']}")
                                st.write(f"**Strategy:** {record['risk_multiplier']}")
                                st.write(f"**Position Size:** {record['position_size']}")
                                st.write(f"**Current Stop Pips:** {record.get('stop_pips', 'None')}")

                            with col2:
                                new_entry_price = st.number_input(
                                    "Entry Price",
                                    value=0.0,
                                    format="%.5f",
                                    key=f"filled_entry_{original_index}"
                                )

                            with col3:
                                new_exit_price = st.number_input(
                                    "Exit Price",
                                    value=0.0,
                                    format="%.5f",
                                    key=f"filled_exit_{original_index}"
                                )

                            with col4:
                                new_target_price = st.number_input(
                                    "Target Price",
                                    value=record.get('target_price', 0.0),
                                    format="%.5f",
                                    key=f"filled_target_{original_index}"
                                )

                            # Status dropdown in a new row
                            status_options = ["Speculation", "Limit Placed", "Order Filled"]
                            current_status = record.get('status', 'Order Filled')
                            new_status = st.selectbox(
                                "Status",
                                status_options,
                                index=status_options.index(current_status) if current_status in status_options else 2,
                                key=f"filled_status_{original_index}"
                            )

                            # Update button for this record
                            col_update, col_delete = st.columns(2)
                            with col_update:
                                if st.button(f"Update Record", key=f"filled_update_{original_index}"):
                                    st.session_state.saved_records[original_index]['entry_price'] = new_entry_price
                                    st.session_state.saved_records[original_index]['exit_price'] = new_exit_price
                                    st.session_state.saved_records[original_index]['target_price'] = new_target_price
                                    st.session_state.saved_records[original_index]['status'] = new_status
                                    st.success(f"Record updated for {record['selected_pair']}!")
                                    st.rerun()

                            with col_delete:
                                if st.button(f"Delete Record", key=f"filled_delete_{original_index}"):
                                    st.session_state.saved_records.pop(original_index)
                                    st.success(f"Record {original_index + 1} deleted successfully!")
                                    st.rerun()

        st.markdown("---")

        # Option to clear all records
        if st.button("Clear All Records", type="secondary"):
            st.session_state.saved_records = []
            st.success("All records cleared!")
            st.rerun()


elif st.session_state.current_page == "Stats":

    if st.session_state.uploaded_data is not None:
        def highlight_summary(row):
            return ['background-color: lightyellow; font-weight: bold' if row['Month'] == 'Total' else '' for _ in row]

        df = st.session_state.uploaded_data
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d')
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
