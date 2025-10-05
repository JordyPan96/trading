elif st.session_state.current_page == "Trade Signal":
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
        """Test connection to MetaAPI"""
        try:
            from metaapi_cloud_sdk import MetaApi
            
            config = get_metaapi_config()
            token = config.get("token", "")
            
            if not token:
                return False, "❌ MetaApi token not configured"
            
            api = MetaApi(token)
            # Test by trying to get accounts list
            accounts = await api.metatrader_account_api.get_accounts()
            return True, f"✅ Connected to MetaApi successfully. Found {len(accounts)} accounts"
                
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
                st.info("Deploying account...")
                await account.deploy()

            st.info("Waiting for API server to connect to broker...")
            await account.wait_connected()

            # Get RPC connection
            connection = account.get_rpc_connection()
            await connection.connect()
            
            # Wait for synchronization
            st.info("Waiting for SDK to synchronize to terminal state...")
            await connection.wait_synchronized()
            
            return True, f"✅ Successfully connected to account: {account.name}"
            
        except Exception as e:
            return False, f"❌ Connection error: {str(e)}"

    async def get_account_information():
        """Get detailed account information"""
        try:
            account, error = await get_metaapi_account()
            if error:
                return None
            
            connection = account.get_rpc_connection()
            if not connection.connected:
                await connection.connect()
                await connection.wait_synchronized()
            
            return await connection.get_account_information()
        except Exception as e:
            st.error(f"Error getting account info: {str(e)}")
            return None

    async def get_positions():
        """Get open positions"""
        try:
            account, error = await get_metaapi_account()
            if error:
                return []
            
            connection = account.get_rpc_connection()
            if not connection.connected:
                await connection.connect()
                await connection.wait_synchronized()
            
            return await connection.get_positions()
        except Exception as e:
            st.error(f"Error getting positions: {str(e)}")
            return []

    async def get_symbols():
        """Get available symbols"""
        try:
            account, error = await get_metaapi_account()
            if error:
                return []
            
            connection = account.get_rpc_connection()
            if not connection.connected:
                await connection.connect()
                await connection.wait_synchronized()
            
            # Get symbols from server time or other method
            # For now, return common symbols
            common_symbols = [
                'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
                'EURGBP', 'EURJPY', 'EURCHF', 'GBPJPY', 'XAUUSD', 'XAGUSD'
            ]
            return common_symbols
        except Exception as e:
            st.error(f"Error getting symbols: {str(e)}")
            return []

    async def place_trade(symbol: str, volume: float, order_type: str, sl: float = None, tp: float = None):
        """Place a trade with MetaApi"""
        try:
            account, error = await get_metaapi_account()
            if error:
                return False, error
            
            connection = account.get_rpc_connection()
            if not connection.connected:
                await connection.connect()
                await connection.wait_synchronized()
            
            # Place market order
            if order_type.upper() == "BUY":
                result = await connection.create_market_buy_order(
                    symbol, 
                    volume,
                    stop_loss=sl,
                    take_profit=tp,
                    comment=f"Trade from Streamlit App {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    client_id=f"ST_{symbol}_{int(datetime.now().timestamp())}"
                )
            else:  # SELL
                result = await connection.create_market_sell_order(
                    symbol, 
                    volume,
                    stop_loss=sl,
                    take_profit=tp,
                    comment=f"Trade from Streamlit App {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    client_id=f"ST_{symbol}_{int(datetime.now().timestamp())}"
                )
            
            return True, f"✅ Trade executed successfully (Code: {result.get('stringCode', 'N/A')})"
                
        except Exception as e:
            return False, f"❌ Trade error: {str(e)}"

    # Add trade signal functions next
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

    # Initialize session states
    if 'trade_signals' not in st.session_state:
        st.session_state.trade_signals = []

    if 'metaapi_connected' not in st.session_state:
        st.session_state.metaapi_connected = False

    if 'metaapi_account_id' not in st.session_state:
        st.session_state.metaapi_account_id = "da2226bc-34b9-4304-b294-7c542551e4d3"

    if 'metaapi_connection' not in st.session_state:
        st.session_state.metaapi_connection = None

    # Load from Google Sheets on page load
    if not st.session_state.trade_signals:
        st.session_state.trade_signals = load_trade_signals_from_sheets()

    # Connection Status Section
    st.subheader("🔗 MetaApi Connection")

    col_status1, col_status2, col_status3 = st.columns(3)

    with col_status1:
        status_color_metaapi = "🟢" if st.session_state.metaapi_connected else "🔴"
        status_text_metaapi = "Connected" if st.session_state.metaapi_connected else "Disconnected"
        st.metric("MetaApi Status", f"{status_color_metaapi} {status_text_metaapi}")

    with col_status2:
        status_color_account = "🟢" if st.session_state.metaapi_account_id else "🔴"
        status_text_account = "Connected" if st.session_state.metaapi_account_id else "No Account"
        st.metric("Trading Account", f"{status_color_account} {status_text_account}")

    with col_status3:
        # Display signal count
        signal_count = len(st.session_state.trade_signals)
        status_color_signals = "🟢" if signal_count > 0 else "⚪"
        st.metric("Active Signals", f"{status_color_signals} {signal_count}")

    # Connection Management
    col_conn1, col_conn2 = st.columns(2)

    with col_conn1:
        if st.button("🔄 Check MetaApi Connection", type="primary"):
            import asyncio
            success, message = asyncio.run(test_metaapi_connection())
            if success:
                st.session_state.metaapi_connected = True
                st.success(message)
            else:
                st.session_state.metaapi_connected = False
                st.error(message)

    with col_conn2:
        if st.button("🔄 Refresh Signals", type="secondary"):
            cloud_signals = load_trade_signals_from_sheets()
            st.session_state.trade_signals = cloud_signals
            st.success(f"🔄 Synced {len(cloud_signals)} trade signals")
            st.rerun()

    # MetaApi Account Connection Section
    st.subheader("🔐 MetaApi Account Setup")

    with st.expander("Account Configuration", expanded=not st.session_state.metaapi_connected):
        config = get_metaapi_config()
        token = config.get("token", "")
        
        if not token:
            st.error("❌ MetaApi token not found in secrets.toml")
            st.info("Please add your MetaApi token to secrets.toml under [metaapi] section")
        else:
            st.success("✅ MetaApi token configured")
            
            # Display current account ID
            st.info(f"**Account ID:** {st.session_state.metaapi_account_id}")
            
            # Manual connection
            if st.button("🔗 Connect to Account", type="primary"):
                import asyncio
                with st.spinner("Connecting to MetaApi account..."):
                    success, message = asyncio.run(connect_metaapi_account())
                    if success:
                        st.session_state.metaapi_connected = True
                        st.success(message)
                    else:
                        st.error(message)

    # MetaApi Account Information
    if st.session_state.metaapi_connected:
        st.success(f"✅ MetaApi Account {st.session_state.metaapi_account_id} is connected")

        # Display account information and symbols in columns
        col_info1, col_info2 = st.columns(2)

        with col_info1:
            st.subheader("📊 Account Overview")
            if st.button("🔄 Refresh Account Info"):
                import asyncio
                account_info = asyncio.run(get_account_information())
                if account_info:
                    st.write(f"**Balance:** ${account_info.get('balance', 'N/A')}")
                    st.write(f"**Equity:** ${account_info.get('equity', 'N/A')}")
                    st.write(f"**Margin:** ${account_info.get('margin', 'N/A')}")
                    st.write(f"**Free Margin:** ${account_info.get('freeMargin', 'N/A')}")
                    st.write(f"**Margin Level:** {account_info.get('marginLevel', 'N/A')}%")
                else:
                    st.info("Click 'Refresh Account Info' to load account information")

        with col_info2:
            st.subheader("💱 Available Symbols")
            if st.button("🔄 Refresh Symbols"):
                import asyncio
                symbols = asyncio.run(get_symbols())
                if symbols:
                    # Display symbols in a compact format
                    cols = st.columns(4)
                    for i, symbol in enumerate(symbols[:12]):
                        with cols[i % 4]:
                            st.code(symbol)
                else:
                    st.info("Common trading symbols will be shown here")

        # Open Positions
        st.subheader("📈 Open Positions")
        if st.button("🔄 Refresh Positions"):
            import asyncio
            positions = asyncio.run(get_positions())
            if positions:
                positions_df = pd.DataFrame(positions)
                st.dataframe(positions_df, use_container_width=True)

                # Calculate totals
                if 'profit' in positions_df.columns:
                    total_profit = positions_df['profit'].sum()
                    total_positions = len(positions)

                    col_pos1, col_pos2 = st.columns(2)
                    with col_pos1:
                        st.metric("Total Positions", total_positions)
                    with col_pos2:
                        st.metric("Total P/L", f"${total_profit:.2f}")
            else:
                st.info("No open positions found")

    st.markdown("---")

    # Trade Signals Display Section
    st.subheader("🎯 Active Trade Signals")

    # Export functionality
    if st.session_state.trade_signals:
        # Ensure direction column exists before exporting
        export_data = []
        for signal in st.session_state.trade_signals:
            signal_copy = signal.copy()
            if 'direction' not in signal_copy:
                signal_copy['direction'] = calculate_direction(
                    signal_copy.get('entry_price'),
                    signal_copy.get('exit_price')
                )
            export_data.append(signal_copy)

        csv_data = pd.DataFrame(export_data)
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
                cloud_signals = load_trade_signals_from_sheets()
                st.session_state.trade_signals = cloud_signals
                st.success(f"Synced {len(cloud_signals)} signals from Google Sheets")
                st.rerun()
    else:
        st.button("📤 Export CSV", disabled=True)

    # Display trade signals
    if not st.session_state.trade_signals:
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
        st.success(f"🎯 Found {len(st.session_state.trade_signals)} active trade signals ready for execution")

        # Convert to DataFrame for nice display
        signals_data = []
        for signal in st.session_state.trade_signals:
            signal_copy = signal.copy()
            if 'direction' not in signal_copy:
                signal_copy['direction'] = calculate_direction(
                    signal_copy.get('entry_price'),
                    signal_copy.get('exit_price')
                )
            signals_data.append(signal_copy)

        signals_df = pd.DataFrame(signals_data)

        # Display compact overview
        st.subheader("Signals Overview")
        display_columns = ['selected_pair', 'direction', 'entry_price', 'exit_price', 'position_size', 'timestamp']
        available_columns = [col for col in display_columns if col in signals_df.columns]

        st.dataframe(signals_df[available_columns], use_container_width=True)

        # Detailed view with execution
        st.subheader("📋 Signal Details & Execution")

        for i, signal in enumerate(st.session_state.trade_signals):
            with st.expander(f"🎯 Signal {i + 1}: {signal['selected_pair']} | {signal.get('timestamp', 'N/A')}", expanded=True):
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
                    exit_price = safe_float(signal.get('exit_price'), 0.0)
                    st.write(f"**Stop Loss:** {exit_price:.5f}")

                with col4:
                    target_price = safe_float(signal.get('target_price'), 0.0)
                    st.write(f"**Take Profit:** {target_price:.5f}")

                # Calculate and display Direction
                direction = calculate_direction(signal.get('entry_price'), signal.get('exit_price'))
                direction_color = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
                st.write(f"**Direction:** {direction_color} {direction}")

                # Trading Execution Section
                st.markdown("---")
                st.subheader("🚀 Trade Execution")

                if st.session_state.metaapi_connected:
                    col_exec1, col_exec2, col_exec3 = st.columns([2, 1, 1])

                    with col_exec1:
                        # Trade execution button
                        if st.button(f"🎯 Execute {direction} Order",
                                     key=f"execute_{i}",
                                     type="primary",
                                     use_container_width=True):
                            import asyncio
                            with st.spinner(f"Executing {direction} order for {signal['selected_pair']}..."):
                                success, message = asyncio.run(place_trade(
                                    symbol=signal['selected_pair'],
                                    volume=float(signal.get('position_size', 0.1)),
                                    order_type=direction,
                                    sl=safe_float(signal.get('exit_price'), 0.0),
                                    tp=safe_float(signal.get('target_price'), 0.0)
                                ))
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)

                    with col_exec2:
                        # Quick edit position size
                        new_size = st.number_input(
                            "Size (lots)",
                            min_value=0.01,
                            max_value=100.0,
                            value=float(signal.get('position_size', 0.1)),
                            step=0.01,
                            key=f"size_{i}"
                        )

                    with col_exec3:
                        # Risk calculator
                        entry = safe_float(signal.get('entry_price'), 0.0)
                        stop = safe_float(signal.get('exit_price'), 0.0)
                        if entry > 0 and stop > 0:
                            risk_pips = abs(entry - stop) * 10000
                            st.metric("Risk", f"{risk_pips:.1f} pips")
                else:
                    st.warning("🔒 Please connect to a MetaApi account to execute trades")
                    st.info("Go to the MetaApi Account Setup section above to connect your account")

                # Additional signal information
                if signal.get('variances'):
                    st.write(f"**Variance Analysis:** {signal.get('variances', 'N/A')}")

                if signal.get('notes'):
                    st.write("---")
                    st.write(f"**Notes:** {signal.get('notes')}")

                # Risk Management Information
                st.write("---")
                st.subheader("📊 Risk Management")

                entry_val = safe_float(signal.get('entry_price'), 0.0)
                stop_val = safe_float(signal.get('exit_price'), 0.0)
                target_val = safe_float(signal.get('target_price'), 0.0)

                if entry_val > 0 and stop_val > 0 and target_val > 0:
                    col_risk1, col_risk2, col_risk3 = st.columns(3)

                    with col_risk1:
                        stop_distance = abs(entry_val - stop_val)
                        if signal['selected_pair'] == 'XAUUSD':
                            st.metric("Stop Distance", f"${stop_distance:.2f}")
                        else:
                            st.metric("Stop Distance", f"{stop_distance * 10000:.1f} pips")

                    with col_risk2:
                        target_distance = abs(entry_val - target_val)
                        if signal['selected_pair'] == 'XAUUSD':
                            st.metric("Target Distance", f"${target_distance:.2f}")
                        else:
                            st.metric("Target Distance", f"{target_distance * 10000:.1f} pips")

                    with col_risk3:
                        if stop_distance > 0:
                            reward_ratio = target_distance / stop_distance
                            st.metric("R:R Ratio", f"{reward_ratio:.2f}:1")

    # Installation reminder
    with st.expander("🔧 Setup Instructions", expanded=False):
        st.info("""
        **To use MetaApi, you need to install the SDK:**
        
        ```bash
        pip install metaapi-cloud-sdk
        ```
        
        **Your configuration:**
        - Account ID: `da2226bc-34b9-4304-b294-7c542551e4d3`
        - Token: Configured in secrets.toml
        
        **Features:**
        - ✅ Account deployment and connection
        - ✅ Real-time account information
        - ✅ Position management
        - ✅ Trade execution
        - ✅ Symbol information
        """)
