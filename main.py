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
    col_conn1, col_conn2 = st.columns(2)

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
            cloud_signals = load_trade_signals_from_sheets()
            st.session_state.trade_signals = cloud_signals
            st.success(f"🔄 Synced {len(cloud_signals)} trade signals")
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
        # Removed Signals Overview section

        # Detailed view with execution
        st.subheader("📋 Signal Details & Execution")

        for i, signal in enumerate(st.session_state.trade_signals):
            with st.expander(f"🎯 Signal {i + 1}: {signal['selected_pair']} | {signal.get('timestamp', 'N/A')}",
                             expanded=True):
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

                    # Execution button in col2 below Entry Price
                    if st.session_state.metaapi_connected:
                        validation_ok = entry_price > 0 and safe_float(signal.get('exit_price'),
                                                                       0.0) > 0 and safe_float(
                            signal.get('target_price'), 0.0) > 0
                        if validation_ok:
                            if st.button(
                                    f"🎯 Execute {calculate_direction(signal.get('entry_price'), signal.get('exit_price'))} Order",
                                    key=f"execute_{i}",
                                    type="primary",
                                    use_container_width=True):
                                import asyncio

                                direction = calculate_direction(signal.get('entry_price'), signal.get('exit_price'))
                                formatted_symbol = format_symbol_for_pepperstone(signal['selected_pair'])
                                with st.spinner(f"Placing {direction} limit order for {formatted_symbol}..."):
                                    success, message = asyncio.run(place_trade(
                                        symbol=signal['selected_pair'],
                                        volume=float(signal.get('position_size', 0.1)),
                                        order_type=direction,
                                        entry_price=entry_price,
                                        sl=safe_float(signal.get('exit_price'), 0.0),
                                        tp=safe_float(signal.get('target_price'), 0.0)
                                    ))
                                    if success:
                                        st.success(message)
                                    else:
                                        st.error(message)

                with col3:
                    sl_price = safe_float(signal.get('exit_price'), 0.0)
                    st.write(f"**Stop Loss:** {sl_price:.5f}")

                    # Position size removed from here

                with col4:
                    tp_price = safe_float(signal.get('target_price'), 0.0)
                    st.write(f"**Take Profit:** {tp_price:.5f}")

                    # R:R Ratio in col4 below Take Profit - CHANGED TO 1:XX format
                    entry_val = safe_float(signal.get('entry_price'), 0.0)
                    stop_val = safe_float(signal.get('exit_price'), 0.0)
                    target_val = safe_float(signal.get('target_price'), 0.0)

                    if stop_val > 0 and target_val > 0:
                        stop_distance = abs(entry_val - stop_val)
                        target_distance = abs(entry_val - target_val)
                        if stop_distance > 0:
                            reward_ratio = target_distance / stop_distance
                            st.metric("R:R Ratio", f"1:{reward_ratio:.2f}")  # CHANGED TO 1:XX format

                # Calculate and display Direction
                direction = calculate_direction(signal.get('entry_price'), signal.get('exit_price'))
                direction_color = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
                st.write(f"**Direction:** {direction_color} {direction}")

                # Show formatted symbol for trading
                formatted_symbol = format_symbol_for_pepperstone(signal['selected_pair'])
                if formatted_symbol != signal['selected_pair']:
                    st.info(f"**Trading Symbol:** {formatted_symbol} (Pepperstone format)")

                # Connection status and validation messages
                if not st.session_state.metaapi_connected:
                    st.warning("🔒 Not connected to trading account")
                    st.info(
                        "The system will automatically try to reconnect. You can also manually reconnect using the button above.")
                else:
                    # Validation check
                    validation_ok = True
                    if entry_val <= 0:
                        st.error("❌ Entry price missing or invalid")
                        validation_ok = False
                    if stop_val <= 0:
                        st.error("❌ Stop loss price missing or invalid")
                        validation_ok = False
                    if target_val <= 0:
                        st.error("❌ Take profit price missing or invalid")
                        validation_ok = False

                    if not validation_ok:
                        st.warning("⚠️ Cannot execute trade - missing required parameters")

                # Variance Analysis section REMOVED

                if signal.get('notes'):
                    st.write("---")
                    st.write(f"**Notes:** {signal.get('notes')}")
