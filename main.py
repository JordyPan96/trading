col1, col2 = st.columns([2, 1])
with col1:
    if st.button("🔄 Refresh Red News", key="refresh_red_news_json", use_container_width=True):
        with st.spinner("Checking for high impact news..."):
            st.session_state.red_events = get_red_news_from_json_with_rate_limit()
            st.session_state.last_news_fetch = datetime.now(timezone.utc)

with col2:
    if st.session_state.last_news_fetch:
        st.write(f"Last: {st.session_state.last_news_fetch.strftime('%H:%M UTC')}")
    else:
        st.write("Click refresh")

# Initial load
if not st.session_state.red_events:
    with st.spinner("Loading high impact news..."):
        st.session_state.red_events = get_red_news_from_json_with_rate_limit()
        st.session_state.last_news_fetch = datetime.now(timezone.utc)

red_events = st.session_state.red_events

# Debug info to check how many events were fetched
st.write(f"Fetched {len(red_events)} high-impact news events")

if red_events:
    now_utc = datetime.now(timezone.utc)
    now_melb = now_utc.astimezone(melbourne_tz)
    today_melb = now_melb.date()
    yesterday = today_melb - timedelta(days=1)
    tomorrow = today_melb + timedelta(days=1)
    valid_dates = {yesterday, today_melb, tomorrow}

    filtered = []
    for e in red_events:
        try:
            # Parse event date
            event_date = datetime.strptime(e['Date'], '%Y-%m-%d').date()
            if event_date not in valid_dates:
                continue

            # Safer ISO datetime parsing
            dt_utc = isoparse(e['TimeUTC'])
            dt_local = dt_utc.astimezone(melbourne_tz)

            # Commented out time filtering to avoid missing events
            # if dt_local > now_melb:
            filtered.append(e)

        except Exception as ex:
            st.write(f"Error parsing event: {ex}")
            continue

    if filtered:
        with st.expander("Upcoming Red News", expanded=False):
            highlight_keywords = [
                "FOMC", "Cash Rate", "Interest Rate", "Unemployment Rate",
                "GDP", "Non-Farm", "CPI", "election", "non farm", "PMI"
            ]

            for e in filtered:
                try:
                    dt_utc = isoparse(e['TimeUTC'])
                    dt_local = dt_utc.astimezone(melbourne_tz)
                    date_str = dt_local.strftime('%Y-%m-%d')
                    time_str = dt_local.strftime('%I:%M %p')  # 12-hour with AM/PM
                    datetime_display = f"{date_str} {time_str}"
                except Exception:
                    datetime_display = "N/A"

                event_name = e['Event']
                should_highlight = any(keyword.lower() in event_name.lower() for keyword in highlight_keywords)

                if should_highlight:
                    st.markdown("""
                        <div style="
                            background-color: #ff4444;
                            color: white;
                            padding: 12px;
                            border-radius: 10px;
                            margin-bottom: 10px;
                        ">
                    """, unsafe_allow_html=True)

                st.write(f"**{datetime_display}** - **[{e['Currency']}] {event_name}**")

                details = []
                if e.get('Forecast') and e['Forecast'] != 'N/A':
                    details.append(f"Forecast: {e['Forecast']}")
                if e.get('Previous') and e['Previous'] != 'N/A':
                    details.append(f"Previous: {e['Previous']}")
                if e.get('Actual') and e['Actual'] != 'N/A':
                    details.append(f"Actual: {e['Actual']}")
                if details:
                    st.caption(" | ".join(details))

                st.markdown("""
                    <div style="
                      background: #ff0000;
                      color: white;
                      padding: 4px 8px;
                      border-radius: 12px;
                      font-size: 12px;
                      text-align: center;
                      font-weight: bold;
                      width: 50px;
                    ">
                      HIGH
                    </div>
                """, unsafe_allow_html=True)

                if should_highlight:
                    st.markdown("</div>", unsafe_allow_html=True)

                st.divider()
    else:
        st.info("No high-impact events found for yesterday, today or tomorrow.")
else:
    st.info("No high-impact events found.")
