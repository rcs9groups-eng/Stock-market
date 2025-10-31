import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config FIRST - must be first Streamlit command
st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #2563eb;
    }
    .buy-signal { border-left-color: #10b981; background: #f0fdf4; }
    .sell-signal { border-left-color: #ef4444; background: #fef2f2; }
    .hold-signal { border-left-color: #f59e0b; background: #fffbeb; }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self):
        self.all_symbols = {
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS', 
            'INFY': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'ITC': 'ITC.NS',
            'HUL': 'HINDUNILVR.NS'
        }
    
    def get_stock_data(self, symbol, period="6mo"):
        """Get stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data if not data.empty else None
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    def calculate_indicators(self, data):
        """Calculate basic technical indicators"""
        if data is None or len(data) < 20:
            return None
            
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume SMA
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
        return df

    def calculate_score(self, df):
        """Calculate trading score"""
        if df is None:
            return 0, []
            
        current_price = df['Close'].iloc[-1]
        score = 0
        reasons = []
        
        # Trend analysis
        if 'SMA_20' in df and 'SMA_50' in df:
            if current_price > df['SMA_20'].iloc[-1]:
                score += 25
                reasons.append("✅ Above 20-day SMA")
            if current_price > df['SMA_50'].iloc[-1]:
                score += 25
                reasons.append("✅ Above 50-day SMA")
        
        # RSI analysis
        if 'RSI' in df:
            rsi = df['RSI'].iloc[-1]
            if 30 <= rsi <= 70:
                score += 20
                reasons.append(f"✅ RSI in good range: {rsi:.1f}")
            elif rsi < 30:
                score += 25
                reasons.append(f"🎯 RSI oversold: {rsi:.1f}")
        
        # MACD analysis
        if 'MACD' in df and 'MACD_Signal' in df:
            if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                score += 20
                reasons.append("✅ MACD bullish")
        
        # Volume analysis
        if 'Volume' in df and 'Volume_SMA_20' in df:
            volume_ratio = df['Volume'].iloc[-1] / df['Volume_SMA_20'].iloc[-1]
            if volume_ratio > 1.2:
                score += 10
                reasons.append(f"✅ High volume: {volume_ratio:.1f}x")
        
        return min(score, 100), reasons

    def get_signal(self, score):
        """Get trading signal"""
        if score >= 80:
            return "STRONG BUY", "buy-signal"
        elif score >= 60:
            return "BUY", "buy-signal"
        elif score >= 40:
            return "HOLD", "hold-signal"
        else:
            return "SELL", "sell-signal"

    def create_chart(self, df, symbol):
        """Create stock chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Add moving averages
        if 'SMA_20' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
        if 'SMA_50' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
        
        # Volume chart
        colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)
        return fig

def main():
    # Simple authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔒 Stock Analyzer Pro")
        st.write("Please enter the password to continue")
        
        password = st.text_input("Password:", type="password")
        if st.button("Login"):
            if password == "StockMaster2024":  # Change this password
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        return
    
    # Main application
    analyzer = StockAnalyzer()
    
    st.title("📈 Stock Analyzer Pro")
    st.write("Advanced technical analysis for Indian stocks")
    
    # Stock selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_stock = st.selectbox("Select Stock:", list(analyzer.all_symbols.keys()))
        symbol = analyzer.all_symbols[selected_stock]
        
        # Manual symbol
        custom_symbol = st.text_input("Or enter custom symbol:")
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        # Analysis parameters
        st.subheader("Settings")
        stop_loss = st.slider("Stop Loss %", 1.0, 20.0, 5.0)
        target = st.slider("Target %", 1.0, 50.0, 15.0)
        
        if st.button("Analyze Stock", type="primary"):
            with st.spinner("Analyzing..."):
                data = analyzer.get_stock_data(symbol)
                
                if data is not None and not data.empty:
                    df = analyzer.calculate_indicators(data)
                    
                    if df is not None:
                        score, reasons = analyzer.calculate_score(df)
                        signal, signal_class = analyzer.get_signal(score)
                        current_price = df['Close'].iloc[-1]
                        
                        # Display results
                        st.markdown(f'<div class="card {signal_class}">', unsafe_allow_html=True)
                        st.subheader(f"Signal: {signal}")
                        st.metric("AI Score", f"{score}/100")
                        st.metric("Current Price", f"₹{current_price:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Trading recommendation
                        stop_loss_price = current_price * (1 - stop_loss/100)
                        target_price = current_price * (1 + target/100)
                        
                        st.subheader("📊 Trading Plan")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Entry", f"₹{current_price:.2f}")
                        with col_b:
                            st.metric("Stop Loss", f"₹{stop_loss_price:.2f}")
                        with col_c:
                            st.metric("Target", f"₹{target_price:.2f}")
                        
                        # Reasons
                        st.subheader("Analysis Details")
                        for reason in reasons:
                            st.write(f"• {reason}")
    
    with col2:
        if 'df' in locals():
            st.plotly_chart(analyzer.create_chart(df, selected_stock), use_container_width=True)
    
    # Sidebar
    st.sidebar.header("Quick Actions")
    if st.sidebar.button("Scan Top Stocks"):
        with st.spinner("Scanning..."):
            results = []
            for stock_name, stock_symbol in list(analyzer.all_symbols.items())[:5]:
                try:
                    data = analyzer.get_stock_data(stock_symbol, "3mo")
                    if data is not None:
                        df = analyzer.calculate_indicators(data)
                        if df is not None:
                            score, _ = analyzer.calculate_score(df)
                            if score >= 70:
                                results.append((stock_name, score))
                except:
                    continue
            
            if results:
                st.subheader("🔥 Top Picks")
                for stock, score in sorted(results, key=lambda x: x[1], reverse=True)[:3]:
                    st.write(f"**{stock}** - Score: {score}/100")
    
    # Logout
    st.sidebar.header("Account")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

if __name__ == "__main__":
    main()
