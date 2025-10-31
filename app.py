import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Stock Analyzer", layout="wide")

def calculate_advanced_indicators(data):
    """Advanced indicators manual calculation"""
    df = data.copy()
    
    # Multiple Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{period}'] = df['Close'].rolling(period).mean()
    
    # MACD Calculation
    exp12 = df['Close'].ewm(span=12).mean()
    exp26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume Indicators
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    
    return df

def calculate_ai_score(df):
    """AI Scoring System"""
    if df is None:
        return 0, []
    
    current_price = df['Close'].iloc[-1]
    score = 0
    reasons = []
    
    # Trend Analysis
    if current_price > df['MA_20'].iloc[-1]:
        score += 20
        reasons.append("✅ Above 20-day MA")
    if current_price > df['MA_50'].iloc[-1]:
        score += 20
        reasons.append("✅ Above 50-day MA")
    
    # RSI Analysis
    rsi = df['RSI'].iloc[-1]
    if 40 <= rsi <= 60:
        score += 20
        reasons.append(f"✅ Good RSI: {rsi:.1f}")
    elif rsi < 30:
        score += 25
        reasons.append(f"📈 Oversold RSI: {rsi:.1f}")
    
    # MACD Analysis
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
        score += 20
        reasons.append("✅ MACD Bullish")
    
    # Volume Analysis
    if df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1]:
        score += 15
        reasons.append("💰 High Volume")
    
    return min(score, 100), reasons

def main():
    st.title("🚀 ADVANCED STOCK ANALYZER")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        symbol = st.text_input("Stock Symbol:", "RELIANCE.NS")
        
        if st.button("🚀 ADVANCED ANALYSIS"):
            try:
                # Get data
                stock = yf.Ticker(symbol)
                data = stock.history(period="6mo")
                
                if data.empty:
                    st.error("No data found")
                    return
                
                # Calculate RSI
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                # Calculate advanced indicators
                data = calculate_advanced_indicators(data)
                
                # AI Scoring
                score, reasons = calculate_ai_score(data)
                current_price = data['Close'].iloc[-1]
                
                # Display Results
                st.success(f"✅ AI Score: {score}/100")
                st.metric("Current Price", f"₹{current_price:.2f}")
                
                # Trading Signal
                if score >= 80:
                    st.markdown("**🎯 SIGNAL: STRONG BUY**")
                elif score >= 60:
                    st.markdown("**📈 SIGNAL: BUY**")
                elif score >= 40:
                    st.markdown("**🔄 SIGNAL: HOLD**")
                else:
                    st.markdown("**📉 SIGNAL: SELL**")
                
                # Reasons
                st.subheader("Analysis Reasons:")
                for reason in reasons:
                    st.write(f"• {reason}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if 'data' in locals() and not data.empty:
            # Advanced Chart
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price & MAs', 'Volume', 'RSI', 'MACD'),
                row_heights=[0.4, 0.15, 0.2, 0.25]
            )
            
            # Price Chart
            fig.add_trace(go.Candlestick(
                x=data.index, open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'], name="Price"
            ), row=1, col=1)
            
            # Moving Averages
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], 
                                   name='MA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], 
                                   name='MA 50', line=dict(color='green')), row=1, col=1)
            
            # Volume
            colors = ['#00C805' if row['Close'] >= row['Open'] else '#FF0000' 
                     for _, row in data.iterrows()]
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], 
                               name='Volume', marker_color=colors), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                                   name='RSI', line=dict(color='purple')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                                   name='MACD', line=dict(color='blue')), row=4, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], 
                                   name='Signal', line=dict(color='red')), row=4, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], 
                               name='Histogram', marker_color='orange'), row=4, col=1)
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
