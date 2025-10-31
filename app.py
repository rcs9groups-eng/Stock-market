import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Stock Analyzer", layout="wide")

def main():
    st.title("🚀 STOCK ANALYZER")
    
    # Input
    col1, col2 = st.columns([1, 2])
    
    with col1:
        symbol = st.text_input("Stock Symbol:", "RELIANCE.NS")
        
        if st.button("ANALYZE STOCK"):
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                data = stock.history(period="3mo")
                
                if data.empty:
                    st.error("No data found for this symbol")
                    return
                
                # Calculate basic indicators
                data['MA_20'] = data['Close'].rolling(20).mean()
                data['MA_50'] = data['Close'].rolling(50).mean()
                
                # RSI Calculation
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                current_price = data['Close'].iloc[-1]
                
                # Display results
                st.success(f"✅ Analysis Complete!")
                st.metric("Current Price", f"₹{current_price:.2f}")
                st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
                
                # Create chart
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    subplot_titles=('Price Chart', 'RSI'),
                    vertical_spacing=0.1
                )
                
                # Price chart
                fig.add_trace(go.Candlestick(
                    x=data.index, open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'], name="Price"
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], 
                                       name='MA 20', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], 
                                       name='MA 50', line=dict(color='green')), row=1, col=1)
                
                # RSI chart
                fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                                       name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
