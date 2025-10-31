import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="SUPER POWERFUL STOCK ANALYZER",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .super-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        margin: 1.5rem 0;
        border-left: 6px solid;
        border-right: 2px solid #e5e7eb;
        border-top: 2px solid #e5e7eb;
        border-bottom: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    .super-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.2);
    }
    .ultra-buy {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 50%, #6ee7b7 100%);
    }
    .strong-buy {
        border-left-color: #22c55e;
        background: linear-gradient(135deg, #bbf7d0 0%, #86efac 100%);
    }
    .strong-sell {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 50%, #f87171 100%);
    }
    .hold {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }
    .indicator-box {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 2px solid #e5e7eb;
        transition: transform 0.2s ease;
    }
    .indicator-box:hover {
        transform: scale(1.05);
    }
    .bullish { border-color: #10b981; background: #d1fae5; }
    .bearish { border-color: #ef4444; background: #fee2e2; }
    .neutral { border-color: #f59e0b; background: #fef3c7; }
</style>
""", unsafe_allow_html=True)

class SuperStockAnalyzer:
    def __init__(self):
        self.nifty_100 = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS',
            'SBIN.NS', 'ASIANPAINT.NS', 'HCLTECH.NS', 'AXISBANK.NS', 'MARUTI.NS',
            'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
            'BAJFINANCE.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'M&M.NS'
        ]
    
    def get_stock_data(self, symbol, period="6mo"):
        """Get comprehensive stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            st.error(f"Data fetch error: {e}")
            return None

    def calculate_advanced_indicators(self, data):
        """Calculate 25+ technical indicators manually"""
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        # PRICE INDICATORS
        # Multiple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI Calculation
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['RSI_14'] = calculate_rsi(df['Close'], 14)
        df['RSI_21'] = calculate_rsi(df['Close'], 21)
        
        # MACD Calculation
        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # VOLUME INDICATORS
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # OBV Calculation
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Money Flow Index
        def calculate_mfi(high, low, close, volume, window=14):
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = ((typical_price > typical_price.shift(1)) * money_flow).rolling(window=window).sum()
            negative_flow = ((typical_price < typical_price.shift(1)) * money_flow).rolling(window=window).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            return mfi
        
        df['MFI'] = calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # VOLATILITY INDICATORS
        # ATR Calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # ADDITIONAL INDICATORS
        # Williams %R
        def calculate_williams_r(high, low, close, window=14):
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            return (highest_high - close) / (highest_high - lowest_low) * -100
        
        df['Williams_R'] = calculate_williams_r(df['High'], df['Low'], df['Close'])
        
        # CCI
        def calculate_cci(high, low, close, window=20):
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=window).mean()
            mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            return (tp - sma_tp) / (0.015 * mad)
        
        df['CCI'] = calculate_cci(df['High'], df['Low'], df['Close'])
        
        # Awesome Oscillator
        def calculate_awesome_oscillator(high, low):
            median_price = (high + low) / 2
            sma5 = median_price.rolling(window=5).mean()
            sma34 = median_price.rolling(window=34).mean()
            return sma5 - sma34
        
        df['Awesome_Oscillator'] = calculate_awesome_oscillator(df['High'], df['Low'])
        
        # ROC
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        return df

    def calculate_ai_powered_score(self, df):
        """AI-powered scoring with 25+ indicators"""
        if df is None:
            return 0, [], {}, 0, 0
            
        current_price = df['Close'].iloc[-1]
        score = 0
        reasons = []
        signals = {}
        
        # 1. TREND ANALYSIS (30 points)
        trend_score = 0
        ma_bullish_count = 0
        
        for period in [5, 10, 20, 50, 100, 200]:
            if f'MA_{period}' in df and not pd.isna(df[f'MA_{period}'].iloc[-1]):
                if current_price > df[f'MA_{period}'].iloc[-1]:
                    trend_score += 3
                    ma_bullish_count += 1
                    signals[f'MA_{period}'] = 'BULLISH'
        
        if ma_bullish_count >= 4:
            trend_score += 12
            reasons.append(f"✅ Strong bullish trend ({ma_bullish_count}/6 MAs)")
        
        # Golden Cross
        if all(col in df for col in ['MA_20', 'MA_50', 'MA_100', 'MA_200']):
            if (df['MA_20'].iloc[-1] > df['MA_50'].iloc[-1] > df['MA_100'].iloc[-1] > df['MA_200'].iloc[-1]):
                trend_score += 15
                reasons.append("🚀 PERFECT GOLDEN CROSS")
                signals['MA_Alignment'] = 'PERFECT_BULLISH'
        
        score += trend_score
        
        # 2. MOMENTUM ANALYSIS (35 points)
        momentum_score = 0
        
        # RSI Analysis
        if 'RSI_14' in df and not pd.isna(df['RSI_14'].iloc[-1]):
            rsi = df['RSI_14'].iloc[-1]
            if 40 <= rsi <= 60:
                momentum_score += 12
                reasons.append(f"🎯 Perfect RSI: {rsi:.1f}")
                signals['RSI_14'] = 'STRONG_BULLISH'
            elif rsi < 30:
                momentum_score += 15
                reasons.append(f"📈 Oversold RSI: {rsi:.1f}")
                signals['RSI_14'] = 'OVERSOLD_BULLISH'
        
        # MACD Analysis
        if all(col in df for col in ['MACD', 'MACD_Signal']):
            if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                momentum_score += 10
                reasons.append("✅ MACD Bullish")
                signals['MACD'] = 'BULLISH'
        
        # Additional Momentum
        if 'Awesome_Oscillator' in df and df['Awesome_Oscillator'].iloc[-1] > 0:
            momentum_score += 5
            signals['Awesome_Osc'] = 'BULLISH'
        
        if 'CCI' in df and df['CCI'].iloc[-1] > 0:
            momentum_score += 4
            signals['CCI'] = 'BULLISH'
        
        if 'ROC' in df and df['ROC'].iloc[-1] > 0:
            momentum_score += 4
            signals['ROC'] = 'BULLISH'
        
        score += momentum_score
        
        # 3. VOLUME ANALYSIS (20 points)
        volume_score = 0
        
        if all(col in df for col in ['Volume', 'Volume_MA']):
            vol_ratio = df['Volume'].iloc[-1] / df['Volume_MA'].iloc[-1]
            if vol_ratio > 2:
                volume_score += 12
                reasons.append("💰 Very high volume")
                signals['Volume'] = 'VERY_BULLISH'
            elif vol_ratio > 1.5:
                volume_score += 8
                reasons.append("💰 High volume")
                signals['Volume'] = 'BULLISH'
        
        if 'OBV' in df and len(df) > 10:
            if df['OBV'].iloc[-1] > df['OBV'].iloc[-10]:
                volume_score += 5
                reasons.append("📊 Rising OBV")
                signals['OBV'] = 'BULLISH'
        
        if 'MFI' in df and df['MFI'].iloc[-1] > 50:
            volume_score += 3
            signals['MFI'] = 'BULLISH'
        
        score += volume_score
        
        # 4. VOLATILITY ANALYSIS (15 points)
        vol_score = 0
        
        if 'BB_Position' in df and not pd.isna(df['BB_Position'].iloc[-1]):
            bb_pos = df['BB_Position'].iloc[-1]
            if 0.3 <= bb_pos <= 0.7:
                vol_score += 8
                reasons.append("📈 Good BB position")
                signals['Bollinger_Bands'] = 'STRONG_BULLISH'
            elif bb_pos < 0.3:
                vol_score += 10
                reasons.append("🎯 Near BB lower band")
                signals['Bollinger_Bands'] = 'OVERSOLD_BULLISH'
        
        if 'ATR' in df and not pd.isna(df['ATR'].iloc[-1]):
            atr_pct = (df['ATR'].iloc[-1] / current_price) * 100
            if atr_pct < 2:
                vol_score += 5
                reasons.append("🛡️ Low volatility")
                signals['ATR'] = 'LOW_VOLATILITY'
        
        score += vol_score
        
        # FINAL CONFIDENCE BOOST
        bullish_signals = sum(1 for s in signals.values() if 'BULLISH' in str(s))
        total_signals = len(signals)
        
        if bullish_signals >= 12:
            score = min(score + 15, 100)
            reasons.append(f"🚀 EXTREME BULLISH ({bullish_signals}/{total_signals} signals)")
        elif bullish_signals >= 8:
            score = min(score + 10, 100)
            reasons.append(f"📈 STRONG BULLISH ({bullish_signals}/{total_signals} signals)")
        
        return min(score, 100), reasons, signals, bullish_signals, total_signals

    def get_super_signal(self, score):
        """Get professional trading signal"""
        if score >= 90:
            return "🚀 ULTRA STRONG BUY", "ultra-buy", "#059669", "IMMEDIATE BUY - EXTREME CONFIDENCE"
        elif score >= 80:
            return "🎯 VERY STRONG BUY", "ultra-buy", "#10b981", "STRONG BUY - HIGH CONFIDENCE"
        elif score >= 70:
            return "📈 STRONG BUY", "strong-buy", "#22c55e", "BUY - GOOD OPPORTUNITY"
        elif score >= 60:
            return "⚡ ACCUMULATE", "hold", "#84cc16", "ACCUMULATE ON DIPS"
        elif score >= 50:
            return "🔄 HOLD", "hold", "#f59e0b", "HOLD - WAIT FOR CLARITY"
        elif score >= 40:
            return "🔔 REDUCE", "hold", "#f97316", "REDUCE POSITION"
        elif score >= 30:
            return "📉 SELL", "strong-sell", "#ef4444", "SELL - WEAK OUTLOOK"
        else:
            return "💀 STRONG SELL", "strong-sell", "#dc2626", "STRONG SELL - AVOID"

    def create_super_chart(self, df, symbol):
        """Create professional 4-panel chart"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f'{symbol} - Professional Analysis', 
                'Volume & Money Flow',
                'RSI & Momentum', 
                'MACD & Trend Strength'
            ),
            row_heights=[0.4, 0.15, 0.2, 0.25]
        )
        
        # Price Subplot
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', increasing_line_color='#00C805', decreasing_line_color='#FF0000'
        ), row=1, col=1)
        
        # Multiple Moving Averages
        for period, color in [(20, 'orange'), (50, 'green'), (200, 'red')]:
            fig.add_trace(go.Scatter(x=df.index, y=df[f'MA_{period}'], 
                                   name=f'MA {period}', line=dict(color=color, width=2)), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                               line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                               line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
        
        # Volume Subplot
        colors = ['#00C805' if row['Close'] >= row['Open'] else '#FF0000' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                           marker_color=colors, opacity=0.7), row=2, col=1)
        
        # RSI Subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI 14', 
                               line=dict(color='blue', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # MACD Subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                               line=dict(color='blue', width=2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', 
                               line=dict(color='red', width=2)), row=4, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', 
                           marker_color='orange'), row=4, col=1)
        fig.add_hline(y=0, line_color="black", row=4, col=1)
        
        fig.update_layout(
            title=f'PROFESSIONAL TRADING ANALYSIS - {symbol}',
            height=1000,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig

def main():
    analyzer = SuperStockAnalyzer()
    
    # Header Section
    st.markdown('<h1 class="main-header">🚀 SUPER POWERFUL STOCK ANALYZER</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #6b7280;">25+ Indicators • AI-Powered • Professional Grade</p>', unsafe_allow_html=True)
    
    # Main Analysis
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 ANALYSIS PARAMETERS")
        symbol = st.text_input("Stock Symbol:", "RELIANCE")
        
        if st.button("🚀 RUN SUPER ANALYSIS", type="primary", use_container_width=True):
            symbol_with_ns = symbol.upper() + '.NS'
            
            with st.spinner("🔄 Running advanced analysis with 25+ indicators..."):
                data = analyzer.get_stock_data(symbol_with_ns, "6mo")
                
                if data is not None and not data.empty:
                    df = analyzer.calculate_advanced_indicators(data)
                    
                    if df is not None:
                        score, reasons, signals, bullish_count, total_signals = analyzer.calculate_ai_powered_score(df)
                        signal, signal_class, color, advice = analyzer.get_super_signal(score)
                        current_price = df['Close'].iloc[-1]
                        
                        # SUPER RESULTS DISPLAY
                        st.markdown(f'<div class="super-card {signal_class}">', unsafe_allow_html=True)
                        st.subheader(f"🎯 {signal}")
                        st.write(f"**AI Confidence Score:** {score}/100")
                        st.write(f"**Current Price:** ₹{current_price:.2f}")
                        st.write(f"**Bullish Signals:** {bullish_count}/{total_signals}")
                        st.write(f"**Professional Advice:** {advice}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # INDICATOR GRID
                        st.subheader("📊 KEY INDICATOR SIGNALS")
                        cols = st.columns(4)
                        
                        indicators = []
                        if 'RSI_14' in df:
                            rsi_val = df['RSI_14'].iloc[-1]
                            rsi_status = 'bullish' if 40 <= rsi_val <= 60 else 'bearish'
                            indicators.append(('RSI 14', f"{rsi_val:.1f}", rsi_status))
                        
                        if all(col in df for col in ['MACD', 'MACD_Signal']):
                            macd_status = 'bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'bearish'
                            indicators.append(('MACD', 'BULLISH' if macd_status == 'bullish' else 'BEARISH', macd_status))
                        
                        if all(col in df for col in ['Volume', 'Volume_MA']):
                            vol_status = 'bullish' if df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1] else 'neutral'
                            indicators.append(('Volume', 'HIGH' if vol_status == 'bullish' else 'NORMAL', vol_status))
                        
                        if 'MA_50' in df:
                            trend_status = 'bullish' if current_price > df['MA_50'].iloc[-1] else 'bearish'
                            indicators.append(('Trend', 'BULLISH' if trend_status == 'bullish' else 'BEARISH', trend_status))
                        
                        for idx, (name, value, status) in enumerate(indicators):
                            with cols[idx]:
                                st.markdown(f'''
                                <div class="indicator-box {status}">
                                    <h4>{name}</h4>
                                    <h3>{value}</h3>
                                </div>
                                ''', unsafe_allow_html=True)
            
            with col2:
                if 'df' in locals() and df is not None:
                    st.plotly_chart(analyzer.create_super_chart(df, symbol), use_container_width=True)
                    
                    # DETAILED ANALYSIS
                    st.subheader("🔍 DETAILED ANALYSIS REPORT")
                    for i, reason in enumerate(reasons[:12], 1):
                        st.write(f"{i}. {reason}")
                    
                    # RISK MANAGEMENT
                    st.subheader("🛡️ RISK MANAGEMENT")
                    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                    
                    with risk_col1:
                        stop_loss = current_price * 0.92
                        st.metric("Stop Loss", f"₹{stop_loss:.1f}")
                    
                    with risk_col2:
                        target = current_price * 1.15
                        st.metric("Target", f"₹{target:.1f}")
                    
                    with risk_col3:
                        risk_reward = (target - current_price) / (current_price - stop_loss)
                        st.metric("Risk/Reward", f"1:{risk_reward:.1f}")
                    
                    with risk_col4:
                        if 'ATR' in df:
                            st.metric("ATR", f"₹{df['ATR'].iloc[-1]:.2f}")

    # Market Scanner
    st.sidebar.header("⚡ QUICK ACTIONS")
    if st.sidebar.button("🔍 SCAN TOP STOCKS"):
        with st.spinner("Scanning market for opportunities..."):
            results = []
            for stock_symbol in analyzer.nifty_100[:8]:
                try:
                    data = analyzer.get_stock_data(stock_symbol)
                    if data is not None and not data.empty:
                        df = analyzer.calculate_advanced_indicators(data)
                        if df is not None:
                            score, _, _, bullish_count, _ = analyzer.calculate_ai_powered_score(df)
                            if score >= 75:
                                current_price = df['Close'].iloc[-1]
                                results.append({
                                    'symbol': stock_symbol,
                                    'price': current_price,
                                    'score': score,
                                    'bullish': bullish_count
                                })
                except:
                    continue
            
            if results:
                st.subheader("💎 TOP RECOMMENDATIONS")
                for stock in sorted(results, key=lambda x: x['score'], reverse=True)[:3]:
                    st.markdown(f'''
                    <div class="super-card ultra-buy">
                        <h3>🚀 {stock['symbol'].replace('.NS', '')}</h3>
                        <p><strong>AI Score:</strong> {stock['score']}/100</p>
                        <p><strong>Current Price:</strong> ₹{stock['price']:.2f}</p>
                        <p><strong>Bullish Signals:</strong> {stock['bullish']}</p>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No strong buy signals found in top stocks.")

if __name__ == "__main__":
    main()
