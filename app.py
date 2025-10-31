import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
            'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS'
        ]
    
    def get_stock_data(self, symbol, period="6mo"):
        """Get stock data from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    def calculate_indicators(self, data):
        """Calculate all technical indicators using pure Python"""
        if data is None or len(data) < 20:
            return None
            
        df = data.copy()
        
        # Price-based Indicators
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI Calculation
        def calculate_rsi(price_data, window=14):
            delta = price_data.diff()
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
        
        # Volume Indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # OBV Calculation
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # ATR Calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Additional Indicators
        # Stochastic RSI
        def stoch_rsi(rsi, window=14):
            lowest_rsi = rsi.rolling(window=window).min()
            highest_rsi = rsi.rolling(window=window).max()
            return (rsi - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
        
        df['Stoch_RSI'] = stoch_rsi(df['RSI_14'])
        
        # Williams %R
        def williams_r(high, low, close, window=14):
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            return (highest_high - close) / (highest_high - lowest_low) * -100
        
        df['Williams_R'] = williams_r(df['High'], df['Low'], df['Close'])
        
        # CCI
        def cci(high, low, close, window=20):
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=window).mean()
            mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            return (tp - sma_tp) / (0.015 * mad)
        
        df['CCI'] = cci(df['High'], df['Low'], df['Close'])
        
        # Awesome Oscillator
        def awesome_oscillator(high, low):
            median_price = (high + low) / 2
            sma5 = median_price.rolling(window=5).mean()
            sma34 = median_price.rolling(window=34).mean()
            return sma5 - sma34
        
        df['Awesome_Oscillator'] = awesome_oscillator(df['High'], df['Low'])
        
        # ROC
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # Money Flow Index
        def mfi(high, low, close, volume, window=14):
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = ((typical_price > typical_price.shift(1)) * money_flow).rolling(window=window).sum()
            negative_flow = ((typical_price < typical_price.shift(1)) * money_flow).rolling(window=window).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            return mfi
        
        df['MFI'] = mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # ADX
        def adx(high, low, close, window=14):
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = true_range.rolling(window=window).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
            
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.rolling(window=window).mean()
            
            return adx, plus_di, minus_di
        
        df['ADX'], df['ADX_Pos'], df['ADX_Neg'] = adx(df['High'], df['Low'], df['Close'])
        
        return df

    def calculate_ai_score(self, df):
        """AI-powered scoring system"""
        if df is None:
            return 0, [], {}, 0, 0
            
        current_price = df['Close'].iloc[-1]
        score = 0
        reasons = []
        signals = {}
        
        # 1. Trend Analysis (25 points)
        trend_score = 0
        ma_bullish = 0
        
        for period in [5, 10, 20, 50, 100, 200]:
            ma_col = f'MA_{period}'
            if ma_col in df and not pd.isna(df[ma_col].iloc[-1]):
                if current_price > df[ma_col].iloc[-1]:
                    trend_score += 3
                    ma_bullish += 1
                    signals[ma_col] = 'BULLISH'
        
        if ma_bullish >= 4:
            trend_score += 10
            reasons.append(f"✅ Strong bullish trend ({ma_bullish}/6 MAs)")
        
        if all(col in df for col in ['MA_20', 'MA_50', 'MA_100', 'MA_200']):
            if (df['MA_20'].iloc[-1] > df['MA_50'].iloc[-1] > df['MA_100'].iloc[-1] > df['MA_200'].iloc[-1]):
                trend_score += 15
                reasons.append("🚀 Perfect Golden Cross")
        
        score += trend_score
        
        # 2. Momentum Analysis (30 points)
        momentum_score = 0
        
        if 'RSI_14' in df and not pd.isna(df['RSI_14'].iloc[-1]):
            rsi = df['RSI_14'].iloc[-1]
            if 40 <= rsi <= 60:
                momentum_score += 10
                reasons.append(f"✅ Good RSI: {rsi:.1f}")
            elif rsi < 30:
                momentum_score += 8
                reasons.append(f"📈 Oversold RSI: {rsi:.1f}")
        
        if all(col in df for col in ['MACD', 'MACD_Signal']):
            if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                momentum_score += 8
                reasons.append("✅ MACD Bullish")
        
        if 'Awesome_Oscillator' in df and df['Awesome_Oscillator'].iloc[-1] > 0:
            momentum_score += 4
            reasons.append("✅ Awesome Oscillator Positive")
        
        score += momentum_score
        
        # 3. Volume Analysis (20 points)
        volume_score = 0
        
        if all(col in df for col in ['Volume', 'Volume_MA']):
            vol_ratio = df['Volume'].iloc[-1] / df['Volume_MA'].iloc[-1]
            if vol_ratio > 1.5:
                volume_score += 10
                reasons.append("💰 High Volume")
            elif vol_ratio > 1:
                volume_score += 5
                reasons.append("💰 Above Average Volume")
        
        if 'OBV' in df and len(df) > 10:
            if df['OBV'].iloc[-1] > df['OBV'].iloc[-10]:
                volume_score += 5
                reasons.append("📊 Rising OBV")
        
        score += volume_score
        
        # 4. Volatility Analysis (15 points)
        vol_score = 0
        
        if 'BB_Position' in df and not pd.isna(df['BB_Position'].iloc[-1]):
            bb_pos = df['BB_Position'].iloc[-1]
            if 0.3 <= bb_pos <= 0.7:
                vol_score += 8
                reasons.append("📈 Good BB Position")
            elif bb_pos < 0.3:
                vol_score += 10
                reasons.append("🎯 Near BB Lower Band")
        
        if 'ATR' in df and not pd.isna(df['ATR'].iloc[-1]):
            atr_pct = (df['ATR'].iloc[-1] / current_price) * 100
            if atr_pct < 2:
                vol_score += 5
                reasons.append("🛡️ Low Volatility")
        
        score += vol_score
        
        # 5. Trend Strength (10 points)
        trend_str_score = 0
        
        if 'ADX' in df and not pd.isna(df['ADX'].iloc[-1]):
            if df['ADX'].iloc[-1] > 25:
                trend_str_score += 8
                reasons.append(f"💪 Strong Trend (ADX: {df['ADX'].iloc[-1]:.1f})")
        
        score += trend_str_score
        
        # Final calculation
        bullish_signals = sum(1 for s in signals.values() if 'BULLISH' in str(s))
        total_signals = len(signals)
        
        if bullish_signals >= 10:
            score = min(score + 10, 100)
            reasons.append(f"🚀 Strong Bullish Confirmation")
        
        return min(score, 100), reasons, signals, bullish_signals, total_signals

    def get_trading_signal(self, score):
        """Get trading signal based on score"""
        if score >= 85:
            return "🚀 STRONG BUY", "ultra-buy", "#10b981", "HIGH CONFIDENCE BUY"
        elif score >= 75:
            return "📈 BUY", "strong-buy", "#22c55e", "GOOD BUY OPPORTUNITY"
        elif score >= 65:
            return "⚡ ACCUMULATE", "hold", "#84cc16", "ACCUMULATE ON DIPS"
        elif score >= 55:
            return "🔄 HOLD", "hold", "#f59e0b", "HOLD POSITION"
        elif score >= 45:
            return "🔔 REDUCE", "hold", "#f97316", "REDUCE EXPOSURE"
        elif score >= 35:
            return "📉 SELL", "strong-sell", "#ef4444", "CONSIDER SELLING"
        else:
            return "💀 STRONG SELL", "strong-sell", "#dc2626", "STRONG SELL SIGNAL"

    def create_chart(self, df, symbol):
        """Create professional trading chart"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f'{symbol} - Price Analysis', 
                'Volume & OBV',
                'RSI & Momentum', 
                'MACD'
            ),
            row_heights=[0.4, 0.15, 0.2, 0.25]
        )
        
        # Price chart
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Moving averages
        for period, color in [(20, 'orange'), (50, 'green'), (200, 'red')]:
            if f'MA_{period}' in df:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[f'MA_{period}'], 
                    name=f'MA {period}', line=dict(color=color, width=2)
                ), row=1, col=1)
        
        # Bollinger Bands
        if 'BB_Upper' in df and 'BB_Lower' in df:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Upper'], name='BB Upper',
                line=dict(color='gray', dash='dash', width=1)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Lower'], name='BB Lower',
                line=dict(color='gray', dash='dash', width=1)
            ), row=1, col=1)
        
        # Volume
        colors = ['#00C805' if row['Close'] >= row['Open'] else '#FF0000' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'], name='Volume',
            marker_color=colors, opacity=0.7
        ), row=2, col=1)
        
        # OBV (normalized)
        if 'OBV' in df:
            obv_norm = (df['OBV'] - df['OBV'].min()) / (df['OBV'].max() - df['OBV'].min()) * df['Volume'].max()
            fig.add_trace(go.Scatter(
                x=df.index, y=obv_norm, name='OBV',
                line=dict(color='purple', width=2)
            ), row=2, col=1)
        
        # RSI
        if 'RSI_14' in df:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['RSI_14'], name='RSI 14',
                line=dict(color='blue', width=2)
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # MACD
        if all(col in df for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD'], name='MACD',
                line=dict(color='blue', width=2)
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_Signal'], name='Signal',
                line=dict(color='red', width=2)
            ), row=4, col=1)
            fig.add_trace(go.Bar(
                x=df.index, y=df['MACD_Histogram'], name='Histogram',
                marker_color='orange'
            ), row=4, col=1)
            fig.add_hline(y=0, line_color="black", row=4, col=1)
        
        fig.update_layout(
            title=f'PROFESSIONAL ANALYSIS - {symbol}',
            height=1000,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig

def main():
    analyzer = SuperStockAnalyzer()
    
    st.markdown('<h1 class="main-header">🚀 SUPER POWERFUL STOCK ANALYZER</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #6b7280;">40+ Indicators • AI-Powered • Professional Grade</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Analysis Parameters")
        symbol = st.text_input("Stock Symbol:", "RELIANCE")
        
        if st.button("🚀 RUN SUPER ANALYSIS", type="primary", use_container_width=True):
            symbol_with_ns = symbol.upper() + '.NS'
            
            with st.spinner("🔄 Running advanced analysis..."):
                data = analyzer.get_stock_data(symbol_with_ns, "6mo")
                
                if data is not None and not data.empty:
                    df = analyzer.calculate_indicators(data)
                    
                    if df is not None:
                        score, reasons, signals, bullish_count, total_signals = analyzer.calculate_ai_score(df)
                        signal, signal_class, color, advice = analyzer.get_trading_signal(score)
                        current_price = df['Close'].iloc[-1]
                        
                        # Display Results
                        st.markdown(f'<div class="super-card {signal_class}">', unsafe_allow_html=True)
                        st.subheader(f"🎯 {signal}")
                        st.write(f"**AI Score:** {score}/100")
                        st.write(f"**Current Price:** ₹{current_price:.2f}")
                        st.write(f"**Bullish Signals:** {bullish_count}/{total_signals}")
                        st.write(f"**Advice:** {advice}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Indicator Grid
                        st.subheader("📊 KEY SIGNALS")
                        cols = st.columns(4)
                        
                        indicators = []
                        if 'RSI_14' in df:
                            rsi_val = df['RSI_14'].iloc[-1]
                            rsi_status = 'bullish' if 40 <= rsi_val <= 60 else 'bearish'
                            indicators.append(('RSI', f"{rsi_val:.1f}", rsi_status))
                        
                        if all(col in df for col in ['MACD', 'MACD_Signal']):
                            macd_status = 'bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'bearish'
                            indicators.append(('MACD', 'BULL' if macd_status == 'bullish' else 'BEAR', macd_status))
                        
                        if all(col in df for col in ['Volume', 'Volume_MA']):
                            vol_status = 'bullish' if df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1] else 'neutral'
                            indicators.append(('Volume', 'HIGH' if vol_status == 'bullish' else 'NORM', vol_status))
                        
                        if 'MA_50' in df:
                            trend_status = 'bullish' if current_price > df['MA_50'].iloc[-1] else 'bearish'
                            indicators.append(('Trend', 'BULL' if trend_status == 'bullish' else 'BEAR', trend_status))
                        
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
                    st.plotly_chart(analyzer.create_chart(df, symbol), use_container_width=True)
                    
                    # Analysis Report
                    st.subheader("🔍 ANALYSIS REPORT")
                    for i, reason in enumerate(reasons[:10], 1):
                        st.write(f"{i}. {reason}")
                    
                    # Risk Management
                    st.subheader("🛡️ RISK MANAGEMENT")
                    r1, r2, r3, r4 = st.columns(4)
                    
                    with r1:
                        stop_loss = current_price * 0.92
                        st.metric("Stop Loss", f"₹{stop_loss:.1f}")
                    
                    with r2:
                        target = current_price * 1.15
                        st.metric("Target", f"₹{target:.1f}")
                    
                    with r3:
                        rr_ratio = (target - current_price) / (current_price - stop_loss)
                        st.metric("Risk/Reward", f"1:{rr_ratio:.1f}")
                    
                    with r4:
                        if 'ATR' in df:
                            st.metric("ATR", f"₹{df['ATR'].iloc[-1]:.2f}")

    # Market Scanner
    st.sidebar.header("⚡ QUICK ACTIONS")
    if st.sidebar.button("🔍 SCAN TOP STOCKS"):
        with st.spinner("Scanning..."):
            results = []
            for stock_symbol in analyzer.nifty_100[:5]:
                try:
                    data = analyzer.get_stock_data(stock_symbol)
                    if data is not None:
                        df = analyzer.calculate_indicators(data)
                        if df is not None:
                            score, _, _, bullish, _ = analyzer.calculate_ai_score(df)
                            if score >= 70:
                                price = df['Close'].iloc[-1]
                                results.append({
                                    'symbol': stock_symbol,
                                    'price': price,
                                    'score': score,
                                    'bullish': bullish
                                })
                except:
                    continue
            
            if results:
                st.subheader("💎 TOP PICKS")
                for stock in sorted(results, key=lambda x: x['score'], reverse=True)[:3]:
                    st.markdown(f'''
                    <div class="super-card ultra-buy">
                        <h3>🚀 {stock['symbol'].replace('.NS', '')}</h3>
                        <p><strong>Score:</strong> {stock['score']}/100</p>
                        <p><strong>Price:</strong> ₹{stock['price']:.2f}</p>
                        <p><strong>Bullish Signals:</strong> {stock['bullish']}</p>
                    </div>
                    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
