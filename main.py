def calculate_rsi(df, period=14):
    """Calculates RSI checking for both lowercase and uppercase dataframe columns."""
    # Fallback assignment selector rule
    close_col = 'close' if 'close' in df.columns else 'Close'
    if close_col not in df.columns or len(df) < period:
        return np.nan
        
    delta = df[close_col].astype(float).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.iloc[-1]

def calculate_atr(df, period=14):
    """Calculates ATR checking for both lowercase and uppercase dataframe columns."""
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'
    close_col = 'close' if 'close' in df.columns else 'Close'
    
    if not {high_col, low_col, close_col}.issubset(df.columns) or len(df) < period:
        return np.nan
        
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close_prev = df[close_col].astype(float).shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_series.iloc[-1]

def calculate_macd_signal_str(df, fast=12, slow=26, signal=9):
    """Calculates MACD trends checking for both lowercase and uppercase dataframe columns."""
    close_col = 'close' if 'close' in df.columns else 'Close'
    if close_col not in df.columns or len(df) < slow + signal:
        return "N/A"
        
    prices = df[close_col].astype(float)
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    c_macd, c_sig = macd_line.iloc[-1], signal_line.iloc[-1]
    p_macd, p_sig = macd_line.iloc[-2], signal_line.iloc[-2]
    
    if c_macd > c_sig and p_macd <= p_sig:
        return "Bullish Crossover"
    elif c_macd < c_sig and p_macd >= p_sig:
        return "Bearish Crossover"
    else:
        return "Bullish Trend" if c_macd > c_sig else "Bearish Trend"
