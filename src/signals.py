
def detect_crossovers(data, short_col, long_col):
    signals = []

    for i in range(1, len(data)):
        prev_short = data[short_col].iloc[i-1]
        prev_long = data[long_col].iloc[i-1]

        curr_short = data[short_col].iloc[i]
        curr_long = data[long_col].iloc[i]

        # 🟢 Golden Cross → BUY
        if prev_short <= prev_long and curr_short > curr_long:
            signals.append(("BUY", data.index[i]))

        # 🔴 Death Cross → SELL
        elif prev_short >= prev_long and curr_short < curr_long:
            signals.append(("SELL", data.index[i]))

    return signals
