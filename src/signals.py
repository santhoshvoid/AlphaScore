def detect_crossovers(data, short_col, long_col):
    signals = data.copy()

    signals["prev_short"] = signals[short_col].shift(1)
    signals["prev_long"]  = signals[long_col].shift(1)

    signals["signal"] = None

    signals.loc[
        (signals["prev_short"] <= signals["prev_long"]) &
        (signals[short_col] > signals[long_col]),
        "signal"
    ] = "BUY"

    signals.loc[
        (signals["prev_short"] >= signals["prev_long"]) &
        (signals[short_col] < signals[long_col]),
        "signal"
    ] = "SELL"

    # 🔥 Convert back to SAME FORMAT your app expects
    result = []

    for idx, row in signals.iterrows():
        if row["signal"] is not None:
            result.append((row["signal"], idx))

    return result