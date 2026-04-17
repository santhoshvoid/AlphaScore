def signal_outcome_analysis(data, signals):
    results = []

    for signal, date in signals:
        try:
            idx = data.index.get_loc(date)
            entry_price = data.iloc[idx]['Close']

            def get_return(days):
                if idx + days < len(data):
                    future_price = data.iloc[idx + days]['Close']
                    return (future_price - entry_price) / entry_price * 100
                return None

            result = {
                "date": date,
                "type": "GC" if signal == "BUY" else "DC",
                "1D": get_return(1),
                "3D": get_return(3),
                "1W": get_return(5),
                "1M": get_return(21)
            }

            results.append(result)

        except:
            continue

    return results


# ---------------- DIRECTION-AWARE SUMMARY ----------------

def summarize_results(results):
    summary = {}
    horizons = ["1D", "3D", "1W", "1M"]

    for h in horizons:
        valid = [r for r in results if r[h] is not None]

        if not valid:
            continue

        correct = 0

        for r in valid:
            val = r[h]

            if r["type"] == "GC":
                if val > 0:
                    correct += 1
            else:  # DC
                if val < 0:
                    correct += 1

        summary[h] = (correct / len(valid)) * 100

    return summary


# ---------------- SIGNAL INTELLIGENCE HELPERS ----------------

def calculate_strength(price, ema_s, ema_l):
    try:
        return abs(ema_s - ema_l) / price * 100
    except:
        return 0


def evaluate_signal(row):
    correctness = {}

    for key in ['+1D', '+3D', '+1W', '+1M']:
        val = row.get(key)

        if val is None:
            correctness[key] = None
        else:
            if row['type'] == "GC":
                correctness[key] = val > 0
            else:
                correctness[key] = val < 0

    return correctness


def calculate_confidence(correctness):
    weights = {'+1D': 1, '+3D': 2, '+1W': 3, '+1M': 4}

    score = 0
    total = 0

    for k, w in weights.items():
        if correctness[k] is not None:
            total += w
            if correctness[k]:
                score += w

    if total == 0:
        return 0

    return (score / total) * 100


def label_confidence(c):
    if c < 30:
        return "Weak"
    elif c < 60:
        return "Moderate"
    elif c < 80:
        return "Strong"
    else:
        return "Very Strong"


# ---------------- FINAL SIGNAL TABLE ----------------

def build_signal_table(data, signals, short_col, long_col):
    table = []

    for signal, date in signals:
        try:
            row = data.loc[date]

            close_price = row['Close']
            ema_short = row[short_col]
            ema_long = row[long_col]

            def future_return(days):
                future_index = data.index.get_loc(date) + days
                if future_index >= len(data):
                    return None
                future_price = data.iloc[future_index]['Close']
                return ((future_price - close_price) / close_price) * 100

            entry = {
                "Date": date.date(),
                "type": "GC" if signal == "BUY" else "DC",
                "Price": round(close_price, 2),
                "EMA_Short": round(ema_short, 2),
                "EMA_Long": round(ema_long, 2),
                "+1D": future_return(1),
                "+3D": future_return(3),
                "+1W": future_return(5),
                "+1M": future_return(21),
            }

            # ---------------- INTELLIGENCE ----------------
            strength = calculate_strength(close_price, ema_short, ema_long)
            correctness = evaluate_signal(entry)
            confidence = calculate_confidence(correctness)
            label = label_confidence(confidence)

            entry["Strength"] = round(strength, 2)
            entry["Confidence"] = round(confidence, 2)
            entry["Label"] = label

            table.append(entry)

        except:
            continue

    return table