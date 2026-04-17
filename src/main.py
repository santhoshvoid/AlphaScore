# imports
from data_fetch import get_stock_data
from indicators import calculate_ema
from signals import detect_crossovers
from backtest import backtest, calculate_metrics
from analysis import signal_outcome_analysis, summarize_results, build_signal_table
from visualization import plot_chart
import pandas as pd


# ---------------- HELPERS ----------------

def format_symbol(symbol):
    symbol = symbol.upper().strip()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    return symbol


def format_period(user_input):
    user_input = user_input.lower().strip().replace(" ", "")

    if user_input == "max":
        return "max"

    # Extract number
    num = ""
    for char in user_input:
        if char.isdigit():
            num += char
        else:
            break

    if num == "":
        print("⚠️ Invalid timeframe. Using default: 1y")
        return "1y"

    num = int(num)

    if num <= 0:
        print("⚠️ Invalid timeframe. Using default: 1y")
        return "1y"

    # Detect unit
    if "mo" in user_input or "m" in user_input or "month" in user_input:
        return f"{num}mo"

    elif "y" in user_input or "year" in user_input:
        return f"{num}y"

    # Default logic
    if num <= 12:
        return f"{num}mo"
    else:
        return f"{num}y"


def get_user_input():
    print("\n" + "="*50)
    print("📊 SmartSignal Strategy Analyzer")
    print("="*50)

    # ---- STOCK ----
    while True:
        symbol_input = input("Enter stock (e.g. TCS, RELIANCE): ").strip()
        if symbol_input:
            symbol = format_symbol(symbol_input)
            break
        print("❌ Invalid stock. Try again.")

    # ---- TIMEFRAME ----
    while True:
        period_input = input("Enter timeframe (1, 6, 1y): ").strip()
        period = format_period(period_input)
        if period:
            break

    # ---- EMA ----
    while True:
        try:
            short_ema = int(input("Enter short EMA: "))
            long_ema = int(input("Enter long EMA: "))

            if short_ema <= 0 or long_ema <= 0:
                print("❌ EMA must be positive numbers")
                continue

            if short_ema >= long_ema:
                print("❌ Short EMA must be less than Long EMA")
                continue

            return symbol, period, short_ema, long_ema

        except ValueError:
            print("❌ Please enter valid numbers")


# ---------------- INPUT ----------------

symbol, period, short_ema, long_ema = get_user_input()

print(f"\n📌 Using: {symbol} | Period: {period} | EMA: {short_ema}/{long_ema}")


# ---------------- DATA ----------------

data = get_stock_data(symbol, period)

if data is None or len(data) < long_ema:
    print("❌ Not enough data for selected EMA/timeframe")
    exit()


# Show actual data range
start_date = data.index.min()
end_date = data.index.max()

days = (end_date - start_date).days
years = round(days / 365, 2)

print(f"\n📅 Data Range: {start_date.date()} → {end_date.date()} (~{years} years)")


# Handle mismatch
if period != "max" and "y" in period:
    requested_years = int(period.replace("y", ""))
    if years < requested_years:
        print("⚠️ Requested data exceeds available history. Showing maximum available data.")


# ---------------- INDICATORS ----------------

short_col = f'EMA{short_ema}'
long_col = f'EMA{long_ema}'

data[short_col] = calculate_ema(data['Close'].tolist(), short_ema)
data[long_col] = calculate_ema(data['Close'].tolist(), long_ema)


# ---------------- SIGNALS ----------------

signals = detect_crossovers(data, short_col, long_col)

if len(signals) == 0:
    print("\n⚠️ No crossover signals found for this setup")


# ---------------- BACKTEST ----------------

trades = backtest(data, signals)

if len(trades) == 0:
    print("\n⚠️ No trades executed")


# ---------------- METRICS ----------------

metrics = calculate_metrics(trades)


# ---------------- OUTPUT ----------------

print("\n📊 Trades:")
if trades:
    for i, t in enumerate(trades, 1):
        print(f"{i}. {round(t, 2)}%")
else:
    print("No trades available")

print("\n📊 Metrics:")
for key, value in metrics.items():
    print(f"{key}: {round(value, 2)}")


# ---------------- SIGNAL ANALYSIS ----------------

analysis_results = signal_outcome_analysis(data, signals)
summary = summarize_results(analysis_results)

print("\n🧠 Signal Analysis:")
if summary:
    for k, v in summary.items():
        print(f"{k}: {round(v, 2)}% positive")
else:
    print("No analysis available")


# ---------------- SIGNAL TABLE ----------------

table = build_signal_table(data, signals)

print("\n📋 Signal Table:")
df = pd.DataFrame(table)

if df.empty:
    print("No signals to display")
else:
    print(df.to_string(index=False))


# ---------------- VISUALIZATION ----------------

plot_chart(data, signals, short_col, long_col)