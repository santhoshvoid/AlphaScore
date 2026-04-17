import matplotlib.pyplot as plt

def plot_chart(data, signals, short_col, long_col):
    plt.figure(figsize=(12, 6))

    # Plot price and EMAs (DYNAMIC)
    plt.plot(data.index, data['Close'], label='Price', color='black')
    plt.plot(data.index, data[short_col], label=short_col, linestyle='--')
    plt.plot(data.index, data[long_col], label=long_col, linestyle='--')

    # Plot signals
    buy_plotted = False
    sell_plotted = False

    for signal, date in signals:
        price = data.loc[data.index == date, 'Close'].values[0]

        if signal == "BUY":
            if not buy_plotted:
                plt.scatter(date, price, marker='^', color='green', label='BUY')
                buy_plotted = True
            else:
                plt.scatter(date, price, marker='^', color='green')

        elif signal == "SELL":
            if not sell_plotted:
                plt.scatter(date, price, marker='v', color='red', label='SELL')
                sell_plotted = True
            else:
                plt.scatter(date, price, marker='v', color='red')

    # Labels and styling
    plt.title(f"EMA Crossover Strategy ({short_col} vs {long_col})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    plt.show()