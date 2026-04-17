def calculate_ema(prices, span):
    ema = []
    multiplier = 2 / (span + 1)

    for i in range(len(prices)):
        if i == 0:
            ema.append(prices[i])
        else:
            value = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
            ema.append(value)

    return ema