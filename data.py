import yfinance as yf
import pandas as pd
import numpy as np


# =========================
# Config: full asset categories
# =========================
TICKERS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication": "XLC"
}
MARKET_TICKER = "^GSPC"   # S&P 500
VIX_TICKER = "^VIX"


# =========================
# Robust price extraction
# =========================
def extract_adj_close(df):
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)

        if "Adj Close" in level0:
            data = df["Adj Close"]
        elif "Close" in level0:
            data = df["Close"]
        else:
            raise KeyError("No Close/Adj Close found")

        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                return data.iloc[:, 0]
            else:
                raise ValueError("Multiple columns found")

        return data

    else:
        if "Adj Close" in df.columns:
            return df["Adj Close"]
        elif "Close" in df.columns:
            return df["Close"]
        else:
            raise KeyError("No Close/Adj Close found")


# =========================
# Download data
# =========================
def download_prices(tickers=TICKERS, start="2015-01-01", end="2024-01-01"):
    data = {}
    tickers_ = {k: v for (k, v) in TICKERS.items() if v in tickers}    
    print(tickers_)
    for name, ticker in tickers_.items():
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            raise ValueError(f"No data for {ticker}")

        series = extract_adj_close(df)

        if not isinstance(series, pd.Series):
            raise ValueError(f"{ticker} is not a Series")

        data[name] = series

    prices = pd.concat(data, axis=1)

    # Market index
    sp500_df = yf.download(MARKET_TICKER, start=start, end=end, progress=False)
    sp500 = extract_adj_close(sp500_df)

    # VIX
    vix_df = yf.download(VIX_TICKER, start=start, end=end, progress=False)
    vix = extract_adj_close(vix_df)

    return prices, sp500, vix


# =========================
# Clean & align
# =========================
def preprocess_data(prices, sp500, vix):
    df = prices.copy()
    df["SP500"] = sp500
    df["VIX"] = vix

    df = df.ffill().dropna()
    df = df[~df.index.duplicated()]

    return df


# =========================
# Log returns
# =========================
def compute_log_returns(df):
    returns = np.log(df / df.shift(1))
    return returns.dropna()


# =========================
# Volatility features
# =========================
def compute_volatility_features(sp500_returns):
    vol20 = sp500_returns.rolling(20).std()
    vol60 = sp500_returns.rolling(60).std()

    vol_ratio = vol20 / (vol60 + 1e-8)

    features = pd.DataFrame({
        "vol20": vol20,
        "vol_ratio": vol_ratio
    })

    return features


# =========================
# Normalize
# =========================
def normalize(df):
    mean = df.mean()
    std = df.std() + 1e-8
    return (df - mean) / std


# =========================
# Build dataset
# =========================
def build_dataset(tickers=TICKERS, start="2015-01-01", end="2024-01-01", normalize_features=True):
    prices, sp500, vix = download_prices(tickers, start, end)
    tickers_={k:v for (k, v) in TICKERS.items() if v in tickers}

    df = preprocess_data(prices, sp500, vix)

    # Split components
    asset_prices = df[list(tickers_.keys())]
    sp500_series = df["SP500"]
    vix_series = df["VIX"]

    # ✅ Compute returns ONLY for features
    asset_returns = compute_log_returns(asset_prices)
    sp500_returns = compute_log_returns(sp500_series.to_frame()).iloc[:, 0]

    # ✅ Volatility features (correct input)
    vol_features = compute_volatility_features(sp500_returns)

    # ✅ Align everything
    features = pd.concat([
        asset_returns,
        vol_features,
        vix_series.loc[asset_returns.index]
    ], axis=1).dropna()

    # Normalize features (recommended)
    if normalize_features:
        features = normalize(features)

    # ✅ Final outputs
    prices_np = asset_prices.loc[features.index].values.astype(np.float32)
    features_np = features.values.astype(np.float32)

    # =========================
    # Sanity checks (IMPORTANT)
    # =========================
    print("Data sanity check:")
    print("Prices mean:", prices_np.mean())
    print("Returns mean:", asset_returns.mean().mean())
    print("Returns std:", asset_returns.std().mean())

    return prices_np, features_np, features, asset_prices.loc[features.index]


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    prices_np, features_np, features_df, aligned_prices = build_dataset()

    print("\nShapes:")
    print("Prices:", prices_np.shape)
    print("Features:", features_np.shape)
    print("\nHead:")
    print(features_df.head())
