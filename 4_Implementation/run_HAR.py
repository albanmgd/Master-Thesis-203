from market_data import MarketData
from HAR_model import HarModel
from stock_universe import StockUniverse
import polars as pl

# First we import the data
tickers = StockUniverse().run()
dic_data = MarketData().run(tickers)

rows = []
for ticker in tickers:
    plot_graph = False
    if ticker == 'AAPL':
        plot_graph = True
    # Get the df of data for the ticker
    df_data = dic_data.get(ticker)
    # Fit a model, get potentially a plot & both mae and mse
    har_model_results = HarModel().run(df_data, method="WLS", plot=plot_graph)
    har_model_results["symbol"] = ticker
    rows.append(har_model_results)

print(1)
