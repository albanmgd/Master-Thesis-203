from market_data import MarketData
from XGBoost_model import XGBoostModel
from scipy.stats import uniform, randint, loguniform
from stock_universe import StockUniverse

# First we import the data
tickers = StockUniverse().run()
dic_data = MarketData().run(tickers)

# Depending on the optimization method we pick, need to define != set of parameters
# Random Search
random_search_space = {
    'learning_rate': loguniform(0.0001, 0.1),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0.0001, 120),  # can't be 0
    'n_estimators': randint(50, 200),
}
# Grid Search
grid_search_space = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'max_depth': [3, 6, 10],
    'subsample': [0.5, 1],
    'colsample_bytree': [0.5, 1],
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 1, 5],
    'n_estimators': [100, 200],
}

# Bayesian Opt
bay_opt_search_space = {
    'learning_rate': (0.0001, 0.25),
    'max_depth': (3, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'min_child_weight': (1, 10),
    'gamma': (0, 120),
    'n_estimators': (50, 200),
}

rows = []
for ticker in tickers:
    plot_graph = False
    if ticker == 'AAPL':
        plot_graph = False

    # Get the df of data for the ticker
    df_data = dic_data.get(ticker)
    # Fit a model, get potentially a plot & both mae and mse
    xgboost_model_results = XGBoostModel().run(df_data, optim_method="Grid Search",
                                               params_space=grid_search_space, plot_graph=plot_graph)
    xgboost_model_results["symbol"] = ticker
    rows.append(xgboost_model_results)

print(1)



















# Depending on the optimization method we pick, need to define != set of parameters
# search_space = {'max_depth': Integer(3, 10),
#                 'learning_rate': Real(0.01, 0.1),
#                 'n_estimators': Integer(50, 300),
#                 'gamma': Real(0, 1)}


# xgboost_model = XGBoostModel().run(df_data, optim_method="Bayesian Opt", params_space=bay_opt_search_space)

# # Fitting the model, computing MAE & MSE, plotting predictions and testing data
# xgboost_model = XGBoostModel().run(df_data, optim_method="Bayesian Opt", params_space=bay_opt_search_space)

