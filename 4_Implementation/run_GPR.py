from market_data import MarketData
from GPR_model import GPRModel
from skopt.space import Real, Integer
from sklearn.gaussian_process.kernels import Matern
import numpy as np
from sklearn.utils.fixes import loguniform


# First getting the data
tickers = ['AAPL', 'ADBE', 'AMD', 'AMZN', 'BA', 'BAC', 'COST', 'HD', 'INTC', 'JNJ', 'JPM', 'MSFT', 'NEE', 'NVDA', 'XOM']
dic_data = MarketData().run(tickers)

# Bayesian Optimization
bay_opt_search_space = {
    'alpha': Real(1e-10, 1, prior='log-uniform'),
    'n_restarts_optimizer': Integer(2, 10)
}

# Grid search
grid_search_space = {
    'alpha': [1e-10, 1e-5, 1e-3, 1e-1, 1],
    'kernel': [Matern(nu=1.5, length_scale=l)
               for l in np.logspace(1e-2, 1e-1, 1)],
    'n_restarts_optimizer': [2, 5, 10,]

}

# Random search
random_search_space = {
    "alpha": loguniform(1e-10, 1e0),
    'kernel': [Matern(nu=1.5, length_scale=l)
               for l in np.logspace(1e-2, 1e-1, 1,)],
    "n_restarts_optimizer": [2, 5, 10],
}

# Defining the kernel
kernel = Matern(nu=1.5)
rows = []
for ticker in tickers:
    plot_graph = False
    if ticker == 'AAPL':
        plot_graph = True
    # Get the df of data for the ticker
    df_data = dic_data.get(ticker)
    # Fit a model, get potentially a plot & both mae and mse
    gpr_model_results = GPRModel().run(df_data, optim_method="Bayesian Opt", params_space=bay_opt_search_space, kernel=kernel,
                                       plot_predictions=plot_graph)
    gpr_model_results["symbol"] = ticker
    rows.append(gpr_model_results)

print(1)