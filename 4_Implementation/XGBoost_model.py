import polars as pl
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics
import time
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
import xgboost as xgb


class XGBoostModel:
    def __init__(self):
        pass

    @staticmethod
    def prepare_data(df: pl.DataFrame()):
        """
        Preparing the data by splitting into training/testing datasets & cleaning it before.
        :param df:
        :return:
        """
        df = df.drop_nulls()
        df = df.to_pandas()
        # Splitting the data & selecting the features
        X_train, X_test, y_train, y_test = train_test_split(df[['daily_rv', '5_days_mean_rv', '22_days_mean_rv']],
                                                            df['next_day_daily_rv'], test_size=0.2, random_state=2023)

        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "index": df["Date"]}

    @staticmethod
    def fit_xgboost_bayesian_opt(param_space: dict, X_train, y_train):
        """
        Fitting an XGBoost model using Bayesian Optimization to tune the hyperparameters
        :param param_space:
        :param X_train:
        :param y_train:
        :return XGBoost object:
        """
        start = time.time()
        print('Initialized the Bayesian Optimization')
        xgboost = xgb.XGBRegressor(objective='reg:squarederror',  random_state=203)
        bayes_opt = BayesSearchCV(xgboost, param_space, n_iter=20, cv=5, random_state=203)
        bayes_opt.fit(X_train, y_train)
        time_taken = time.time() - start
        print(f"Bayesian Optimization & fitting done after: {time_taken:.6f}")
        print('The best set of hyperparameters is: ', bayes_opt.best_params_)
        return bayes_opt

    @staticmethod
    def fit_xgboost_grid_search(param_space: dict, X_train, y_train):
        """
        Fitting an XGBoost model using Grid search to tune the hyperparameters
        :param param_space:
        :param X_train:
        :param y_train:
        :return XGBoost regressor object:
        """
        start = time.time()
        print('Initialized the Grid Search')
        xgboost = xgb.XGBRegressor(objective='reg:squarederror', random_state=203)
        grid_search = GridSearchCV(xgboost, param_space, cv=5, n_jobs=-1,)
        grid_search.fit(X_train, y_train)
        time_taken = time.time() - start
        print(f"Grid Search & fitting done after: {time_taken:.6f}")
        print('The best set of hyperparameters is: ', grid_search.best_params_)
        return grid_search

    @staticmethod
    def fit_xgboost_random_search(param_space: dict, X_train, y_train):
        """
        Fitting an XGBoost model using random search to tune the hyperparameters
        :param param_space:
        :param X_train:
        :param y_train:
        :return XGBoost object:
        """
        start = time.time()
        print('Initialized the Random Search')
        xgboost = xgb.XGBRegressor(objective='reg:squarederror', random_state=203)
        random_search = RandomizedSearchCV(xgboost, param_space, n_iter=100, cv=5, random_state=203)
        random_search.fit(X_train, y_train)
        time_taken = time.time() - start
        print(f"Random Search & fitting done after: {time_taken:.6f}")
        print('The best set of hyperparameters is: ', random_search.best_params_)
        return random_search

    @staticmethod
    def make_predictions(xgboost, X_test, y_test):
        y_pred = xgboost.predict(X_test)
        mse = metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)
        mae = metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred)

        # Print results rounded to two decimal places
        print(f"The MSE is: {mse:.6f}")
        print(f"The MAE is: {mae:.6f}")
        return {"predictions": y_pred, "MSE": mse, "MAE": mae}

    @staticmethod
    def plot_predictions(index, y_test, y_pred):
        """
        Preliminary function to do a sanity check on the results by plotting them against observed data.
        :param index:
        :param y_test:
        :param y_pred:
        :return:
        """
        # plot the actual realized volatility
        plt.plot(index, y_test, label='Centered Realized Volatility')
        # plot the predictions made by the model
        plt.plot(index, y_pred, label='XGBoost Predictions')

        # add a legend and title to the plot
        plt.legend()
        plt.title('XGBoost Model Predictions vs. Actual Realized Volatility')
        plt.show()

    def run(self, df: pl.DataFrame(), optim_method, params_space, plot_graph=False):
        dic_data = self.prepare_data(df)
        X_train, X_test, y_train, y_test, index = dic_data.get("X_train"), dic_data.get("X_test"), dic_data.get(
            "y_train"), dic_data.get("y_test"), dic_data.get("index")
        if optim_method == "Bayesian Opt":
            xgboost = self.fit_xgboost_bayesian_opt(params_space, X_train, y_train)
        elif optim_method == "Grid Search":
            xgboost = self.fit_xgboost_grid_search(params_space, X_train, y_train)
        elif optim_method == "Random Search":
            xgboost = self.fit_xgboost_random_search(params_space, X_train, y_train)
        else:
            print("This method has not been implemented yet.")
            return
        results = self.make_predictions(xgboost, X_test, y_test)
        y_pred = results["predictions"]
        if plot_graph:
            self.plot_predictions(index[len(y_train):], y_test, y_pred)
        results["model"] = xgboost
        return results








