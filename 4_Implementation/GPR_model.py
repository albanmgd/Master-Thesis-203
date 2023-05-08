import polars as pl
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
import sklearn.metrics as metrics
import time
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor


class GPRModel:
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
                                                            df['next_day_daily_rv'], test_size=0.2, random_state=203)

        # Normalizing the datasets
        normalized_X_train = preprocessing.StandardScaler().fit_transform(X_train)
        normalized_X_test = preprocessing.StandardScaler().fit_transform(X_test)
        normalized_y_train = preprocessing.StandardScaler().fit_transform(y_train.values.reshape(-1, 1))
        normalized_y_test = preprocessing.StandardScaler().fit_transform(y_test.values.reshape(-1, 1))

        return {"normalized_X_train": normalized_X_train, "normalized_X_test": normalized_X_test, "normalized_y_train":
            normalized_y_train, "normalized_y_test": normalized_y_test, "index": df["Date"]}

    @staticmethod
    def fit_gpr_bayesian_opt(kernel, param_space: dict, X_train, y_train):
        """
        Fitting a GPR using Bayesian Optimization to tune the hyperparameters
        :param param_space:
        :param X_train:
        :param y_train:
        :return GaussianProcessRegressor object:
        """
        gpr = GaussianProcessRegressor(kernel=kernel)
        bayes_cv = BayesSearchCV(gpr, param_space, n_iter=50, cv=5)

        start = time.time()
        print('Initialized the Bayesian fitting')
        bayes_cv.fit(X_train, y_train)
        time_taken = time.time() - start
        print(f"Bayesian fitting done after: {time_taken:.6f}")
        # Get the best hyperparameters
        best_params = bayes_cv.best_params_
        print('The best set of hyperparameters is: ' + str(best_params))
        # Train a new model with the best hyperparameters
        gpr.set_params(**best_params)
        gpr.fit(X_train, y_train)
        return gpr

    @staticmethod
    def fit_gpr_grid_search(param_space: dict, X_train, y_train):
        """
        Fitting a GPR using Grid search to tune the hyperparameters
        :param param_space:
        :param X_train:
        :param y_train:
        :return GaussianProcessRegressor object:
        """
        gpr = GaussianProcessRegressor()
        start = time.time()
        print('Initialized the Grid Search')
        grid_search = GridSearchCV(
            estimator=gpr,
            param_grid=param_space,
            cv=5,
            scoring='neg_mean_squared_error', )

        grid_search.fit(X_train, y_train)
        print('The best set of hyperparameters is: ', grid_search.best_params_)
        time_taken = time.time() - start
        print(f"Grid Search & fitting done after: {time_taken:.6f}")

        return grid_search

    @staticmethod
    def fit_gpr_random_search(kernel, param_space: dict, X_train, y_train):
        """
        Fitting a GPR using random search to tune the hyperparameters
        :param kernel:
        :param param_space:
        :param X_train:
        :param y_train:
        :return GaussianProcessRegressor object:
        """
        gpr = GaussianProcessRegressor(kernel=kernel)
        start = time.time()
        print('Initialized the Random Search')
        random_search = RandomizedSearchCV(
            gpr,
            param_space,
            n_iter=10,
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=203
        )
        # Fit the model to the training data
        random_search.fit(X_train, y_train)
        time_taken = time.time() - start
        print(f"Random Search & fitting done after: {time_taken:.6f}")
        print('The best set of hyperparameters is: ', random_search.best_params_)
        return random_search

    @staticmethod
    def make_predictions(gpr, X_test, y_test):
        y_pred = gpr.predict(X_test)
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
        plt.plot(index, y_pred, label='GPR Predictions')

        # add a legend and title to the plot
        plt.legend()
        plt.title('GPR Model Predictions vs. Actual Centered Realized Volatility')
        plt.show()

    def run(self, df: pl.DataFrame(), optim_method, params_space, kernel=None, plot_predictions=False):
        dic_data = self.prepare_data(df)
        X_train, X_test, y_train, y_test, index = dic_data.get("normalized_X_train"), dic_data.get(
            "normalized_X_test"), dic_data.get(
            "normalized_y_train"), dic_data.get("normalized_y_test"), dic_data.get("index")
        if optim_method == "Bayesian Opt":
            gpr = self.fit_gpr_bayesian_opt(kernel, params_space, X_train, y_train)
        elif optim_method == "Grid Search":
            gpr = self.fit_gpr_grid_search(params_space, X_train, y_train)
        elif optim_method == "Random Search":
            gpr = self.fit_gpr_random_search(kernel, params_space, X_train, y_train)
        else:
            print("This method has not been implemented yet.")
            return
        results = self.make_predictions(gpr, X_test, y_test)
        y_pred = results["predictions"]
        if plot_predictions:
            self.plot_predictions(index[len(y_train):], y_test, y_pred)
        results["model"] = gpr
        return results
