import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Note: most of sklearn functions work with pandas dataframes. We now switch back from pl to pd for convenience.
# To me, the value added of polars in this project lies in the preparation of the data and its cleaning


class HarModel:
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

        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "index": df["Date"]}

    @staticmethod
    def fit_model(X_train, y_train, method: str, add_constant=True):
        """
        Fitting a model according to a specified method to determine the weights
        :param X_train:
        :param y_train:
        :param method:
        :return model:
        """
        # Adding a constant
        if add_constant:
            X_train = sm.add_constant(X_train)

        if method == "OLS":
            har_model = sm.OLS(y_train, X_train).fit()
            print(har_model.summary())
        elif method == "WLS":
            mod_sm = sm.OLS(y_train, X_train)
            res_sm = mod_sm.fit()
            y_resid = [abs(resid) for resid in res_sm.resid]
            X_resid = sm.add_constant(res_sm.fittedvalues)
            mod_resid = sm.OLS(y_resid, X_resid)
            res_resid = mod_resid.fit()
            mod_fv = res_resid.fittedvalues
            # Calculate weights
            weights = 1 / (mod_fv ** 2)
            model = sm.WLS(y_train, X_train, weights=weights)
            har_model = model.fit()
            print(har_model.summary())
        elif method == "NNLS":
            har_model = LinearRegression(positive=True).fit(X_train, y_train)
            print(har_model.coef_)
        else:
            print("This method has not been defined.")
            return

        return har_model

    @staticmethod
    def make_predictions(model, X_test, y_test, add_constant=True):
        """
        For a given model & testing dataset, making predictions. USer can specify whether to add a constant or not
        :param model:
        :param X_test:
        :param y_test:
        :param add_constant: boolean
        :return y_pred:
        """
        if add_constant:
            X_test = sm.add_constant(X_test)
        y_pred = model.predict(X_test)
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
        plt.plot(index, y_test, label='Realized Volatility')

        # plot the predictions made by the HAR model
        plt.plot(index, y_pred, label='HAR Predictions')

        # add a legend and title to the plot
        plt.legend()
        plt.title('HAR Model Predictions vs. Actual Realized Volatility')
        plt.show()

    def run(self, df: pl.DataFrame(), method: str, plot: bool):
        dic_data = self.prepare_data(df)
        X_train, X_test, y_train, y_test = dic_data.get("X_train"), dic_data.get("X_test"), dic_data.get(
            "y_train"), dic_data.get("y_test")
        har_model = self.fit_model(X_train, y_train, method)
        results = self.make_predictions(har_model, X_test, y_test)
        y_pred = results["predictions"]
        index = dic_data.get('index')[len(y_train):]
        if plot:
            self.plot_predictions(index, y_test, y_pred)
        results.pop("predictions")
        return results

