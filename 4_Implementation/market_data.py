from datetime import datetime, time
import polars as pl
import yfinance as yf
import glob


class MarketData:

    def __init__(self):
        self.dic_data = {}
        self.path = r"C:\Users\mager\Desktop\Master's Thesis\Market_Data"

    def load_data(self, tickers: list, start_date: str, end_date: str):
        """
        Loading the data from parquet files if already exists or fetching it from Yahoo Finance otherwise
        :param tickers: list of tickers for which we want to fetch the data
        :param start_date:
        :param end_date:
        :return:
        """
        # Avoids downloading the data every time
        for ticker in tickers:
            filename = ticker + "_" + start_date + "_" + end_date + ".parquet"
            file = glob.glob(self.path + '/' + filename)
            if len(file) == 1:  # Means data exists
                df = pl.read_parquet(file[0])
            else:
                pd_df = yf.download(ticker, start=start_date, end=end_date)
                pd_df['Date'] = pd_df.index
                df = pl.from_dataframe(pd_df)
                path_filename = self.path + "/" + filename
                df.write_parquet(path_filename)  # Storing the data
            self.dic_data[ticker] = df
        return self.dic_data

    @staticmethod
    def compute_returns(df):
        """
        Computing the daily log returns for close prices
        :param df:
        :return df:
        """
        # Computing the returns on close for each day
        df = df.with_columns(
            [
                (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_returns"),
            ]
        )
        return df

    @staticmethod
    def compute_daily_rv(df):
        """
        Used to compute our proxy of daily vol, i.e. log(close/open)
        :param df:
        :return df:
        """
        df = df.with_columns(
            [
                (pl.col("Close") / pl.col("Open")).log().abs().alias("daily_rv"),
            ]
        )
        return df

    @staticmethod
    def compute_averages_rv(df, n_days: int):
        """
        Computing the averages of volatility over x days.
        :param n_days: number of days for the rolling mean
        :param df:
        :return df:
        """
        col_name = str(n_days) + '_days_mean_rv'
        df = df.with_columns(
            [
                pl.col("daily_rv").rolling_mean(window_size=n_days).alias(col_name),
            ]
        )
        return df

    @staticmethod
    def shift_data(df):
        """
        Method to shift the realized R.V. of one period since we're predicting daily R.V.
        :param df:
        :return df:
        """
        df = df.with_columns(
            pl.col('daily_rv').shift().alias('next_day_daily_rv')
        )
        return df

    @staticmethod
    def compute_vol_close_intraday_data(df):
        """
        This method was written initially for 5-min intraday data in order to study the impact of vol near the close.
        :param df:
        :return df_vol_close:
        """
        # We also want to compute the vol near the close "volatility near the close (15:30-16:00) in the previous day
        # (lag=1) is the most important predictor", see research paper
        df_close_rv = (
            df
            .filter(
                pl.col("timestamp").cast(pl.Time).is_between(time(15, 30), time(16), closed="both")
            )
            .groupby([pl.col("timestamp").dt.date()])
            .agg(
                [
                    pl.col("log_returns").std().alias("close_session_realized_vol"),
                ]
            )
            .sort([pl.col("timestamp").dt.date()])
        )
        return df_close_rv

    def run(self, tickers: list, start_date="2000-01-01", end_date="2023-03-03"):
        """
        Executing all the previously coded methods in one place.
        :param tickers:
        :param start_date:
        :param end_date:
        :return:
        """
        self.load_data(tickers, start_date, end_date)
        for ticker, df_data in self.dic_data.items():
            df_data = self.compute_returns(df_data)
            df_data = self.compute_daily_rv(df_data)
            df_data = self.compute_averages_rv(df_data, 5)
            df_data = self.compute_averages_rv(df_data, 22)
            df_data = self.shift_data(df_data)
            self.dic_data.update({ticker: df_data})
        return self.dic_data

    def test_1(self):
        self.run(tickers=["AAPL", "TSLA"])
