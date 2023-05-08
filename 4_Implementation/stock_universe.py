import requests
import bs4 as bs
import yfinance as yf
import numpy as np
import glob
import pandas as pd

class StockUniverse:

    def __init__(self):
        self.start_date = '2000-01-03'  # actual B.D. after the 1st January
        self.end_date = "2023-03-03"
        self.size_universe = 15
        self.path = r"C:\Users\mager\Desktop\Master's Thesis\Market_Data"
        self.filename = self.path + '/sp500_historical_data.parquet'

    def get_sp500_history_data(self):
        """
        Gets stock data for each stock in the SP from start date to end date.
        :return df:
        """
        file = glob.glob(self.filename)
        if len(file) == 1:  # Means data exists
            df = pd.read_parquet(file[0])
        else:
            # Requesting the composition of the SP
            resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})

            # Building the tickers
            tickers = []

            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                tickers.append(ticker)

            tickers = [s.replace('\n', '') for s in tickers]
            data = yf.download(tickers, start=self.start_date, end=self.end_date)  # Note for improvement; check in the
            # yf module if requests are done async

            df = data.stack().reset_index().rename(index=str, columns={"level_1": "Symbol"}).sort_values(['Symbol', 'Date'])
            df["market_cap"] = df["Close"] * df["Volume"]
        return df

    def get_list_tickers(self, df):
        """
        Gets the list of 15 stocks which have been present in the SP from start date to end date and have the highest
        market capitalization as of end date.
        :param df:
        :return tickers:
        """
        first_date_df = df[df["Date"] == self.start_date]
        ticker_list = first_date_df['Symbol'].tolist()
        # Get all tickers present throughout the whole period
        for date in df["Date"].unique():
            date_df = df[df["Date"] == date]
            date_ticker_list = date_df['Symbol'].tolist()
            ticker_list = list(set(ticker_list).intersection(date_ticker_list))
        last_date_df = df[df["Date"] == self.end_date]
        last_date_df = last_date_df[last_date_df["Symbol"].isin(ticker_list)].sort_values(by="market_cap", ascending=False)["Symbol"].head(self.size_universe)
        tickers = last_date_df.to_list()
        return tickers

    @staticmethod
    def get_summary_universe(df, tickers):
        """
        Computes the min, max and mean log close prices & log vol for the selected stocks
        :param df:
        :param tickers:
        :return grouped:
        """
        # filter the DataFrame to include only the symbols in symbol_list
        symbol_df = df[df['Symbol'].isin(tickers)]
        symbol_df["Log Close"] = np.log(symbol_df["Close"])
        symbol_df["Log Open"] = np.log(symbol_df["Open"])

        symbol_df["Daily RV"] = abs(symbol_df["Log Close"] - symbol_df["Log Open"])

        # calculate the desired statistics by group
        grouped = symbol_df.groupby('Symbol').agg(
            {'Log Close': ['min', 'max', 'mean'], 'Daily RV': ['min', 'max', 'mean']}
        )

        print(grouped.head().to_string())
        # rename columns
        grouped.columns = ['Min log close price', 'Max log close price', 'Mean log close price',
                           'Min daily volatility', 'Max daily volatility', 'Mean daily volatility']

        # sort the DataFrame by market capitalization
        grouped = grouped.sort_values(by='Symbol', ascending=False)

        # reset the index to have a clean DataFrame with one row per symbol
        grouped = grouped.reset_index()
        return grouped

    def run(self):
        df_sp = self.get_sp500_history_data()
        tickers = self.get_list_tickers(df_sp)
        return tickers
