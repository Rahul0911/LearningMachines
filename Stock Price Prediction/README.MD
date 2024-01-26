# Data Source
I have extracted the stock data from Yahoo Finance website using yfinance API. I have also attached the website and the yfinance API documentation.

https://ca.finance.yahoo.com/topic/news/

https://pypi.org/project/yfinance/

# About the dataset
This dataset is teh collection of stock data for various companies and each individual ticker has 6 columns - Open, Low, High, Close, Adj Close and Volume and it is indexed by dates which are going back to 1967. I have used the data from late 1900's to present for a semiconductor company called AMD.

* Ticker - A ticker symbol or stock symbol is an abbreviation used to uniquely identify publicly traded shares of a particular stock on a particular stock market.

# Goal
My main purpose for using this dataset was to get a good understanding about LSTMs -a  recurrent neural networks which is good at retaining long term information, hence the name long short term memory and works really well with time series data such as stock price changes. In the ipynb file attached, I have filtered out one company and used its stock data to feel my LSTM model.

# Results
After the training was completed, the model was able to effective learn from the data and predict the behaviours trends of the stock. Although the model is not able to predict the exact change in price of the stock, it definitely is predicting the behavious of price surge and drop, which is a valuable information when you're investing in certain stock.

# Future Improvements
Next challange will be to use the stock data of multiple organization and make a generalized model, where I can select the backtrack duration(how far back in time we are feeding the data to LSTM model to train) and forwardtrack duration(how far in the future, we are predicting the values)
