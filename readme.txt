readme

There are two parts of the code
The first one is constructed with python, and the second one is constructed with pyspark.

Down df_all_close_price.csv file from https://drive.google.com/file/d/1qA3x8_9tG6mnF73W6BDENmRmsPbtmAXE/view?usp=sharing
and put it in "data" folder and "MR raw data" folder.

To run the first part, run ML.py first, it will do the prediction.
Then run all the stocks.py, it will do backtest using all strategies.

To run the second part, put all files into gdrive. 
Use Colab to open price_predict.ipynb, backtest_spark.ipynb, and change the paths. 
Run price_predict.ipynb first, and then backtest_spark.ipynb