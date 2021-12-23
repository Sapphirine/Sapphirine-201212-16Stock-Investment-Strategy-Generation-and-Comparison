**Description：**  
ML.py:
Machine learning module using pandas package. Import all the data from csv files, including data price and all the factors. Then uses three machine learning models to predict the trend of stock price in the next month. The accuracy figure is put in ./MR result fig, and the prediction is put in ./MR result data. 

All the stocks.py:
Backtesting Module using pandas package. Using three strategies that was designed by us to generate buy/sell signals. Then use backtesting algorithm to evaluate all the strategies.
 
Backtest2.py:
Help functions for backtesting.
 
Price_predict.ipynb:
Machine learning module using pyspark. Import all the data from csv files, including data price and all the factors. Then uses three machine learning models to predict the trend of stock price in the next month. Files are put in ./result2. 
 
Backtest_spark.ipynb:
Backtesting module using pyspark. Using signals and price of stocks to do backtesting.

**run:**
There are two parts of the code
The first one is constructed with python, and the second one is constructed with pyspark.

Download df_all_close_price.csv file from https://drive.google.com/file/d/1qA3x8_9tG6mnF73W6BDENmRmsPbtmAXE/view?usp=sharing
and put it in "data" folder and "MR raw data" folder.

To run the first part, run ML.py first, it will do the prediction.
Then run all the stocks.py, it will do backtest using all strategies.

To run the second part, put all files into gdrive. 
Use Colab to open price_predict.ipynb, backtest_spark.ipynb, and change the paths. 
Run price_predict.ipynb first, and then backtest_spark.ipynb

./  
├── BT raw data  
│   └── time.xlsx  
├── BT result fig  
│   ├── backtest resultstrategy1_GB.png  
│   ├── backtest resultstrategy1_LR.png  
│   ├── backtest resultstrategy1_NB.png  
│   ├── backtest resultstrategy2_GB.png  
│   ├── backtest resultstrategy2_LR.png  
│   ├── backtest resultstrategy2_NB.png  
│   ├── backtest resultstrategy_GB.png  
│   ├── backtest resultstrategy_LR.png  
│   └── backtest resultstrategy_NB.png  
├── ML.py  
├── MR raw data  
│   ├── df_all_close_price.csv  
│   └── timelist_m.xlsx  
├── MR result data  
│   ├── price.csv  
│   ├── result_GB.csv  
│   ├── result_LR.csv  
│   └── result_NB.csv  
├── MR result fig  
│   ├── The Accuracy of the Gradient Boosting Classifier Model.png  
│   ├── The Accuracy of the Logistic Regression Model.png  
│   └── The Accuracy of the Naive Bayes Bernoulli Model.png  
├── README.md  
├── __pycache__  
├── all the stocks.py  
├── backtest.py  
├── backtest2.py  
├── backtest_spark.ipynb  
├── data  
│   ├── backtestingDate.csv  
│   ├── close.csv  
│   ├── df_all_close_price.csv  
│   ├── signal.csv  
│   └── timelist_m.csv  
├── idea  
│   ├── all the stocks.iml  
│   ├── inspectionProfiles  
│   │   └── profiles_settings.xml  
│   ├── misc.xml  
│   ├── modules.xml  
│   └── workspace.xml  
├── price_predict.ipynb  
├── pycache  
│   ├── backtest.cpython-37.pyc  
│   ├── backtest2.cpython-37.pyc  
│   └── backtest2.cpython-38.pyc  
├── result2  
└── singal factors  
  
12 directories, 39 files  
  
