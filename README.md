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
  