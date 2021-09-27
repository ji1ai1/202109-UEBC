# python 3.9.6
# python套件 lightgbm 3.2.1
# python套件 numpy 1.21.2
# python套件 pandas 1.3.3
#
# 輸入：
#	train_data.csv
#	A_test_data.csv
#
# 輸出：
# 	result.csv
#
import datetime
import gc
import lightgbm
import numpy
import pandas
import random
import sklearn

def 統計特徴(某表, 鍵, 統計字典, 前綴=""):
	某統計表 = 某表.groupby(鍵).aggregate(統計字典)
	某統計表.columns = ["%s%s之%s%s" % (前綴, "".join(iter(鍵)), 欄名, 丑 if isinstance(丑, str) else 丑.__name__) for 欄名, 函式 in 統計字典.items() for 丑 in (函式 if isinstance(函式, list) else [函式])]
	return 某統計表

訓練表 = pandas.read_csv("train_data.csv", encoding="gbk", header=0, names=["標識", "帳號", "組", "地址", "鏈接", "埠", "虚擬網域編號", "交換機地址", "時間", "標籤"])
訓練表["秒序"] = (pandas.to_datetime(訓練表.時間) - datetime.datetime.strptime("2021-06-30", "%Y-%m-%d")).dt.total_seconds()
訓練表["曜日"] = 1 + pandas.to_datetime(訓練表.時間).dt.day_of_week
訓練表["日之秒序"] = 訓練表.秒序 % 86400
測試表 = pandas.read_csv("A_test_data.csv", encoding="gbk", header=0, names=["標識", "帳號", "組", "地址", "鏈接", "埠", "虚擬網域編號", "交換機地址", "時間"])
測試表["標籤"] = numpy.nan
測試表["秒序"] = (pandas.to_datetime(測試表.時間) - datetime.datetime.strptime("2021-06-30", "%Y-%m-%d")).dt.total_seconds()
測試表["曜日"] = 1 + pandas.to_datetime(測試表.時間).dt.day_of_week
測試表["日之秒序"] = 測試表.秒序 % 86400

測訓表 = pandas.concat([測試表, 訓練表], ignore_index=True)
統計表清單 = []
for 甲 in ["帳號", "地址", "鏈接", ["帳號", "地址"], ["帳號", "鏈接"], ["組", "地址"], ["組", "鏈接"]]:
	統計表清單.append((甲, 統計特徴(測訓表, 甲, {"標識": "count", "地址": "nunique", "鏈接": "nunique", "埠": "nunique", "虚擬網域編號": "nunique", "交換機地址": "nunique", "秒序": "nunique"})))

def 構建資料表(某表, 某特徴表):
	某資料表 = 某表.copy()

	for 甲 in ["帳號", "地址", "鏈接", ["帳號", "地址"], ["帳號", "鏈接"]]:
		某資料表 = 某資料表.merge(統計特徴(某特徴表, 甲, {"標籤": ["mean", "median", "min", "max"],}), on=甲, how="left")
	for 甲, 甲統計表 in 統計表清單:
		某資料表 = 某資料表.merge(甲統計表, on=甲, how="left")
		
	某資料表 = 某資料表.loc[:, ["標識", "標籤"] + [子 for 子 in 某資料表.columns if 子 not in ["標識", "標籤", "帳號", "組", "地址", "鏈接", "埠", "虚擬網域編號", "交換機地址", "時間"]]]
	return 某資料表

預測表 = None
測試資料表 = 構建資料表(測試表, 訓練表)
for 癸 in range(7, 13):
	print(str(datetime.datetime.now()) + "\t%s" % 癸)
	折數 = 癸
	索引 = random.sample(range(len(訓練表)), len(訓練表))
	癸訓練資料表 = None
	for 甲 in range(折數):
		甲標籤表 = 訓練表.iloc[[索引[子] for 子 in range(len(索引)) if 子 % 折數 == 甲]].reset_index(drop=True)
		甲特徴表 = 訓練表.iloc[[索引[子] for 子 in range(len(索引)) if 子 % 折數 != 甲]].reset_index(drop=True)
	
		甲資料表 = 構建資料表(甲標籤表, 甲特徴表)
		癸訓練資料表 = pandas.concat([癸訓練資料表, 甲資料表], ignore_index=True)

	癸輕模型 = lightgbm.train(train_set=lightgbm.Dataset(癸訓練資料表.iloc[:, 2:], label=癸訓練資料表.標籤)
		, num_boost_round=32768, params={"objective": "regression", "learning_rate": 0.05, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
	)
	
	癸預測表 = 測試資料表.loc[:, ["標識"]]
	癸預測表["預測"] = 癸輕模型.predict(測試資料表.iloc[:, 2:])
	癸預測表.loc[癸預測表.預測 < 0, "預測"] = 0
	癸預測表.loc[癸預測表.預測 > 1, "預測"] = 1
	預測表 = pandas.concat([預測表, 癸預測表], ignore_index=True)

預測表 = 預測表.groupby("標識").aggregate({"預測": "mean"}).reset_index()

提交表 = 預測表.loc[:, ["標識", "預測"]]
提交表.columns = ["id", "ret"]
提交表.to_csv("result.csv", index=False)
