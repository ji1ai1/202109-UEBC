# python 3.9.6
# python套件 lightgbm 3.2.1
# python套件 numpy 1.21.2
# python套件 pandas 1.3.3
#
# 输入：
#	train_data.csv
#	A_test_data.csv
#
# 输出：
# 	result.csv
#
import datetime
import gc
import lightgbm
import numpy
import pandas
import random
import sklearn

def 统计特征(某表, 键, 统计字典, 前缀=""):
	某统计表 = 某表.groupby(键).aggregate(统计字典)
	某统计表.columns = ["%s%s之%s%s" % (前缀, "".join(iter(键)), 栏名, 丑 if isinstance(丑, str) else 丑.__name__) for 栏名, 函式 in 统计字典.items() for 丑 in (函式 if isinstance(函式, list) else [函式])]
	return 某统计表

训练表 = pandas.read_csv("train_data.csv", encoding="gbk", header=0, names=["标识", "账号", "组", "地址", "链接", "埠", "虚拟网域编号", "交换机地址", "时间", "标签"])
训练表["秒序"] = (pandas.to_datetime(训练表.时间) - datetime.datetime.strptime("2021-06-30", "%Y-%m-%d")).dt.total_seconds()
训练表["曜日"] = 1 + pandas.to_datetime(训练表.时间).dt.day_of_week
训练表["日之秒序"] = 训练表.秒序 % 86400
测试表 = pandas.read_csv("A_test_data.csv", encoding="gbk", header=0, names=["标识", "账号", "组", "地址", "链接", "埠", "虚拟网域编号", "交换机地址", "时间"])
测试表["标签"] = numpy.nan
测试表["秒序"] = (pandas.to_datetime(测试表.时间) - datetime.datetime.strptime("2021-06-30", "%Y-%m-%d")).dt.total_seconds()
测试表["曜日"] = 1 + pandas.to_datetime(测试表.时间).dt.day_of_week
测试表["日之秒序"] = 测试表.秒序 % 86400

测训表 = pandas.concat([测试表, 训练表], ignore_index=True)
统计表清单 = []
for 甲 in ["账号", "地址", "链接", ["账号", "地址"], ["账号", "链接"], ["组", "地址"], ["组", "链接"]]:
	统计表清单.append((甲, 统计特征(测训表, 甲, {"标识": "count", "地址": "nunique", "链接": "nunique", "埠": "nunique", "虚拟网域编号": "nunique", "交换机地址": "nunique", "秒序": "nunique"})))

def 构建数据表(某表, 某特征表):
	某数据表 = 某表.copy()

	for 甲 in ["账号", "地址", "链接", ["账号", "地址"], ["账号", "链接"]]:
		某数据表 = 某数据表.merge(统计特征(某特征表, 甲, {"标签": ["mean", "median", "min", "max"],}), on=甲, how="left")
	for 甲, 甲统计表 in 统计表清单:
		某数据表 = 某数据表.merge(甲统计表, on=甲, how="left")
		
	某数据表 = 某数据表.loc[:, ["标识", "标签"] + [子 for 子 in 某数据表.columns if 子 not in ["标识", "标签", "账号", "组", "地址", "链接", "埠", "虚拟网域编号", "交换机地址", "时间"]]]
	return 某数据表

预测表 = None
测试数据表 = 构建数据表(测试表, 训练表)
for 癸 in range(7, 13):
	print(str(datetime.datetime.now()) + "\t%s" % 癸)
	折数 = 癸
	索引 = random.sample(range(len(训练表)), len(训练表))
	癸训练数据表 = None
	for 甲 in range(折数):
		甲标签表 = 训练表.iloc[[索引[子] for 子 in range(len(索引)) if 子 % 折数 == 甲]].reset_index(drop=True)
		甲特征表 = 训练表.iloc[[索引[子] for 子 in range(len(索引)) if 子 % 折数 != 甲]].reset_index(drop=True)
	
		甲数据表 = 构建数据表(甲标签表, 甲特征表)
		癸训练数据表 = pandas.concat([癸训练数据表, 甲数据表], ignore_index=True)

	癸轻模型 = lightgbm.train(train_set=lightgbm.Dataset(癸训练数据表.iloc[:, 2:], label=癸训练数据表.标签)
		, num_boost_round=32768, params={"objective": "regression", "learning_rate": 0.05, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
	)
	
	癸预测表 = 测试数据表.loc[:, ["标识"]]
	癸预测表["预测"] = 癸轻模型.predict(测试数据表.iloc[:, 2:])
	癸预测表.loc[癸预测表.预测 < 0, "预测"] = 0
	癸预测表.loc[癸预测表.预测 > 1, "预测"] = 1
	预测表 = pandas.concat([预测表, 癸预测表], ignore_index=True)

预测表 = 预测表.groupby("标识").aggregate({"预测": "mean"}).reset_index()

提交表 = 预测表.loc[:, ["标识", "预测"]]
提交表.columns = ["id", "ret"]
提交表.to_csv("result.csv", index=False)
