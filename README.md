#参考代码
https://github.com/AI4Finance-Foundation/FinGPT/tree/master

# 环境安装
环境为environment.yml，可以用conda直接安装，python为3.8版本
# 代码说明：
数据集创建 进入目录 fingpt/FinGPT_Sentiment_Analysis_v3/data
step1_make_my_datasets.py  
step2_make_my_datasets.py

# 模型训练：
1 需要下载base_model  chatglm2-6b
2 进入目录 fingpt/FinGPT_Sentiment_Analysis_v3/training_parallel
执行 sh train.sh

# 预测 
fingpt/FinGPT_Sentiment_Analysis_v3/training_parallel/FinGPT-Green 为微调好的模型参数
predict.py
predict_zip.py # csv文件在压缩文件里面，结果保留在 results目录下
# FinGPT_EnvironmentalProtection_Analysis
