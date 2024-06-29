#参考代码
https://github.com/AI4Finance-Foundation/FinGPT/tree/master

# 环境安装
环境为environment.yml，可以用conda直接安装，python为3.8版本
# 代码说明：
数据集创建 进入目录 fingpt/FinGPT_Sentiment_Analysis_v3/data
<br /> 依次执行如下py文件
<br /> step1_make_my_datasets.py  
step2_make_my_datasets.py

# 模型训练：
1 需要下载base_model  chatglm2-6b
<br /> 2 进入目录 fingpt/FinGPT_Sentiment_Analysis_v3/training_parallel
<br />  执行 sh train.sh

# 预测 
fingpt/FinGPT_Sentiment_Analysis_v3/training_parallel/FinGPT-Green 为微调好的模型参数
<br /> predict.py
<br /> predict_zip.py # csv文件在压缩文件里面，结果保留在 results目录下

