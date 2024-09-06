# Reference Code:
https://github.com/AI4Finance-Foundation/FinGPT/tree/master

# Environment installation:
The environment is environment.yml, which can be installed directly with conda, and Python is version 3.8.

# Code Description:
Dataset creation  
Go to the directory fingpt/FinGPT_Sentiment_Analysis_v3/data
Execute the following py files
Step1_make_my_datasets.py  
step2_make_my_datasets.py

# Model Training:
1 Requires base_model download of base_model chatglm2-6b
Enter the directory fingpt/FinGPT_Sentiment_Analysis_v3/training_parallel
Run sh train.sh

# Forecasting:
Using fine-tuned model: training_parallel/FinGPT-Green
Run predict.py

