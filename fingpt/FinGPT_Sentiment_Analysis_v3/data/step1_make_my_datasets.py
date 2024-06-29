import pandas as pd

data_list = []
file_path = 'dataset3/GreenTagtoTrain.csv'

# 0 487219,1 12780 标签0太多,text,Green
df = pd.read_csv(file_path)
df = df.rename(columns={'Green': 'label'})
# 数据统计 begin
groups = df.groupby('label')
print(groups)
for i, j in groups.size().items():
    print(i, j)
    account = j / df.shape[0]
    dd = "%.2f%%" % (account * 100)
    print(f"{i}占比为{dd}")
print('------------')
print(groups.size())

max_size = 0  # 228
# for index, row in df.iterrows():
#     size = len(row['text'].split(' '))
#     if size > max_size:
#         max_size = size

print('max_size:', max_size)

# 数据统计 end
df0 = df[df['label'] == 0]
df1 = df[df['label'] == 1]
# test cnt default = 12780
df0 = df0.sample(n=12780, random_state=0)

df0 = df0.sample(frac=1.0)  # 全部打乱
df1 = df1.sample(frac=1.0)  # 全部打乱

# 训练集
index = int(12780 * 0.02)  # 划分数据集
# edit begin
# index = 10
# train_data = pd.concat([df0.iloc[:index, :], df1.iloc[:index, :]], axis=0, ignore_index=True).sample(frac=1)
# train_data.to_csv('dataset3/sent_train.csv', index=False)
#
# # edit end
train0 = df0.iloc[:index, :]
train1 = df1.iloc[:index, :]
train_data = pd.concat([train0, train1], axis=0, ignore_index=True).sample(frac=1)

# 验证集
valid0 = df0.iloc[index:, :]
valid1 = df1.iloc[index:, :]
valid_data = pd.concat([valid0, valid1], axis=0, ignore_index=True).sample(frac=1)

train_data.to_csv('dataset3/sent_train.csv', index=False)
valid_data.to_csv('dataset3/sent_test.csv', index=False)
# 后续会自动 分割
# train_data = pd.concat([train_data, valid_data], axis=0, ignore_index=True).sample(frac=1)
# train_data.to_csv('dataset3/sent_train.csv', index=False)
