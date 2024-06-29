from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel  # Model,Tokenizer
from transformers import TrainingArguments, Trainer

from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
import os
from datasets import load_dataset
import datasets
from sklearn.metrics import classification_report
import zipfile
import pandas as pd

# edit begin
zip_file_name = '/root/autodl-tmp/workspace/data.zip'

os.environ['CUDA_VISIBLE_DEVICE'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dic = {
    0: "no",  # 与环保无关
    1: 'yes'
}

# 加载模型 begin
base_model = "/root/autodl-tmp/workspace/fingpt/pretrained_models/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# begin
# Load Models
from peft import PeftModel

# peft_model = "finetuned_model"
peft_model = "./FinGPT-Green"

model = AutoModel.from_pretrained(base_model, trust_remote_code=True, device_map=device)
model = PeftModel.from_pretrained(model, peft_model)
model = model.eval()
max_l = 512


# 加载模型 end

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    # print(context)
    # target = example["output"]
    # print(len(context.split(' ')))
    return {"context": context}


with zipfile.ZipFile(zip_file_name, 'r') as zip_file:
    csv_file = 'split.csv'
    # csv_file = 'split_head.csv'

    chunkSize = 100000  # 每次处理10万行数据
    # -----edit begin
    # pandas的chunk分块读取
    # iterator = True ：开启迭代器
    # chunksize = chunkSize：指定一个chunkSize分块的大小来读取文件，此处是读取65535个数据为一个块。
    read_chunks = pd.read_csv(zip_file.open(csv_file),
                              encoding='utf-8',
                              iterator=True,
                              chunksize=chunkSize)
    name_index = 0
    for chunk in read_chunks:
        label_lst = []
        companyid_lst = []
        yearmonth1_lst = []
        countvar1_lst = []
        # sent_lst = []
        prompt_lst = []
        # df = chunk[['companyid', 'yearmonth1', 'countvar1', 'sentences']]
        test_lst = []
        for index, row in chunk.iterrows():
            # if index % 100 == 0:
            #     print(index)

            sent = row['sentences']
            # 大致检查文本数据是否太长 begin
            words = sent.split(' ')
            if len(words) > 350:  # 避免文本太长
                words = words[0:350]
                sent = ' '.join(words)

            # 大致检查文本数据是否太长 end

            # sent_lst.append(sent)
            companyid_lst.append(row['companyid'])
            yearmonth1_lst.append(row['yearmonth1'])
            countvar1_lst.append(row['countvar1'])

            # edit begin
            tmp_d = dict()
            tmp_d['instruction'] = 'Does this news have anything to do with environmental protection? ' \
                                   'Please select an answer from {yes/no}.'

            tmp_d['input'] = sent

            prompt = format_example(tmp_d)['context']
            prompt_lst.append(prompt)

        batch_size = 32
        for i in range(0, len(prompt_lst), batch_size):
            print(i)
            prompts = prompt_lst[i:i + batch_size]
            # 预测类别 begin
            # print('\n')
            # print(prompt)
            # 必须要to(device)
            tokens = tokenizer(prompts, return_tensors='pt', padding=True, max_length=max_l).to(device)
            res = model.generate(**tokens, max_length=max_l)
            res_sentences = [tokenizer.decode(i) for i in res]
            outs = [o.split("Answer: ")[1] for o in res_sentences]
            batch_preds = []
            for tmp_out in outs:
                tmp_out = tmp_out.strip().lower()
                # 因为0类多很多
                if 'yes' in tmp_out:
                    batch_preds.append(1)
                else:
                    batch_preds.append(0)

            label_lst.extend(batch_preds)

        df = pd.DataFrame({
            'label': label_lst,
            'companyid': companyid_lst,
            'yearmonth1': yearmonth1_lst,
            'countvar1': countvar1_lst,
            # 'sentences': sent_lst
        }
        )
        # 将 'label' 列移动到第一列
        # first_col = 'label'
        # df = pd.concat([df[first_col], df.drop(columns=first_col)], axis=1)

        # print(df.iloc[-1]['sentences'])
        name_index += 1
        file_name = 'results/data_' + str(name_index) + '.csv'
        df.to_csv(file_name, index=False)
        print('next')

print('end!')
