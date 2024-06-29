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

os.environ['CUDA_VISIBLE_DEVICE'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dic = {
    0: "no",  # 与环保无关
    1: 'yes'
}

# 解析数据 begin
# social_media_dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
social_media_dataset = load_dataset('../data/dataset3')
social_media_dataset = social_media_dataset['test']
social_media_dataset = social_media_dataset.to_pandas()
social_media_dataset['label'] = social_media_dataset['label'].apply(lambda x: dic[x])
social_media_dataset[
    'instruction'] = 'Does this news have anything to do with environmental protection? Please select an answer from {yes/no}.'
social_media_dataset.columns = ['input', 'output', 'instruction']
social_media_dataset = datasets.Dataset.from_pandas(social_media_dataset)
print('....')

# %%

import json
from tqdm.notebook import tqdm


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    # print(context)
    target = example["output"]
    # print(len(context.split(' ')))
    return {"context": context, "target": target}


# %%

data_list = []
for item in social_media_dataset.to_pandas().itertuples():
    tmp = {}
    tmp["instruction"] = item.instruction
    tmp["input"] = item.input
    tmp["output"] = item.output
    data_list.append(tmp)

# %%

test_lst = []
cnt = 100
for example in tqdm(data_list, desc="formatting.."):
    test_data = format_example(example)
    test_lst.append(test_data)
    if len(test_lst) >= cnt:
        break

# 解析数据 end


# Model,Tokenizer, Datacollator
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

# Make prompts
# prompt = [
#     '''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
#     Input: FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs .
#     Answer: ''',
#     '''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
#     Input: According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
#     Answer: ''',
#     '''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
#     Input: A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser .
#     Answer: ''',
# ]

max_l = 512

labels = []
preds = []
for tmp_d in test_lst:
    # Generate results
    prompt = tmp_d['context']
    print('\n')
    print(prompt)
    label = tmp_d['target']
    # prompts = [prompt, prompt, prompt, prompt]  #测试batch输入
    # 必须要to(device)
    # tokens = tokenizer(prompts, return_tensors='pt', padding=True, max_length=max_l).to(device)
    tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=max_l).to(device)
    res = model.generate(**tokens, max_length=max_l)
    res_sentences = [tokenizer.decode(i) for i in res]
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
    #
    pred_txt = out_text[0].strip().lower()

    # 因为yes类比no多很多
    labels.append(label)
    if 'yes' in pred_txt and label == 'yes':
        preds.append('yes')
    else:
        preds.append('no')

report = classification_report(labels, preds)
print(report)
# print(out_text)

# edit begin
# end
