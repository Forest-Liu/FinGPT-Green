from datasets import load_dataset
import datasets

# 自己的字典,必须为一个单词，方便后面解析，以及模型预测
dic = {
    0: "no",  # 与环保无关
    1: 'yes'
}

# %%

# social_media_dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
social_media_dataset = load_dataset('dataset3')
social_media_dataset = social_media_dataset['train']
social_media_dataset = social_media_dataset.to_pandas()
social_media_dataset['label'] = social_media_dataset['label'].apply(lambda x: dic[x])
social_media_dataset[
    'instruction'] = 'Does this news have anything to do with environmental protection? ' \
                     'Please select an answer from {yes/no}.'
social_media_dataset.columns = ['input', 'output', 'instruction']
social_media_dataset = datasets.Dataset.from_pandas(social_media_dataset)
print('....')

# tmp_dataset = datasets.concatenate_datasets([social_media_dataset] * 2)
# train_dataset = datasets.concatenate_datasets([train_dataset, tmp_dataset])
# print(tmp_dataset.num_rows)
# train_dataset

## Make Dataset

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

with open("dataset3/dataset_new.jsonl", 'w') as f:
    for example in tqdm(data_list, desc="formatting.."):
        f.write(json.dumps(format_example(example)) + '\n')

# %% md

### Tokenize

# %%

import json
from tqdm.notebook import tqdm

import datasets
from transformers import AutoTokenizer, AutoConfig

model_name = "/root/autodl-tmp/workspace/fingpt/pretrained_models/chatglm2-6b"
jsonl_path = "dataset3/dataset_new.jsonl"
save_path = 'dataset3/dataset_new'
max_seq_length = 512  # 最长的text 为223，在加上instruction的长度
skip_overlength = True


# %%

def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


# %%

dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(jsonl_path, max_seq_length, skip_overlength)
)
dataset.save_to_disk(save_path)

# %%
