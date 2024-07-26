import json
import torch
from transformers import Trainer, TrainingArguments
from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset


# 数据处理
def get_mydata(tokenizer):
    number = [i for i in range(20)]
    my_data = []
    for i in number:
        file_path = rf"C:\Users\86152\Desktop\data\alpaca_chinese_part_{i}.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for dic in data:
                if dic['input'] == '':
                    my_data.append({'input': dic['instruction'], 'output': dic['output']})
                else:
                    my_data.append({'input': dic['instruction'] + ' ' + dic['input'], 'output': dic['output']})

    # 将数据转换为datasets库支持的格式
    dataset = Dataset.from_dict({
        "input": [item["input"] for item in my_data],
        "output": [item["output"] for item in my_data]
    })

    # 数据预处理
    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["output"]
        model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")

        # 设置目标文本的Token
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")

        model_inputs["labels"] = labels['input_ids']
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return tokenized_datasets


model_path = r"E:\Qwen1.5-0.5B-Chat\qwen\Qwen1___5-0___5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 确保模型在GPU上（如果可用），或者留在CPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = get_mydata(tokenizer)

from peft import get_peft_model, LoraConfig, TaskType

# 设置Lora参数
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["self_attn.q_proj", "self_attn.k_proj"]
)

model = get_peft_model(model, peft_config)
# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # 训练轮次
    per_device_train_batch_size=8,  # 每个设备的训练批次大小
    per_device_eval_batch_size=8,  # 每个设备的评估批次大小
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
)
trainer.train()