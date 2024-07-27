from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
import json
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载预训练模型和fine tuned模型
model_pre = AutoModelForCausalLM.from_pretrained('pretrained_model_path').to(device)
model_finetuned = AutoModelForCausalLM.from_pretrained('finetuned_model_path').to(device)
tokenizer = AutoTokenizer.from_pretrained('model_path')


# 加载测试数据集
file_path = rf"C:\Users\86152\Desktop\data\test.json"
with open(file_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)



# 评估模型
def evaluate_model(model, tokenizer, test_data):
    bleu = load_metric('bleu')
    rouge = load_metric('rouge')

    for data in test_data:
        inputs = tokenizer(data['input'], return_tensors='pt')
        outputs = model.generate(**inputs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        references = [data['output']]
        predictions = [generated_text]

        bleu.add_batch(predictions=predictions, references=references)
        rouge.add_batch(predictions=predictions, references=references)

    bleu_score = bleu.compute()
    rouge_score = rouge.compute()

    return bleu_score, rouge_score

# 评估fine tuning前的模型
bleu_pre, rouge_pre = evaluate_model(model_pre, tokenizer, test_data)

# 评估fine tuning后的模型
bleu_finetuned, rouge_finetuned = evaluate_model(model_finetuned, tokenizer, test_data)

# 打印结果
print("BLEU before fine tuning:", bleu_pre)
print("BLEU after fine tuning:", bleu_finetuned)
print("ROUGE before fine tuning:", rouge_pre)
print("ROUGE after fine tuning:", rouge_finetuned)