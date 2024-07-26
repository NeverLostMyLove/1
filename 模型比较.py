from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_metric

# 加载预训练模型和fine tuned模型
model_pre = AutoModelForSeq2SeqLM.from_pretrained('pretrained_model_path')
model_finetuned = AutoModelForSeq2SeqLM.from_pretrained('finetuned_model_path')
tokenizer = AutoTokenizer.from_pretrained('model_path')

# 加载测试数据集
test_data = [
    {"input": "Translate the following text to French: 'Hello, how are you?'", "output": "Bonjour, comment ça va?"}
    # 添加更多测试数据
]

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