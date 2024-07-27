import torch

from modelscope import AutoModelForCausalLM, AutoTokenizer
model_name_or_path = r"E:\Qwen1.5-0.5B-Chat\qwen\Qwen1___5-0___5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


# from peft import AutoPeftModelForCausalLM
#
# model = AutoPeftModelForCausalLM.from_pretrained(
#     './results/checkpoint-7500',
#     device_map="auto",
# ).eval()
# from modelscope import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('./results/checkpoint-7500')
#
#
# 确保模型在GPU上（如果可用），或者留在CPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# prompt = "今年是哪一年？"
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(device)
#
# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
#
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)
# 定义一个函数来生成对话响应
def generate_response(input_text, model, tokenizer, max_length=100, top_p=0.95, top_k=50):
    # 编码输入文本
    encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=1024,
                              return_attention_mask=True)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        # 解码响应
    decoded_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_response


# 对话循环
while True:
    user_input = input("你: ")
    if user_input.lower() in ['exit', 'quit', 'q']:
        break

    response = generate_response(user_input, model, tokenizer)
    print("模型: " + response)

# 退出对话
print("对话结束。")