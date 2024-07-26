import json
from datasets import Dataset


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





