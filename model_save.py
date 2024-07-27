from peft import AutoPeftModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoPeftModelForCausalLM.from_pretrained(
    './results/checkpoint-720',
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
merged_model.save_pretrained('./model', safe_serialization=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./results/checkpoint-720',)
tokenizer.save_pretrained('./model')