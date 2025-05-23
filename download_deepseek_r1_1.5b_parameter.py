from transformers import AutoTokenizer, AutoModelForCausalLM
# to download Deepseek R1 for better Resume Formatting
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
