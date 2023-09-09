from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/LLM/baichuan-7b-sft", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/hy-tmp/LLM/baichuan-7b-sft", trust_remote_code=True,device_map="auto",low_cpu_mem_usage=True,torch_dtype=torch.float16)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

query = "晚上睡不着怎么办"
template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {}\nAssistant: "

inputs = tokenizer([template.format(query)], return_tensors="pt")
inputs = inputs.to("cuda") 
generate_ids = model.generate(**inputs, max_new_tokens=256, streamer=streamer)