# import requests
# import json

# url = "http://localhost:2222/judgment"

# payload = {
#     "messages": [
#         {
#             "role": "user",
#             "content": ""
#         }
#     ],
#     "stream": False
# }

# headers = {
#     'Content-Type': 'application/json'
# }

# response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

# print(response.text)

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/rayjue/SHARE/rayjue/LLM_MODEL/chatglm2-6b-32k", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/rayjue/SHARE/rayjue/LLM_MODEL/chatglm2-6b-32k", trust_remote_code=True).cuda()
print(model)