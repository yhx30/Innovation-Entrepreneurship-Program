# import requests
# import json

# url = "http://localhost:6666/wish_qa"

# payload = {
#     "messages": [
#         {
#             "role": "user",
#             "content": "“用户问题”分为哪几类问题？"
#         }
#     ],
#     "stream": False
# }

# headers = {
#     'Content-Type': 'application/json'
# }

# response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

# print(response.text)



# import requests
# import json

# url = "http://localhost:8000/wish_qa"

# # Replace with actual request payload based on ChatCompletionRequest data model
# payload = {
#     "question": "How can I track my order on WISH?"
# }

# headers = {
#     'Content-Type': 'application/json'
# }

# response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

# print(response.text)
import requests
from pydantic import BaseModel

# 定义请求数据模型
class ResumeMessage(BaseModel):
    use_model: bool
    instruct: str

class ResumeRequest(BaseModel):
    message: ResumeMessage
    stream: bool

# 构造请求体数据
request_data = ResumeRequest(
    message=ResumeMessage(use_model=True, instruct="生成一份完整的简历"),
    stream=False
)

# 请求的URL
url = "http://localhost:7777/resume_generate"  # 修改为你的FastAPI服务器地址和端口

# 发送POST请求
response = requests.post(url, json=request_data.dict())

# 处理响应
if response.status_code == 200:
    response_data = response.json()
    if "choices" in response_data and len(response_data["choices"]) > 0:
        assistant_response = response_data["choices"][0]["message"]["content"]
        print("Assistant's Response:", assistant_response)
    else:
        print("No valid response data found.")
else:
    print("Error:", response.text)




