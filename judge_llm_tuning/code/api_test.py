import requests
import json

url = "http://i-1.gpushare.com:38245/resume/resume_generate"

payload = json.dumps({
  "message": {
    "use_model": True,
    "instruct": ""
  },
  "stream": True
})
headers = {
  'Content-Type': 'application/json',
  'Timeout': '18000'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
