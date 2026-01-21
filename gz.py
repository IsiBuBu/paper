import requests
headers = {
  'Authorization': 'Bearer <OPENROUTER_API_KEY>',
  'HTTP-Referer': '<YOUR_SITE_URL>',
  'X-Title': '<YOUR_SITE_NAME>',
  'Content-Type': 'application/json',
}
response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, json={
  'model': 'qwen/qwen3-8b',
  'messages': [{ 'role': 'user', 'content': 'Hello' }],
  'provider': {
    'quantizations': ['fp8'],
  },
})