import openai
from config import DynaGraphConfig as config

client = openai.OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
response = client.chat.completions.create(
    model=config.MAIN_MODEL,
    messages=[{"role": "user", "content": "Tell me a short story about quantum computing for high school kids"}],
    max_tokens=500
)
content = response.choices[0].message.content
print("LENGTH:", len(content))
print("CONTENT:", content)
print("FINISH REASON:", response.choices[0].finish_reason)
