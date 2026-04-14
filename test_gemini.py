import openai
from config import DynaGraphConfig as config

client = openai.OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
print("Testing Gemini integration via OpenAI SDK...")
try:
    response = client.chat.completions.create(
        model=config.TRIPLET_MODEL,
        messages=[{"role": "user", "content": "Extract the triplets from: Quantum computing is a branch of physics."}],
        temperature=0.1,
        max_tokens=100
    )
    print("Success! Response from Gemini:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
