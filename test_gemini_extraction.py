import openai
from config import DynaGraphConfig as config

client = openai.OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
prompt = """
Extract key facts as a JSON list of [Subject, Predicate, Object] triplets.
Focus on entities, their attributes, and relationships.

Text: "User states their goal is to organize a workshop on quantum computing for high-school"

Output format: {"triplets": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}
"""
response = client.chat.completions.create(
    model=config.TRIPLET_MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_tokens=500
)
print("GEMINI RAW RESPONSE:")
print(repr(response.choices[0].message.content))
