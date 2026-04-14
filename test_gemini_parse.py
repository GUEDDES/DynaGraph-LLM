import openai
from config import DynaGraphConfig as config
import json

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
    max_tokens=1024
)
content = response.choices[0].message.content
print("RAW CONTENT:", repr(content))
        
if content.startswith("`json"):
    content = content.split("`json")[-1]
if content.startswith("`"):
    content = content.split("`")[-1]
if content.endswith("`"):
    content = content.rsplit("`", 1)[0]
    
print("PARSED:", repr(content.strip()))
try:
    result = json.loads(content.strip())
    print("SUCCESSFUL JSON DECODE!", result)
except json.JSONDecodeError as e:
    print("JSON DECODE ERROR:", e)
