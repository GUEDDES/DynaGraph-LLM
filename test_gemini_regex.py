import openai
from config import DynaGraphConfig as config
import json
import re

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
    temperature=0.1
)
content = response.choices[0].message.content
print("RAW CONTENT:", repr(content))
        
match = re.search(r'\{.*\}', content, re.DOTALL)
if match:
    json_str = match.group(0)
    print("PARSED:", repr(json_str))
    try:
        result = json.loads(json_str)
        print("SUCCESS!", result)
    except Exception as e:
        print("ERROR:", e)
else:
    print("No JSON found.")
