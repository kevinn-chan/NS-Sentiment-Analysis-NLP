# Run just 1 row with debug output
import pandas as pd
from openai import OpenAI
import re, json

client = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1')
gold = pd.read_parquet('commitment_testset.parquet')
row = gold.iloc[0]

# Load prompt
exec(open('commitment_v2_prompt.py').read())

resp = client.chat.completions.create(
    model='qwen3:32b',
    temperature=0,
    max_tokens=60,
    messages=[
        {'role': 'system', 'content': '/no_think\n\n' + SYSTEM_PROMPT},
        {'role': 'user', 'content': f'Classify this text:\n\n{row["text"]}'},
    ],
)

raw = resp.choices[0].message.content
print('RAW OUTPUT:')
print(repr(raw))
print()
print('Trying to parse JSON:')
match = re.search(r'\{[^}]+\}', raw)
if match:
    print(json.loads(match.group()))
else:
    print('No JSON found!')

