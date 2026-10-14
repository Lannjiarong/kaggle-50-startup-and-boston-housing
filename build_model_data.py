import json

with open('teacher_model_data.js', 'r', encoding='utf-8') as f:
    startups_raw = f.read().split('=', 1)[1].strip().rstrip(';')
    startups_data = json.loads(startups_raw)

with open('boston_model_data.json', 'r', encoding='utf-8') as f:
    boston_data = json.load(f)

combined = {
    '50startups': startups_data,
    'boston': boston_data
}

js_content = 'window.PROBLEMS_DATA = ' + json.dumps(combined, indent=2) + ';\nwindow.MODEL_DATA = window.PROBLEMS_DATA["50startups"];\n'

with open('model_data.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print("model_data.js generated successfully! Size:", len(js_content))
