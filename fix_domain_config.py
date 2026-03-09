import json

INPUT_FILE = "domain_config.json"
OUTPUT_FILE = "domain_config.json"

clean_lines = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.lstrip()
        # Skip full-line // comments
        if stripped.startswith("//"):
            continue
        # Remove inline // comments
        if "//" in line:
            line = line.split("//")[0] + "\n"
        clean_lines.append(line)

clean_text = "".join(clean_lines).strip()

# Validate JSON
data = json.loads(clean_text)

# Rewrite clean JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("✅ domain_config.json cleaned (// comments removed) and validated")
