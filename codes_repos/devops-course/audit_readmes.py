import os

base_path = r"H:\My Drive\Codes & Repos\codes_repos\devops-course"
placeholders = [
    "[Comprehensive theoretical content",
    "[Detailed explanations",
    "[Industry-standard best practices",
    "[Key concept 1]"
]

files_to_update = []

for root, dirs, files in os.walk(base_path):
    if "README.md" in files:
        file_path = os.path.join(root, "README.md")
        # Skip the root READMEs, only check modules
        if "module-" not in file_path:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                for p in placeholders:
                    if p in content:
                        files_to_update.append(file_path)
                        break
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

print(f"Found {len(files_to_update)} READMEs with placeholders:")
for f in sorted(files_to_update):
    print(f)
