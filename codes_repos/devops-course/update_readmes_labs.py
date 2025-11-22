import os
import re

base_path = r"H:\My Drive\Codes & Repos\codes_repos\devops-course"

phases = [
    "phase1_beginner",
    "phase2_intermediate",
    "phase3_advanced"
]

def get_lab_title(file_path):
    """Extracts title from the first line of the markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith('# '):
                return first_line[2:].strip()
    except:
        pass
    return os.path.basename(file_path)

def update_readme(module_path):
    readme_path = os.path.join(module_path, "README.md")
    labs_path = os.path.join(module_path, "labs")

    if not os.path.exists(readme_path) or not os.path.exists(labs_path):
        return

    # Get list of labs
    labs = []
    for f in sorted(os.listdir(labs_path)):
        if f.endswith(".md"):
            full_path = os.path.join(labs_path, f)
            title = get_lab_title(full_path)
            labs.append(f"- [{title}](./labs/{f})")

    if not labs:
        return

    # Read README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create new Labs section
    new_labs_section = "## 🎯 Hands-on Labs\n\n" + "\n".join(labs) + "\n"

    # Regex to replace existing Hands-on Labs section
    # Matches from "## 🎯 Hands-on Labs" to the next "## " or End of File
    pattern = r"## 🎯 Hands-on Labs.*?(?=## |\Z)"
    
    if re.search(pattern, content, flags=re.DOTALL):
        new_content = re.sub(pattern, new_labs_section, content, flags=re.DOTALL)
    else:
        # Append if not found (unlikely based on template)
        new_content = content + "\n" + new_labs_section

    # Write back
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Updated {readme_path}")

for phase in phases:
    phase_dir = os.path.join(base_path, phase)
    if not os.path.exists(phase_dir):
        continue
        
    for module in sorted(os.listdir(phase_dir)):
        if module.startswith("module-"):
            update_readme(os.path.join(phase_dir, module))
