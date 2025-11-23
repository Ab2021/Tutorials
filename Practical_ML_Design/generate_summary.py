import os
import glob

# Statistics
base_dir = 'ML_LLM_Use_Cases_2022+'

print("=" * 80)
print("FINAL ORGANIZATION SUMMARY")
print("=" * 80)

# Count main folders
main_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
print(f"\n✓ Main Tag Folders: {len(main_folders)}")

# Count all markdown files
all_md_files = glob.glob(f'{base_dir}/**/*.md', recursive=True)
use_case_files = [f for f in all_md_files if 'README' not in f]
readme_files = [f for f in all_md_files if 'README' in f]

print(f"✓ Use Case Files: {len(use_case_files)}")
print(f"✓ README Files: {len(readme_files)}")
print(f"✓ Total Markdown Files: {len(all_md_files)}")

# Count industry subfolders
print(f"\n{'=' * 80}")
print("FOLDER STRUCTURE BREAKDOWN")
print("=" * 80)

for folder in sorted(main_folders):
    folder_path = os.path.join(base_dir, folder)
    industries = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    use_cases_in_folder = glob.glob(f'{folder_path}/**/*.md', recursive=True)
    use_cases_in_folder = [f for f in use_cases_in_folder if 'README' not in f]
    
    print(f"\n{folder}:")
    print(f"  Industries: {len(industries)}")
    print(f"  Use Cases: {len(use_cases_in_folder)}")

# Sample files
print(f"\n{'=' * 80}")
print("SAMPLE USE CASE FILES")
print("=" * 80)

sample_files = use_case_files[:5]
for i, file in enumerate(sample_files, 1):
    rel_path = os.path.relpath(file, base_dir)
    print(f"{i}. {rel_path}")

print(f"\n{'=' * 80}")
print("✅ ORGANIZATION COMPLETE!")
print(f"{'=' * 80}")
print(f"\nAll files are located in: {os.path.abspath(base_dir)}")
print(f"\nYou can start exploring by opening: {os.path.abspath(os.path.join(base_dir, 'README.md'))}")
