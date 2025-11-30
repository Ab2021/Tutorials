import os
import re

ROOT_DIR = r"g:\My Drive\Codes & Repos\Insurance_Analytical_Modelling\Course_Content"

def audit_course_content():
    missing_files = []
    found_files = []
    
    # Regex to parse Day folders (e.g., "Day_01_Topic", "Day_118_120_Topic")
    day_folder_pattern = re.compile(r"Day_(\d+)(?:_(\d+))?.*")
    
    for root, dirs, files in os.walk(ROOT_DIR):
        # Skip Archive and hidden folders
        if "_Archive" in root or ".git" in root:
            continue
            
        for dirname in dirs:
            match = day_folder_pattern.match(dirname)
            if match:
                start_day = int(match.group(1))
                end_day = int(match.group(2)) if match.group(2) else start_day
                
                folder_path = os.path.join(root, dirname)
                
                for day in range(start_day, end_day + 1):
                    # Expected filename format: Day_XX_Theoretical_Deep_Dive.md
                    # Note: XX might be 1 or 2 or 3 digits. Usually padded to match folder? 
                    # Let's check for "Day_{day}_Theoretical_Deep_Dive.md" or "Day_{day:02d}_..." or "Day_{day:03d}_..."
                    
                    expected_name_base = f"Theoretical_Deep_Dive.md"
                    
                    # Try to find the file with any padding
                    found = False
                    possible_names = [
                        f"Day_{day}_{expected_name_base}",
                        f"Day_{day:02d}_{expected_name_base}",
                        f"Day_{day:03d}_{expected_name_base}"
                    ]
                    
                    existing_files = os.listdir(folder_path)
                    
                    for fname in possible_names:
                        if fname in existing_files:
                            found = True
                            found_files.append(os.path.join(folder_path, fname))
                            break
                    
                    if not found:
                        missing_files.append(f"Day {day} in {dirname}")

    print(f"Total Found: {len(found_files)}")
    print(f"Total Missing: {len(missing_files)}")
    if missing_files:
        print("\nMissing Days:")
        for item in sorted(missing_files, key=lambda x: int(re.search(r'Day (\d+)', x).group(1))):
            print(item)

if __name__ == "__main__":
    audit_course_content()
