import pandas as pd
import os
import re
from pathlib import Path
from collections import defaultdict

# Load filtered data
df = pd.read_csv('filtered_use_cases_2022_onwards.csv')

# Define tag groupings with priority order
TAG_GROUPS = {
    '01_AI_Agents_and_LLMs': ['AI agents', 'LLM', 'generative AI', 'RAG', 'chatbot'],
    '02_Recommendations_and_Personalization': ['recommender system', 'content personalization', 'ad ranking / targeting'],
    '03_Search_and_Retrieval': ['search', 'item classification'],
    '04_Computer_Vision': ['CV'],
    '05_NLP_and_Text': ['NLP', 'spam / content moderation'],
    '06_Operations_and_Infrastructure': ['ops', 'fraud detection', 'customer support'],
    '07_Prediction_and_Forecasting': ['demand forecasting', 'ETA prediction', 'pricing', 'churn prediction', 'propensity to buy', 'lead scoring'],
    '08_Product_Features': ['product feature'],
    '09_Causal_Inference': ['causality'],
    '10_Specialized': ['predictive maintenance', 'voice interface']
}

# Base directory
BASE_DIR = 'ML_LLM_Use_Cases_2022+'

def sanitize_filename(text):
    """Convert text to safe filename"""
    # Remove special characters
    text = re.sub(r'[^\w\s-]', '', str(text))
    # Replace spaces with underscores
    text = re.sub(r'\s+', '_', text)
    # Limit length
    return text[:100]

def sanitize_foldername(text):
    """Convert industry name to safe folder name"""
    text = str(text).replace(',', '_').replace(' ', '_')
    text = re.sub(r'[^\w_-]', '', text)
    return text

def get_tag_group(tags_str):
    """Determine which tag group a use case belongs to"""
    if pd.isna(tags_str):
        return '10_Specialized'
    
    tags_lower = str(tags_str).lower()
    
    # Check each group in priority order
    for group_name, group_tags in TAG_GROUPS.items():
        for tag in group_tags:
            if tag.lower() in tags_lower:
                return group_name
    
    return '10_Specialized'

def create_markdown_content(row):
    """Create markdown content for a use case"""
    content = f"""# {row['Title']}

## Metadata
- **Company**: {row['Company']}
- **Industry**: {row['Industry']}
- **Year**: {row['Year']}
- **Tags**: {row['Tag']}

## Short Description
{row['Short Description (< 5 words)']}

## Link
[Read the full article]({row['Link']})

---
*This use case is part of the ML and LLM Use Cases collection (2022+)*
"""
    return content

# Create base directory
os.makedirs(BASE_DIR, exist_ok=True)

# Statistics tracking
stats = {
    'tag_groups': defaultdict(int),
    'industries': defaultdict(int),
    'years': defaultdict(int),
    'total_files': 0
}

# Track cases per folder
folder_contents = defaultdict(lambda: defaultdict(list))

print(f"Organizing {len(df)} use cases into folder structure...")
print("="*80)

# Process each use case
for idx, row in df.iterrows():
    # Determine tag group
    tag_group = get_tag_group(row['Tag'])
    
    # Sanitize industry name
    industry = sanitize_foldername(row['Industry'])
    
    # Create folder path
    folder_path = os.path.join(BASE_DIR, tag_group, industry)
    os.makedirs(folder_path, exist_ok=True)
    
    # Create filename
    company = sanitize_filename(row['Company'])
    title = sanitize_filename(row['Title'])
    filename = f"{row['Year']}_{company}_{title}.md"
    
    # Write markdown file
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(create_markdown_content(row))
    
    # Update statistics
    stats['tag_groups'][tag_group] += 1
    stats['industries'][row['Industry']] += 1
    stats['years'][row['Year']] += 1
    stats['total_files'] += 1
    
    # Track for README generation
    folder_contents[tag_group][industry].append({
        'filename': filename,
        'title': row['Title'],
        'company': row['Company'],
        'year': row['Year']
    })
    
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(df)} use cases...")

print(f"\n{'='*80}")
print(f"✓ Created {stats['total_files']} markdown files")
print(f"\nTag Group Distribution:")
for tag_group, count in sorted(stats['tag_groups'].items()):
    print(f"  {tag_group}: {count} cases")

# Create README files for each tag group folder
print(f"\n{'='*80}")
print("Creating README files for each tag group...")

TAG_GROUP_DESCRIPTIONS = {
    '01_AI_Agents_and_LLMs': 'AI Agents, Large Language Models, Generative AI, RAG, and Chatbots',
    '02_Recommendations_and_Personalization': 'Recommender Systems, Content Personalization, and Ad Targeting',
    '03_Search_and_Retrieval': 'Search Systems and Item Classification',
    '04_Computer_Vision': 'Computer Vision Applications',
    '05_NLP_and_Text': 'Natural Language Processing and Content Moderation',
    '06_Operations_and_Infrastructure': 'Operations, Fraud Detection, and Customer Support',
    '07_Prediction_and_Forecasting': 'Demand Forecasting, ETA Prediction, and Pricing',
    '08_Product_Features': 'Product Features and Capabilities',
    '09_Causal_Inference': 'Causal Inference and Analysis',
    '10_Specialized': 'Specialized Use Cases'
}

for tag_group in sorted(folder_contents.keys()):
    readme_path = os.path.join(BASE_DIR, tag_group, 'README.md')
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"# {tag_group.replace('_', ' ').title()}\n\n")
        f.write(f"**Category**: {TAG_GROUP_DESCRIPTIONS.get(tag_group, 'Various ML/LLM use cases')}\n\n")
        f.write(f"**Total Use Cases**: {stats['tag_groups'][tag_group]}\n\n")
        
        # Industry breakdown
        f.write("## Industry Breakdown\n\n")
        industries = folder_contents[tag_group]
        for industry in sorted(industries.keys()):
            count = len(industries[industry])
            f.write(f"- **{industry.replace('_', ' ')}**: {count} cases\n")
        
        # List all use cases by industry
        f.write("\n## Use Cases\n\n")
        for industry in sorted(industries.keys()):
            f.write(f"### {industry.replace('_', ' ')}\n\n")
            
            cases = sorted(industries[industry], key=lambda x: (x['year'], x['company']))
            for case in cases:
                f.write(f"- **[{case['year']}]** {case['company']}: [{case['title']}]({industry}/{case['filename']})\n")
            f.write("\n")

print(f"✓ Created README files for {len(folder_contents)} tag groups")

# Create main README
print(f"\n{'='*80}")
print("Creating main README.md...")

main_readme_path = os.path.join(BASE_DIR, 'README.md')
with open(main_readme_path, 'w', encoding='utf-8') as f:
    f.write("# ML and LLM Use Cases Collection (2022+)\n\n")
    f.write("A curated collection of 500+ real-world Machine Learning and Large Language Model use cases from leading tech companies, organized by technology and industry.\n\n")
    
    f.write("## 📊 Overview\n\n")
    f.write(f"- **Total Use Cases**: {stats['total_files']}\n")
    f.write(f"- **Years Covered**: {min(stats['years'].keys())} - {max(stats['years'].keys())}\n")
    f.write(f"- **Industries**: {len(stats['industries'])}\n")
    f.write(f"- **Categories**: {len(stats['tag_groups'])}\n\n")
    
    f.write("## 📁 Folder Structure\n\n")
    f.write("Use cases are organized into 10 main categories based on technology/approach:\n\n")
    
    for tag_group in sorted(stats['tag_groups'].keys()):
        count = stats['tag_groups'][tag_group]
        desc = TAG_GROUP_DESCRIPTIONS.get(tag_group, '')
        f.write(f"### [{tag_group.replace('_', ' ').title()}]({tag_group}/)\n")
        f.write(f"{desc}\n\n")
        f.write(f"**{count} use cases**\n\n")
    
    f.write("## 🏭 Industry Distribution\n\n")
    f.write("| Industry | Use Cases |\n")
    f.write("|----------|----------|\n")
    for industry, count in sorted(stats['industries'].items(), key=lambda x: -x[1])[:15]:
        f.write(f"| {industry} | {count} |\n")
    
    f.write("\n## 📅 Year Distribution\n\n")
    f.write("| Year | Use Cases |\n")
    f.write("|------|----------|\n")
    for year, count in sorted(stats['years'].items()):
        f.write(f"| {year} | {count} |\n")
    
    f.write("\n## 🔍 How to Navigate\n\n")
    f.write("1. Browse by **technology category** using the main folders above\n")
    f.write("2. Within each category, find use cases organized by **industry**\n")
    f.write("3. Each use case includes:\n")
    f.write("   - Company name and metadata\n")
    f.write("   - Short description\n")
    f.write("   - Link to the full article/blog post\n\n")
    
    f.write("## 📈 Top Tags\n\n")
    top_tags = [
        ("LLM", 174),
        ("Generative AI", 129),
        ("Operations", 90),
        ("Product Features", 83),
        ("Recommender Systems", 79),
        ("Search", 65)
    ]
    for tag, count in top_tags:
        f.write(f"- **{tag}**: {count} cases\n")
    
    f.write("\n---\n\n")
    f.write("*Data compiled from industry blogs, engineering posts, and technical articles.*\n")

print(f"✓ Created main README.md")

print(f"\n{'='*80}")
print("✅ Organization complete!")
print(f"\nFolder structure created at: {os.path.abspath(BASE_DIR)}")
print(f"\nSummary:")
print(f"  - {len(stats['tag_groups'])} main tag folders")
print(f"  - {stats['total_files']} markdown files")
print(f"  - {len(stats['industries'])} industries represented")
