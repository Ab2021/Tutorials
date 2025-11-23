import pandas as pd
import os
from collections import Counter

# Load the CSV
df = pd.read_csv('650 ML and LLM use cases.csv')

# Filter for 2022 onwards
df_filtered = df[df['Year'] >= 2022]

print(f"Total use cases: {len(df)}")
print(f"Use cases from 2022+: {len(df_filtered)}")
print(f"\n{'='*80}\n")

# Analyze tags
print("TAG ANALYSIS")
print("="*80)

# Split tags to get individual tag counts
all_tags = []
for tag_str in df_filtered['Tag'].dropna():
    tags = [t.strip() for t in str(tag_str).split(',')]
    all_tags.extend(tags)

tag_counts = Counter(all_tags)
print(f"\nTop 30 individual tags:")
for tag, count in tag_counts.most_common(30):
    print(f"  {tag}: {count}")

# Suggest tag groupings
print(f"\n{'='*80}\n")
print("SUGGESTED TAG GROUPINGS")
print("="*80)

tag_groups = {
    'AI_Agents_and_LLMs': ['AI agents', 'LLM', 'generative AI', 'RAG', 'chatbot', 'voice interface'],
    'Recommendations_and_Personalization': ['recommender system', 'content personalization', 'ad ranking / targeting'],
    'Search_and_Retrieval': ['search', 'item classification'],
    'Computer_Vision': ['CV'],
    'NLP_and_Text': ['NLP', 'spam / content moderation'],
    'Operations_and_Infrastructure': ['ops', 'fraud detection', 'customer support'],
    'Prediction_and_Forecasting': ['demand forecasting', 'ETA prediction', 'pricing', 'churn prediction', 'propensity to buy', 'lead scoring'],
    'Product_Features': ['product feature'],
    'Causal_Inference': ['causality'],
    'Specialized': ['predictive maintenance', 'voice interface']
}

# Count cases in each group
for group_name, group_tags in tag_groups.items():
    cases = df_filtered[df_filtered['Tag'].apply(
        lambda x: any(tag in str(x) for tag in group_tags) if pd.notna(x) else False
    )]
    print(f"\n{group_name}: {len(cases)} cases")
    for tag in group_tags:
        count = sum(1 for t in all_tags if tag == t)
        if count > 0:
            print(f"  - {tag}: {count}")

# Industry breakdown
print(f"\n{'='*80}\n")
print("INDUSTRY BREAKDOWN (2022+)")
print("="*80)
industry_counts = df_filtered['Industry'].value_counts()
for industry, count in industry_counts.items():
    print(f"{industry}: {count}")

# Save filtered data
df_filtered.to_csv('filtered_use_cases_2022_onwards.csv', index=False)
print(f"\n{'='*80}\n")
print(f"Filtered data saved to: filtered_use_cases_2022_onwards.csv")
