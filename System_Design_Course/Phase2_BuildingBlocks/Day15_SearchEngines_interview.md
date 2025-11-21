# Day 15 Interview Prep: Search Engines

## Q1: How does Elasticsearch scale?
**Answer:**
*   **Sharding:** Splits index into smaller pieces. Allows parallel writes and storage.
*   **Replication:** Copies shards. Allows parallel reads and high availability.

## Q2: What is the difference between "Term" and "Match" query?
**Answer:**
*   **Term:** Exact match. `user_id = 123`. No analysis.
*   **Match:** Full-text search. `message = "hello world"`. Analyzes input (tokenizes) and searches index.

## Q3: How to handle "Typeahead" (Autocomplete)?
**Answer:**
*   **Trie (Prefix Tree):** Efficient for prefix lookups.
*   **N-Grams:** Store `app`, `appl`, `apple` in index.
*   **Fuzzy Search:** Levenshtein distance (handle typos).

## Q4: Why is Deep Paging slow?
**Answer:**
*   Query: `LIMIT 10 OFFSET 10000`.
*   Each shard must fetch top 10,010 results.
*   Coordinator must merge $N \times 10,010$ results.
*   **Solution:** Search After (Cursor-based). Use the sort value of the last result to fetch next page.
