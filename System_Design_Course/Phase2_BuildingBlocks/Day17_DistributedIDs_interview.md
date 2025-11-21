# Day 17 Interview Prep: Distributed IDs

## Q1: Why not use UUID?
**Answer:**
*   **Size:** 128-bit is 2x larger than 64-bit. Indexes grow bigger.
*   **Performance:** UUIDs are random. Inserting random keys into a B-Tree causes page splits and fragmentation. Sequential IDs (Snowflake) append to the end of the tree (Fast).

## Q2: How to handle clock synchronization in Snowflake?
**Answer:**
*   **NTP:** Network Time Protocol keeps servers synced.
*   **Clock Drift:** If clock moves backwards, the generator should refuse to generate IDs (throw error) to prevent duplicates.

## Q3: What if the Sequence Number overflows (4096 per ms)?
**Answer:**
*   Wait for the next millisecond.
*   Or, allocate more bits to Sequence and fewer to Machine ID.

## Q4: Design a URL Shortener ID generator.
**Answer:**
*   **Requirement:** Short string (7 chars).
*   **Base62:** A-Z, a-z, 0-9. $62^7 \approx 3.5$ Trillion combinations.
*   **Method:** Generate a 64-bit integer (Snowflake/DB), then convert to Base62.
    *   `12345` -> `dnh`.
