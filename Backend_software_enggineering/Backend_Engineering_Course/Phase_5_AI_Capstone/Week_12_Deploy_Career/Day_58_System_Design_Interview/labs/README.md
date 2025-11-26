# Lab: Day 58 - Mock Interview

## Goal
Practice talking while designing.

## Prompt: Design a URL Shortener (TinyURL)

### Requirements
1.  **Functional**:
    *   `shorten(long_url) -> short_url`
    *   `redirect(short_url) -> long_url`
2.  **Non-Functional**:
    *   Highly Available (Redirects must work).
    *   Read-heavy (100:1 Read:Write).

### Checklist
- [ ] Did you ask about traffic volume? (e.g., 100M new URLs/month).
- [ ] Did you calculate storage? (100M * 12 months * 5 years * 500 bytes = ?).
- [ ] Did you choose a DB? (NoSQL vs SQL).
- [ ] Did you explain the Hashing Algorithm? (MD5 vs Base62).
- [ ] Did you handle collisions?

### Self-Review
*   Did I pause too much?
*   Did I draw clear diagrams?
*   Did I mention trade-offs?
