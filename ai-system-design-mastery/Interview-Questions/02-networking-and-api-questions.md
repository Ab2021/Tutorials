# Networking & API — 45 Interview Questions with Model Answers

> **TL;DR** — This is the drill file for the networking and API half of a system-design interview. Every question has a **model answer** written the way you would actually say it, plus, where it matters, a **`Probe:`** follow-up the interviewer will use to find the bottom of your knowledge, and a **`Weak:` vs `Strong:`** contrast so you can calibrate. The four topics that produce the most interview signal are: **DNS** (because it is the first hop and the first thing that breaks), **TLS termination** (because almost nobody can explain where the certificate lives), **HTTP version choice** (because it is a real engineering trade, not a checkbox), and **idempotency + rate limiting** (because those are where API design meets distributed systems). Read the questions in order once; then live in §5 onwards, which is where the hard probes are.

---

## How to use this file

| Symbol | Meaning |
|---|---|
| **Q** | The question, phrased as an interviewer would ask it |
| **A** | A model answer you could say out loud, roughly 30–90 seconds |
| **Probe** | The follow-up the interviewer uses to test depth |
| **Weak / Strong** | A contrast to calibrate your answer |
| **Level** | E = easy/warm-up, M = medium, H = hard, P = pushback |

**The three rules that apply to every answer in this file:**
1. **Name the default and the number.** "TTL is typically 300 seconds" beats "the TTL is configurable".
2. **Name the failure mode.** Every mechanism in networking exists because something breaks without it.
3. **Name the trade.** There is no correct answer, only a correct trade, and saying which one you made is what is being graded.

---

## 1. DNS (Q1–Q8)

### Q1. Walk me through what happens when I type `example.com` into a browser. — E

**A:** Roughly: the browser checks its own cache and the OS resolver cache. If there is no answer, the OS sends a recursive query to the configured resolver — typically the ISP's, or a public one like 8.8.8.8 or 1.1.1.1. That resolver walks the DNS hierarchy: it asks a root server, which returns the `.com` TLD nameservers; it asks a `.com` nameserver, which returns the authoritative nameservers for `example.com`; it asks the authoritative server, which returns the A or AAAA record. The resolver caches each answer according to its TTL and returns the IP to the OS, which hands it to the browser. Then the browser opens a TCP connection — a three-way handshake — does a TLS handshake if it is HTTPS, and sends the HTTP request. With HTTP/2 or HTTP/3 and a warm connection, later requests skip most of that.

**Probe: how many round trips before the first byte of the response?**
Cold, with no caches: DNS is typically 1–2 round trips to a nearby resolver but 4+ if the resolver has to walk the hierarchy; TCP is 1 RTT for the handshake; TLS 1.3 is 1 RTT (TLS 1.2 is 2); the HTTP request is 1 more. So roughly 4–8 RTTs cold, and 1 RTT warm. That is why connection reuse and TLS resumption matter so much — and why HTTP/3 can get to 0-RTT on a resumed connection.

### Q2. Why does DNS use UDP, and when does it use TCP? — E

**A:** DNS uses UDP because the query and response are tiny and a single datagram avoids a handshake — one round trip instead of three. It falls back to TCP in three cases: when the response is larger than 512 bytes (the classic limit) and the server sets the TC — truncated — bit, so the client retries over TCP; for zone transfers (AXFR/IXFR) between nameservers, which are bulk operations; and when DNSSEC signatures push a response past the UDP size. Modern EDNS0 raises the advertised UDP payload size to typically 1232–4096 bytes, which is why large DNSSEC answers can still fit in UDP — and 1232 is the common choice because it avoids IP fragmentation on paths with a 1280-byte MTU.

**Probe: why 1232 specifically?** It is 1280 (the IPv6 minimum MTU) minus 48 bytes of IPv6 and UDP header overhead. Choosing it avoids fragmentation, and fragmentation is dangerous because many middleboxes drop fragmented UDP.

### Q3. What is a TTL and what happens when you lower it? — M

**A:** The TTL is how long a resolver may cache a record, in seconds. To migrate a service, you lower the TTL well in advance of the change — typically to 60 seconds — wait for the old, longer TTL to expire everywhere, then make the change, and raise the TTL back afterwards. The critical detail is the ordering: if you change the record first and lower the TTL second, resolvers that already cached the old record with a 24-hour TTL will keep serving the old answer for up to 24 hours, and your migration appears to work for some users and not others.

**Probe: the TTL was 86400 and you just changed the record. How long is the outage?** Up to 24 hours for the resolvers that cached it, and you cannot force it shorter. Your only levers are: keep the old endpoint running until the TTL has fully drained, or accept partial failure and monitor.

**Weak:** "Set the TTL low so changes propagate fast."
**Strong:** "Lower the TTL at least two old-TTL periods before the change, verify with multiple resolvers, make the change, then raise it back after confirming."

### Q4. A record, CNAME, and the apex problem. — M

**A:** An A record maps a name directly to an IPv4 address; AAAA is the IPv6 equivalent. A CNAME says "this name is an alias for that name" — resolvers follow it and use the target's records. The apex problem: the DNS specification forbids a CNAME at the zone apex (`example.com` itself) because a CNAME cannot coexist with the SOA and NS records the apex requires. So if you want `example.com` to point at a cloud load balancer whose IP is not stable, you cannot CNAME it. The standard solutions are ALIAS or ANAME records (a provider-specific extension that resolves the target server-side and returns an A record), or the provider's own apex flattening, which is what Route 53, Cloudflare and others do.

**Probe: why does a CNAME break email?** Because the MX record lookup is separate, and if the apex is a CNAME the SOA and NS records are shadowed, which breaks the zone semantics; also, a CNAME at the apex commonly breaks the SPF lookup because the TXT record cannot coexist either.

### Q5. How does DNS-based load balancing work, and what are its limits? — M

**A:** The authoritative server returns multiple A records, or rotates the order, or returns a subset based on health or geography. The client picks one, usually the first, though many clients and resolvers ignore the ordering entirely. Its limits are significant: the resolver caches the answer for the TTL, so you cannot react quickly to failure; a single resolver serves many clients, so you cannot count requests per client; clients ignore TTLs and reorder anyway; and the "load balancing" is really load *distribution*, with no feedback loop. It is useful as a coarse first tier — routing a continent to a region — and not as a substitute for a real load balancer.

**Probe: how do you fail over with DNS?** Health checks at the DNS provider remove the unhealthy record, but the failover time is bounded below by the TTL, and any client with a cached record keeps hitting the dead endpoint. That is why any serious failover also needs the endpoint itself to fail fast or the connection to be rerouted — DNS alone is not a failover mechanism.

### Q6. What is anycast and why does it matter? — H

**A:** Anycast advertises the same IP address from many locations; the internet's routing then delivers each user's packets to the topologically nearest advertisement. It is how root DNS servers, public resolvers, and most large CDNs work. The benefits are a shorter path and natural DDoS absorption, because traffic is spread across many points. The catch is that BGP is not stable: a route change can move a user's traffic from one site to another mid-session, which breaks established TCP connections. That is precisely why TCP-based anycast is used for stateless request/response services and UDP-based protocols, and why long-lived TCP sessions over anycast need care.

**Probe: why is HTTP/3 a better fit for anycast than HTTP/2?** HTTP/3 runs over QUIC, which has connection IDs independent of the IP address and port, so a connection can survive an address change — the very thing BGP route changes cause. With TCP, a route change breaks the connection and the client has to reconnect.

### Q7. What is DNSSEC, and why is adoption low? — H

**A:** DNSSEC adds cryptographic signatures to DNS records so a resolver can verify that an answer came from the zone's authoritative server and was not tampered with. It provides authenticity and integrity, not confidentiality. Adoption is low for three reasons: the operational complexity is real — key rotation, signature expiry, and a zone goes dark if signatures lapse; signed responses are larger, which pushes some answers over UDP limits and onto TCP; and it protects the wrong layer for most threat models, since the actual attack is usually a hijacked registrar account or a compromised resolver, neither of which DNSSEC prevents. DoH and DoT solve the confidentiality problem separately, and are more directly useful for most deployments.

### Q8. Your service is unreachable for some users and fine for others. Where do you look? — P

**A:** I would work outward from the client. First, is it DNS — do the affected users resolve to the right IP? Check with several public resolvers, and check whether the TTL change has drained. Second, is it routing — can those users reach the IP at all? A traceroute from an affected network distinguishes "no route" from "route exists but the service rejects". Third, is it TLS — a certificate that is valid for one SNI name and not another produces a hard failure for those users only; check the SNI the client sends and whether the cert covers it. Fourth, is it geo or policy — a WAF or geo-block rejecting a country or an ASN, which shows up as a 403 rather than a timeout. Fifth, is it an MTU or fragmentation issue on a specific access network, which typically manifests as a hang after the TLS handshake rather than a clean failure.

**Strong:** "The distinguishing test is whether the failure is a timeout, a connection reset, or an HTTP error — those three point at routing, TLS, and policy respectively, and separating them cuts the search space immediately."

---

## 2. CDN (Q9–Q15)

### Q9. What does a CDN actually do for you? — E

**A:** Three things, in order of importance. It **reduces latency** by serving content from a point of presence near the user rather than from an origin that may be on another continent. It **absorbs load and bandwidth** — the origin sees a small fraction of the requests, and the CDN's edge capacity is much larger than yours. And it **absorbs attacks**, because a DDoS is spread across hundreds of edge locations rather than concentrated on your origin. The cost model is the reason it matters commercially: origin egress is typically around $0.09/GB while a CDN's is a fraction of that, and CDN-to-origin fetches are usually free, so a high cache hit ratio is a direct cost saving as well as a latency one.

### Q10. Explain cache keys and why they are the most common CDN misconfiguration. — M

**A:** A cache key is the tuple of request attributes the CDN uses to decide whether two requests are the same object. Typically it is the host plus the path, plus, depending on configuration, the query string and selected headers. The classic bug is a cache key that does not vary on something that changes the response: you serve the same cached HTML to logged-in and logged-out users, or you cache a response that contains one user's data. The reverse bug is a cache key that varies on too much — for example, including a `Cookie` header or a per-request signed token in the key — which means every request is a miss and the CDN does nothing. Both are cache-key configuration problems, and they are the most common CDN incident in practice.

**Probe: how do you cache content for logged-in users?** You do not cache personalised HTML at the CDN at all; you cache the static assets and the public API responses, and you render personalisation client-side or via an edge function. If you must cache it, you vary on a normalised, non-secret header and you accept that the cache hit ratio will be poor.

### Q11. Cache-control directives you actually use. — M

**A:** `max-age` is the freshness lifetime in seconds. `s-maxage` overrides `max-age` for shared caches, so you can have a short browser TTL and a long CDN TTL. `public` and `private` control whether a shared cache may store it. `no-cache` means "store it but revalidate before use", which is not the same as `no-store`, which means "do not store it at all" — conflating those two is a common interview slip. `must-revalidate` forbids serving stale when revalidation fails. `stale-while-revalidate` serves stale content for a bounded window while refreshing in the background, which is the single most useful directive for perceived performance. `stale-if-error` serves stale content when the origin is down, which converts an origin outage into a slightly-stale page rather than an error. And `immutable` tells the browser never to revalidate, which is correct for content-addressed filenames.

**Strong:** "My default set is `public, max-age=31536000, immutable` for content-hashed assets, `public, s-maxage=300, stale-while-revalidate=60` for API GETs, and `private, no-store` for anything user-specific."

### Q12. How do you invalidate a CDN cache? — M

**A:** Three ways, in order of preference. **Content-addressed URLs** — put a hash of the content in the filename — so a change produces a new URL and no invalidation is ever needed; this is the right answer for static assets and is why build tools do it. **Purge by URL or by tag/surrogate key**, which is a CDN API call, is fast but is per-object and may be rate-limited or billed. **Short TTLs**, which make the problem disappear by making the cache nearly useless. The important operational point is that a purge is not atomic across a global CDN — different edge locations will be serving different versions for a window of seconds to minutes — so a purge is eventual, and a design that requires instant global consistency cannot rely on it.

**Probe: a purge takes 30 seconds and a user sees the old page. Now what?** Either accept it, or make the URL change (content addressing or a version query parameter), which is instantly and atomically correct because it is a different cache key. Version-in-URL is the standard escape hatch.

### Q13. What is a surrogate key / cache tag? — H

**A:** It is a label attached to a cached response via a header, so that a purge can target the label rather than the URL. If a product page appears at `/product/123` and also inside a category listing and a search result, tagging all of those responses with `product-123` lets you purge all of them with one call when the product changes. Without tags you would have to know every URL that embeds the product, which is usually not knowable. The trade-off is that tags must be attached when the response is cached, so the origin has to compute them, and the number of tags per response is usually capped by the CDN.

### Q14. How does an origin know a request came from a CDN? — M

**A:** Via headers the CDN adds — `X-Forwarded-For` (and `Forwarded` in the standard form) carrying the original client IP, plus CDN-specific headers such as `X-Cache`, `X-Cache-Hits`, `CF-Ray`, or `X-Amz-Cf-Id`. The security-critical point is that **these headers are client-supplied and trivially spoofable unless you lock the origin down**. If your origin accepts connections from anywhere and trusts `X-Forwarded-For`, an attacker sets it to `127.0.0.1` and bypasses your IP-based rate limiting. The fix is to firewall the origin to the CDN's published IP ranges, and ideally to authenticate the CDN with a shared secret header or mTLS.

**Weak:** "We read the client IP from X-Forwarded-For."
**Strong:** "We read it from X-Forwarded-For, but the origin only accepts traffic from the CDN's IP ranges and requires a shared secret header, otherwise the value is attacker-controlled."

### Q15. When would you NOT use a CDN? — H

**A:** When the content is genuinely uncacheable and the connection setup dominates — a purely dynamic API where every response is unique and the origin already has a short path to the user, in which case the CDN adds a hop without a hit. When the data is so sensitive that you cannot put it in third-party infrastructure at all, though most CDNs now offer private origin or edge functions for that. When the workload is a small internal tool with users on the corporate network, where the CDN does nothing. And when the traffic is so small that the CDN's fixed cost exceeds the egress it saves. That said, the modern answer is usually "use it anyway", because even an uncacheable API benefits from the CDN's TLS termination, connection reuse from edge to origin, and DDoS absorption — the point being that you do not have to cache to benefit.

---

## 3. Proxies (Q16–Q21)

### Q16. Forward proxy versus reverse proxy. — E

**A:** A forward proxy sits in front of **clients** and acts on their behalf — the origin server sees the proxy's IP, not the client's. It is used for egress filtering, caching for a client population, and anonymisation. A reverse proxy sits in front of **servers** and acts on their behalf — the client thinks it is talking to the origin. It is used for TLS termination, load balancing, caching, compression, request routing and protection. The naming is from the origin's point of view: a forward proxy is forward of the client; a reverse proxy is in front of the server. Many pieces of software — nginx, Envoy, HAProxy — can do both; the difference is deployment and configuration, not the binary.

### Q17. What does a reverse proxy give you that a load balancer does not? — M

**A:** Overlapping but not identical roles. A load balancer's job is distribution and health; a reverse proxy's job is request handling. In practice a modern L7 load balancer *is* a reverse proxy, and the useful distinction is which features you rely on. The reverse-proxy-specific value is: TLS termination and certificate management, request and response transformation (header rewriting, compression, buffering), caching, connection pooling to the backend, protocol translation (HTTP/1.1 or HTTP/2 on the client side, HTTP/1.1 or gRPC on the backend), and applying policy — auth, rate limiting, WAF rules. A pure L4 load balancer gives you none of that.

**Probe: what is the cost of TLS termination at the proxy?** The proxy has to do the handshake and symmetric crypto for every connection. Modern hardware does this at tens of thousands of handshakes per second per core, so it is rarely the bottleneck, but it does mean the proxy sees plaintext, which has compliance implications for regulated data — the answer being re-encryption to the backend or an end-to-end TLS pass-through.

### Q18. Explain the `X-Forwarded-*` headers and the traps. — M

**A:** `X-Forwarded-For` is a comma-separated list of IPs, appended to by each proxy, so the leftmost entry is nominally the client and the rightmost is the last proxy. `X-Forwarded-Proto` says whether the original request was HTTP or HTTPS, which matters for redirect logic and for generating absolute URLs. `X-Forwarded-Host` preserves the original host. The traps: (1) they are append-only and client-supplied, so you cannot trust the leftmost entry unless your edge proxy overwrites rather than appends; (2) each proxy in the chain must be configured to append, and a misconfigured one will drop the header entirely; (3) the correct answer for "what is the client IP" is to take the *rightmost* entry you do not control, not the leftmost. The standard `Forwarded` header (RFC 7239) has the same semantics with better structure, and the same trust problem.

### Q19. What is TLS termination, and what are the alternatives? — M

**A:** TLS termination is where the TLS connection ends — where the certificate lives and where the plaintext begins. There are three placements. **At the edge** (load balancer or CDN): simplest, lets you inspect and route on the request, and concentrates certificate management in one place; the trade is that everything behind the edge is plaintext, so you must trust the internal network. **Re-encryption** (terminate at the edge, then open a new TLS connection to the backend): keeps the internal traffic encrypted, at the cost of two handshakes and more CPU. **Pass-through** (the edge forwards encrypted bytes and the backend terminates): end-to-end encryption, but the edge cannot route on anything in the request, cannot cache, and cannot apply L7 policy — you are back to an L4 load balancer.

**Probe: how does the edge know which certificate to present in pass-through?** It uses SNI — the hostname the client sends in the TLS ClientHello — to pick a backend to route to, without decrypting. That works, but it means routing is coarse: only the hostname, nothing in the path or headers.

### Q20. What is SNI, and what breaks without it? — M

**A:** SNI — Server Name Indication — is an extension in the TLS ClientHello where the client states the hostname it is trying to reach, in the clear, before the certificate is presented. Without it, a server with one IP address could only serve one certificate, so you would need one IP per hostname. SNI is what allows shared hosting and CDNs to serve millions of certificates from shared infrastructure. What breaks: very old clients that do not send SNI (a vanishing set); and, more relevantly, the fact that SNI is **plaintext**, so an observer can see which hostname you are visiting even over HTTPS. That is what Encrypted Client Hello (ECH) addresses, and it is why SNI-based blocking works at the network level.

### Q21. A proxy adds 40 ms of latency. How do you find out why? — P

**A:** I would break the added latency into its components and measure each. First, connection establishment — is the proxy opening a new backend connection per request, or pooling? If it is not pooling, that is a TCP plus TLS handshake on every request, which is easily 20–40 ms. Second, the proxy's own processing — enabling access logs with response-time breakdowns (`$request_time` versus `$upstream_response_time` in nginx) separates "the proxy is slow" from "the backend is slow". Third, buffering settings — a proxy that buffers the whole response before forwarding to the client adds the backend's full latency to the client's time-to-first-byte; `proxy_buffering off` or streaming helps for large responses. Fourth, DNS resolution on the proxy — if the upstream is a hostname and the proxy resolves it per request, that is a resolver round trip. Fifth, TLS to the backend — re-encryption doubles the handshake cost.

**Strong:** "`request_time` and `upstream_response_time` give the split in one line; if `request_time` is much larger than `upstream_response_time`, it is the proxy's own work or the client connection, not the backend."

---

## 4. API gateway & rate limiting (Q22–Q29)

### Q22. What is an API gateway and why not just a reverse proxy? — M

**A:** A reverse proxy routes and terminates; an API gateway adds API-level concerns: authentication and authorisation, API key and quota management, request validation against a schema, request and response transformation, per-route rate limiting, versioning and deprecation, analytics and logging, and often a developer portal. The practical difference is that the gateway understands the *API*, not just the HTTP. That said, the distinction has blurred — Envoy, Kong and AWS API Gateway are all reverse proxies with an API-management layer — and the honest answer is that an API gateway is a reverse proxy with policy and lifecycle features attached.

**Probe: what is the downside of a gateway?** It is a shared dependency and a potential single point of failure and a latency hop, and it concentrates configuration in one place where a mistake affects every service. Mitigations: run it in a highly available configuration, keep its per-request work small, and treat its configuration as code with review.

### Q23. Compare rate-limiting algorithms. — M

**A:** Four, with distinct trade-offs.

| Algorithm | How it works | Strength | Weakness |
|---|---|---|---|
| **Fixed window** | A counter per window (e.g. per minute) | Trivial, cheap | The boundary burst problem: 100 requests at 12:00:59 and 100 at 12:01:00 gives 200 in one second |
| **Sliding window log** | Store every request timestamp, count in the last N seconds | Exact | Memory scales with the request count |
| **Sliding window counter** | Weighted sum of the current and previous window | Approximate, cheap, smooths the boundary | Slightly approximate |
| **Token bucket** | A bucket refills at a rate R, capacity B; each request takes a token | **Allows bursts up to B while enforcing an average of R** | Two parameters to tune |
| **Leaky bucket** | A queue drains at a fixed rate | Perfectly smooth output | Adds latency; bursts are queued, not absorbed |

**The answer I would give:** token bucket for almost everything, because "average rate plus bounded burst" matches how real clients behave. Sliding window counter if I need a cheap exact-ish count for a quota. Leaky bucket only when downstream capacity genuinely requires smooth output.

**Probe: how do you rate limit across many gateway instances?** The counter must be shared — Redis with `INCR` and `EXPIRE`, or a Lua script for atomicity. That adds a network hop and a dependency; the local alternative is per-instance limits, which give you N× the intended global limit and are acceptable only when approximate limiting is fine. A middle path is consistent hashing of the limit key to a specific Redis shard to reduce contention.

### Q24. Which key do you rate limit on? — M

**A:** It depends on what you are protecting. **Per user** (from the authenticated identity) for fairness and quota. **Per IP** for unauthenticated traffic, accepting that NAT and mobile carriers put thousands of users behind one IP, so the limit must be generous or it punishes a whole corporate network. **Per API key** for partner quotas. **Per endpoint** to protect an expensive operation specifically. **Global** as a circuit breaker to protect a downstream dependency. In practice you layer them: a generous per-IP limit to stop crude abuse, a tighter per-user limit for fairness, and per-endpoint limits on anything expensive. And you always apply them at the edge, before the request costs you anything.

**Probe: a user is behind a NAT shared with 10,000 others. What happens?** They get rate limited because of their neighbours. That is why per-IP limits have to be generous, and why authenticated per-user limits are the real mechanism. There is no perfect answer at the IP layer; the honest statement is that IP-based limiting is a blunt first filter, not a fairness mechanism.

### Q25. How do you communicate a rate limit to a client? — M

**A:** With `429 Too Many Requests` plus headers. The IETF draft standard is `RateLimit-Limit`, `RateLimit-Remaining` and `RateLimit-Reset`, with the older, still widely-used `X-RateLimit-*` variants. `Retry-After` tells the client when to come back — in seconds or as an HTTP date — and it is the header well-behaved clients actually honour. The design advice that matters: **return the limit headers on successful responses too**, not just on 429s, so a client can pace itself and never hit the limit at all; and **make `Retry-After` accurate**, because a client that retries too early wastes a request and one that waits too long under-uses its quota.

### Q26. What is an API key, and is it authentication? — E

**A:** An API key is a bearer token identifying an application, not a user. It is **identification, not authentication** — anybody who has the string is treated as the application, and keys routinely leak through client-side code, logs and repositories. So: use keys for identification, quota and analytics; put them in a header (`Authorization` or a dedicated header), never in a URL where they land in logs and referrers; rotate them; and require a real authentication mechanism — OAuth 2.0 with short-lived tokens, or mTLS for machine-to-machine — for anything that actually needs to be secure.

**Probe: where should an API key never appear?** In a URL query string, in a browser-side JavaScript bundle, in a mobile app binary, or in a log line. The first three are extraction; the fourth is leakage.

### Q27. Explain idempotency in an API and how you implement it. — M

**A:** An operation is idempotent if applying it repeatedly has the same effect as applying it once. GET, PUT and DELETE are idempotent by HTTP semantics; POST is not. For POST, the implementation is an **idempotency key**: the client generates a unique key per logical operation and sends it in a header; the server stores the key with the response, and a repeat of the same key returns the stored response rather than re-executing. The storage is typically Redis with a TTL — 24 hours is common — and the write must be atomic (a `SET NX`, or an insert with a unique constraint) so two simultaneous requests with the same key cannot both execute.

The details that matter: the key must be scoped to the user or account, or one user's key can collide with another's; the stored response must include the status code, not just the body; and a request with the same key but a different body should be rejected as a conflict rather than silently returning the first result.

**Probe: what if the first request is still in flight when the second arrives?** Then `SET NX` fails but there is no stored response yet. The correct behaviour is to return `409 Conflict` or a `Retry-After`, not to execute again. Some implementations store an "in progress" marker and return 409 while it is set.

### Q28. Where do you enforce authorisation — gateway or service? — H

**A:** Both, for different reasons. The gateway enforces **coarse** authorisation — is this token valid, does this client have access to this API at all, has the quota been exceeded — because that is cheap, applies uniformly, and protects the services. The service enforces **fine-grained** authorisation — can this user read this specific record — because only the service knows the domain. The failure mode of doing it only at the gateway is that any request that reaches the service directly bypasses it, and services are frequently reachable internally; the failure of doing it only at the service is that every service reimplements token validation and they diverge. So: gateway for identity and coarse scope, service for record-level checks.

**Strong:** "And the service must never assume the gateway checked anything, because the gateway might be bypassed or misconfigured — defence in depth means the service validates the token too, even if it is the same token."

### Q29. A partner integrates with your API badly — retry storms, duplicate writes. Design for it. — P

**A:** Assume the client is buggy and make the server defensive. Concretely: (1) **idempotency keys on every write endpoint**, so their retries cannot create duplicates — and make the key required, not optional, for writes that matter. (2) **Rate-limit and shape their traffic** with a token bucket that allows a reasonable burst; a client that retries immediately after a 429 should get another 429, not a queue slot. (3) **`Retry-After` on every 429 and 503**, with an accurate value. (4) **Circuit-break per partner**, so a misbehaving integration degrades only itself, not the shared tier. (5) **Backpressure**, so under sustained overload you shed the partner's low-priority traffic rather than degrading everyone. (6) **Observability per client**, so you can see which key is generating the storm and contact them.

**Strong:** "The design principle is that the server must be correct in the presence of a client that retries at the wrong time, because that client exists and you cannot change it."

---

## 5. HTTP versions, TLS, and protocols (Q30–Q36)

### Q30. Compare HTTP/1.1, HTTP/2 and HTTP/3. — M

**A:**

| | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| Transport | TCP | TCP | QUIC over UDP |
| Multiplexing | None — one request per connection at a time | Yes — many streams over one connection | Yes, with no head-of-line blocking |
| Head-of-line blocking | At the connection level | At the TCP level (one lost packet stalls all streams) | Eliminated for independent streams |
| Header compression | None | HPACK | QPACK |
| Server push | No | Yes (largely deprecated in practice) | Yes |
| Connection setup | TCP + TLS | TCP + TLS | QUIC handshake includes TLS 1.3; 0-RTT on resumption |
| Typical use | Legacy, simple services | **The default for APIs and web** | Latency-sensitive and mobile |

The one-line version: HTTP/1.1's problem was connections, HTTP/2 fixed that but inherited TCP's loss behaviour, and HTTP/3 fixes the loss behaviour by moving off TCP.

**Probe: what is the classic HTTP/1.1 workaround and why does it hurt?** Domain sharding — serving assets from multiple hostnames to get more parallel connections — plus concatenating files. Both are workarounds for the 6-connections-per-host limit, and both hurt under HTTP/2 because they defeat multiplexing and break caching granularity.

### Q31. Why does HTTP/2 need HPACK and what is the security catch? — H

**A:** HTTP/2 multiplexes many requests on one connection, so header overhead — which is often 500–800 bytes of cookies, user agent and accept headers per request — is paid on every stream. HPACK compresses headers using a static table of common header names and values plus a dynamic table that both ends maintain, so a repeated header is sent as a short index. The security catch is that compression plus a shared dynamic table leaks information: if an attacker can inject content into the request and observe the compressed length, they can learn things about secrets in the headers — this is CRIME and BREACH for TLS compression and HTTP compression respectively. The mitigation is to disable header compression for sensitive values or to avoid reflecting attacker-controlled input into a compressed response alongside a secret.

### Q32. TLS 1.2 versus TLS 1.3. — M

**A:** TLS 1.3 reduces the handshake from two round trips to one, and to zero on resumption with a pre-shared key. It removes the legacy algorithms — RSA key exchange, CBC-mode ciphers, RC4 — leaving only forward-secret constructions with AEAD. It encrypts more of the handshake, so the certificate is no longer sent in the clear. It simplifies the cipher suite negotiation, which removed a whole class of downgrade attacks. And it supports 0-RTT, which allows data in the first flight — with the important caveat that 0-RTT data is replayable, so it must only be used for idempotent requests.

**Probe: what is forward secrecy and why does it matter?** A captured session cannot be decrypted later even if the server's long-term private key is compromised, because each session uses an ephemeral key. It matters because the realistic threat is a bulk capture now and a key compromise later.

### Q33. What is mTLS and when do you use it? — M

**A:** Mutual TLS: both sides present certificates, so the client authenticates to the server and the server to the client. It is used for machine-to-machine communication inside a service mesh, for high-security partner APIs, and for device identity. Its advantages are that it authenticates at the transport layer before any application code runs, and that it is not a bearer token, so there is nothing to steal and replay from a log. Its costs are real: certificate issuance, distribution, rotation, and revocation, plus the operational pain of revocation checking — CRL is slow, OCSP adds a round trip and leaks browsing patterns, and OCSP stapling solves the leak but requires server support. In practice, a service mesh with short-lived certificates issued by a local CA is the workable model.

### Q34. REST, gRPC, or GraphQL? — M

**A:** They solve different problems.

| | REST | gRPC | GraphQL |
|---|---|---|---|
| Transport | HTTP/1.1 or 2 | HTTP/2 (required) | HTTP/1.1 or 2 |
| Payload | JSON (usually) | Protobuf (binary) | JSON |
| Schema | OpenAPI, optional | `.proto`, enforced | SDL, enforced |
| Streaming | Limited (chunked, SSE) | **Bidirectional streaming** | Subscriptions (usually over WS) |
| Over- and under-fetching | Both possible | Both possible | **Solved by the client choosing fields** |
| Caching | HTTP caching works naturally | Hard — POST-like semantics | Hard — usually a single endpoint, POST |
| Browser support | Native | Needs a proxy (gRPC-Web) | Native |
| Best for | Public APIs, CRUD, cacheable resources | Internal service-to-service, low latency, streaming | Aggregating many sources for a flexible client |

The practical advice: **REST for public APIs** because caching, tooling and familiarity dominate; **gRPC internally** because the schema, code generation and streaming are worth the loss of HTTP caching; **GraphQL at the aggregation layer** when a client genuinely needs to fetch a variable shape from many backends — with the caveat that it makes caching and rate limiting significantly harder, and query cost analysis becomes mandatory.

**Probe: how do you rate limit a GraphQL API?** Not by request count, because one request can be arbitrarily expensive. You compute a query cost from the schema (depth and field multipliers) and limit on the computed cost, with a depth limit and a persisted-query allow-list to prevent arbitrary queries.

### Q35. Why is gRPC a poor fit for a browser-facing API? — H

**A:** Because browsers do not expose the HTTP/2 framing that gRPC needs — no access to trailers, and no control over the framing — so gRPC-Web exists as a translation layer, requiring a proxy at the edge that converts between gRPC-Web and gRPC. On top of that: protobuf payloads are not human-readable, which makes debugging harder; HTTP caching does not apply meaningfully; and the binary format means you cannot use standard HTTP tooling to inspect traffic without a decoder. So gRPC-Web plus a proxy is workable for internal dashboards, and REST or JSON-over-HTTP remains the pragmatic choice for anything public.

### Q36. What is SSE, and when is it better than WebSocket? — M

**A:** Server-Sent Events is a one-way stream from server to client over a plain HTTP connection, with a simple text format (`data:`, `event:`, `id:`, `retry:`), automatic reconnection built into the browser's `EventSource`, and a `Last-Event-ID` header for resuming. It is better than WebSocket when the data flow is genuinely one-way — notifications, live scores, progress updates, a log tail — because it is plain HTTP, so it passes through proxies and CDNs without upgrade handling, it reconnects automatically, and it needs no custom protocol. WebSocket is better when you need bidirectional messaging or binary frames, or when the per-message overhead matters, since SSE is text-only and each event carries some framing.

**Probe: what is SSE's practical limit?** It holds a connection open, and browsers limit connections per origin (which matters over HTTP/1.1 — six per host — but not over HTTP/2, where multiplexing removes the limit). It is also text-only, so binary requires base64.

---

## 6. API design craft (Q37–Q45)

### Q37. Offset pagination versus cursor pagination. — M

**A:** Offset pagination is `?page=3&limit=20` or `?offset=40&limit=20`. It is simple and it allows jumping to an arbitrary page, and it is wrong for any dataset that changes. The failure is concrete: if five items are inserted before the current page between page 2 and page 3, page 3 shifts and the user sees five duplicate items; if five are deleted, they miss five. Cursor pagination is `?after=<opaque>` where the cursor encodes the last item's sort key — a keyset — so the next page is "everything after this position in the sort order", which is stable regardless of insertions. The costs: you cannot jump to page N, you need a stable and unique sort key (typically `(created_at, id)` to break ties), and the cursor must be opaque and validated so a client cannot inject arbitrary sort keys.

**Strong:** "For a feed or a log, cursor pagination is mandatory; offset pagination is only acceptable for an admin table over a static dataset."

### Q38. How do you version an API? — M

**A:** Three common placements: in the path (`/v2/orders`), in a header (`Accept: application/vnd.acme.v2+json`), or in the query string. Path versioning is the most common and the easiest to route and cache, at the cost of polluting the URL space. Header versioning is more RESTful and keeps URLs stable, but it is invisible in logs and harder to test by hand. The deeper advice: **version only on breaking changes**, and design so that most changes are not breaking — add optional fields rather than renaming, never remove a field without a deprecation cycle, treat enums as open, and ignore unknown fields on input. Most "version 2" launches are avoidable. When you do version, run both versions concurrently with a shared implementation behind a translation layer, and publish a deprecation timeline with `Deprecation` and `Sunset` headers so clients can see it programmatically.

### Q39. How do you handle a breaking change without a new version? — H

**A:** By making it non-breaking. **Additive changes only**: a new optional field, a new endpoint, a new enum value. **Tolerant readers**: clients must ignore unknown fields, and you must treat unknown enum values as valid rather than rejecting them. **Expand/contract for schema changes**: add the new field, write both, migrate readers, then stop writing the old one — never rename in place. **Content negotiation** for a genuinely different representation. **A compatibility test suite** that runs the previous version's client against the new server, because the only reliable test of compatibility is the old client. The one thing that is genuinely not solvable is a semantic change — the same field meaning something different — which is why the field should get a new name rather than a new meaning.

### Q40. How do you design a webhook? — M

**A:** A webhook is an HTTP POST you send to a customer's URL when an event happens, so the design is a delivery system, not an API. The requirements: **sign the payload** with an HMAC over the body plus a timestamp, using a shared secret, so the receiver can verify authenticity and reject replays. **Include an event ID and make delivery at-least-once**, so receivers must deduplicate — and say so in the documentation. **Retry with exponential backoff and jitter** on failure, for a bounded period (typically 24–72 hours), then dead-letter. **Use a fast timeout** — 5 to 10 seconds — because a slow receiver must not hold your workers. **Treat non-2xx as failure** and any 2xx as success. **Order is not guaranteed**, so include a sequence or timestamp and tell the receiver to sort. **Provide a replay API and a delivery log**, because every integrator will eventually need to re-fetch an event. And **allow the receiver to disable** — an endpoint failing consistently should be paused rather than hammered.

**Probe: how does the receiver verify the signature?** It computes the HMAC over the raw request body — not the parsed and re-serialised body, which is a classic bug — and compares in constant time. The timestamp in the signed payload bounds the replay window.

### Q41. What is the difference between authentication and authorisation, and where does OAuth 2.0 fit? — E

**A:** Authentication establishes who you are; authorisation decides what you may do. OAuth 2.0 is an **authorisation delegation** framework: it lets a user grant a third-party application limited access to their resources without sharing credentials, by issuing an access token. It is not an authentication protocol — that is OpenID Connect, which is a layer on top of OAuth 2.0 that adds an ID token and a userinfo endpoint. The distinction matters because using OAuth 2.0 alone for "log in with X" was a real and common vulnerability class. And in an API: authenticate at the edge with a token, authorise at the service against the resource.

### Q42. How would you design pagination for a feed where items can be inserted and deleted? — H

**A:** A keyset cursor over a stable, unique sort key. The key needs to be unique, so `(created_at, id)` rather than `created_at` alone, because two items with the same timestamp would make the boundary ambiguous and could skip or repeat items. The cursor is opaque — base64 of the key — so clients cannot inject arbitrary ordering. Deleted items are filtered at read time, and the design must tolerate a page returning fewer than `limit` items without concluding it is the end — the `has_more` flag is computed by asking for `limit + 1` and discarding the extra. If ranking reorders items, the cursor has to encode the ranking position, which is why a ranked feed snapshots its ordering per scroll session and shows new items as a separate banner rather than shifting the order ([see Instagram News Feed](../Case-Studies/HLD/04-instagram-news-feed.md) §3.5).

### Q43. Design a public API for a URL shortener. — M

**A:**

```http
POST /v1/links
Idempotency-Key: <uuid>                 # so a retry does not create two links
{ "long_url": "https://example.com/very/long/path",
  "custom_slug": "launch",              # optional
  "expires_at": "2027-01-01T00:00:00Z", # optional
  "tags": ["campaign-q4"] }
→ 201 { "slug":"launch", "short_url":"https://sho.rt/launch",
        "long_url":"...", "created_at":"..." }
→ 409 { "error":"slug_taken" }
→ 422 { "error":"invalid_url" }         # after validating the scheme and the host

GET /v1/links/{slug}
→ 200 { "slug":"launch", "long_url":"...", "clicks": 12043,
        "created_at":"...", "expires_at": null }
→ 404 { "error":"not_found" }           # or 410 Gone if it expired

DELETE /v1/links/{slug}
→ 204

GET /v1/links?cursor=<opaque>&limit=50  # list, keyset paginated
GET /v1/links/{slug}/stats?from=...&to=...&interval=day
→ { "clicks": [ {"day":"2026-09-01","count":842}, ... ] }
```

Design points: `301` for a permanent redirect is cacheable by browsers and saves origin traffic, but it means you lose the click event — so use `302` (or `307`) if you need to count clicks, and say why. Rate-limit link creation per account, because link shorteners are abused for phishing. Never blindly redirect to any URL — check the scheme (`http`/`https` only, never `javascript:` or `data:`) and check against a blocklist, or you have built an open redirect. And reserve slugs so a customer cannot take a word your own routes need.

### Q44. You need to add a field to a JSON response without breaking clients. What do you check? — M

**A:** Whether clients tolerate unknown fields. In practice: JSON parsers in most languages ignore unknown fields by default, but strict schema validators — some generated clients, and `additionalProperties: false` in an OpenAPI-generated validator — will reject the response. So the checks are: does the client use a generated client from my OpenAPI spec, and does that generator set strict validation; does the client validate responses at all; and is the new field nullable or always present — because if clients deserialise into a struct with a required field, adding a field is fine but making an existing field nullable is not. And adding an enum value is the sneaky one: clients that switch exhaustively on an enum will break, so new enum values must be treated as "unknown category" by clients rather than rejected.

### Q45. How do you test an API contract? — H

**A:** Three layers. **Schema validation** — the response conforms to the OpenAPI or protobuf schema, checked in CI, so a change that violates the published contract fails the build. **Consumer-driven contract tests** — the consumer declares what it needs and the provider verifies it, which catches the case where the provider changes something the consumer relies on but the schema does not capture. **Compatibility testing with the previous version's client** — run the actual old client against the new server; this is the only test that catches a semantic break. Plus runtime checks: log responses that fail validation in production at a low sample rate, so you learn about a real client's assumptions even when your tests pass. The gap none of these covers is a client that depends on undocumented behaviour, which is why the contract must document what it guarantees, not just what it happens to send.

---

## 7. Rapid-fire numbers

| Thing | Number | Note |
|---|---|---|
| UDP DNS payload ceiling without EDNS0 | 512 bytes | Beyond it, the TC bit forces TCP |
| EDNS0 advertised size | 1232 bytes typically | 1280 MTU − 48 bytes of header, avoiding fragmentation |
| DNS TTL before a migration | Lower to 60 s, two old-TTLs early | Otherwise propagates over the old TTL |
| Typical API cache `s-maxage` | 300 s | With `stale-while-revalidate` |
| Immutable asset cache | `max-age=31536000` + `immutable` | Content-addressed filenames |
| TCP handshake | 1 RTT | Plus TLS 1.3 = 1 RTT; TLS 1.2 = 2 |
| HTTP/3 resumed connection | 0 RTT | Replayable — idempotent requests only |
| HTTP/1.1 per-host connection limit | ~6 in browsers | The reason for domain sharding, which HTTP/2 makes harmful |
| HPACK header compression | Typically 80–90% for repeated headers | Dynamic table is the CRIME/BREACH surface |
| nginx default proxy timeout | 60 s | `proxy_read_timeout` |
| nginx upstream keepalive | Off by default | Enable it, or every request pays a handshake |
| Typical rate-limit response | `429` + `Retry-After` + `RateLimit-*` | Return the headers on 200s too |
| Idempotency key TTL | 24 h | Covers client retry windows |
| Webhook timeout | 5–10 s | Slow receivers must not hold workers |
| Webhook retry window | 24–72 h, exponential + jitter | Then dead-letter, with a replay API |
| API gateway added latency | 1–5 ms typically | Budget it; alert above 10 ms |
| TLS handshake CPU | Tens of thousands/sec/core | Rarely the bottleneck |
| OCSP staple validity | Usually 24 h, must be refreshed | A stale staple fails the connection |

---

## 8. Cheat-sheet summary

| Concept | One-line truth |
|---|---|
| DNS failure mode | TTL governs everything; lower it before the change, not after |
| DNS load balancing | Coarse and slow; not a substitute for a real load balancer |
| Anycast | Same IP everywhere; breaks long-lived TCP on BGP changes |
| CDN cache key | The most common misconfiguration in both directions |
| CDN invalidation | Content-addressing beats purging; purges are eventually consistent |
| `no-cache` vs `no-store` | Revalidate versus do not store at all |
| `stale-while-revalidate` | The single best directive for perceived performance |
| Forward vs reverse proxy | Whose behalf it acts on, not which software |
| TLS termination | Edge = simple and routable; pass-through = opaque and end-to-end |
| SNI | Plaintext hostname; what makes shared certificates possible |
| `X-Forwarded-For` | Client-supplied; only trustworthy if the origin is locked to the CDN |
| Rate limiting | Token bucket by default: average rate plus bounded burst |
| Rate-limit key | Layer per-IP (generous), per-user (fair), per-endpoint (expensive) |
| `429` etiquette | `Retry-After` plus limit headers on successes too |
| Idempotency key | Mandatory on writes; store the response; `SET NX` for atomicity |
| HTTP/2 | Fixes connections, inherits TCP head-of-line blocking |
| HTTP/3 | QUIC removes that; survives IP changes; 0-RTT is replayable |
| TLS 1.3 | 1 RTT, forward secret, more of the handshake encrypted |
| mTLS | Nothing to steal, but certificate lifecycle is the cost |
| REST / gRPC / GraphQL | Public cacheable / internal fast streaming / flexible aggregation |
| GraphQL rate limiting | On computed query cost, not request count |
| Cursor pagination | Keyset over a unique sort key; offsets break on insert and delete |
| Versioning | Version only on breaking changes; make most changes additive |
| Webhooks | Signed, at-least-once, retried with jitter, logged, replayable |

**"If the interviewer says X → answer Y"**

| Interviewer says | You say |
|---|---|
| "Users can't reach it" | "Separate timeout vs reset vs HTTP error — routing, TLS, policy — then check DNS TTL drain" |
| "Speed up the site" | "CDN with the right cache key, immutable assets, `stale-while-revalidate`, connection reuse" |
| "Cache invalidation" | "Content-addressed URLs first; purges are eventual, so never depend on them for consistency" |
| "Where does TLS terminate?" | "At the edge for routing and caching; re-encrypt if the internal network isn't trusted" |
| "Rate limit design" | "Token bucket, layered keys, `429` + `Retry-After`, headers on success too" |
| "Client retries duplicate writes" | "A required idempotency key with `SET NX` and a 24 h TTL, scoped per account" |
| "HTTP/2 or 3?" | "3 for latency-sensitive mobile; 2 is the safe default; 0-RTT only for idempotent requests" |
| "REST or gRPC?" | "REST public for caching and tooling, gRPC internal for schema and streaming" |
| "Pagination" | "Keyset cursor over `(created_at, id)`; offsets duplicate and skip" |
| "Add a field safely" | "Additive only; check strict client validators; never repurpose an existing field's meaning" |
| "Design a webhook" | "HMAC-signed, at-least-once, jittered retries, a delivery log, and a replay API" |
| "Trade-offs?" | "Every networking choice trades latency for correctness, cache efficiency, or operational simplicity" |

---

## 9. Cross-links

- [DNS Deep Dive](../HLD/02-networking/01-dns-deep-dive.md)
- [CDN Deep Dive](../HLD/02-networking/02-cdn-deep-dive.md)
- [Forward and Reverse Proxies](../HLD/02-networking/03-forward-and-reverse-proxy.md)
- [API Gateway and Rate Limiting](../HLD/02-networking/04-api-gateway-and-rate-limiting.md)
- [Scaling, Caching and DB Questions](03-scaling-caching-db-questions.md)
- [Scaling, Caching and DB Cheatsheet](../Cheat-Sheets/03-scaling-caching-db-cheatsheet.md)
- [Networking Cheatsheet](../Cheat-Sheets/02-networking-cheatsheet.md)
