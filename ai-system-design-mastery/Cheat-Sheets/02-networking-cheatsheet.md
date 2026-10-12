# Networking Cheatsheet — DNS, CDN, Proxies, Gateways, HTTP & TLS

> **TL;DR** — One revision page for the networking quarter of a system-design interview. Every number, default, decision rule and failure mode from [DNS](../HLD/02-networking/01-dns-deep-dive.md), [CDN](../HLD/02-networking/02-cdn-deep-dive.md), [proxies](../HLD/02-networking/03-forward-and-reverse-proxy.md) and the [API gateway](../HLD/02-networking/04-api-gateway-and-rate-limiting.md), compressed to tables. Skim the **Decision rules** section if you have two minutes; read the **If the interviewer says X → answer Y** table if you have one.

---

## 1. DNS

### 1.1 The five record types you must know

| Type | Holds | Points to | Notes |
|---|---|---|---|
| **A** | IPv4 address | 192.0.2.1 | The workhorse |
| **AAAA** | IPv6 address | 2001:db8::1 | "quad-A" |
| **CNAME** | An alias | Another **name** | Cannot coexist with other records at the same name |
| **MX** | Mail exchanger | Hostname + priority | Needs an A/AAAA, never a CNAME |
| **TXT** | Arbitrary text | — | SPF, DKIM, DMARC, domain verification |
| **NS** | Delegation | Nameserver hostnames | Defines the zone cut |
| **SOA** | Zone authority | — | Serial, refresh, retry, expire, minimum (the negative-cache TTL) |
| **SRV** | Service location | Host + port + priority + weight | Service discovery |
| **CAA** | Which CAs may issue | CA domain | Certificate issuance control |
| **PTR** | Reverse | Hostname | `in-addr.arpa`; mostly for mail reputation |
| **ALIAS / ANAME** | A CNAME at the apex | Another name | Provider-specific; the apex workaround |

### 1.2 The hierarchy and the cold lookup

```
Browser cache    →  OS cache  →  /etc/hosts  →  Stub resolver
   ↓ (on miss)
Recursive resolver (ISP / 8.8.8.8 / 1.1.1.1)
   ↓  Root server        ("." → where is .com?")     ~1 RTT
   ↓  TLD server         (".com" → where is example.com?")   ~1 RTT
   ↓  Authoritative NS   ("example.com → A 192.0.2.1")       ~1 RTT
   ↑ answer cached per the record's TTL
```

**Negative answers are cached too** — for the SOA `minimum` field, commonly 60–300 s. This is why "I created the record and it still does not resolve" is often a cached NXDOMAIN, not a propagation delay.

### 1.3 Cold-request round trips

| Step | Cold | Warm |
|---|---|---|
| DNS (recursive, uncached) | 1–3 RTT | 0 (OS/browser cache) |
| TCP handshake (HTTP/1.1, HTTP/2) | 1 RTT | 0 (keep-alive) |
| TLS 1.3 handshake | 1 RTT | 0 (session resumption / 0-RTT) |
| HTTP request + first byte | 1 RTT | 1 RTT |
| **Total to first byte** | **~4–6 RTT** | **~1 RTT** |

At 50 ms RTT that is 200–300 ms cold versus 50 ms warm, which is the entire justification for keeping connections alive and for preconnecting (`<link rel="preconnect">`) to third-party origins.

### 1.4 TTL: the rules

| Situation | Action |
|---|---|
| Normal record | 300–3,600 s. Long TTLs cut resolver load; short TTLs allow change |
| Before a planned migration | **Lower the TTL to 60 s at least two old-TTL periods before** the change |
| After the migration is verified | Raise it back |
| Do NOT | Change the TTL at the same moment as the record — resolvers already hold the old long TTL |
| Failure drill (failover) | A 24 h TTL means a 24 h outage window when the address is wrong |

**The two-period rule is the single most-tested DNS operational fact.** Resolvers that cached the old record hold it for the old TTL, so you must wait for the old TTL to drain *before* the new record matters — hence two periods of the *old* TTL.

### 1.5 CNAME rules and traps

| Rule | Why it exists |
|---|---|
| A CNAME cannot coexist with any other record at the same name | RFC 1034: the name *is* the alias; there is nothing to hold other records |
| Never put a CNAME at the apex (`example.com`) | The apex must hold SOA and NS; this is why ALIAS/ANAME/flattening exists |
| Never CNAME an MX or NS target | Mail servers and delegations need address records |
| Never CNAME a TXT-carrying name (SPF) | You break SPF silently |
| CNAME chains cost RTTs | Each link is another lookup, up to the resolution limit |

### 1.6 UDP versus TCP for DNS

| | UDP/53 | TCP/53 |
|---|---|---|
| Default | Yes — one datagram, no handshake | Used on truncation, AXFR/IXFR, DNSSEC-heavy responses |
| Truncation | Sets the TC bit when the answer exceeds the advertised size | The resolver then retries over TCP |
| Classic size limit | 512 bytes | — |
| EDNS0 | Advertises a larger buffer; **1,232 bytes is the safe modern value** (1,280 IPv6 MTU − 48 for headers) | — |
| Why not always TCP | The handshake triples the cost for the common case of a tiny answer | |

### 1.7 DNS as a load balancer, and its limits

| Can do | Cannot do |
|---|---|
| Round-robin A records | Know how many clients are behind a resolver — caching collapses distinct clients into one |
| Geographic steering (with EDNS Client Subnet, imprecisely) | React quickly: the TTL is the control loop, and it is minutes |
| Failover by removing a record | Health-check at request granularity, or drain connections |
| Weighted answers | Guarantee even distribution — resolvers and clients cache and reorder |
| **Latency-based routing** | Do sticky sessions or per-request routing |

**The core limitation:** DNS is a *coarse, cached, per-resolver* signal. Real load balancing needs L4/L7 and a fast control loop. DNS-LB is for the first hop and for regional steering, not for balancing within a region.

### 1.8 DNS failure modes

| Failure | Symptom | Fix |
|---|---|---|
| **TTL too long during a change** | "Propagation is slow" — actually the old TTL is still draining | The two-period rule |
| **Resolver cache poisoning of your own test** | You see the old answer, everyone else sees the new | Query the authoritative server directly (`dig @ns1`) |
| **NXDOMAIN negative caching** | A new record does not resolve for the SOA minimum | Understand the SOA minimum; test with a fresh resolver |
| **Serving stale on resolver failure** | An unreachable authoritative NS leads some resolvers to keep serving the expired answer rather than fail | Multi-provider authoritative DNS; monitor from many vantage points |
| **A record points at a decommissioned IP** | Partial outage — only some clients | Never reuse IPs without a DNS change first, and drain before decommissioning |
| **CNAME to a third-party host that changes** | You inherit their change, unannounced | Monitor; consider a stable CNAME you control |
| **Split-horizon / internal zone conflicts** | Internal clients resolve differently from external | Be explicit about which resolver each environment uses |
| **DNS is the single point of failure** | Everything is down | Multiple authoritative providers, and a long-ish TTL so a provider outage does not immediately empty every resolver |
| **Case where DNS is not the problem** | Slow first byte with warm DNS | Look at TCP/TLS/application instead — see the triage table in §1.9 |

### 1.9 Latency triage — DNS vs TCP vs TLS vs app

| Observation | Likely layer | Check |
|---|---|---|
| Slow first connection, fast subsequent | TCP/TLS setup | Reuse connections; enable TLS 1.3 and session resumption |
| Slow *every* request, `request_time` ≈ `upstream_response_time` + constant | Application or upstream | Break down server-side timings |
| Slow first byte on a new hostname only | DNS | `dig +trace`; check the TTL and the resolver |
| Intermittent timeouts on long-lived connections | LB / anycast route change / idle timeout | Idle timeouts, keepalive pings, and the fact that a BGP change can break a long TCP connection |
| Errors only from one geography | Anycast or regional routing | Per-region probes |
| Connection reset, not timeout | Policy — a firewall, an MTU/PMTU black hole, or an LB refusing | Check MSS clamping and firewall logs |

### 1.10 DNSSEC, and its three adoption blockers

**What it does:** signs DNS answers so a resolver can verify authenticity and integrity — defeating cache poisoning.

**Blockers:** (1) **Operational complexity** — key rollovers, KSK/ZSK ceremonies, and a mistake takes the whole domain offline (a signed zone with a bad signature is worse than an unsigned one); (2) **Larger responses** — signatures push answers over 512 bytes, forcing TCP fallback or EDNS0, and some middleboxes break on that; (3) **Broken validation in the wild** — a misconfigured validating resolver returns SERVFAIL for the whole domain, and it is common enough that operators are cautious.

---

## 2. CDN

### 2.1 What a CDN actually gives you

| Benefit | Mechanism | Magnitude |
|---|---|---|
| **Latency** | Serve from an edge near the user | 200 ms origin RTT → 20 ms edge |
| **Origin offload** | Cache hits never reach the origin | 80–95% typical for static assets |
| **Bandwidth absorption** | Traffic terminates at the edge | The origin's egress drops to misses only |
| **Attack absorption** | Anycast + scrubbing capacity | Terabits of capacity the origin does not need |
| **TLS termination** | The edge terminates, the origin sees one connection | Fewer handshakes on the origin |
| **Cost** | CDN egress is cheaper than origin egress | Origin egress ~$0.09/GB, CDN ~$0.02–0.08/GB |

### 2.2 Cache headers, and the two pairs people confuse

| Header | Meaning | Trap |
|---|---|---|
| `Cache-Control: max-age=N` | Fresh for N seconds | Applies to both browser and CDN |
| `Cache-Control: s-maxage=N` | Fresh for N seconds **for shared caches only** | Overrides `max-age` at the CDN; the browser still uses `max-age` |
| `Cache-Control: no-cache` | **Store, but revalidate before every use** | Not "do not cache" |
| `Cache-Control: no-store` | **Do not store at all** | This is the "do not cache" one |
| `Cache-Control: private` | Browsers may store; shared caches may not | For per-user responses |
| `Cache-Control: public` | Any cache may store | |
| `Cache-Control: must-revalidate` | Do not serve stale after expiry | Without it, stale may be served |
| `Cache-Control: immutable` | Never revalidate during the freshness lifetime | For content-addressed assets only |
| `Cache-Control: stale-while-revalidate=N` | Serve stale for N s while refreshing in the background | The single biggest perceived-latency win for CDN content |
| `Cache-Control: stale-if-error=N` | Serve stale for N s if the origin errors | Turns an origin outage into a latency non-event |
| `Age` (response) | Seconds the object has been in the CDN | The CDN tells you its age — use it |
| `ETag` / `If-None-Match` | Validator | Cheap `304` revalidation |
| `Last-Modified` / `If-Modified-Since` | Coarser validator | 1-second granularity |
| `Vary` | The response varies by a request header | **`Vary: *` makes an object uncacheable**; `Vary: User-Agent` fragments the cache into thousands of variants |
| `Surrogate-Control` / `CDN-Cache-Control` | CDN-only directives | Lets you set a short edge TTL with a long browser TTL |

**The recipe that matters:** `Cache-Control: public, max-age=31536000, immutable` on content-addressed filenames (`app.a3f9c1.js`), and `Cache-Control: no-cache` on the HTML that references them. Long TTLs everywhere, with the invalidation handled by changing the *name* rather than purging.

### 2.3 Stale-while-revalidate in one line

```
Fresh window ────────►│◄──── stale-while-revalidate window ────►│◄── must revalidate
   serve from cache    serve STALE immediately, refresh in background   block on origin
```

A user never waits for a revalidation, and only one request refreshes. On a busy edge this removes almost all the tail latency of cache expiry.

### 2.4 Cache-key design — the failure in both directions

| Mistake | Effect |
|---|---|
| Key includes a header that varies per user (`User-Agent`, `Cookie`) | Cache fragmentation — a near-zero hit ratio while the origin sees the same load |
| Key omits something that changes the response (e.g. `Accept-Encoding`, `Accept-Language`) | **Wrong content served to the wrong user** — the dangerous direction |
| Key includes the full query string when most parameters do not affect the response | Cache misses on `?utm_source=` and similar |
| Per-user signed URLs in the key | Every viewer is a distinct object — zero sharing |
| `Vary: *` | The object is effectively uncacheable |
| Cookies forwarded and in the key | Same as per-user signed URLs |

**The rule:** the cache key must be exactly the set of inputs that changes the bytes — no more, no less. And for signed URLs, scope the token to a *class* — `(asset_id, quality, region, time_bucket)` — so thousands of viewers share one cached object (`[see YouTube](../Case-Studies/HLD/05-youtube-upload-and-recommendation.md) §6.4`).

### 2.5 Invalidation — three methods and their honest limits

| Method | Mechanism | Latency | Limit |
|---|---|---|---|
| **TTL** | Simply expire | 0 to the TTL | Blunt; you cannot force an update |
| **Purge by URL** | Tell the CDN to drop specific URLs | Seconds | **Not atomic across the fleet** — a purge is a message to many edges, and some serve stale for a few seconds; there may be a per-request purge cost and a rate limit |
| **Versioned URLs / content addressing** | Change the name (`app.a3f9.js` → `app.b7c2.js`) | Instant and total | Requires the referring document (HTML) to be fresh; the HTML must therefore be short-TTL |
| **Surrogate keys / cache tags** | Tag objects, purge by tag | Seconds | The most powerful option — "purge everything tagged `product:1123`" — but it must be designed in |

**The statement to make:** purges are **not globally atomic**. A purge propagates to the edges over seconds, so if correctness depends on an invalidation being simultaneous everywhere, a purge is the wrong tool — use a versioned URL or a short TTL.

### 2.6 CDN failure modes

| Failure | Cause | Defence |
|---|---|---|
| **Cache poisoning** | A request with a header that the key does not include changes the response, and that response is cached for everyone | Normalise the key; strip unknown headers; `Vary` correctly |
| **Cache stampede at the origin after a purge** | Purging a hot object means every edge misses simultaneously | `stale-while-revalidate`; a request-collapsing layer; purge gradually |
| **Origin overload on cold start** | A new region or a new deploy with an empty edge cache | Pre-warm; rate-limit the origin; `stale-if-error` |
| **Wrong content served** | Key too coarse — a shared cache serving user A's page to user B | Never cache authenticated responses in a shared cache; `Cache-Control: private` |
| **Private data cached** | Missing `private`/`no-store` | Audit; a shared cache is not a private cache |
| **`X-Forwarded-For` spoofing** | The origin trusts a header the client can set | Firewall the origin to the CDN only, and require a shared secret header |
| **Origin reachable directly** | Bypassing the CDN, bypassing WAF and rate limits | Firewall to the CDN's egress ranges; authenticate the CDN |
| **Cache key ignoring `Accept-Encoding`** | A gzip response served to a client that did not ask for it, or vice versa | Correct `Vary` |
| **Purge rate limits** | A purge loop exhausting the API quota | Versioned URLs instead of purges |

### 2.7 The edge security checklist

- Origin firewalled to the CDN's published IP ranges, so the CDN cannot be bypassed.
- A shared secret header (`X-CDN-Auth`) validated at the origin, in case the firewall is misconfigured.
- `X-Forwarded-For` parsed from the **rightmost** entry your own infrastructure wrote, never the leftmost, which is client-supplied.
- Real client IP taken from the CDN's own header (`CF-Connecting-IP`, `X-Real-IP`, `True-Client-IP`) and never from a client-set header.
- TLS between edge and origin (re-encrypt or pass-through) if the data is sensitive — an edge-to-origin plaintext hop is a real exposure.

---

## 3. Forward vs reverse proxy

### 3.1 The one-line distinction

| | Forward proxy | Reverse proxy |
|---|---|---|
| Acts on behalf of | The **client** | The **server** |
| Client knows about it | Yes — configured explicitly | No — it looks like the server |
| Typical purpose | Filtering, anonymity, egress control, caching for many clients | Load balancing, TLS termination, caching, WAF, routing |
| Examples | Squid, corporate egress proxy, a VPN with a proxy | Nginx, Envoy, HAProxy, Cloudflare, ALB |

**A reverse proxy is not a load balancer, and a load balancer is often a reverse proxy.** The reverse proxy is the broader category: it terminates the client connection and makes a new one to the origin, which is what enables everything else.

### 3.2 What a reverse proxy adds

| Function | Detail |
|---|---|
| TLS termination | One place to manage certificates; the client's handshake ends at the proxy |
| Load balancing | Distributes across backends |
| Caching | A shared response cache in front of the origin |
| Compression | gzip/brotli applied once instead of per backend |
| Connection pooling | Many client connections multiplexed onto few backend connections |
| Request routing | By path, host or header |
| Rate limiting and WAF | Centralised policy |
| Response buffering | Absorbs a slow client without holding a backend thread |
| Retries and timeouts | Independent of the application |
| Observability | One place for access logs, tracing and metrics |
| Header hygiene | Strips hop-by-hop headers, adds `X-Forwarded-*` |

### 3.3 Forwarded headers — get these right

| Header | Meaning |
|---|---|
| `X-Forwarded-For` | Comma-separated chain of client IPs, appended to by each proxy |
| `X-Forwarded-Proto` | The original scheme (`http`/`https`) — needed so the app generates correct absolute URLs |
| `X-Forwarded-Host` | The original `Host` |
| `X-Real-IP` | Single client IP (nginx convention) |
| `Forwarded` (RFC 7239) | The standardised `for=`, `by=`, `proto=`, `host=` form |
| `Via` | Which proxies the message traversed |

**The trap:** `X-Forwarded-For` is **client-supplied and spoofable**. A client can send `X-Forwarded-For: 1.2.3.4` and a naive origin that reads the leftmost entry believes it. The rule: the proxy **appends** the peer address, so the trustworthy entry is the **rightmost one added by your own infrastructure**, and the number of trusted proxies from the right is a configuration value, not a guess. If the origin is only reachable through the proxy, this is safe; if it is directly reachable, the whole header is attacker-controlled.

### 3.4 TLS termination — the three placements

| Placement | How | Pros | Cons |
|---|---|---|---|
| **Terminate at the edge** | Proxy holds the cert; plaintext to the origin | Simplest, cheapest, full L7 visibility (caching, WAF, routing) | Plaintext inside the network; needs a trusted network |
| **Re-encrypt (TLS bridging)** | Edge terminates, opens a *new* TLS connection to the origin | L7 features plus encryption in transit internally | Two handshakes; certificate management in two places |
| **Pass-through (TLS tunnelling)** | The proxy forwards encrypted bytes; the origin holds the cert | End-to-end encryption; the proxy never sees plaintext | No L7 inspection, no caching, no path-based routing, no header rewriting |

**Pass-through routing still needs the SNI** — the hostname in the TLS ClientHello, which is **plaintext** — so the proxy routes by SNI without decrypting. That SNI is also visible to every network observer, which is why Encrypted Client Hello (ECH) exists; it hides the SNI by encrypting it so only the resolver and the edge can read it.

**mTLS costs, when asked:** certificate issuance, distribution, expiry and rotation for every client; revocation via CRL or OCSP (and OCSP stapling to avoid a per-handshake round trip); and an operational process for lost keys. It is worth it for service-to-service inside a trusted perimeter; it is heavy for user-facing clients.

---

## 4. API gateway and rate limiting

### 4.1 Gateway versus reverse proxy

| Gateway adds | Why it belongs there |
|---|---|
| **Authentication** (JWT verification, API keys, OAuth introspection) | Every request needs it; doing it once at the edge saves every service |
| **Authorisation, coarse-grained** | Scope and role checks; fine-grained stays in the service |
| **Rate limiting and quotas** | Needs a global view no single service has |
| **Request/response transformation** | Protocol translation, header mapping, body shaping |
| **API versioning and routing** | `/v1` vs `/v2` to different backends |
| **Aggregation / BFF** | One client call fans out to several services (use sparingly) |
| **Request validation** | Reject malformed requests before they reach a service |
| **Observability and analytics** | Per-consumer usage, latency, error rates |
| **Idempotency handling** | A central place to store and replay idempotency keys |
| **Circuit breaking, retries, timeouts** | Consistent policy across all backends |

**The principle:** the gateway does the **coarse** checks; the service does the **fine** ones and **validates anyway**. A gateway is a policy enforcement point, not the only one — defence in depth, because a misconfiguration or a bypass must not be a full compromise.

### 4.2 The four rate-limiting algorithms

| Algorithm | How | Burst | Memory | Notes |
|---|---|---|---|---|
| **Fixed window** | Counter per window, reset at the boundary | Up to 2× at the boundary | Tiny | The boundary problem: 100 requests at 00:59 and 100 at 01:00 = 200 in two seconds |
| **Sliding window log** | Store every request timestamp | Exact | O(requests) — expensive | Precise, memory-hungry |
| **Sliding window counter** | Weighted blend of the current and previous window | Approximate, smooth | Tiny | The usual production compromise |
| **Token bucket** | Tokens refill at rate R, bucket size B; each request takes one | Allows a burst of B | Tiny | **The default.** R = sustained rate, B = burst |
| **Leaky bucket** | Queue drains at a fixed rate | None — smooths everything | Queue size | Right when the downstream needs a constant rate |
| **GCRA** | Theoretical arrival time; allows a burst then a strict rate | One burst | Tiny | The mathematically clean version of token bucket |

**Token bucket in Redis, correctly:**

```lua
-- KEYS[1] = bucket key   ARGV = rate, burst, now_ms, cost
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or tonumber(ARGV[2])
local ts     = tonumber(b[2]) or tonumber(ARGV[3])
local delta  = math.max(0, tonumber(ARGV[3]) - ts) / 1000.0
tokens = math.min(tonumber(ARGV[2]), tokens + delta * tonumber(ARGV[1]))
if tokens < tonumber(ARGV[4]) then
  local wait = (tonumber(ARGV[4]) - tokens) / tonumber(ARGV[1])
  return {0, math.ceil(wait * 1000)}
end
tokens = tokens - tonumber(ARGV[4])
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', ARGV[3])
redis.call('PEXPIRE', KEYS[1], math.ceil(tonumber(ARGV[2]) / tonumber(ARGV[1]) * 1000) + 1000)
return {1, 0}
```

The `Lua` script matters: check-and-decrement must be **atomic**, or two concurrent requests both see enough tokens and both pass. A `GET` then `SET` is a race; `EVAL` is not.

### 4.3 Layering rate limits

| Layer | Key | Limit style | Trap |
|---|---|---|---|
| **Per-IP** | Client IP | Generous (e.g. 100/s) | **NAT** — a corporate or mobile carrier NAT means thousands of users share one IP; a per-IP limit locks out a whole office |
| **Per-user / per-API-key** | Authenticated identity | Per plan tier | Requires auth first, so it cannot protect the auth endpoint |
| **Per-endpoint** | User + route | Weighted by cost | A login endpoint gets 5/min; a read gets 1,000/min |
| **Per-tenant** | Account | Quota with burst | Prevents one tenant starving others |
| **Global** | Service | Circuit-breaker style | Protects the service from the sum of all callers |
| **Cost-based** | Computed weight | GraphQL and expensive queries | Rate limit on *computed query cost*, not request count |

**Always layer at least two.** A per-IP limit alone breaks behind NAT; a per-user limit alone cannot protect the unauthenticated endpoints.

### 4.4 What to return

| Element | Value |
|---|---|
| Status | `429 Too Many Requests` |
| `Retry-After` | Seconds (integer) or an HTTP date — tell the client when to come back |
| `RateLimit-Limit` | The ceiling for the window |
| `RateLimit-Remaining` | What is left |
| `RateLimit-Reset` | When it resets (seconds or an epoch time) |
| **On successes too** | Yes — the client needs to know its budget before it hits the wall |

**The detail that separates a good answer:** send the `RateLimit-*` headers on **2xx responses as well**, so a well-behaved client can slow down preemptively rather than discovering the limit by being rejected. And for a per-tenant quota, the limit is the *plan*, so it belongs in the response body on rejection with a link to upgrade — not just in a header.

### 4.5 API keys are identification, not authentication

| Rule | Reason |
|---|---|
| Never in a URL | URLs land in logs, browser history, `Referer` headers and proxy caches |
| Never in a JS bundle or mobile binary | Both are trivially extractable |
| Never in a log | Redact at the logging layer, not by convention |
| Send as a header | `Authorization: Bearer <key>` or a dedicated header |
| Treat as a low-secret | Prefer short-lived tokens derived from a key, or mTLS, for anything sensitive |
| Rotate and revoke | Store a **hash** of the key, not the key, and support overlapping validity during rotation |

### 4.6 Idempotency keys — the mechanics

| Aspect | Correct behaviour |
|---|---|
| Scope | Per account (or per customer), never global |
| Store | Key → (status code, response body, request hash), with a TTL of ~24 h |
| Same key, same body | Return the **original** response — do not re-execute |
| Same key, **different body** | `422` (or `400`) — a client bug; do not silently serve the old result |
| Same key, request still in flight | `409 Conflict` — the client should retry later, not duplicate the work |
| First request failed | Decide explicitly: does a 5xx consume the key? Usually **no** for a retryable failure and **yes** for a completed execution |
| Storage | Redis with persistence for the fast path, or a unique constraint in the database as the authority |

**The invariant:** the idempotency key must be committed **in the same transaction** as the effect, or a crash between them either loses the record (a duplicate) or records without completing (a phantom). A unique index on the key column is the usual way to make this atomic.

### 4.7 Gateway failure modes

| Failure | Cause | Defence |
|---|---|---|
| Gateway is a single point of failure | One instance, or one region | Multiple instances behind an L4 balancer; multi-region |
| Gateway CPU saturation | TLS handshakes, JWT verification, regex WAF rules | Terminate TLS at the L4/edge, cache JWKS and verification results, avoid catastrophic regex |
| Rate-limit store unavailable | Redis down | **Fail open or fail closed — decide deliberately.** For a public API the usual choice is fail-open with a conservative local limit; for an auth endpoint, fail-closed |
| Thundering herd on the limiter | One Redis key per user across a fleet | Sharded counters; local pre-check with a global confirmation |
| Limit keyed on a spoofable header | `X-Forwarded-For` from the client | Derive the key from a trusted value only |
| Clock skew across limiters | Fixed windows computed from local time | Centralise the time source in the store (`TIME`), not in the gateway process |
| Retry storm | Gateway retries plus client retries | Bounded retries, jitter, and a retry budget per route |
| Hot partition on a shared counter | One global counter for all traffic | Shard the counter and sum; or use a token bucket per key instead of one counter |

---

## 5. HTTP and TLS in one table each

### 5.1 HTTP versions

| | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| Transport | TCP | TCP | **QUIC over UDP** |
| Connections per origin | 6 per browser host | 1 (multiplexed streams) | 1 |
| Head-of-line blocking | **Application-level** — a slow response blocks the connection | **TCP-level** — a lost packet stalls all streams | **Solved** — streams are independent at the transport level |
| Header compression | None | HPACK | QPACK |
| Server push | No | Yes (deprecated in practice) | Yes |
| Handshake | TCP + TLS = 2 RTT | TCP + TLS 1.3 = 2 RTT | 1 RTT, 0-RTT on resumption |
| Migration | Connection breaks on IP change | Same | **Connection ID survives an IP change** — matters for mobile |

**Domain sharding** (`static1.example.com`, `static2.example.com`) was an HTTP/1.1 workaround for the 6-connection limit. Under HTTP/2 it is actively harmful: it defeats multiplexing, costs extra DNS lookups and extra TLS handshakes, and fragments the cache.

**HPACK and CRIME/BREACH:** HPACK compresses headers across requests on the same connection, which means an attacker who can both inject input and observe ciphertext length can infer header contents. The defences are to disable compression of sensitive headers (`Authorization`, `Cookie`), to mask secrets with randomness, and to consider disabling HTTP compression on responses that mix secrets with attacker-controlled input.

### 5.2 TLS

| Version | Handshake | Notes |
|---|---|---|
| TLS 1.2 | 2 RTT | Still in the wild; needs careful cipher configuration |
| TLS 1.3 | **1 RTT** | Forward secret by default; fewer plaintext fields; removes RSA key exchange, CBC, RC4, SHA-1 |
| TLS 1.3 0-RTT | **0 RTT** on resumption | **Replayable** — use only for idempotent requests, and the server must dedupe |

**What 1.3 fixed:** it removed the cipher suites with known weaknesses, made forward secrecy mandatory (an ephemeral key per session, so a stolen long-term key cannot decrypt past traffic), and encrypted more of the handshake (the certificate is now encrypted, so the hostname is largely hidden — except the SNI in the ClientHello, which is why ECH exists).

**Session resumption:** session tickets (a server-encrypted ticket the client presents) or session IDs. This is what turns a repeat visit into a 0-RTT or 1-RTT setup instead of a full handshake.

---

## 6. Decision rules

| Question | Answer |
|---|---|
| Where do I terminate TLS? | At the edge for L7 features; pass-through only when end-to-end encryption is a requirement |
| Do I need a CDN? | If content is cacheable and users are geographically spread — yes. If every response is per-user and uncacheable, the CDN gives you TLS and DDoS absorption, not offload |
| Long TTL or short TTL on DNS? | Long by default, **lowered two old-TTL periods before a planned change** |
| Which rate-limiting algorithm? | Token bucket with atomic Lua; sliding window counter if you need smoothness without burst |
| Per-IP or per-user? | Both — per-IP breaks behind NAT, per-user cannot protect login |
| Do I need a gateway or is a reverse proxy enough? | A gateway if you need auth, quotas, per-consumer analytics or protocol translation across many services. A reverse proxy alone for a single service |
| Is DNS-based load balancing enough? | For the first hop and regional steering. Not for per-request balancing or fast failover |
| Where does `X-Forwarded-For` come from? | Your own proxies, appended; read it from the right, with a known count of trusted proxies |
| Purge or version? | Version, when you control the filename. Purge when you do not |
| Fail open or fail closed on the rate limiter? | Decided per endpoint: open for public reads, closed for auth and payments |
| HTTP/2 or HTTP/3? | HTTP/2 as the baseline; HTTP/3 for mobile and lossy networks, where the connection-ID migration and the removal of TCP HOL blocking pay off |

---

## 7. Numbers to have memorised

| Thing | Number |
|---|---|
| DNS authoritative record TTL | 300–3,600 s typical |
| DNS TTL before a change | 60 s, set two old-TTL periods ahead |
| DNS negative-cache TTL | The SOA `minimum`, commonly 60–300 s |
| EDNS0 safe buffer size | 1,232 bytes (1,280 − 48) |
| Classic DNS UDP limit | 512 bytes, then the TC bit and TCP |
| Cold HTTP request | ~4–6 RTT (DNS 1–3 + TCP 1 + TLS 1 + request 1) |
| Warm HTTP request | 1 RTT |
| Browser connections per origin (HTTP/1.1) | 6 |
| TLS 1.3 handshake | 1 RTT; 0-RTT on resumption, replayable |
| CDN cache hit ratio target | 80–95% for static assets |
| `stale-while-revalidate` typical | 30–300 s |
| `stale-if-error` typical | 300–86,400 s |
| Immutable asset TTL | 31,536,000 s (1 year), content-addressed |
| HTML TTL | 0–60 s (must revalidate) |
| Origin egress cost | ~$0.09/GB (CDN typically cheaper) |
| nginx `proxy_connect_timeout` | 60 s default — too long, lower it |
| nginx `proxy_read_timeout` | 60 s default — set per route |
| Keepalive idle timeout (typical LB) | 60–300 s; make the client timeout **shorter** than the LB's |
| Rate limit headers | `RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset` |
| Idempotency key TTL | 24 h |
| Token bucket burst | 2–10× the per-second rate, chosen deliberately |
| Default API quota | 1,000 req/hour per key is a common public default |
| JWT clock skew allowance | 30–60 s |

**The keepalive rule that prevents a whole class of bug:** the client must time out **before** the load balancer does. If the LB closes an idle connection at exactly the moment the client sends on it, the client sees a reset on a request that should have succeeded — the classic sporadic `502`/`connection reset` on an idle-then-active pool. Client idle timeout < LB idle timeout < server idle timeout.

---

## 8. Cheat-sheet summary

| Topic | One-line truth |
|---|---|
| DNS hierarchy | Root → TLD → authoritative; 1–3 RTT cold, cached after |
| DNS TTL change | Lower it two old-TTL periods **before** the change |
| CNAME | Never at the apex, never on an MX/NS/SPF name |
| DNS as a load balancer | Cached and per-resolver — coarse steering only |
| DNSSEC | Authenticity, blocked by complexity, size, and broken validators |
| CDN value | Latency, offload, absorption, TLS, cost — in that order of importance to the user |
| `no-cache` vs `no-store` | Revalidate versus never store — a classic mis-answer |
| `s-maxage` | The shared-cache TTL; overrides `max-age` at the CDN |
| `stale-while-revalidate` | Serve stale, refresh behind — removes expiry from the critical path |
| Cache key | Exactly the inputs that change the bytes; too narrow is worse than too wide |
| Purging | **Not globally atomic** — prefer versioned URLs |
| Origin security | Firewall to the CDN AND a shared secret header |
| Forward vs reverse proxy | Whose behalf — client versus server |
| `X-Forwarded-For` | Client-supplied; read from the right with a known trusted-hop count |
| TLS termination | Edge for L7, re-encrypt for internal privacy, pass-through for end-to-end |
| SNI | Plaintext in the ClientHello, so pass-through routing works — and so does passive observation |
| Rate limiting | Token bucket by default; atomic in Lua; layer per-IP **and** per-user |
| `429` response | `Retry-After` plus `RateLimit-*`, and the headers on successes too |
| API keys | Identification, not authentication — header only, hashed at rest |
| Idempotency key | Per account, stored with the response, same key + different body = `422`, in flight = `409` |
| Gateway versus proxy | Auth, quotas, per-consumer analytics, transformation |
| Rate limiter down | Fail open for public reads, closed for auth — decided, not defaulted |
| HTTP/1.1 → 2 → 3 | Six connections → one multiplexed TCP → one multiplexed QUIC |
| HTTP/3's real win | Independent streams plus a connection ID that survives an IP change |
| TLS 1.3 | 1 RTT, forward secret, encrypted certificate; 0-RTT is replayable |

**"If the interviewer says X → answer Y"**

| Interviewer says | You say |
|---|---|
| "How does DNS resolve a name?" | "Root → TLD → authoritative, 1–3 RTT, cached per the TTL, and negative answers cached for the SOA minimum" |
| "We need to change the IP next week" | "Lower the TTL to 60 s **now**, two old-TTL periods before, and raise it back after" |
| "Use DNS round robin for load balancing" | "It is cached per resolver and cannot count connections — fine for regional steering, not for balancing" |
| "Why did the CNAME break mail?" | "A CNAME cannot coexist with MX, and it breaks SPF's TXT lookup" |
| "Add a CDN" | "Cache-key design first: too wide fragments the cache, too narrow serves the wrong bytes" |
| "`no-cache` means do not cache, right?" | "No — it means store and revalidate. `no-store` is the do-not-cache directive" |
| "Purge it and it is instantly gone" | "Purging is not globally atomic — it propagates in seconds. Version the URL if it must be instant" |
| "Users get stale content after a deploy" | "The HTML is cached too long; the assets should be content-addressed with a long TTL and the HTML short-TTL" |
| "Set up a proxy" | "Forward or reverse? — whose behalf it acts on decides the entire design" |
| "Read the client IP from `X-Forwarded-For`" | "From the right, with a known trusted-hop count, and only if the origin is firewalled to the proxy" |
| "Terminate TLS at the proxy" | "Fine for L7 features; the internal hop is then plaintext, so re-encrypt if the data is sensitive" |
| "Rate limit by IP" | "Per-IP breaks behind carrier NAT — layer it with per-user, and weight by endpoint cost" |
| "Return 429 and drop it" | "Add `Retry-After` and `RateLimit-*`, and send those headers on the 2xx responses too" |
| "The client says it charged twice" | "The retry lacked an idempotency key scoped per account, stored with the original response" |
| "Redis is down, so rate limiting is down" | "Fail open for public reads, closed for auth — and decide that before the incident" |
| "The connection randomly resets" | "Client idle timeout is longer than the load balancer's — make the client time out first" |
| "Why HTTP/3?" | "No TCP head-of-line blocking, and the connection ID survives an IP change — that is the mobile win" |

---

## 9. Cross-links

- [DNS Deep Dive](../HLD/02-networking/01-dns-deep-dive.md)
- [CDN Deep Dive](../HLD/02-networking/02-cdn-deep-dive.md)
- [Forward and Reverse Proxy](../HLD/02-networking/03-forward-and-reverse-proxy.md)
- [API Gateway and Rate Limiting](../HLD/02-networking/04-api-gateway-and-rate-limiting.md)
- [Networking and API Interview Questions](../Interview-Questions/02-networking-and-api-questions.md)
- [Load Balancers](../HLD/03-scaling/01-load-balancers.md)
- [Scaling, Caching and DB Cheatsheet](03-scaling-caching-db-cheatsheet.md)
