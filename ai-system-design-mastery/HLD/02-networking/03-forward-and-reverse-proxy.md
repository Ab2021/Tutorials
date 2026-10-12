# Forward Proxy vs Reverse Proxy — TLS, HTTP/2/3, Streaming and Sidecars

> **TL;DR** — A **forward proxy** sits in front of *clients* and makes requests on their behalf; the server does not know (and usually does not care) who the real client is. A **reverse proxy** sits in front of *servers* and accepts requests on their behalf; the client thinks it is talking to the origin. Because a reverse proxy terminates client connections and opens its own connections to the backend, it becomes the natural place to do TLS termination, compression, connection pooling, buffering, retries, auth, rate limiting, and observability — and also the natural place for a whole class of failures (502, 504, slowloris, buffering breaks streaming, `X-Forwarded-For` spoofing).

> **60-second interview answer:** "A forward proxy represents the client — it hides who the client is, and it is what a corporate egress proxy or a VPN looks like. A reverse proxy represents the server — it hides the backend topology and is what Nginx, Envoy and every cloud load balancer's L7 mode actually are. The reason interviews love it is that one mechanism explains a dozen behaviours: because the proxy terminates the client's TCP and TLS and opens separate connections upstream, you get TLS termination at the edge, connection pooling and keep-alive to the backend, HTTP/2 or HTTP/3 to the client and HTTP/1.1 upstream, request buffering and therefore backpressure, and a new failure surface — 502 when the backend is unreachable, 504 when it is too slow, and `X-Forwarded-For` for client IP which I must only trust from a trusted hop."

---

## 1. Definition & mental model

**Plain English.**

| | Forward proxy | Reverse proxy |
|---|---|---|
| Sits in front of | The **client** | The **server** |
| Configured by | The client (browser settings, PAC file, VPN, corporate policy) | The service operator (DNS points at it) |
| Knows the identity of | The client | The backend fleet |
| The other side sees | The proxy's IP, not the client's | The proxy's IP, not the origin's |
| Typical examples | Squid, corporate egress proxies, Tor, VPN gateways, a scraping proxy pool | Nginx, HAProxy, Envoy, Traefik, Cloudflare, AWS ALB, an API gateway |
| Primary purpose | Egress control, privacy, caching for many clients, access policy | Ingress control, load balancing, TLS, security, caching for many servers |

**Memory hook:** the *direction the anonymity points*. **Forward proxy hides the client from the server. Reverse proxy hides the server from the client.** Everything else follows.

**The analogy.** Think of a **company mailroom**.

- **Forward proxy** = the outgoing mailroom in your office building. You drop a letter there; it re-addresses the envelope with the building's return address and mails it. The recipient sees "Acme Corp, 5th floor," never your desk. The building can also inspect, log, or refuse to send your letter. That is a forward proxy: egress control and client anonymity. (If the building also keeps a copy of every catalogue you order from, that is forward-proxy caching.)
- **Reverse proxy** = the **reception desk of a large company**. Customers call one public number. The receptionist answers, decides which department should handle it, transfers the call, and the customer never learns the internal extension. The receptionist also verifies the caller, can queue callers when all lines are busy, and can hang up on abusive callers. That is a reverse proxy: one public identity, many hidden backends, with policy in the middle.

The transcript uses the same idea with Nginx as the reverse proxy in front of application servers, and a VPN as the forward proxy example — "VPN hides the client; Nginx hides the server."

---

## 2. Why it exists — the problem it solves

### For the reverse proxy

1. **You cannot expose 50 app servers to the internet.** You need one address, one certificate, one firewall rule, and a place to swap the fleet underneath.
2. **TLS must terminate somewhere.** Terminating at the edge means the app servers do not each need certificates, private keys, and CPU for handshakes. A single modern CPU core can do roughly 1,000–5,000 TLS 1.3 handshakes/second; on 50 app servers that is 50× the key management and 50× the patching burden.
3. **Naive app servers make terrible internet-facing servers.** Node.js/Python app servers handle slow clients badly (each connection occupies a worker or event-loop slot); they often lack graceful restart, connection limits, and request-size limits. A proxy that speaks HTTP correctly and buffers protects them (**slowloris** defence, §6.4).
4. **You need shared behaviour across heterogeneous backends.** Compression, caching, retry, CORS headers, request IDs, auth — implement once at the proxy rather than in six languages.
5. **Connection pooling.** Backends get a small number of long-lived, warm connections instead of thousands of new TCP/TLS handshakes (which also avoids ephemeral-port exhaustion and TIME_WAIT storms).
6. **Rolling deploys with zero downtime.** The proxy can drain connections from an old instance, keep serving, and shift traffic gradually.
7. **Protocol translation and upgrade.** HTTP/3 from the client, HTTP/1.1 to a legacy backend; WebSocket upgrade; gRPC over HTTP/2 internally while the edge speaks HTTP/1.1 to browsers.

### For the forward proxy

1. **Egress control**: allow-list which external hosts employees/services may reach.
2. **Audit and DLP**: log and inspect outbound traffic.
3. **Client privacy / bypassing geo restrictions**: VPN and Tor.
4. **Shared caching for many clients** in a bandwidth-constrained network (the original 1990s use case — CERN's proxy cache).
5. **Scraping and crawling fleets** that need many egress IPs to avoid rate limits.
6. **NAT/egress IP stability**: all of a VPC's outbound traffic exits through a known IP so partners can allow-list you.

### What breaks without a reverse proxy

- Every app server exposed to the internet → larger attack surface, more certs, harder patching.
- No connection pooling → TLS handshake cost per request, ephemeral port exhaustion under load (`TIME_WAIT` accumulation can exhaust ~28k ports per destination in ~60 s under 10k req/s).
- No buffering → a slow client can hold an app worker, and a 10 Mbit/s uploader can pin a Node.js event loop slot. **Slowloris takes the whole service down with a handful of sockets.**
- No graceful drain → every deploy drops in-flight requests.
- No single place for retries/timeouts → each service invents its own, usually wrong.

---

## 3. Internal mechanics — how it actually works

### 3.1 The connection model (the thing everything else derives from)

```
Client                Reverse Proxy                     Backend
  │  TCP + TLS(1)        │                                │
  ├─────────────────────▶│                                │
  │  HTTP request        │  TCP (pooled, reused)          │
  │                      ├───────────────────────────────▶│
  │                      │  HTTP/1.1 request              │
  │                      │◀───────────────────────────────┤
  │                      │  response                      │
  │◀─────────────────────┤                                │
  │  response            │                                │
```

**Six consequences of the proxy owning two separate connections:**

1. **Client IP is lost** → hence `X-Forwarded-For`, `Forwarded`, `X-Real-IP`.
2. **TLS is terminated at the proxy** → hence re-encryption or plaintext upstream, and hence SNI-based routing.
3. **Client and upstream protocols can differ** → HTTP/3 client, HTTP/2 upstream, or WebSocket upgrade handled at the edge.
4. **The proxy can buffer** → hence backpressure, request-size limits, and the ability to serve a slow client from a fast backend. Also hence broken streaming if configured wrong.
5. **The proxy can retry** → because it has the request in memory and the upstream connection may be new. (Careful: retrying non-idempotent requests duplicates side effects.)
6. **The proxy becomes a single point of failure and a bottleneck** → it must be redundant, and its own timeouts/pools must be sized for peak.

### 3.2 TLS termination, SNI and re-encryption

**TLS termination** = the proxy performs the handshake and then talks plaintext (or, in a service mesh, mTLS) to the backend.

```
Client ──TLS──▶ Proxy ──plaintext HTTP──▶ App server
                (holds cert + key)
```

Three topologies:

| Topology | Where TLS ends | Where it is re-established | Use when |
|---|---|---|---|
| **Termination at the edge** | Proxy | Nowhere — plaintext inside the trusted network | Default; simplest and fastest |
| **Re-encryption (TLS bridging)** | Proxy | Proxy → backend, new TLS session | Compliance requires encryption in transit inside the network |
| **TLS passthrough** | Backend | — | The proxy must not see plaintext (e.g. legal), or you need end-to-end mTLS to a third party; **loses L7 routing and caching**, since the proxy cannot read the request |

**SNI (Server Name Indication)** is the TLS ClientHello extension that carries the hostname, letting one IP serve many certificates. It is what makes multi-tenant hosting possible, and what a proxy uses to choose a certificate and a route:

```
ClientHello
  ├─ SNI: shop.example.com       ← plaintext, even in HTTPS
  ├─ ALPN: h2, http/1.1          ← protocol negotiation
  └─ supported_versions: TLS1.3, TLS1.2
```

Consequences:
- **SNI is exposed pre-handshake.** Anyone on the path sees the hostname even with ECH absent. Encrypted Client Hello (ECH) exists but has limited deployment.
- **No SNI = no routing.** Clients that omit SNI (older Java, some IoT, a raw IP `curl`) hit the default certificate, which then fails to validate → handshake error before any HTTP status exists.
- **ALPN determines the protocol**: `h2` → HTTP/2, `h3` → HTTP/3, absent → HTTP/1.1.

**Certificate automation:** ACME with an HTTP-01 challenge requires the proxy to pass `/.well-known/acme-challenge/` through unauthenticated and *not* redirect it. DNS-01 avoids that but needs DNS API credentials. In a large fleet, certificates must be issued and rotated automatically — a manual renewal is an outage waiting on a calendar reminder.

### 3.3 HTTP/1.1 vs HTTP/2 vs HTTP/3 — what the proxy must handle

| Property | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| Transport | TCP | TCP (+TLS required in practice) | QUIC over UDP |
| Multiplexing | None — one request at a time per connection | Streams over one connection | Streams over one connection |
| Head-of-line blocking | Yes, at connection level | Yes, at TCP level (lost packet stalls all streams) | **No** — loss only stalls its own stream |
| Header compression | None | HPACK (stateful, dynamic table) | QPACK |
| Server push | No | Yes (deprecated in browsers) | Yes (rarely used) |
| Handshake | 1 RTT TCP (+1–2 TLS) | 1 RTT TCP + 1 RTT TLS; 0-RTT resumption possible | 1 RTT QUIC; 0-RTT resumption |
| Connection setup cost | High; browsers open 6 parallel connections per host | One connection per origin | One connection per origin, survives IP change (connection migration) |
| Typical browser usage | Fallback | Default for HTTPS | Growing, especially on lossy mobile |
| Proxy considerations | Needs `Connection`, `Keep-Alive`, `Transfer-Encoding` handling; hop-by-hop headers must be stripped | Must translate from HTTP/1.1 headers; must not coalesce across cert boundaries; `:authority` replaces `Host` | Must handle UDP/443; some enterprise firewalls block UDP, so fallback to H2 must be configured |

**The classic HTTP/1.1 problem the proxy fixes:** a browser opens ~6 connections per host, so 6 concurrent requests. Each is a full request-response serialisation on that connection. With HTTP/2, hundreds of streams share one connection. With HTTP/3 on a lossy mobile link, one lost packet no longer stalls everything.

**What a proxy must do to translate correctly:**
- Strip **hop-by-hop headers** on every hop: `Connection`, `Keep-Alive`, `Proxy-Authenticate`, `Proxy-Authorization`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade`.
- Rebuild `Host`/`:authority`.
- Handle `Transfer-Encoding: chunked` from a backend that does not set `Content-Length`.
- Not send HTTP/1.1's `Connection: close` upstream on every request (kills pooling).

### 3.4 Keep-alive and connection pooling

**Keep-alive** reuses a TCP connection for multiple HTTP requests, avoiding a handshake per request.

- Latency saving: ~1 RTT (TCP) + 1–2 RTT (TLS) per request avoided.
- Cost: a held-open socket and a server-side timeout (`keepalive_timeout`, often 60–75 s; AWS ALB default idle timeout **60 s**).
- The classic bug: **timeout mismatch**. If the proxy's idle timeout is 65 s but the backend's is 60 s, the proxy will occasionally send a request down a connection the backend has just closed — a **502 or a spurious retry** every few minutes, in a pattern that looks random. **Rule: the client-side idle timeout must be shorter than the server-side one.** (Or the backend must send `Connection: close` before closing.)
- Another classic bug: a load balancer or firewall silently drops idle connections after 350 s (AWS NLB default idle timeout is **350 s**) while the client thinks the connection is alive. Set client keep-alive and application timeouts below it.

**Connection pooling (upstream):**

```
Proxy upstream pool to backend:backend-port
  ├─ max_connections: 100 (per worker)
  ├─ idle_timeout: 60 s
  ├─ keepalive_requests: 1000
  └─ queue when exhausted → this is your real backpressure
```

Pool sizing rule of thumb: **connections ≈ backend concurrency limit**, not "as many as possible." If the backend has 4 workers × 32 threads = 128 concurrent requests, a pool of 128 with a bounded queue is right. A pool of 1,000 just moves the queue from the proxy into the backend, where it consumes memory and hides the overload.

Nginx caveat: `keepalive` in an `upstream` block is *per worker process*. With 8 workers and `keepalive 32`, you get up to 256 upstream connections.

### 3.5 Request buffering and backpressure

By default, Nginx **buffers the request body** to disk/memory before forwarding (`client_body_buffer_size`, then `client_max_body_size`; overflow goes to `client_body_temp_path`). This means:

- A slow uploader cannot hold a fast backend worker. **This is the core slowloris defence.**
- The backend sees the full body immediately.
- But: for **large uploads** (a 5 GB video), buffering to disk doubles I/O and adds latency; you disable it with `proxy_request_buffering off;` so the body streams straight through.
- And: for **streaming responses**, `proxy_buffering on` (the default) causes the proxy to accumulate response bytes before sending them to the client — which destroys server-sent events, LLM token streams, and WebSocket-adjacent patterns. Set `proxy_buffering off;` and send `X-Accel-Buffering: no` from the app.

| Directive | Default | Set to | Because |
|---|---|---|---|
| `proxy_buffering` | `on` | `off` for SSE/LLM streaming | Buffering delays tokens |
| `proxy_request_buffering` | `on` | `off` for huge uploads | Avoid disk staging of GBs |
| `client_max_body_size` | `1m` | Set explicitly (e.g. `100m`, or `0` for unlimited at the edge with app-level limits) | Default 1 MB silently 413s uploads |
| `client_body_timeout` | `60s` | Lower (e.g. 15–30 s) | Slow request-body attacks |
| `proxy_read_timeout` | `60s` | Raise for LLM streaming (e.g. 300 s) | Default kills long generations |
| `proxy_connect_timeout` | `60s` | Lower (e.g. 5 s) | Fail fast on unreachable backends |
| `proxy_send_timeout` | `60s` | Tune | Slow backend reads |

### 3.6 WebSocket upgrade through a proxy

WebSocket starts life as an HTTP request with an upgrade:

```http
GET /ws HTTP/1.1
Host: chat.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
```

and the server answers:

```http
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

**What the proxy must do:**
- Forward `Upgrade` and `Connection` (Nginx: `proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`). If you set `Connection: close` unconditionally, WebSockets break — this is the single most common Nginx WebSocket bug.
- **Disable buffering** for that location.
- **Raise the read timeout above the idle period.** A chat connection may be idle for minutes; the 60 s default will kill it. Use `proxy_read_timeout 3600s;` or send application-level pings (every 20–30 s) — pings are better because they keep *every* hop alive, including NAT gateways and load balancers.
- **HTTP/2 cannot carry a WebSocket upgrade in the RFC 6455 form.** It uses **RFC 8441 extended CONNECT** (`:method CONNECT`, `:protocol websocket`). Many proxies do not support it; browsers may therefore force HTTP/1.1 for WebSocket, or you use a separate subdomain for the WS endpoint. In practice: **put WebSocket on its own hostname, terminate HTTP/1.1 there, keep HTTP/2/3 for everything else.**
- **Sticky routing or a shared registry.** Once the socket is open, that connection is pinned to one backend. The reverse proxy must not rebalance mid-connection. This is why chat systems need a connection registry (§06-case-studies/02).

### 3.7 gRPC proxying

gRPC is HTTP/2 with:
- Mandatory TLS in most deployments,
- A binary framed body (`Content-Type: application/grpc+proto`),
- **Streaming semantics** — unary, server-streaming, client-streaming, bidirectional,
- **Trailers** (`grpc-status`, `grpc-message`) sent in HTTP/2 trailers.

**The requirements this places on a proxy:**
1. **Must speak HTTP/2 end-to-end.** A proxy that terminates HTTP/2 and re-emits HTTP/1.1 upstream will break gRPC entirely (HTTP/1.1 has no trailers and no multiplexing).
2. **Must forward trailers.**
3. **Must not buffer** (streaming RPCs are long-lived).
4. **Load balancing must be per-request (L7), not per-connection (L4).** With L4, one long-lived HTTP/2 connection from a client pins *all* its RPCs to one backend — so a client with 10 RPCs per second sends all of them to a single server. Nginx needs `grpc_pass`; Envoy handles it natively; HAProxy needs HTTP/2 support enabled; AWS ALB supports gRPC.
5. **`grpc-timeout` header** must be respected and propagated, and the proxy's own read timeout must exceed it.
6. **Health checking** must use the gRPC health checking protocol, not an HTTP GET.

**Common failure:** putting gRPC behind an L4 load balancer and wondering why traffic is grossly unbalanced. Another: an L7 proxy with a 60 s `proxy_read_timeout` killing long-lived bidirectional streams.

### 3.8 Forward proxy mechanics: tunnelling with `CONNECT`

A browser behind a forward proxy sends the full URL:

```http
GET http://example.com/index.html HTTP/1.1
Proxy-Connection: keep-alive
```

And for HTTPS it asks for a raw tunnel, because the proxy must not see the plaintext:

```http
CONNECT example.com:443 HTTP/1.1
Host: example.com:443

← HTTP/1.1 200 Connection Established
← (then opaque TLS bytes flow both ways)
```

`CONNECT` is also the mechanism for proxying any TCP protocol (SSH, database connections) through an HTTP proxy. It is why corporate proxies can enforce policy on *which* hosts you reach but not on *what* you send to them (assuming TLS).

**Proxy chaining:** proxies can chain (client → proxy A → proxy B → origin). Policy, auth, and rate limits may apply at each hop; latency grows linearly. Detect and reject loops by checking `Via` and `Max-Forwards`.

**Transparent proxy** = interception without client configuration, usually via network routing (WCCP, policy routing, or a service mesh's iptables rules). The client does not know a proxy exists. Common in ISPs, corporate networks, and service meshes (Istio's sidecar intercepts via iptables). Caveat: with TLS, the transparent proxy cannot read anything unless it does MITM with an installed CA — a practice with real security and privacy consequences, and a frequent cause of "my app fails only on the corporate network" bugs (certificate pinning breaks).

**PAC files / WPAD:** JavaScript that tells a browser which proxy to use per URL. Legacy, fragile, and historically a security hole (WPAD DNS hijacking).

### 3.9 Sidecars and service meshes

A **sidecar** is a reverse proxy (usually Envoy) running in the same pod as your application, with all inbound and outbound traffic transparently redirected through it by iptables/eBPF rules.

```
Pod
┌─────────────────────────────────────┐
│  ┌──────────┐      ┌─────────────┐  │
│  │  app     │◀────▶│ Envoy       │◀────┼── inbound from mesh
│  │ :8080    │      │ sidecar     │─────┼──▶ outbound to mesh
│  └──────────┘      └─────────────┘  │
└─────────────────────────────────────┘
```

What the mesh gives you, implemented uniformly without touching app code: mTLS between all services with automatic certificate rotation (SPIFFE identities), retries with budgets, timeouts, circuit breaking, outlier detection, traffic shifting for canaries, observability (RED metrics per service pair), and authorisation policy (`AuthorizationPolicy`).

| Cost | Detail |
|---|---|
| Latency | ~0.5–3 ms per hop pair added by the sidecar |
| Resource | ~50–100 MB RAM and ~0.1–0.5 vCPU per sidecar; at 1,000 pods that is 50–100 GB of RAM |
| Complexity | Debugging now spans app + sidecar + control plane (istiod) |
| Failure mode | The sidecar's iptables rules can black-hole traffic; a bad control-plane push can degrade the whole mesh |
| Upgrade risk | You upgrade the data plane and the control plane on someone else's release schedule |

**When it is worth it:** you have many services (>20), multiple languages, and a real mTLS/compliance requirement. **When it is not:** three services in one language — use a library.

### 3.10 Nginx vs HAProxy vs Envoy

| | **Nginx** | **HAProxy** | **Envoy** |
|---|---|---|---|
| Primary role | Web server + reverse proxy | Pure L4/L7 load balancer | L7 proxy for service meshes and edge |
| Config model | Static config file + reload (`nginx -s reload`) | Static config + runtime API (Data Plane API) | **Dynamic** via xDS APIs — no reload |
| Performance | Excellent static file serving; event-driven (`epoll`) | Best-in-class raw LB throughput, very low latency | Excellent, designed for high churn |
| Health checks | Passive-ish; `max_fails`/`fail_timeout`; needs Plus for active | Active + passive, very fine-grained | Active + passive + outlier detection |
| L7 features | Rich: rewrite, caching, gzip/brotli, sub_filter, Lua (OpenResty) | Strong ACLs, stick tables, agent checks | Rich: retries, circuit breaking, shadowing, fault injection, WASM |
| gRPC | `grpc_pass` | Supported (with HTTP/2) | Native, best support |
| Dynamic config | Requires reload (OpenResty/Plus improves) | Runtime API | xDS, zero reload |
| Best for | Front door, static + API, simple LB, caching | TCP/UDP LB, very high connection counts, precise health checking | Kubernetes ingress, service mesh, complex traffic policy |
| Weakness | Reload-based config; fewer LB algorithms; caching is basic | Config is terse and hard to read; no caching; UI is commercial | Steep learning curve, verbose config, no caching in the classic sense |

**Choosing:** Nginx for the web front door (static + proxy + cache + TLS); HAProxy when the job is "move millions of connections with precise health checking and no caching"; Envoy when traffic policy is dynamic and Kubernetes-native or when you need a mesh.

---

## 4. Variants & taxonomy

| Variant | In front of | Notes |
|---|---|---|
| Forward proxy (explicit) | Clients | Client-configured; supports `CONNECT` tunnelling |
| Transparent proxy | Clients | Interception, no client config; TLS-blind unless MITM |
| Reverse proxy | Servers | The default meaning of "proxy" in system design |
| TLS-terminating proxy | Servers | Holds certs; plaintext (or mTLS) upstream |
| TLS-passthrough proxy | Servers | Routes on SNI only; no L7 features |
| Re-encrypting / TLS-bridging proxy | Servers | Cert on both sides |
| Load balancer (L4) | Servers | TCP/UDP-level; no HTTP awareness |
| Load balancer (L7) | Servers | Is a reverse proxy with balancing + health checks |
| API gateway | Servers | A reverse proxy with auth, quotas, transformation, versioning |
| Service mesh sidecar | Both | Per-pod proxy; east-west traffic |
| Ingress controller | Servers | Kubernetes-native reverse proxy (Nginx/Envoy/Traefik behind an Ingress resource) |
| CDN edge | Servers | A globally distributed caching reverse proxy |
| Caching proxy | Either | Squid forward-caching for clients; Nginx/Varnish reverse-caching for servers |
| WAF | Servers | A reverse proxy with a rules engine (ModSecurity, AWS WAF) |
| Egress gateway | Clients (internal) | NAT + allow-list for VPC egress; a forward proxy for machines |
| Circuit-breaking proxy | Servers | Envoy/mesh; also Resilience4j at the library level |
| Protocol-translating proxy | Both | gRPC↔REST (grpc-gateway), HTTP/3↔HTTP/1.1, HTTP↔WebSocket |

---

## 5. Pros / Cons

| Pros | Cons |
|---|---|
| Single public entry point; hides backend topology | A new single point of failure — must itself be redundant |
| TLS termination in one place: one cert, one rotation, one patch | Plaintext inside the network unless you re-encrypt |
| Connection pooling to backends; avoids handshake storms and port exhaustion | Pool exhaustion becomes the bottleneck and the queue is invisible without metrics |
| Buffering protects backends from slow clients (slowloris) | Buffering *breaks* streaming (SSE, LLM tokens) if not disabled |
| Shared cross-cutting behaviour: gzip/brotli, caching, retries, headers, request IDs | Configuration drift between proxy and app; two places to reason about timeouts |
| Zero-downtime deploys via graceful drain | Draining is only graceful if the drain timeout exceeds the longest request |
| Protocol agility: HTTP/3 to client, HTTP/1.1 to legacy backend | Translation is where subtle bugs live (hop-by-hop headers, trailers) |
| Observability: one place for access logs, RED metrics, tracing | Access logs are high-volume and can themselves be a cost/perf problem |
| WAF/rate limiting/bot mitigation at the edge | A misconfigured WAF blocks legitimate traffic; tuning takes real work |
| Forward proxy: egress control, audit, IP stability, client privacy | Forward proxy: MITM breaks pinning; single egress IP becomes a rate-limit magnet |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 502 Bad Gateway

**Meaning:** the proxy could not get a valid response from the upstream. It is a *proxy-generated* status — the backend never sent it.

**Causes, ranked by frequency in production:**

1. **Backend process is down or crashed** (OOM-killed, bad deploy, port not listening).
2. **Wrong upstream address or port** in config after a rename.
3. **Backend closed the connection mid-request** — often a keep-alive race (§3.4): the proxy reused a pooled connection the backend had already closed.
4. **Response headers too large** — `upstream sent too big header while reading response header` (Nginx). Fix: raise `proxy_buffer_size` and `large_client_header_buffers`, but first see §7 of the DNS file: really big cookies are the usual culprit.
5. **Backend sent a malformed response** — invalid chunked encoding, a bad status line, HTTP/2 with a plaintext backend.
6. **Backend restarted during a deploy** without a drain.
7. **File descriptor exhaustion** on the backend (`EMFILE` → accept failing → connection refused/closed).
8. **MTU / packet fragmentation** issues: small requests work, large ones 502 — a classic hard-to-find case caused by a broken PMTU path.

**Debug order:** `curl` the backend directly from the proxy host → check the proxy error log for the specific upstream message → check backend process health and logs → check for a keep-alive timeout mismatch → check header sizes.

### 6.2 504 Gateway Timeout

**Meaning:** the proxy's connection to the upstream established, but the upstream did not respond within the configured timeout.

**Causes:** slow database query, slow downstream dependency, GC pause, thread pool saturation (all workers busy → requests queue), a lock convoy, unbounded result sets, a network partition that black-holes packets (SYN retries rather than a refusal).

**Diagnostic discipline:**
- Distinguish **connect timeout** (`proxy_connect_timeout`, "is it reachable?") from **read timeout** (`proxy_read_timeout`, "is it alive?"). A 504 with an immediately-established connection is a read timeout.
- Check whether the timeout is *at the proxy* or *at the client's own client*. A client with a 10 s timeout in front of a proxy with a 60 s timeout means the user gives up first, and the proxy keeps working on a request nobody wants ("work amplification" / wasted work). **Timeouts must decrease as you go outward, not increase.**
- **Set the proxy timeout *below* the client's timeout and *above* the backend's internal timeout.** Example ladder: client 20 s → LB 15 s → reverse proxy 12 s → app server 10 s → DB statement timeout 8 s → DB `lock_timeout` 3 s. If any layer exceeds the one outside it, you get work that nobody is waiting for.

**The timeout ladder (canonical example):**

```
Client 20 s
  └─ CDN/edge 18 s
      └─ ALB idle 60 s, target read 15 s
          └─ Nginx proxy_read_timeout 12 s
              └─ App gRPC/HTTP client deadline 10 s
                  └─ DB statement_timeout 8 s, lock_timeout 3 s
```

### 6.3 The retry storm and non-idempotent duplication

When the proxy retries a `POST` that actually succeeded but whose response was lost, you get a **duplicate payment, duplicate order, duplicate message**. Rules:

- **Only retry idempotent methods** (`GET`, `HEAD`, `PUT`, `DELETE` with proper semantics) *unless* the request carries an idempotency key.
- **Retry only on connection-level failures and safe statuses** (connection refused, `502`, `503` with `Retry-After`), never on `500` blindly and never on `4xx`.
- **Budget the retries.** Envoy's retry budget caps retries as a percentage of active requests (default ~20%) so that retries cannot amplify a partial outage into a full one. Without a budget, a 10% backend failure at 3 retries becomes a 33% traffic increase on the healthy backends — killing them too.
- **Jitter the backoff.** Without jitter, all clients retry simultaneously.

### 6.4 Slowloris and slow-body attacks

**Slowloris:** the attacker opens many connections and sends *partial* HTTP headers, one byte at a time, never completing the request. A thread-per-connection server (Apache prefork, a naive Java servlet stack) exhausts its worker pool with a few hundred sockets.

**Slow body (RUDY):** the same trick with `Content-Length` declared and the body dribbled out.

**Defences at the proxy:**
- Buffer the request body (`proxy_request_buffering on`, the default) so the backend never sees a partial body.
- `client_header_timeout 10s; client_body_timeout 15s;` — bound the time to send headers/body.
- `limit_conn` / `limit_req` per IP.
- Cap simultaneous connections per IP.
- `send_timeout` to bound how long the proxy will wait to write to a slow client.
- Nginx's event-driven model is itself immune-ish (connections are cheap), but the *upstream* is not — which is exactly why the proxy must buffer.

### 6.5 Buffering breaks streaming

Symptom: an LLM chat UI shows nothing for 20 seconds and then the whole answer at once; SSE events arrive in batches; a progress bar jumps.

Cause: an intermediary is buffering. Check every hop:
- Nginx: `proxy_buffering off;` and `X-Accel-Buffering: no` from the app.
- Cloudflare/CloudFront: disable compression for the streaming route (compression forces buffering) and confirm the cache policy is pass-through.
- A CDN that buffers chunked responses.
- An ALB: fine, ALBs stream, but the *idle* timeout (60 s default) will kill a long generation with no bytes.
- Application frameworks: some WSGI/ASGI servers buffer by default; some HTTP clients too.

Also verify `Cache-Control: no-store` so no intermediary caches a stream, and disable gzip on that route — `gzip on` with a chunked response is one of the most common causes.

### 6.6 `X-Forwarded-For` and real client IP

```
X-Forwarded-For: client, proxy1, proxy2
```

Each hop appends the IP it received from. The **leftmost** entry is claimed to be the client; the **rightmost** is the last hop before you.

**The trap:** `X-Forwarded-For` is **client-supplied and trivially spoofable**. If your rate limiter reads the leftmost value and you do not strip client-supplied headers at the edge, an attacker sends `X-Forwarded-For: 1.2.3.4` on every request and bypasses per-IP rate limiting entirely, or poisons your audit logs.

**Correct approach:**
1. At the **outermost trusted hop**, *delete* any client-supplied `X-Forwarded-For`/`X-Real-IP` and set your own.
2. Downstream, take the **rightmost untrusted** value, or configure a trusted-hop count and index from the right.
3. Or use the standard `Forwarded: for=192.0.2.60;proto=https;by=203.0.113.43` header, which supports `for=`, `by=`, `host=`, `proto=`.

```nginx
# At the edge: replace, do not append, for untrusted input
proxy_set_header X-Forwarded-For $remote_addr;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_set_header X-Forwarded-Host $host;
```

**Application-side parsing done right (conceptual):**

```python
TRUSTED_HOPS = 2   # CDN + our own edge proxy
def client_ip(request):
    xff = [ip.strip() for ip in request.headers.get("X-Forwarded-For", "").split(",") if ip.strip()]
    # walk from the right, skipping the IPs our own infrastructure appended
    chain = xff + [request.remote_addr]
    return chain[-(TRUSTED_HOPS + 1)] if len(chain) > TRUSTED_HOPS else chain[0]
```

**Related headers and their pitfalls:**
- `X-Forwarded-Proto` — required so the app knows the original scheme is `https`, otherwise redirect loops (app redirects to https → proxy sees http → redirect again).
- `X-Forwarded-Host` — needed for canonical URLs, but **never trust it for security decisions** (§6.7, cache poisoning).
- `X-Real-IP` — Nginx-specific, single value.
- `Forwarded` — RFC 7239, rarely fully implemented.

### 6.7 Host header attacks through a proxy

Because the proxy forwards the `Host` header, an attacker who can control it can cause:
- **Password reset poisoning** — the app builds the reset link from `Host`, so the attacker sets `Host: evil.com` and the victim's reset token is mailed pointing at the attacker.
- **Cache poisoning** (see the CDN file) — `Host` or `X-Forwarded-Host` reflected into a cached response.
- **SSRF / routing bypass** — some proxies route by `Host`; a spoofed value could reach an internal vhost.

**Defences:** an allow-list of valid `Host` values at the proxy (return `421 Misdirected Request` or `400` otherwise); the app builds absolute URLs from a configured canonical base URL, never from the request.

### 6.8 Hop-by-hop header leakage

Headers that are meaningful for one hop only must be dropped and, where applicable, regenerated:

```
Connection, Keep-Alive, Proxy-Authenticate, Proxy-Authorization,
TE, Trailer, Transfer-Encoding, Upgrade
```

Failures if you do not: a `Transfer-Encoding: chunked` forwarded to an HTTP/2 backend is a protocol error; a `Connection: close` forwarded upstream kills pooling; a `TE: gzip` leaked upstream confuses intermediaries.

Also: **`Connection: keep-alive, X-Custom-Header`** is a legitimate but rarely-seen pattern that marks `X-Custom-Header` as hop-by-hop. Proxies that blindly forward it leak an internal header.

### 6.9 Proxy as a single point of failure

The proxy is on the critical path of every request. Consequences:
- A bad config reload takes everything down (test with `nginx -t` and a canary instance).
- A certificate expiry takes everything down.
- A DNS change for the proxy hostname is now a global availability event.
- The proxy's own resource limits (file descriptors, worker connections, upstream pool) become the system's limit.

**Mitigations:** run ≥2 proxy instances across AZs behind anycast or a DNS name with multiple A records; make reloads atomic and validated; alert on cert expiry at 30/14/7 days; load-test the proxy itself, not just the app; and size `worker_connections` and `ulimit -n` deliberately (`worker_connections 1024` × `worker_processes auto` is the concurrency ceiling).

### 6.10 Other traps worth naming

- **`421 Misdirected Request`** with HTTP/2 connection coalescing: a client reuses one connection for two hostnames that resolve to the same IP; the server's cert does not cover the second name → `421`. Fix by ensuring the certificate is valid for all coalesced names, or by setting `Connection: close` / disabling coalescing.
- **Header size limits.** `large_client_header_buffers 4 8k;` and `proxy_buffer_size 8k;`. Cookies are the usual cause of "upstream sent too big header" 502s; large JWTs in cookies are a common modern trigger (a 4 KB JWT in a cookie plus several others exceeds the 8 KB default).
- **Proxy protocol.** When a TCP proxy (NLB, HAProxy in TCP mode) sits in front of an L7 proxy, the L7 proxy sees only the TCP proxy's IP. Send the original via the **PROXY protocol** (`proxy_protocol on;` on the receiver, `send-proxy` on the sender) — or use `X-Forwarded-For` at the HTTP layer. Mismatch here means every request looks like it comes from one IP, breaking rate limits and geo routing.
- **Trailing slash and path normalisation.** `/api/../admin`, `//api`, `%2e%2e/` can bypass path-based routing rules if the proxy and the app normalise differently. A well-known class of auth bypass.
- **WebSocket connections and connection limits.** Each WebSocket is a long-lived connection counting against `worker_connections`. A million-user chat service needs the proxy's connection ceiling raised well above its request concurrency.
- **Graceful shutdown.** On `SIGTERM`, a proxy should stop accepting new connections, finish in-flight requests, keep WebSockets for a bounded grace period (usually you *do* want to cut them and let clients reconnect), and exit. Nginx's `worker_shutdown_timeout` bounds this. Without it, deploys cause visible errors.
- **DNS resolution of upstreams.** Nginx resolves `upstream` hostnames at startup and caches them forever unless you use variables with a `resolver` directive. In Kubernetes, where a Service IP can change, this produces the classic "Nginx is sending traffic to a dead pod IP" bug. Fix: use a `resolver` + variable `proxy_pass`, or an Ingress controller that watches endpoints.
- **Buffer/disk exhaustion.** `proxy_max_temp_file_size` (default 1 GB) and `client_body_temp_path` can fill disk under sustained large uploads. Monitor the temp path.

---

## 7. When to use / When NOT to use

### Use a reverse proxy when
- You have **more than one backend instance** and want a single address.
- You need **TLS termination**, certificate management, or SNI-based routing.
- You need **shared cross-cutting behaviour** (compression, caching, auth, headers, CORS).
- You need **connection pooling** or **slow-client protection**.
- You want **zero-downtime deploys** with connection draining.
- You need **protocol flexibility** (HTTP/3 to clients, HTTP/1.1 or gRPC upstream).
- You are exposing anything to the internet at all.

### Use a forward proxy when
- You need **egress control or audit** (corporate, regulated environments).
- You need a **stable egress IP** for partner allow-lists.
- You need **client anonymity** (VPN, Tor) or **many egress IPs** (scraping fleets).
- You need **shared caching** for a bandwidth-constrained client population.

### Do NOT use a reverse proxy when
- There is exactly **one backend, in the same process** — a library-level middleware is simpler.
- You need **true end-to-end TLS with no intermediary able to read traffic** — then passthrough only, and accept losing L7 features.
- You are **adding a proxy only to add a proxy** — every hop adds latency (~0.5–3 ms), a config surface, and a failure domain.
- You need **sub-millisecond internal RPC** — a mesh sidecar's overhead may not be worth it for two services.
- Your **streaming path cannot tolerate buffering** and you cannot reconfigure the proxy to pass through — route around it.

### Decision table

| Situation | Answer |
|---|---|
| Web front door, static + API | Nginx (TLS, gzip/brotli, cache, proxy) |
| Millions of TCP connections, precise health checks | HAProxy |
| Kubernetes ingress + dynamic traffic policy | Envoy / Ingress controller |
| gRPC microservices | Envoy or `grpc_pass`, L7 balancing, HTTP/2 end-to-end |
| WebSocket chat | Dedicated hostname, HTTP/1.1, buffering off, long read timeout, shared connection registry |
| LLM token streaming | Buffering off, compression off, read timeout 300 s, `no-store` |
| Two services, one language | A library (Resilience4j, Polly), not a sidecar |
| 50 services, multiple languages, mTLS mandate | Service mesh with sidecars |
| Corporate egress policy | Forward proxy with allow-list + `CONNECT` |
| Partner requires a fixed IP | NAT gateway / egress proxy with elastic IP |

---

## 8. Numbers, defaults & tuning

| Quantity | Default / typical | Notes |
|---|---|---|
| Nginx `worker_connections` | 512 (older) / 1024 | × `worker_processes` = max simultaneous connections |
| Nginx `keepalive_timeout` (client) | 75 s | Keep below the LB's idle timeout |
| Nginx `keepalive` (upstream, per worker) | unset | Set 32–128; multiply by worker count |
| Nginx `client_header_timeout` | 60 s | Lower to 10–15 s for internet-facing |
| Nginx `client_body_timeout` | 60 s | Lower to 15–30 s |
| Nginx `client_max_body_size` | **1 MB** | The cause of most surprise 413s |
| Nginx `proxy_connect_timeout` | 60 s | Lower to 3–5 s |
| Nginx `proxy_read_timeout` | 60 s | Raise to 300 s+ for SSE/LLM, 3600 s for WebSocket |
| Nginx `proxy_send_timeout` | 60 s | Tune for slow backends |
| Nginx `proxy_buffering` | `on` | `off` for streaming |
| Nginx `proxy_buffer_size` | 4k/8k | Raise for large headers/cookies |
| Nginx `proxy_buffers` | 8 × 4k/8k | Raise for large responses |
| Nginx `proxy_max_temp_file_size` | 1024 MB | Response overflow goes to disk |
| Nginx `large_client_header_buffers` | 4 × 8k | Raise if cookies are large |
| Nginx `worker_shutdown_timeout` | unset | Set to bound graceful shutdown |
| AWS ALB idle timeout | **60 s** | Raise for long streams; it is an idle, not total, timeout |
| AWS NLB idle timeout | **350 s** | TCP flows; keep client keep-alive below it |
| AWS ALB target group deregistration delay | 300 s | Lower it (30–60 s) for faster deploys |
| Default HTTP/1.1 connections per host (browser) | 6 | The reason HTTP/2 matters |
| TLS 1.3 handshake | 1 RTT (0-RTT resumption) | TLS 1.2 is 2 RTT |
| TLS handshakes/sec/core | ~1,000–5,000 | With session resumption and ECDSA certs |
| Proxy added latency per hop | 0.5–3 ms | Sidecar or Nginx; L4 is lower |
| Envoy sidecar memory | 50–100 MB | × pod count = real cost |
| Envoy retry budget | ~20% of active requests by default | Prevents retry amplification |
| Ephemeral port range (Linux) | 32768–60999 ≈ 28k ports | Exhaustible in ~60 s at 500 conn/s to one destination with TIME_WAIT |
| `TIME_WAIT` duration | 60 s (2×MSL) | `net.ipv4.tcp_tw_reuse=1` helps outbound |
| Recommended client timeout ladder | App 10 s < proxy 12 s < LB 15 s < client 20 s | Decreasing toward the user |
| `X-Forwarded-For` max length | Unbounded by spec | Bound it at the edge to prevent header-size abuse |
| Proxy protocol version | v1 (text) / v2 (binary) | v2 is preferred, supports TLVs |
| ACME cert alerting | 30 / 14 / 7 days | Automate; do not rely on memory |

---

## 9. Interview Q&A

**Q1 (easy). What is the difference between a forward proxy and a reverse proxy?**
A forward proxy represents the client: it is configured by the client, and the destination server sees the proxy's IP instead of the client's. That is a corporate egress proxy, a VPN, or a scraping proxy pool. A reverse proxy represents the server: the client is pointed at it by DNS and thinks it is talking to the origin, while the proxy forwards to a hidden backend fleet. That is Nginx, Envoy, and every cloud load balancer in L7 mode. So: forward hides the client, reverse hides the server.

**Q2 (easy). Why does every production system have a reverse proxy?**
Because it centralises everything that should not be in the application: TLS termination with one certificate to rotate, connection pooling to backends so you do not pay a handshake per request, buffering that protects the app from slow clients, compression and caching, a single place for access logs and tracing, rate limiting and WAF, and graceful draining so deploys do not drop requests. Also it hides the topology — the client sees one address and you can replace the fleet underneath without them knowing.

**Q3 (easy). What does TLS termination mean and where do you put it?**
It means the proxy performs the TLS handshake, holds the certificate, and forwards plaintext or a new TLS session to the backend. I put it at the outermost hop I own — the CDN edge or the load balancer — because that minimises handshake round trips for the user and means one certificate and one key to manage instead of one per app server. If compliance requires encryption inside the network I re-encrypt from the proxy to the backend, which costs a second handshake but keeps the compliance story. I avoid passthrough unless there is a hard requirement, because passthrough gives up all L7 routing and caching.

**Q4 (medium). Explain HTTP/2 and HTTP/3 and why a proxy cares.**
HTTP/2 multiplexes many streams over one TCP connection with HPACK header compression, which removes the browser's six-connection-per-host limit and the per-request handshake. But it still runs over TCP, so a single lost packet stalls all streams — head-of-line blocking at the transport layer. HTTP/3 runs over QUIC on UDP, so streams are independent: a lost packet only delays its own stream, and it supports 0-RTT resumption and connection migration when a phone changes network. The proxy cares because it has to translate: I typically terminate HTTP/3 or H2 at the edge and speak H2 to gRPC backends or HTTP/1.1 to legacy ones, and I must strip hop-by-hop headers and rebuild `Host`/`:authority` at each translation, or I break things subtly.

**Q5 (medium). How does a WebSocket connection traverse a reverse proxy?**
It starts as an HTTP/1.1 GET with `Upgrade: websocket` and `Connection: Upgrade`. The proxy must forward both headers, disable buffering, and raise the read timeout well above the idle period — otherwise the connection dies after the default 60 seconds. I also send application-level pings every 20–30 seconds so every hop including NAT gateways stays alive. Critically, HTTP/2 cannot carry the RFC 6455 upgrade in the normal way, so in practice I put WebSocket on a dedicated hostname and keep it on HTTP/1.1, and I make sure the backend that accepts the socket is reachable by a shared connection registry so any node can deliver a message to a socket held elsewhere.

**Q6 (medium). What is `X-Forwarded-For` and what is the common mistake with it?**
It is a header where each proxy appends the IP it received the request from, so the list grows left-to-right. The mistake is trusting the leftmost entry: it is client-supplied and trivially spoofable, so if a rate limiter keys on it, an attacker just sends a different fake value on every request and bypasses the limit entirely. The correct approach is to strip any client-supplied value at the outermost hop I control, set my own, and then at the application read from the right using a configured trusted-hop count. I also use `X-Forwarded-Proto` so the app knows the original scheme, otherwise I get redirect loops behind an HTTPS-terminating proxy.

**Q7 (medium). How do connection pools at the proxy interact with the backend?**
The proxy holds warm, pooled connections upstream, so backends get a small number of long-lived connections instead of a handshake per request. The sizing rule is that the pool should match the backend's real concurrency limit, not be as large as possible — if the backend has 128 concurrent request slots, a pool of 128 with a bounded queue is right, and anything larger just moves the queue into the backend where it consumes memory and hides overload. The classic operational bug is a timeout mismatch: the backend closes idle connections at 60 seconds while the proxy's idle timeout is 75, so occasionally the proxy sends a request down a connection that was just closed and you get random 502s every few minutes. The client side must time out first.

**Q8 (hard). Walk me through debugging a 502 versus a 504.**
Both are proxy-generated, so the backend never sent them. A 502 means I could not get a valid response: the backend is down, the address is wrong, it closed the connection mid-request — usually a keep-alive race — the response headers were too large, or the response was malformed. A 504 means the connection established but the backend did not answer in time, so it is a slowness problem: a slow query, a lock convoy, GC, or thread-pool saturation. My first step in both cases is to `curl` the backend directly from the proxy host to separate "the backend is broken" from "the proxy-to-backend path is broken," then read the proxy error log because it names the specific upstream failure — `upstream timed out` versus `connection refused` versus `upstream sent too big header`. For 504s I then check whether the timeout is a connect timeout or a read timeout, and I compare my timeout ladder to make sure the layers decrease toward the user.

**Q9 (hard). Explain the timeout ladder and why it matters.**
Every layer has a timeout, and if an inner layer's timeout is longer than the outer layer's, work continues on requests nobody is waiting for — wasted capacity, and worse, the outer layer may retry, so the backend sees double load for one user request. So the timeouts must strictly decrease as you go inward: client 20 seconds, load balancer 15, reverse proxy 12, application 10, database statement 8, database lock timeout 3. That way each layer gives up before its caller does, and you get one clean error at the outermost layer instead of a cascade of retries. I also distinguish connect timeouts, which should be short — three to five seconds — from read timeouts, which are workload-specific: 300 seconds for LLM streaming, an hour for WebSocket.

**Q10 (hard). What is slowloris and how does a reverse proxy defend against it?**
Slowloris is a denial-of-service where the attacker opens many connections and sends HTTP headers one byte at a time, never completing the request, so a thread-per-connection server exhausts its worker pool with a few hundred sockets. The proxy defends by buffering: it fully reads the request headers and body before opening an upstream connection, so a slow client never holds a backend worker. I also bound it with short header and body timeouts, a per-IP connection limit, and a bounded header size. Nginx's event-driven model makes the connections themselves cheap, but the real protection is that the backend never sees a partial request.

**Q11 (hard, testing §6). Our LLM chat streams tokens and users complain the answer appears all at once after 20 seconds. What do you check?**
Something is buffering. I check each hop. In Nginx, `proxy_buffering` defaults to on, which accumulates response bytes — it must be off for that location, and the app should send `X-Accel-Buffering: no`. Gzip must be off for that route, because compressing a chunked stream forces buffering. At the CDN I need the cache policy to be pass-through and compression disabled on the streaming path. At the ALB the idle timeout must exceed the longest gap between tokens; the default is 60 seconds. Then I check the application layer — some ASGI/WSGI servers and some HTTP clients buffer by default. And `Cache-Control: no-store` so nothing caches the completion. My rule is: any hop that can buffer, will buffer, until you tell it not to.

**Q12 (hard, testing §6). An attacker is bypassing our per-IP rate limit. How, and what do we do?**
Almost certainly we are keying on a spoofable client-IP header. The attacker sends a different `X-Forwarded-For` on every request, and if our limiter reads the leftmost value or trusts the header entirely, each request looks like a new client. The fix is to strip client-supplied forwarding headers at the outermost hop we control — the CDN or load balancer — and set them ourselves, then read the client IP at the app using a trusted-hop count from the right. I would also verify we are not keying on something even more spoofable like a client-supplied device ID or API key we do not validate. And I would add a test that sends a spoofed header and asserts the limiter still buckets correctly.

**Q13 (hard, testing §6). When is retrying at the proxy dangerous?**
When the request is not idempotent, and when retries are unbounded. If a `POST /payments` succeeds at the backend but the response is lost, a proxy retry creates a second payment — so I only auto-retry idempotent methods, or I require an idempotency key and make the backend deduplicate on it. The second danger is amplification: if 10% of backends are failing and each request retries three times, I have increased load on the healthy backends by roughly a third, which can tip them over and turn a partial outage into a total one. That is why Envoy has a retry budget — retries are capped as a percentage of active requests — and why I only retry on connection-level errors and `503` with `Retry-After`, never blindly on `500`, and always with jittered exponential backoff.

**Q14 (hard). What is the PROXY protocol and when do you need it?**
It is a small header a TCP proxy prepends to a connection so the next hop learns the original client's source and destination address. I need it when an L4 proxy or NLB sits in front of an L7 proxy: without it, the L7 proxy only sees the TCP balancer's IP for every request, which breaks per-IP rate limiting, geo routing, and audit logs, and makes all traffic look like it comes from one client. The alternatives are doing it at the HTTP layer with `X-Forwarded-For`, or using an L7 balancer end to end. The failure mode is a mismatch — sending proxy protocol to something that does not expect it produces a parse error on every connection.

**Q15 (hard). Would you put a service mesh sidecar in front of three services?**
No. For three services in one language, a library for retries, timeouts and circuit breaking is simpler, cheaper, and easier to debug. A mesh earns its keep when you have many services, multiple languages, and a real mTLS or compliance requirement, because then the alternative is implementing identical policy six times and drifting. The costs are real though: each sidecar adds latency per hop, 50 to 100 megabytes of memory per pod, and a control plane that can degrade the whole mesh with a bad push.

---

## 10. Common mistakes candidates make

1. **Confusing forward and reverse proxy.** State the rule: forward hides the client, reverse hides the server.
2. **Treating a reverse proxy and a load balancer as unrelated.** An L7 load balancer *is* a reverse proxy with balancing and health checks.
3. **Forgetting that the proxy owns two connections.** Almost every proxy behaviour and bug derives from this.
4. **Not mentioning TLS termination** — and its cost, key management, or the passthrough/re-encrypt trade-off.
5. **Saying "just raise the timeout"** for a 504 instead of finding the slow dependency, and not knowing the ladder must decrease inward.
6. **Assuming `X-Forwarded-For` is trustworthy.** It is the classic rate-limiting bypass.
7. **Leaving out slowloris/buffering.** Buffering is a security feature as well as a performance one.
8. **Forgetting hop-by-hop headers** and HTTP version translation issues.
9. **Proposing WebSocket over HTTP/2 without knowing about RFC 8441.** In practice, WebSocket goes on its own HTTP/1.1 hostname.
10. **Not knowing that `client_max_body_size` defaults to 1 MB** — a real, frequently-hit production default.
11. **Retrying everything.** Duplicated payments; retry amplification.
12. **Ignoring the proxy as a single point of failure** and a capacity ceiling (fds, `worker_connections`).
13. **Buffering breaking streaming** — the single most common real-world LLM-app bug at the infra layer.
14. **Forgetting connection draining** so every deploy drops requests.

---

## 11. Cheat-sheet summary

| Concept | One-line truth |
|---|---|
| Forward proxy | Represents the client; hides the client; client-configured |
| Reverse proxy | Represents the server; hides the backend; DNS-configured |
| Two connections | Client↔proxy and proxy↔backend; source of nearly every behaviour and bug |
| TLS termination | Proxy holds the cert; plaintext or re-encrypted upstream |
| TLS passthrough | Routes on SNI only; loses all L7 features |
| SNI | Hostname in the ClientHello, in the clear; enables multi-tenant TLS |
| ALPN | Negotiates `h2`/`h3`/`http/1.1` during the handshake |
| HTTP/1.1 | One request per connection at a time; 6 connections per host in browsers |
| HTTP/2 | Multiplexed streams over TCP; HPACK; still TCP head-of-line blocking |
| HTTP/3 | QUIC over UDP; independent streams; 0-RTT; connection migration |
| Keep-alive | Reuse connections; **client idle timeout must be shorter than server's** |
| Connection pooling | Warm upstream connections; size to backend concurrency, not to "max" |
| Buffering on | Protects backend from slow clients (slowloris) |
| Buffering off | Required for SSE/LLM streaming and WebSocket |
| `client_max_body_size` | Defaults to 1 MB; the usual cause of surprise 413s |
| `proxy_read_timeout` | Defaults to 60 s; raise to 300 s for LLM, 3600 s for WebSocket |
| WebSocket upgrade | Forward `Upgrade`+`Connection`; no buffering; long read timeout; app-level pings |
| gRPC | Needs HTTP/2 end-to-end, trailers, no buffering, L7 (per-request) balancing |
| `CONNECT` | Forward-proxy tunnelling method for HTTPS and arbitrary TCP |
| Transparent proxy | Interception without client config; TLS-blind without MITM |
| Sidecar | Per-pod Envoy; mTLS, retries, policy; costs latency, RAM, complexity |
| 502 | Bad Gateway — no valid upstream response (down, refused, closed, huge headers) |
| 504 | Gateway Timeout — connected but no timely response (slow dependency, saturation) |
| Timeout ladder | 20 s client > 15 s LB > 12 s proxy > 10 s app > 8 s DB > 3 s lock |
| Slowloris | Partial requests exhausting workers; defeated by proxy buffering + short timeouts |
| `X-Forwarded-For` | Client-supplied and spoofable; strip at the edge, read from the right |
| `X-Forwarded-Proto` | Prevents redirect loops behind TLS-terminating proxies |
| Host header attack | Password-reset poisoning and cache poisoning; allow-list `Host` at the proxy |
| Hop-by-hop headers | Strip `Connection`, `Keep-Alive`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade` |
| PROXY protocol | Pass the real client IP through an L4 hop to an L7 proxy |
| Retry budget | Cap retries (~20%) so retries cannot amplify an outage |
| Nginx vs HAProxy vs Envoy | Web front door vs raw L4/L7 throughput vs dynamic mesh traffic policy |

**"If the interviewer says X → answer Y"**

| Interviewer says | You say |
|---|---|
| "Terminate TLS at the edge" | "One cert, lower handshake RTT, but re-encrypt if compliance requires in-network encryption" |
| "WebSocket chat behind a proxy" | "Dedicated HTTP/1.1 hostname, buffering off, 3600 s read timeout, app pings, connection registry" |
| "LLM tokens stalling" | "Buffering off at the proxy, compression off, `X-Accel-Buffering: no`, LB idle timeout raised" |
| "504s under load" | "Find the slow dependency; check thread-pool saturation; verify the timeout ladder decreases inward" |
| "502s intermittently" | "Keep-alive timeout mismatch first; then backend health, header size, connection close mid-request" |
| "Rate limit by IP" | "Strip client-supplied XFF at the edge; key on the rightmost trusted hop or the authenticated identity" |
| "Should we add a service mesh?" | "Only with many services, multiple languages and an mTLS requirement — otherwise a library" |
| "Deploys drop requests" | "Connection draining: stop accepting, finish in-flight, bound with a shutdown timeout, deregistration delay down" |

---

## 12. Cross-links

- [DNS Deep Dive](01-dns-deep-dive.md) — resolving the proxy's hostname; TTL effects on failover
- [CDN Deep Dive](02-cdn-deep-dive.md) — a CDN edge is a caching reverse proxy
- [API Gateway and Rate Limiting](04-api-gateway-and-rate-limiting.md) — a reverse proxy plus auth, quotas and transformation
- [Load Balancers](../03-scaling/01-load-balancers.md) — L4 vs L7, algorithms, health checks, draining
- [Caching Strategies and Layers](../03-scaling/02-caching-strategies-and-layers.md) — the reverse-proxy cache layer
- [Redis Deep Dive](../03-scaling/03-redis-deep-dive.md) — the shared connection registry a WebSocket fleet needs
- [Blob and Object Storage](../04-data/04-blob-and-object-storage.md) — large uploads and `proxy_request_buffering off`
- [WhatsApp Chat System](../../Case-Studies/HLD/02-whatsapp-chat-system.md) — WebSocket proxying, presence, connection registry
- [Networking Cheatsheet](../../Cheat-Sheets/02-networking-cheatsheet.md)
