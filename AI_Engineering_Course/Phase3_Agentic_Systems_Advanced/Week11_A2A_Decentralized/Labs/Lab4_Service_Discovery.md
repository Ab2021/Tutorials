# Lab 4: Service Discovery (DHT)

## Objective
How do agents find each other without a central server?
**DHT (Distributed Hash Table)**.

## 1. The DHT (`dht.py`)

```python
nodes = {} # Mock Network

def register(service_name, address):
    # Hash the service name to find the node
    key = hash(service_name) % 10
    if key not in nodes:
        nodes[key] = []
    nodes[key].append(address)
    print(f"Registered {service_name} at Node {key}")

def find(service_name):
    key = hash(service_name) % 10
    return nodes.get(key, [])

# Usage
register("weather_service", "192.168.1.5")
register("stock_service", "192.168.1.6")

providers = find("weather_service")
print(f"Weather Providers: {providers}")
```

## 2. Analysis
Kademlia DHT is used in IPFS and BitTorrent.
Agents use it to advertise capabilities.

## 3. Submission
Submit the node index where "weather_service" was stored.
