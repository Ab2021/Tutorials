# Lab 4.4: Network Troubleshooting

## Objective
Diagnose and resolve network issues.

## Learning Objectives
- Use network diagnostic tools
- Troubleshoot connectivity issues
- Analyze network traffic
- Resolve DNS problems

---

## Connectivity Testing

```bash
# Ping test
ping -c 4 google.com

# Traceroute
traceroute google.com
tracepath google.com

# Test specific port
telnet example.com 80
nc -zv example.com 80

# Check route
ip route show
route -n
```

## DNS Troubleshooting

```bash
# DNS lookup
nslookup example.com
dig example.com
host example.com

# Check DNS servers
cat /etc/resolv.conf

# Flush DNS cache
sudo systemd-resolve --flush-caches

# Test DNS resolution
dig @8.8.8.8 example.com
```

## Network Analysis

```bash
# View connections
netstat -tulpn
ss -tulpn

# Monitor traffic
sudo tcpdump -i eth0
sudo tcpdump -i eth0 port 80

# Analyze packets
sudo tcpdump -i eth0 -w capture.pcap
wireshark capture.pcap

# Check bandwidth
iftop
nethogs
```

## Common Issues

```bash
# Check firewall
sudo iptables -L
sudo ufw status

# Test HTTP
curl -v http://example.com
wget -O- http://example.com

# Check listening ports
sudo lsof -i :80
sudo netstat -tlnp | grep :80
```

## Success Criteria
✅ Connectivity tested  
✅ DNS issues resolved  
✅ Traffic analyzed  
✅ Problems diagnosed  

**Time:** 40 min
