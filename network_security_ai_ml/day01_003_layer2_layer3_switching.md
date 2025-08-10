# Day 1.3: Layer 2 vs Layer 3 Switching - VLANs and Routing Protocols

## 🎯 Learning Objectives
By the end of this section, you will understand:
- Fundamental differences between Layer 2 and Layer 3 switching
- VLAN implementation and security in AI/ML environments
- Routing protocols (OSPF, BGP) for GPU cluster networks
- Performance and security trade-offs between switching layers

---

## 📚 Theoretical Foundation

### 1. Introduction to Switching Layers in AI/ML Networks

In AI/ML environments, the choice between Layer 2 and Layer 3 switching significantly impacts:

1. **Broadcast Domain Management**: Large GPU clusters can generate significant broadcast traffic
2. **Security Segmentation**: Isolating different ML workloads and environments
3. **Performance Optimization**: Reducing latency for time-sensitive AI operations
4. **Scalability**: Supporting growth from small research clusters to production-scale systems
5. **Multi-tenancy**: Supporting multiple teams, projects, or customers on shared infrastructure

### 2. Layer 2 Switching Fundamentals

#### 2.1 Basic Operation Principles

**MAC Address Learning**:
Layer 2 switches build and maintain MAC address tables to forward frames efficiently.

**Forwarding Decision Process**:
1. **Unicast**: Forward to specific port based on destination MAC
2. **Broadcast**: Forward to all ports in same VLAN
3. **Multicast**: Forward to registered multicast listeners
4. **Unknown Unicast**: Flood to all ports (learning opportunity)

**Bridge Protocol Data Unit (BPDU)**:
Used by Spanning Tree Protocol (STP) to prevent loops and build loop-free topology.

#### 2.2 AI/ML Specific Considerations

**GPU-to-GPU Communication**:
Many AI frameworks rely on Layer 2 adjacency for optimal performance:
- **NCCL (NVIDIA Collective Communications Library)**: Leverages Ethernet for multi-GPU communication
- **MPI (Message Passing Interface)**: Often optimized for Layer 2 networks
- **RDMA over Converged Ethernet (RoCE)**: Requires Layer 2 adjacency

**Broadcast Storm Implications**:
Large AI clusters can suffer from:
- **ARP Broadcasting**: Address resolution for numerous GPU nodes
- **DHCP Discovery**: Dynamic IP assignment in elastic clusters
- **Service Discovery**: Applications broadcasting for service location

**Example Scenario**:
A 512-GPU cluster generating ARP traffic:
```
ARP Rate = 512 nodes × 1 ARP/minute = 512 ARP packets/minute
In large broadcast domain: All nodes process every ARP packet
CPU overhead: ~0.1ms per ARP × 512 = 51.2ms/minute CPU time lost
```

#### 2.3 Layer 2 Performance Characteristics

**Latency Benefits**:
- No IP routing lookup delay
- Direct MAC-based forwarding
- Hardware-based switching in ASICs

**Bandwidth Utilization**:
- Full line-rate forwarding possible
- No IP header processing overhead
- Optimal for east-west traffic patterns

**Scalability Limitations**:
- MAC address table size limits (typically 8K-64K entries)
- Broadcast domain size impacts performance
- STP convergence time increases with network size

### 3. Layer 3 Switching and Routing

#### 3.1 Routing Fundamentals

**IP Routing Process**:
1. **Destination Lookup**: Examine destination IP address
2. **Routing Table Consultation**: Find best matching route
3. **Next Hop Determination**: Identify next router in path
4. **TTL Decrement**: Decrease Time To Live by 1
5. **Forward Packet**: Send to appropriate interface

**Routing Table Structure**:
```
Destination Network | Subnet Mask | Next Hop | Interface | Metric
192.168.1.0        | /24         | Direct   | eth0      | 0
10.0.0.0           | /8          | 10.1.1.1 | eth1      | 10
0.0.0.0            | /0          | 10.1.1.1 | eth1      | 20
```

#### 3.2 AI/ML Network Routing Requirements

**Multi-Path Load Balancing**:
Equal-Cost Multi-Path (ECMP) routing distributes traffic across multiple paths:
- **Hash-based Distribution**: Based on source/destination IP, port numbers
- **Per-Flow Consistency**: Ensure packets in same flow take same path
- **Bandwidth Aggregation**: Combine multiple links for higher throughput

**Quality of Service (QoS)**:
Differentiated Services Code Point (DSCP) marking for AI traffic:
- **Real-time Inference**: High priority (DSCP 46)
- **Distributed Training**: Medium priority (DSCP 26)
- **Data Movement**: Best effort (DSCP 0)
- **Management Traffic**: Low priority (DSCP 8)

**Traffic Engineering**:
Advanced routing for optimal path selection:
- **Constraint-based Routing**: Consider bandwidth, latency constraints
- **Traffic Matrix Optimization**: Route based on expected traffic patterns
- **Failure Recovery**: Fast reroute mechanisms for high availability

### 4. Virtual Local Area Networks (VLANs)

#### 4.1 VLAN Fundamentals

**VLAN Tagging (802.1Q)**:
```
Ethernet Frame with VLAN Tag:
[Destination MAC][Source MAC][VLAN Tag][EtherType][Payload][FCS]

VLAN Tag Structure:
[TPID: 0x8100][PCP: 3 bits][DEI: 1 bit][VID: 12 bits]

Where:
- TPID: Tag Protocol Identifier
- PCP: Priority Code Point (QoS)
- DEI: Drop Eligible Indicator
- VID: VLAN Identifier (1-4094)
```

**VLAN Types**:
1. **Access VLAN**: Untagged traffic from end devices
2. **Trunk VLAN**: Tagged traffic between switches
3. **Native VLAN**: Untagged traffic on trunk ports
4. **Management VLAN**: Network device management traffic

#### 4.2 AI/ML VLAN Design Patterns

**Workload-Based Segmentation**:
```
VLAN 100: Production Training Cluster
VLAN 200: Development/Testing Environment  
VLAN 300: Inference Serving Infrastructure
VLAN 400: Data Storage and ETL Systems
VLAN 500: Management and Monitoring
VLAN 999: Out-of-Band Management
```

**Security Zone Implementation**:
- **DMZ VLAN**: External-facing inference APIs
- **Internal VLAN**: Protected training environments
- **Management VLAN**: Administrative access only
- **Guest VLAN**: Limited access for visitors/contractors

**Multi-Tenant Isolation**:
```
Organization A: VLANs 100-199
Organization B: VLANs 200-299
Organization C: VLANs 300-399
Shared Services: VLANs 900-999
```

#### 4.3 VLAN Security Features

**Private VLANs (PVLANs)**:
Subdivide VLANs for additional isolation:
- **Primary VLAN**: Contains all secondary VLANs
- **Community VLAN**: Members can communicate with each other
- **Isolated VLAN**: Members cannot communicate with each other
- **Promiscuous Ports**: Can communicate with all VLANs

**VLAN Access Control Lists (VACLs)**:
Layer 2 filtering within VLANs:
```
VACL Example:
permit ip any host 192.168.1.100  # Allow access to ML model server
deny ip any 192.168.1.0/24        # Deny access to other training nodes
permit ip any any                  # Allow all other traffic
```

**Port Security**:
Limit MAC addresses per switch port:
- **Maximum MAC addresses**: Prevent MAC flooding attacks
- **Sticky MAC learning**: Remember authorized devices
- **Violation actions**: Shutdown, restrict, or protect port

#### 4.4 VLAN Performance Considerations

**VLAN Scaling Limits**:
- **802.1Q Standard**: 4094 usable VLAN IDs (1-4094, excluding 0 and 4095)
- **Switch VLAN Database**: Typically supports all 4094 VLANs
- **Active VLAN Limit**: May be restricted by hardware resources

**Inter-VLAN Routing Performance**:
- **Hardware-based Routing**: Line-rate performance in modern switches
- **Software-based Routing**: May introduce latency bottlenecks
- **VLAN Interface Limits**: Number of Layer 3 VLAN interfaces

**Broadcast Domain Optimization**:
Smaller VLANs reduce broadcast traffic but increase management complexity.

### 5. Routing Protocols for AI/ML Networks

#### 5.1 Open Shortest Path First (OSPF)

**OSPF Fundamentals**:
Link-state routing protocol that builds complete network topology.

**OSPF Areas**:
Hierarchical design for scalability:
- **Area 0 (Backbone)**: Central area connecting all other areas
- **Standard Areas**: Regular OSPF areas
- **Stub Areas**: Don't receive external routes
- **Totally Stubby Areas**: Only receive default route
- **NSSA**: Not-So-Stubby Areas with limited external routes

**AI/ML OSPF Design**:
```
Area 0 (Backbone): Core/Spine switches
Area 1: Training Cluster 1 (GPUs 1-128)
Area 2: Training Cluster 2 (GPUs 129-256)  
Area 3: Inference Serving Farm
Area 4: Storage and Data Systems
```

**OSPF Metrics and Path Selection**:
```
OSPF Cost = Reference Bandwidth / Interface Bandwidth

Example Costs:
10 Gigabit Ethernet: 100,000,000 / 10,000,000,000 = 1
1 Gigabit Ethernet: 100,000,000 / 1,000,000,000 = 10
100 Megabit Ethernet: 100,000,000 / 100,000,000 = 100
```

**OSPF Convergence Optimization**:
- **Hello Timers**: Faster neighbor detection (1 second instead of 10)
- **Dead Timers**: Quicker failure detection (4 seconds instead of 40)
- **LSA Flooding**: Efficient topology update propagation
- **SPF Throttling**: Control CPU usage during convergence

#### 5.2 Border Gateway Protocol (BGP)

**BGP Overview**:
Path vector protocol designed for inter-autonomous system routing.

**BGP in Data Center Networks**:
- **eBGP**: Between different autonomous systems
- **iBGP**: Within same autonomous system
- **BGP Confederations**: Hierarchical iBGP design
- **Route Reflectors**: Reduce iBGP mesh complexity

**BGP Attributes for Traffic Engineering**:
1. **AS_PATH**: Autonomous system path length
2. **LOCAL_PREF**: Local preference (higher is better)
3. **MED**: Multi-Exit Discriminator (lower is better)
4. **COMMUNITY**: Tag routes for policy application

**AI/ML BGP Use Cases**:

**Multi-Homed Data Centers**:
```
Primary ISP: AS 65001
Secondary ISP: AS 65002
Internal AS: AS 65100

BGP Policy:
- Prefer primary ISP for outbound traffic (higher LOCAL_PREF)
- Load balance inbound traffic using AS_PATH prepending
- Maintain backup paths for redundancy
```

**Cloud Connectivity**:
- **AWS Direct Connect**: BGP for hybrid cloud AI workloads
- **Azure ExpressRoute**: Private connectivity to cloud ML services
- **Google Cloud Interconnect**: Dedicated network connections

#### 5.3 Enhanced Interior Gateway Routing Protocol (EIGRP)

**EIGRP Characteristics**:
- **Hybrid Protocol**: Combines distance vector and link-state features
- **DUAL Algorithm**: Diffusing Update Algorithm for loop-free paths
- **Composite Metric**: Bandwidth, delay, reliability, load, MTU
- **Unequal Cost Load Balancing**: Traffic distribution across different cost paths

**EIGRP Metric Calculation**:
```
Metric = (K1 × Bandwidth + K2 × Bandwidth / (256 - Load) + K3 × Delay) × (K5 / (Reliability + K4))

Default K values: K1=1, K2=0, K3=1, K4=0, K5=0
Simplified: Metric = Bandwidth + Delay
```

**AI/ML EIGRP Applications**:
- **Campus Networks**: Connecting research labs and GPU clusters
- **Branch Connectivity**: Remote sites with limited IT staff
- **Legacy Integration**: Connecting to existing EIGRP networks

### 6. Advanced Switching Technologies

#### 6.1 Virtual Switching Technologies

**Virtual Extensible LAN (VXLAN)**:
Overlay network technology for cloud and virtualized environments.

**VXLAN Header Structure**:
```
[Outer MAC][Outer IP][UDP][VXLAN Header][Inner Ethernet Frame]

VXLAN Header:
[Flags: 8 bits][Reserved: 24 bits][VNI: 24 bits][Reserved: 8 bits]

Where VNI = VXLAN Network Identifier (16.7M possible networks)
```

**Network Virtualization using Generic Routing Encapsulation (NVGRE)**:
Microsoft's overlay network solution.

**Software-Defined Networking (SDN)**:
- **OpenFlow**: Protocol for centralized network control
- **Open vSwitch (OVS)**: Software-based virtual switching
- **P4**: Programming language for data plane customization

#### 6.2 Multi-Chassis Link Aggregation

**Virtual Port Channel (vPC)**:
Cisco technology for dual-homed devices.

**Multi-Chassis Trunk (MCT)**:
Avaya's implementation of multi-chassis LAG.

**Virtual Router Redundancy Protocol (VRRP)**:
Provides gateway redundancy for Layer 3 networks.

### 7. Performance Analysis and Optimization

#### 7.1 Switching Performance Metrics

**Forwarding Rate**:
Measured in packets per second (pps) or frames per second (fps).

**Latency Measurements**:
- **Store-and-Forward**: Entire frame received before forwarding
- **Cut-Through**: Forwarding begins after destination address received
- **Fragment-Free**: Forward after first 64 bytes received

**Buffer Management**:
- **Shared Buffers**: Dynamic allocation across ports
- **Dedicated Buffers**: Fixed allocation per port
- **Priority Queuing**: Separate buffers for different traffic classes

#### 7.2 Layer 2 vs Layer 3 Performance Trade-offs

**Layer 2 Advantages**:
- Lower latency (no IP lookup)
- Higher throughput (simpler processing)
- Better for local communication

**Layer 3 Advantages**:
- Better scalability (routing hierarchy)
- Enhanced security (routing policies)
- Support for multiple subnets

**Hybrid Approaches**:
- **Layer 3 to the access**: IP routing down to individual servers
- **Layer 2 for GPU clusters**: Direct Layer 2 connectivity within clusters
- **Overlay networks**: Layer 3 underlay with Layer 2 overlay for applications

### 8. Security Implementation

#### 8.1 Layer 2 Security Features

**MAC Address Security**:
- **Port Security**: Limit MAC addresses per port
- **DHCP Snooping**: Prevent rogue DHCP servers
- **Dynamic ARP Inspection**: Validate ARP packets
- **IP Source Guard**: Bind IP addresses to MAC addresses

**Storm Control**:
Rate limiting for broadcast, multicast, and unknown unicast traffic.

**Root Guard and BPDU Guard**:
Prevent unauthorized changes to spanning tree topology.

#### 8.2 Layer 3 Security Features

**Access Control Lists (ACLs)**:
```
Extended ACL Example:
access-list 100 permit tcp 192.168.1.0 0.0.0.255 host 10.1.1.100 eq 443
access-list 100 permit tcp 192.168.1.0 0.0.0.255 host 10.1.1.100 eq 80
access-list 100 deny ip any any log
```

**Unicast Reverse Path Forwarding (uRPF)**:
Prevents IP spoofing by verifying source addresses.

**Control Plane Protection**:
Rate limiting and filtering for routing protocol traffic.

#### 8.3 AI/ML Specific Security Considerations

**Model Protection**:
- Network segmentation between training and inference
- Encrypted communication for model updates
- Access control for model repositories

**Data Protection**:
- Isolation of sensitive training datasets
- Network monitoring for data exfiltration
- Compliance with data protection regulations

**Infrastructure Protection**:
- Secure management of GPU clusters
- Monitoring for unauthorized access
- Protection against DDoS attacks on inference endpoints

---

## 🔍 Key Questions

### Beginner Level

1. **Q**: What is the main difference between a Layer 2 switch and a Layer 3 switch?
   **A**: Layer 2 switches forward frames based on MAC addresses within the same network segment, while Layer 3 switches route packets between different network segments based on IP addresses.

2. **Q**: Why might you choose Layer 2 switching for a GPU cluster?
   **A**: Layer 2 switching provides lower latency, higher throughput, and is often required for technologies like RDMA and certain MPI implementations that need Layer 2 adjacency.

3. **Q**: What is the purpose of VLANs in an AI/ML environment?
   **A**: VLANs provide logical segmentation to isolate different workloads (training vs. inference), environments (production vs. development), and tenants while sharing the same physical infrastructure.

### Intermediate Level

4. **Q**: Calculate the maximum number of nodes that can be supported in a single VLAN before broadcast traffic becomes problematic. Assume each node generates 1 ARP request per minute and 10ms processing overhead per ARP is acceptable.
   **A**: 
   ```
   Acceptable overhead: 10ms per minute = 600ms per hour
   Processing time per ARP: 0.1ms
   Maximum ARPs per hour: 600ms / 0.1ms = 6000
   Maximum nodes: 6000 (assuming each node ARPs once per minute)
   
   However, this is simplistic. Real networks have ARP caching, so practical limits are higher.
   ```

5. **Q**: How does ECMP routing benefit distributed AI training, and what are potential drawbacks?
   **A**: Benefits: Load distribution across multiple paths, higher aggregate bandwidth, automatic failover. Drawbacks: Potential for flow reordering, uneven load distribution with hash-based algorithms, complexity in troubleshooting.

6. **Q**: Design a VLAN scheme for a multi-tenant AI research facility with 3 organizations, each needing production and development environments.
   **A**:
   ```
   Org A Production: VLAN 100
   Org A Development: VLAN 101
   Org B Production: VLAN 200  
   Org B Development: VLAN 201
   Org C Production: VLAN 300
   Org C Development: VLAN 301
   Shared Services: VLAN 900 (DNS, NTP, etc.)
   Management: VLAN 999
   ```

### Advanced Level

7. **Q**: Explain how you would implement a zero-trust network model using Layer 2 and Layer 3 technologies in an AI training environment.
   **A**: 
   - **Layer 2**: Private VLANs for node isolation, 802.1X authentication, MAC address whitelisting
   - **Layer 3**: Micro-segmentation with ACLs, application-aware firewalling, encrypted tunnels between segments
   - **Continuous verification**: Dynamic VLAN assignment based on device trust level, real-time monitoring and policy enforcement

8. **Q**: How would you troubleshoot a scenario where some nodes in a distributed training job experience 10x higher latency than others, despite being on the same VLAN?
   **A**: 
   - Check spanning tree topology for suboptimal paths
   - Verify ECMP hash distribution isn't causing imbalanced flows
   - Analyze switch buffer utilization and congestion
   - Check for hardware issues (bad cables, transceivers)
   - Verify consistent switch configuration across infrastructure
   - Monitor for broadcast storms or Layer 2 loops

### Tricky Questions

9. **Q**: In a spine-leaf network using BGP for routing, you observe that traffic between certain leaf switches is taking suboptimal paths through multiple spines. The BGP tables show equal-cost paths available. What could cause this behavior and how would you resolve it?
   **A**: Possible causes:
   - **BGP route selection**: Different BGP attributes causing path preference
   - **ECMP hashing**: Hash algorithm not distributing flows evenly
   - **BGP timer misalignment**: Inconsistent convergence across switches
   - **Hardware ECMP limitations**: Switch unable to load balance properly
   
   Resolution: Verify BGP attributes consistency, adjust ECMP hash parameters, check hardware ECMP capabilities, implement traffic engineering with BGP communities.

10. **Q**: You need to migrate a large AI training cluster from Layer 2 to Layer 3 without disrupting ongoing training jobs that may run for weeks. Describe a migration strategy.
    **A**: 
    - **Phase 1**: Deploy Layer 3 infrastructure in parallel, maintain Layer 2 for existing jobs
    - **Phase 2**: Implement VXLAN overlay to provide Layer 2 services over Layer 3 underlay
    - **Phase 3**: Gradually migrate applications to use Layer 3 directly during maintenance windows
    - **Phase 4**: Use blue-green deployment for new training jobs on Layer 3 infrastructure
    - **Phase 5**: Decommission Layer 2 infrastructure after all jobs complete migration

---

## 🛡️ Security Deep Dive

### Layer 2 Security Threats and Mitigations

#### Common Layer 2 Attacks

**VLAN Hopping**:
- **Switch Spoofing**: Attacker configures device to act as switch
- **Double Tagging**: Exploiting native VLAN configuration
- **Mitigation**: Disable DTP, configure trunk ports explicitly, use dedicated VLAN for trunks

**MAC Flooding**:
- **Attack**: Overwhelm switch MAC table with fake entries
- **Result**: Switch fails open and floods all traffic
- **Mitigation**: Port security, MAC address limits, storm control

**ARP Spoofing/Poisoning**:
- **Attack**: Send fake ARP responses to redirect traffic
- **Result**: Man-in-the-middle attacks, traffic interception
- **Mitigation**: Dynamic ARP Inspection, DHCP snooping, static ARP entries

#### Layer 2 Security Best Practices

**Access Port Configuration**:
```
interface FastEthernet0/1
 switchport mode access
 switchport access vlan 100
 switchport port-security
 switchport port-security maximum 2
 switchport port-security violation shutdown
 switchport port-security mac-address sticky
 spanning-tree portfast
 spanning-tree bpduguard enable
```

**Trunk Port Security**:
```
interface GigabitEthernet0/1
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 100,200,300
 switchport trunk native vlan 999
 spanning-tree guard root
```

### Layer 3 Security Implementation

#### Routing Protocol Security

**OSPF Authentication**:
```
router ospf 1
 area 0 authentication message-digest
 
interface FastEthernet0/0
 ip ospf message-digest-key 1 md5 SecurePassword123
```

**BGP Security**:
```
router bgp 65001
 neighbor 10.1.1.2 password SecureBGPPassword
 neighbor 10.1.1.2 ttl-security hops 1
 bgp log-neighbor-changes
```

#### Access Control Implementation

**Standard ACL Example**:
```
access-list 10 permit 192.168.1.0 0.0.0.255
access-list 10 deny any log

interface FastEthernet0/0
 ip access-group 10 in
```

**Extended ACL for AI/ML Traffic**:
```
ip access-list extended AI_TRAINING_ACL
 permit tcp 192.168.100.0 0.0.0.255 192.168.100.0 0.0.0.255 eq 22
 permit tcp 192.168.100.0 0.0.0.255 192.168.100.0 0.0.0.255 range 8000 8100
 permit udp 192.168.100.0 0.0.0.255 192.168.100.0 0.0.0.255 gt 1024
 deny ip any any log
```

---

## 🚀 Performance Optimization

### Layer 2 Performance Tuning

#### Spanning Tree Optimization

**Rapid Spanning Tree Protocol (RSTP)**:
```
spanning-tree mode rapid-pvst
spanning-tree vlan 1-4094 priority 4096
```

**Multiple Spanning Tree (MST)**:
```
spanning-tree mst configuration
 name REGION1
 instance 1 vlan 1-100
 instance 2 vlan 101-200
spanning-tree mst 1 priority 4096
spanning-tree mst 2 priority 8192
```

#### Switch Buffer Tuning

**Buffer Allocation**:
- Monitor buffer utilization during peak traffic
- Adjust buffer allocation based on traffic patterns
- Consider different buffer strategies for different port types

### Layer 3 Performance Optimization

#### OSPF Tuning for Large Networks

**Area Design**:
- Limit LSA flooding with proper area boundaries
- Use stub areas to reduce routing table size
- Implement area filtering to control route advertisement

**Timer Optimization**:
```
interface FastEthernet0/0
 ip ospf hello-interval 1
 ip ospf dead-interval 4
 ip ospf retransmit-interval 1
```

#### BGP Performance Tuning

**Route Filtering**:
```
ip prefix-list TRAINING_NETWORKS permit 192.168.0.0/16 le 24
ip prefix-list INFERENCE_NETWORKS permit 10.0.0.0/8 le 24

router bgp 65001
 neighbor 10.1.1.2 prefix-list TRAINING_NETWORKS out
 neighbor 10.1.1.3 prefix-list INFERENCE_NETWORKS out
```

**Route Aggregation**:
```
router bgp 65001
 aggregate-address 192.168.0.0 255.255.0.0 summary-only
```

---

## 📝 Practical Exercises

### Exercise 1: VLAN Design and Implementation
Design a comprehensive VLAN scheme for a 500-node AI research facility that includes:
- 3 different research groups
- Production and development environments
- Shared storage and services
- Management and monitoring infrastructure
- Guest and visitor access

Include security considerations and inter-VLAN routing requirements.

### Exercise 2: Routing Protocol Selection
Compare OSPF and BGP for the following scenarios:
1. Single data center with 200 GPU nodes
2. Multi-site deployment across 3 locations
3. Hybrid cloud environment with on-premises and cloud resources

Justify your choice based on scalability, convergence time, and management complexity.

### Exercise 3: Performance Troubleshooting
Given the following symptoms in an AI training cluster:
- Random training job failures
- Inconsistent gradient synchronization times
- High CPU utilization on some switches
- Occasional broadcast storms

Develop a systematic troubleshooting approach using Layer 2 and Layer 3 diagnostic tools.

---

## 🔗 Next Steps
In the next section (day01_004), we'll explore data center fabric designs, focusing on leaf-spine architectures and multi-tier designs optimized for AI/ML workloads, including detailed analysis of oversubscription ratios, traffic patterns, and scaling considerations.