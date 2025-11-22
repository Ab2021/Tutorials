# Lab 15: Security Hardening

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
- Enable SSL/TLS encryption
- Configure SASL authentication
- Implement ACLs

## Problem Statement
Secure your Kafka cluster by enabling SSL encryption and SASL/SCRAM authentication. Create ACLs to restrict topic access to specific users.

## Starter Code
```properties
# server.properties
listeners=SASL_SSL://localhost:9093
security.inter.broker.protocol=SASL_SSL
sasl.mechanism.inter.broker.protocol=SCRAM-SHA-512
```

## Hints
<details>
<summary>Hint 1</summary>
Generate SSL certificates using `keytool`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**Step 1: Generate SSL Certificates**
```bash
# Create CA
openssl req -new -x509 -keyout ca-key -out ca-cert -days 365

# Create broker keystore
keytool -keystore kafka.server.keystore.jks -alias localhost \
  -validity 365 -genkey -keyalg RSA

# Sign certificate
keytool -keystore kafka.server.keystore.jks -alias localhost \
  -certreq -file cert-file
openssl x509 -req -CA ca-cert -CAkey ca-key -in cert-file \
  -out cert-signed -days 365 -CAcreateserial

# Import signed cert
keytool -keystore kafka.server.keystore.jks -alias CARoot -import -file ca-cert
keytool -keystore kafka.server.keystore.jks -alias localhost -import -file cert-signed

# Create truststore
keytool -keystore kafka.server.truststore.jks -alias CARoot -import -file ca-cert
```

**Step 2: Configure Broker**
```properties
# server.properties
listeners=SASL_SSL://0.0.0.0:9093
advertised.listeners=SASL_SSL://localhost:9093
security.inter.broker.protocol=SASL_SSL

# SSL
ssl.keystore.location=/var/ssl/kafka.server.keystore.jks
ssl.keystore.password=password
ssl.key.password=password
ssl.truststore.location=/var/ssl/kafka.server.truststore.jks
ssl.truststore.password=password

# SASL
sasl.mechanism.inter.broker.protocol=SCRAM-SHA-512
sasl.enabled.mechanisms=SCRAM-SHA-512
```

**Step 3: Create Users**
```bash
kafka-configs --zookeeper localhost:2181 --alter \
  --add-config 'SCRAM-SHA-512=[password=alice-secret]' \
  --entity-type users --entity-name alice
```

**Step 4: Configure ACLs**
```bash
# Allow alice to produce to topic 'secure-topic'
kafka-acls --bootstrap-server localhost:9093 \
  --add --allow-principal User:alice \
  --operation Write --topic secure-topic \
  --command-config client-ssl.properties

# Allow alice to consume from 'secure-topic'
kafka-acls --bootstrap-server localhost:9093 \
  --add --allow-principal User:alice \
  --operation Read --topic secure-topic \
  --group alice-group \
  --command-config client-ssl.properties
```

**Step 5: Client Configuration**
```properties
# client-ssl.properties
security.protocol=SASL_SSL
sasl.mechanism=SCRAM-SHA-512
sasl.jaas.config=org.apache.kafka.common.security.scram.ScramLoginModule required \
  username="alice" \
  password="alice-secret";

ssl.truststore.location=/var/ssl/kafka.client.truststore.jks
ssl.truststore.password=password
```

**Verification:**
```bash
# Produce (should succeed)
kafka-console-producer --bootstrap-server localhost:9093 \
  --topic secure-topic \
  --producer.config client-ssl.properties

# Produce to unauthorized topic (should fail)
kafka-console-producer --bootstrap-server localhost:9093 \
  --topic other-topic \
  --producer.config client-ssl.properties
# Error: Not authorized to access topics: [other-topic]
```
</details>
