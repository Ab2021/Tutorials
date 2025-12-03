# Day 86: BSM (Basic Safety Messages)
## Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication

---

> **📝 Day 86 Focus:**
> In V2X, we don't send JSON or XML. It's too big. We use **ASN.1 (Abstract Syntax Notation One)** to pack data into tiny binary packets. The most important packet is the **BSM (Basic Safety Message)**. It's the "Heartbeat" of the connected car.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Analyze** the structure of SAE J2735 BSM (Part I and Part II).
2.  **Explain** the role of ASN.1 and UPER (Unaligned Packed Encoding Rules).
3.  **Implement** a BSM Encoder/Decoder in Python using `asn1tools`.
4.  **Simulate** the transmission of vehicle kinematics (Lat, Lon, Speed, Heading, Accel).
5.  **Visualize** the decoded BSM data on a map.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 85:** V2X Basics.
-   **Binary Data:** Bits, Bytes, Endianness.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `asn1tools`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SAE J2735 Standard

This standard defines the "Dictionary" of V2X messages.
-   **BSM (Basic Safety Message):** Vehicle status (10Hz).
-   **SPAT (Signal Phase and Timing):** Traffic light status.
-   **MAP:** Intersection geometry.
-   **TIM (Traveler Information Message):** Road signs, warnings.

### 🔹 Part 2: Structure of a BSM

**Part I (Mandatory, Sent every 100ms):**
-   `msgID`: DSRC_Message_ID_BSM
-   `id`: TemporaryID (4 bytes, changes every 5 mins).
-   `secMark`: Time (milliseconds within the minute).
-   `pos`: Latitude, Longitude, Elevation.
-   `accuracy`: GNSS accuracy.
-   `transmission`: Gear (Park, Reverse, Drive).
-   `speed`: m/s.
-   `heading`: Degrees.
-   `angle`: Steering wheel angle.
-   `accelSet`: Long, Lat, Vert, YawRate.
-   `brakes`: Brake system status (ABS, Traction Control).
-   `size`: Vehicle Length/Width.

**Part II (Optional, Sent rarely or on event):**
-   `safetyExt`: ABS active, Airbag deployed, Wipers on.
-   `status`: Lights, Door status.

### 🔹 Part 3: ASN.1 UPER

JSON: `{"speed": 25.5}` (15 bytes).
UPER: `110011...` (2 bytes).
-   **Schema:** Defines the types and ranges.
-   **Encoding:** Removes all field names. Packs bits tightly (no padding).
-   **Result:** Extremely efficient bandwidth usage.

---

## 💻 Implementation: BSM Encoder/Decoder

**Scenario:**
-   **Task:** Define a simplified ASN.1 schema for BSM. Compile it. Encode a message. Decode it.

### 🛠️ Setup
Create `week13_day86` and `bsm_codec.py`.

```bash
mkdir -p ~/ros2_ws/src/week13_day86
cd ~/ros2_ws/src/week13_day86
touch bsm_codec.py
touch bsm.asn
```

### 👨‍💻 Code: ASN.1 Definition (`bsm.asn`)

```asn
BSM-Schema DEFINITIONS AUTOMATIC TAGS ::= BEGIN

BasicSafetyMessage ::= SEQUENCE {
    coreData    BSMcoreData,
    partII      SEQUENCE (SIZE(1..8)) OF PartIIcontent OPTIONAL
}

BSMcoreData ::= SEQUENCE {
    msgCnt      INTEGER (0..127),
    id          OCTET STRING (SIZE(4)),
    secMark     INTEGER (0..65535),
    lat         INTEGER (-900000000..900000001),
    long        INTEGER (-1800000000..1800000001),
    elev        INTEGER (-4096..61439),
    accuracy    INTEGER (0..255),
    transmission ENUMERATED {neutral, park, forwardGears, reverseGears, reserved1, reserved2, reserved3, unavailable},
    speed       INTEGER (0..8191),
    heading     INTEGER (0..28800),
    angle       INTEGER (-127..127),
    accelSet    AccelerationSet4Way,
    brakes      BrakeSystemStatus,
    size        VehicleSize
}

AccelerationSet4Way ::= SEQUENCE {
    long        INTEGER (-2000..2001),
    lat         INTEGER (-2000..2001),
    vert        INTEGER (-127..127),
    yaw         INTEGER (-32767..32767)
}

BrakeSystemStatus ::= BIT STRING {
    unavailable(0),
    leftFront(1),
    leftRear(2),
    rightFront(3),
    rightRear(4),
    scsActive(5),
    absActive(6)
} (SIZE(7))

VehicleSize ::= SEQUENCE {
    width       INTEGER (0..1023),
    length      INTEGER (0..4095)
}

PartIIcontent ::= SEQUENCE {
    partII-Id   INTEGER (0..63),
    value       OCTET STRING
}

END
```

### 👨‍💻 Code: Python Codec (`bsm_codec.py`)

```python
import asn1tools
import time
import binascii

# --- Load Schema ---
# In real life, use the full J2735.asn file
bsm_spec = asn1tools.compile_files('bsm.asn', codec='uper')

def encode_bsm(lat, lon, speed, heading):
    # Prepare Data Dictionary matching the ASN.1 structure
    
    # Scaling factors (J2735)
    # Lat/Lon: 1/10 micro degree
    # Speed: 0.02 m/s
    # Heading: 0.0125 degree
    
    data = {
        'coreData': {
            'msgCnt': 1,
            'id': b'\x01\x02\x03\x04', # Random 4 bytes
            'secMark': int((time.time() * 1000) % 60000), # ms in minute
            'lat': int(lat * 10000000),
            'long': int(lon * 10000000),
            'elev': 100,
            'accuracy': 0,
            'transmission': 'forwardGears',
            'speed': int(speed / 0.02),
            'heading': int(heading / 0.0125),
            'angle': 0,
            'accelSet': {
                'long': 0,
                'lat': 0,
                'vert': 0,
                'yaw': 0
            },
            'brakes': (b'\x00', 7), # 7 bits, all zero
            'size': {
                'width': 200, # 200cm
                'length': 500 # 500cm
            }
        }
    }
    
    encoded = bsm_spec.encode('BasicSafetyMessage', data)
    return encoded

def decode_bsm(encoded_data):
    decoded = bsm_spec.decode('BasicSafetyMessage', encoded_data)
    
    # Extract readable values
    core = decoded['coreData']
    readable = {
        'lat': core['lat'] / 10000000.0,
        'lon': core['long'] / 10000000.0,
        'speed': core['speed'] * 0.02,
        'heading': core['heading'] * 0.0125
    }
    return readable

def main():
    print("--- BSM Encoder/Decoder (J2735 UPER) ---")
    
    # 1. Simulate Vehicle Data
    my_lat = 37.4220
    my_lon = -122.0841
    my_speed = 25.5 # m/s
    my_heading = 90.0 # East
    
    print(f"Original: Lat={my_lat}, Lon={my_lon}, Spd={my_speed}, Hdg={my_heading}")
    
    # 2. Encode
    payload = encode_bsm(my_lat, my_lon, my_speed, my_heading)
    print(f"Encoded ({len(payload)} bytes): {binascii.hexlify(payload)}")
    
    # 3. Decode
    decoded = decode_bsm(payload)
    print(f"Decoded:  Lat={decoded['lat']}, Lon={decoded['lon']}, Spd={decoded['speed']}, Hdg={decoded['heading']}")
    
    # 4. Verify Size Efficiency
    # JSON equivalent size
    json_str = f'{{"lat":{my_lat},"lon":{my_lon},"speed":{my_speed},"heading":{my_heading}}}'
    print(f"JSON Size: {len(json_str)} bytes")
    print(f"UPER Size: {len(payload)} bytes")
    print(f"Savings: {(1 - len(payload)/len(json_str))*100:.1f}%")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Bit Shift

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   UPER payload is tiny (~20-30 bytes).
    -   JSON is ~80 bytes.
    -   Decoded values match original values (within quantization error).
3.  **Experiment:**
    -   Change `speed` to `8191 * 0.02` (Max speed ~163 m/s).
    -   Change `speed` to `8192 * 0.02`.
    -   **Result:** `asn1tools.EncodeError`. The value is out of range defined in ASN.1 `INTEGER (0..8191)`.
    -   **Insight:** ASN.1 enforces strict constraints, preventing buffer overflows and invalid data.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Endianness
**Symptom:** Decoded values are garbage (e.g., Lat = 1.5e9).
**Cause:** Network byte order (Big Endian) vs Host byte order (Little Endian).
**Solution:** ASN.1 tools usually handle this, but if doing manual bit packing, use `struct.pack('!...')`.

#### 2. Quantization Error
**Symptom:** Input 25.55, Output 25.54.
**Cause:** Speed resolution is 0.02 m/s.
**Solution:** This is expected. ADAS apps must handle this tolerance.

---

## ⚡ Optimization & Best Practices

### 1. Hardware Acceleration
Encoding/Decoding ASN.1 in Python is slow.
-   **C/C++:** Use `asn1c` to compile the schema into C code.
-   **HSM (Hardware Security Module):** Some V2X chips handle signing and encoding in hardware.

### 2. Congestion Control (DCC)
-   If channel busy ratio > 60%, drop `Part II` data (Optional fields).
-   Only send `coreData`.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is `secMark` (ms within minute) used instead of full UTC timestamp?
    *   **A:** To save bits. 2 bytes (0..60000) vs 8 bytes for full timestamp. The receiver knows the current minute.
2.  **Q:** What is the `TemporaryID`?
    *   **A:** A random 4-byte ID that changes every 5 minutes (or distance traveled) to prevent tracking the vehicle.
3.  **Q:** What happens if I send a value outside the ASN.1 range?
    *   **A:** Encoding fails. The standard ensures all devices speak the exact same language with valid ranges.

### Challenge Task
**Task:** Brake Status Bitmask.
1.  Modify the encoder to set `absActive` to 1.
2.  Decode the `brakes` BIT STRING.
3.  Verify that the 7th bit is 1.

---

## 📚 Further Reading & References
-   [SAE J2735 Standard (Paid)](https://www.sae.org/standards/content/j2735_201603/)
-   [Pycrate (ASN.1 Library)](https://github.com/P1sec/pycrate)

---

**Day 86 Complete** | Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication
