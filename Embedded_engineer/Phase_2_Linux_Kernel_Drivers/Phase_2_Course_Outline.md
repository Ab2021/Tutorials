# Phase 2: Linux Kernel & Device Drivers
## Days 121-260 (Weeks 18-37)

---

## 📋 Phase Overview

**Duration:** 140 Days (20 Weeks)  
**Focus:** Linux kernel internals, device driver development, V4L2 subsystem, device tree, kernel debugging, and embedded Linux system development.

**Learning Objectives:**
- Master Linux kernel architecture and internals
- Develop character, block, and network device drivers
- Understand and work with V4L2 (Video for Linux 2) subsystem
- Configure device trees for hardware description
- Debug kernel-level code effectively
- Build and customize embedded Linux systems
- Implement power management in Linux
- Work with kernel synchronization primitives

---

## 🗓️ Week-by-Week Breakdown

### **Week 18: Linux Kernel Fundamentals** (Days 121-127)

#### **Day 121: Introduction to Linux Kernel**
**Topics:**
- Linux kernel architecture overview
- Kernel vs user space
- Monolithic vs microkernel
- Kernel source tree organization

**Sections:**
1. Linux Kernel History and Evolution
2. Kernel Architecture Components
3. System Call Interface
4. Kernel Source Code Structure

**Labs:**
- Lab 121.1: Kernel source download and exploration
- Lab 121.2: Building your first kernel
- Lab 121.3: Kernel version identification

---

#### **Day 122: Kernel Build System**
**Topics:**
- Kbuild system
- Kernel configuration (menuconfig)
- Cross-compilation
- Kernel modules

**Sections:**
1. Kconfig and Kbuild
2. Configuration Options
3. Cross-Compilation Setup
4. Module Compilation

**Labs:**
- Lab 122.1: Kernel configuration
- Lab 122.2: Cross-compilation for ARM
- Lab 122.3: Custom kernel build

---

#### **Day 123: Kernel Boot Process**
**Topics:**
- Bootloader to kernel handoff
- Kernel decompression
- Early initialization
- Init process

**Sections:**
1. Boot Sequence
2. Kernel Parameters
3. Initramfs/Initrd
4. Init Systems (systemd, init)

**Labs:**
- Lab 123.1: Analyzing boot logs
- Lab 123.2: Custom kernel parameters
- Lab 123.3: Initramfs creation

---

#### **Day 124: Kernel Memory Management**
**Topics:**
- Virtual memory
- Page tables
- Memory zones
- Slab allocator

**Sections:**
1. Virtual Memory Concept
2. Memory Management Unit (MMU)
3. Kernel Memory Allocation
4. Memory Zones (DMA, Normal, High)

**Labs:**
- Lab 124.1: Memory map analysis
- Lab 124.2: kmalloc vs vmalloc
- Lab 124.3: Memory debugging

---

#### **Day 125: Process Management**
**Topics:**
- Process creation
- Process scheduling
- Context switching
- Process states

**Sections:**
1. Task Structure (task_struct)
2. Process Creation (fork, exec)
3. Scheduler Classes
4. Process Lifecycle

**Labs:**
- Lab 125.1: Process inspection
- Lab 125.2: Scheduler analysis
- Lab 125.3: Process monitoring

---

#### **Day 126: Kernel Synchronization Primitives**
**Topics:**
- Spinlocks
- Mutexes
- Semaphores
- Atomic operations

**Sections:**
1. Concurrency in Kernel
2. Spinlock Usage
3. Mutex vs Semaphore
4. Atomic Variables

**Labs:**
- Lab 126.1: Spinlock implementation
- Lab 126.2: Mutex usage
- Lab 126.3: Race condition debugging

---

#### **Day 127: Week 18 Review and Mini-Project**
**Topics:**
- Kernel fundamentals review
- Kernel module basics

**Sections:**
1. Week Recap
2. Module Programming Introduction
3. Best Practices

**Labs:**
- Lab 127.1: First kernel module (Hello World)
- Lab 127.2: Module parameters
- Lab 127.3: Module dependencies

---

### **Week 19: Character Device Drivers** (Days 128-134)

#### **Day 128: Character Driver Basics**
**Topics:**
- Character device concept
- Major and minor numbers
- File operations structure
- Device registration

**Sections:**
1. Character Device Overview
2. Device Numbers
3. cdev Structure
4. Registration/Unregistration

**Labs:**
- Lab 128.1: Simple character driver
- Lab 128.2: Device number allocation
- Lab 128.3: Device file creation

---

#### **Day 129: File Operations**
**Topics:**
- open/release operations
- read/write operations
- ioctl implementation
- seek and other operations

**Sections:**
1. File Operations Structure
2. Read/Write Implementation
3. ioctl Design
4. File Position Management

**Labs:**
- Lab 129.1: Read/write implementation
- Lab 129.2: ioctl commands
- Lab 129.3: User-space testing

---

#### **Day 130: Memory Management in Drivers**
**Topics:**
- Kernel memory allocation
- DMA memory
- Memory mapping (mmap)
- Cache coherency

**Sections:**
1. kmalloc/kfree
2. DMA API
3. mmap Implementation
4. Cache Management

**Labs:**
- Lab 130.1: Memory allocation in driver
- Lab 130.2: DMA buffer allocation
- Lab 130.3: mmap implementation

---

#### **Day 131: Interrupt Handling in Drivers**
**Topics:**
- Requesting IRQs
- Interrupt handlers
- Tasklets and workqueues
- Threaded interrupts

**Sections:**
1. IRQ Registration
2. Top Half/Bottom Half
3. Deferred Work Mechanisms
4. Threaded IRQ Handlers

**Labs:**
- Lab 131.1: GPIO interrupt driver
- Lab 131.2: Tasklet usage
- Lab 131.3: Workqueue implementation

---

#### **Day 132: Waiting and Completion**
**Topics:**
- Wait queues
- Completion API
- Blocking I/O
- Poll/select implementation

**Sections:**
1. Wait Queue Mechanism
2. Completion Usage
3. Blocking Operations
4. Poll Implementation

**Labs:**
- Lab 132.1: Wait queue usage
- Lab 132.2: Completion API
- Lab 132.3: Poll-based driver

---

#### **Day 133: Debugging Character Drivers**
**Topics:**
- printk and log levels
- Dynamic debug
- Debugfs
- Kernel oops analysis

**Sections:**
1. Kernel Logging
2. Debug Techniques
3. Debugfs Usage
4. Crash Analysis

**Labs:**
- Lab 133.1: Debugging with printk
- Lab 133.2: Debugfs interface
- Lab 133.3: Oops debugging

---

#### **Day 134: Week 19 Review and Project**
**Topics:**
- Character driver review
- Complete driver development

**Sections:**
1. Driver Development Summary
2. Best Practices
3. Testing Strategies

**Labs:**
- Lab 134.1: Complete character driver
- Lab 134.2: Multi-instance driver
- Lab 134.3: Driver testing suite

---

### **Week 20: Platform Drivers and Device Tree** (Days 135-141)

#### **Day 135: Platform Device Model**
**Topics:**
- Platform bus
- Platform devices and drivers
- Probe and remove functions
- Platform data

**Sections:**
1. Platform Bus Architecture
2. Device-Driver Binding
3. Probe Mechanism
4. Resource Management

**Labs:**
- Lab 135.1: Platform driver creation
- Lab 135.2: Platform device registration
- Lab 135.3: Resource handling

---

#### **Day 136: Device Tree Fundamentals**
**Topics:**
- Device tree concept
- DTS and DTB
- Device tree syntax
- Bindings documentation

**Sections:**
1. Device Tree Overview
2. DTS Format
3. Compilation (dtc)
4. Binding Specifications

**Labs:**
- Lab 136.1: Device tree analysis
- Lab 136.2: DTS modification
- Lab 136.3: DTB compilation

---

#### **Day 137: Device Tree Properties**
**Topics:**
- Standard properties
- Compatible strings
- Reg and ranges
- Interrupts and GPIOs

**Sections:**
1. Property Types
2. Compatible Property
3. Address Translation
4. Interrupt Mapping

**Labs:**
- Lab 137.1: Property parsing
- Lab 137.2: GPIO from DT
- Lab 137.3: Interrupt from DT

---

#### **Day 138: OF (Open Firmware) API**
**Topics:**
- OF API functions
- Property reading
- Resource extraction
- Node traversal

**Sections:**
1. OF API Overview
2. of_property_read_* Functions
3. of_get_* Functions
4. Device Tree Walking

**Labs:**
- Lab 138.1: OF API usage
- Lab 138.2: Property extraction
- Lab 138.3: Resource parsing

---

#### **Day 139: Platform Driver with Device Tree**
**Topics:**
- DT-based platform drivers
- Compatible matching
- Multiple instances
- Overlay support

**Sections:**
1. DT Integration
2. of_match_table
3. Multi-Instance Handling
4. Device Tree Overlays

**Labs:**
- Lab 139.1: DT-based platform driver
- Lab 139.2: Multi-instance support
- Lab 139.3: Overlay creation

---

#### **Day 140: Pinctrl and GPIO Subsystems**
**Topics:**
- Pinctrl subsystem
- GPIO subsystem
- Pin configuration
- GPIO consumer interface

**Sections:**
1. Pinctrl Framework
2. GPIO Framework
3. Pin Multiplexing
4. GPIO Descriptor API

**Labs:**
- Lab 140.1: Pinctrl configuration
- Lab 140.2: GPIO driver
- Lab 140.3: GPIO consumer

---

#### **Day 141: Week 20 Review and Project**
**Topics:**
- Platform driver and DT review
- Complete DT-based driver

**Sections:**
1. Week Summary
2. DT Best Practices
3. Integration Testing

**Labs:**
- Lab 141.1: Complete platform driver with DT
- Lab 141.2: Multi-peripheral DT
- Lab 141.3: DT overlay testing

---

### **Week 21: I2C and SPI Drivers** (Days 142-148)

#### **Day 142: I2C Subsystem**
**Topics:**
- I2C subsystem architecture
- I2C adapter drivers
- I2C client drivers
- I2C core API

**Sections:**
1. I2C Framework
2. Adapter vs Client
3. I2C Core Functions
4. Device Tree Integration

**Labs:**
- Lab 142.1: I2C client driver
- Lab 142.2: I2C device registration
- Lab 142.3: I2C communication

---

#### **Day 143: I2C Device Drivers**
**Topics:**
- I2C sensor drivers
- EEPROM drivers
- RTC drivers
- I2C debugging

**Sections:**
1. Sensor Driver Development
2. EEPROM Access
3. RTC Integration
4. I2C Tools

**Labs:**
- Lab 143.1: Temperature sensor driver
- Lab 143.2: I2C EEPROM driver
- Lab 143.3: I2C debugging

---

#### **Day 144: SPI Subsystem**
**Topics:**
- SPI subsystem architecture
- SPI master drivers
- SPI device drivers
- SPI core API

**Sections:**
1. SPI Framework
2. Master Controller Drivers
3. SPI Device Drivers
4. Transfer Mechanisms

**Labs:**
- Lab 144.1: SPI device driver
- Lab 144.2: SPI transfer
- Lab 144.3: SPI DMA

---

#### **Day 145: SPI Device Drivers**
**Topics:**
- SPI Flash drivers
- SPI ADC drivers
- SPI display drivers
- MTD subsystem basics

**Sections:**
1. Flash Driver Development
2. ADC Integration
3. Display Drivers
4. MTD Framework

**Labs:**
- Lab 145.1: SPI Flash driver
- Lab 145.2: SPI ADC driver
- Lab 145.3: MTD integration

---

#### **Day 146: Regmap API**
**Topics:**
- Regmap subsystem
- I2C/SPI register access
- Cache mechanisms
- IRQ handling with regmap

**Sections:**
1. Regmap Overview
2. Regmap Configuration
3. Register Access
4. Regmap IRQ

**Labs:**
- Lab 146.1: Regmap I2C driver
- Lab 146.2: Regmap SPI driver
- Lab 146.3: Regmap cache

---

#### **Day 147: Input Subsystem**
**Topics:**
- Input subsystem overview
- Input device registration
- Event reporting
- Input handlers

**Sections:**
1. Input Framework
2. Event Types
3. Device Registration
4. Handler Development

**Labs:**
- Lab 147.1: Button input driver
- Lab 147.2: Touchscreen driver basics
- Lab 147.3: Event reporting

---

#### **Day 148: Week 21 Review and Project**
**Topics:**
- I2C/SPI driver review
- Multi-bus driver development

**Sections:**
1. Bus Subsystems Summary
2. Driver Design Patterns
3. Integration Strategies

**Labs:**
- Lab 148.1: Multi-sensor I2C/SPI driver
- Lab 148.2: Input device integration
- Lab 148.3: Complete peripheral driver

---

### **Week 22: V4L2 Subsystem Basics** (Days 149-155)

#### **Day 149: V4L2 Architecture Overview**
**Topics:**
- Video4Linux2 introduction
- V4L2 architecture
- Media controller framework
- Subdevice concept

**Sections:**
1. V4L2 History and Purpose
2. Framework Architecture
3. Media Controller
4. Subdevice Model

**Labs:**
- Lab 149.1: V4L2 device enumeration
- Lab 149.2: Media controller exploration
- Lab 149.3: Subdevice inspection

---

#### **Day 150: V4L2 Device Nodes**
**Topics:**
- /dev/video* devices
- /dev/v4l-subdev* devices
- /dev/media* devices
- Device capabilities

**Sections:**
1. Device Node Types
2. Capability Querying
3. Device Opening
4. Node Relationships

**Labs:**
- Lab 150.1: Device node identification
- Lab 150.2: Capability inspection
- Lab 150.3: Device tree for V4L2

---

#### **Day 151: V4L2 Controls**
**Topics:**
- Control framework
- Control types
- Control operations
- Extended controls

**Sections:**
1. V4L2 Control API
2. Standard Controls
3. Custom Controls
4. Control Handlers

**Labs:**
- Lab 151.1: Control enumeration
- Lab 151.2: Control get/set
- Lab 151.3: Control handler implementation

---

#### **Day 152: V4L2 Formats**
**Topics:**
- Pixel formats
- Format negotiation
- Format enumeration
- Format conversion

**Sections:**
1. Pixel Format Types
2. FOURCC Codes
3. Format Negotiation
4. Format Capabilities

**Labs:**
- Lab 152.1: Format enumeration
- Lab 152.2: Format setting
- Lab 152.3: Format validation

---

#### **Day 153: V4L2 Buffer Management**
**Topics:**
- Buffer types
- Memory types (MMAP, USERPTR, DMABUF)
- Buffer allocation
- Queue management

**Sections:**
1. Buffer Framework
2. Memory Management
3. Buffer Lifecycle
4. Queue Operations

**Labs:**
- Lab 153.1: Buffer allocation
- Lab 153.2: MMAP buffers
- Lab 153.3: DMABUF usage

---

#### **Day 154: V4L2 Streaming**
**Topics:**
- Stream on/off
- Buffer queueing
- Frame capture
- Streaming modes

**Sections:**
1. Streaming Lifecycle
2. QBUF/DQBUF
3. Capture Loop
4. Continuous Capture

**Labs:**
- Lab 154.1: Simple capture application
- Lab 154.2: Continuous streaming
- Lab 154.3: Frame processing

---

#### **Day 155: Week 22 Review and Project**
**Topics:**
- V4L2 basics review
- Simple capture application

**Sections:**
1. V4L2 Concepts Summary
2. Application Development
3. Debugging Techniques

**Labs:**
- Lab 155.1: Complete V4L2 capture app
- Lab 155.2: Multi-format capture
- Lab 155.3: Performance testing

---

### **Week 23: V4L2 Subdevices** (Days 156-162)

#### **Day 156: Subdevice Architecture**
**Topics:**
- Subdevice concept
- Subdevice operations
- Pad-level operations
- Routing

**Sections:**
1. Subdevice Model
2. v4l2_subdev Structure
3. Pad Operations
4. Media Links

**Labs:**
- Lab 156.1: Subdevice exploration
- Lab 156.2: Pad enumeration
- Lab 156.3: Link configuration

---

#### **Day 157: Camera Sensor Drivers**
**Topics:**
- Sensor driver structure
- Register configuration
- Exposure and gain control
- Frame rate control

**Sections:**
1. Sensor Driver Architecture
2. I2C Communication
3. Control Implementation
4. Timing Configuration

**Labs:**
- Lab 157.1: Simple sensor driver
- Lab 157.2: Control implementation
- Lab 157.3: Format support

---

#### **Day 158: CSI-2 Receiver Drivers**
**Topics:**
- MIPI CSI-2 overview
- CSI-2 receiver configuration
- Lane configuration
- Virtual channels

**Sections:**
1. CSI-2 Protocol Basics
2. Receiver Architecture
3. Lane Mapping
4. VC Handling

**Labs:**
- Lab 158.1: CSI-2 receiver setup
- Lab 158.2: Lane configuration
- Lab 158.3: VC routing

---

#### **Day 159: ISP (Image Signal Processor) Basics**
**Topics:**
- ISP pipeline overview
- Bayer pattern processing
- Demosaicing
- Color correction

**Sections:**
1. ISP Architecture
2. RAW Processing
3. Demosaic Algorithms
4. Color Space Conversion

**Labs:**
- Lab 159.1: ISP pipeline configuration
- Lab 159.2: RAW to RGB conversion
- Lab 159.3: Color correction

---

#### **Day 160: Media Controller API**
**Topics:**
- Media device
- Entities and pads
- Links and pipelines
- Pipeline configuration

**Sections:**
1. Media Controller Framework
2. Entity Graph
3. Link Management
4. Pipeline Setup

**Labs:**
- Lab 160.1: Media graph exploration
- Lab 160.2: Link configuration
- Lab 160.3: Pipeline setup

---

#### **Day 161: V4L2 Async Framework**
**Topics:**
- Async subdevice registration
- Notifier mechanism
- Probe ordering
- Device tree integration

**Sections:**
1. Async Framework Purpose
2. Notifier Setup
3. Subdevice Binding
4. DT Integration

**Labs:**
- Lab 161.1: Async notifier setup
- Lab 161.2: Subdevice binding
- Lab 161.3: Multi-device system

---

#### **Day 162: Week 23 Review and Project**
**Topics:**
- V4L2 subdevice review
- Camera pipeline development

**Sections:**
1. Subdevice Summary
2. Pipeline Design
3. Integration Testing

**Labs:**
- Lab 162.1: Complete camera driver
- Lab 162.2: Multi-subdevice pipeline
- Lab 162.3: End-to-end capture

---

### **Week 24: Advanced V4L2 Topics** (Days 163-169)

#### **Day 163: V4L2 Memory-to-Memory Devices**
**Topics:**
- M2M framework
- M2M device structure
- Job scheduling
- Hardware acceleration

**Sections:**
1. M2M Concept
2. M2M Context
3. Job Queue
4. Hardware Integration

**Labs:**
- Lab 163.1: M2M device basics
- Lab 163.2: Format conversion M2M
- Lab 163.3: Hardware encoder/decoder

---

#### **Day 164: V4L2 Events**
**Topics:**
- Event framework
- Event types
- Event subscription
- Event handling

**Sections:**
1. Event Mechanism
2. Standard Events
3. Subscription API
4. Event Processing

**Labs:**
- Lab 164.1: Event subscription
- Lab 164.2: Control change events
- Lab 164.3: Custom events

---

#### **Day 165: V4L2 Selection API**
**Topics:**
- Selection targets
- Cropping
- Composing
- Scaling

**Sections:**
1. Selection Framework
2. Crop/Compose
3. Rectangle Handling
4. Scaling Configuration

**Labs:**
- Lab 165.1: Crop configuration
- Lab 165.2: Scaling setup
- Lab 165.3: ROI selection

---

#### **Day 166: V4L2 Metadata**
**Topics:**
- Metadata formats
- Metadata buffers
- Embedded data
- Statistics

**Sections:**
1. Metadata Concept
2. Metadata Formats
3. Buffer Management
4. Statistics Extraction

**Labs:**
- Lab 166.1: Metadata capture
- Lab 166.2: Embedded data parsing
- Lab 166.3: Statistics processing

---

#### **Day 167: V4L2 Request API**
**Topics:**
- Request API concept
- Request creation
- Control association
- Request submission

**Sections:**
1. Request Framework
2. Request Lifecycle
3. Per-Frame Controls
4. Request Queue

**Labs:**
- Lab 167.1: Request creation
- Lab 167.2: Per-frame control
- Lab 167.3: Request-based capture

---

#### **Day 168: V4L2 Compliance Testing**
**Topics:**
- v4l2-compliance tool
- Compliance requirements
- Test coverage
- Debugging failures

**Sections:**
1. Compliance Testing
2. v4l2-compliance Usage
3. Common Issues
4. Fixing Compliance

**Labs:**
- Lab 168.1: Running compliance tests
- Lab 168.2: Analyzing failures
- Lab 168.3: Fixing compliance issues

---

#### **Day 169: Week 24 Review and Project**
**Topics:**
- Advanced V4L2 review
- Complete camera system

**Sections:**
1. Advanced Topics Summary
2. System Integration
3. Performance Optimization

**Labs:**
- Lab 169.1: Full-featured camera driver
- Lab 169.2: M2M processing pipeline
- Lab 169.3: Compliance-ready driver

---

### **Week 25: DMA and Videobuf2** (Days 170-176)

#### **Day 170: DMA API in Linux**
**Topics:**
- DMA concepts in Linux
- Coherent vs streaming DMA
- DMA mapping
- DMA pools

**Sections:**
1. Linux DMA Framework
2. DMA Allocation
3. Mapping Types
4. Cache Management

**Labs:**
- Lab 170.1: Coherent DMA allocation
- Lab 170.2: Streaming DMA
- Lab 170.3: DMA pool usage

---

#### **Day 171: DMA-BUF Framework**
**Topics:**
- DMA-BUF concept
- Buffer sharing
- Attachment and mapping
- Synchronization

**Sections:**
1. DMA-BUF Overview
2. Buffer Export/Import
3. Attachment Mechanism
4. Fence Synchronization

**Labs:**
- Lab 171.1: DMA-BUF export
- Lab 171.2: DMA-BUF import
- Lab 171.3: Zero-copy pipeline

---

#### **Day 172: Videobuf2 Framework**
**Topics:**
- VB2 architecture
- Queue operations
- Buffer operations
- Memory allocators

**Sections:**
1. VB2 Overview
2. Queue Setup
3. Buffer Lifecycle
4. Allocator Types

**Labs:**
- Lab 172.1: VB2 queue setup
- Lab 172.2: Buffer handling
- Lab 172.3: Custom allocator

---

#### **Day 173: VB2 Memory Types**
**Topics:**
- MMAP allocator
- USERPTR handling
- DMABUF integration
- Contiguous memory

**Sections:**
1. Memory Type Comparison
2. MMAP Implementation
3. USERPTR Usage
4. DMABUF in VB2

**Labs:**
- Lab 173.1: MMAP implementation
- Lab 173.2: USERPTR support
- Lab 173.3: DMABUF integration

---

#### **Day 174: VB2 Queue Operations**
**Topics:**
- Queue start/stop
- Buffer queueing
- Buffer completion
- Error handling

**Sections:**
1. Queue Lifecycle
2. Start/Stop Streaming
3. Buffer Management
4. Error Recovery

**Labs:**
- Lab 174.1: Queue operations
- Lab 174.2: Buffer flow
- Lab 174.3: Error handling

---

#### **Day 175: CMA (Contiguous Memory Allocator)**
**Topics:**
- CMA concept
- CMA configuration
- CMA usage
- Reserved memory

**Sections:**
1. CMA Overview
2. Device Tree Configuration
3. CMA Allocation
4. Reserved Memory Regions

**Labs:**
- Lab 175.1: CMA setup
- Lab 175.2: CMA allocation
- Lab 175.3: Reserved memory usage

---

#### **Day 176: Week 25 Review and Project**
**Topics:**
- DMA and VB2 review
- Zero-copy video pipeline

**Sections:**
1. Memory Management Summary
2. Performance Optimization
3. Best Practices

**Labs:**
- Lab 176.1: Complete VB2 driver
- Lab 176.2: Zero-copy implementation
- Lab 176.3: Performance benchmarking

---

### **Week 26-37: Continuation...**

Due to length constraints, I'll summarize the remaining weeks:

**Week 26:** Power Management (Runtime PM, System Sleep)  
**Week 27:** Kernel Debugging (KGDB, Ftrace, Perf)  
**Week 28:** Device Model and Sysfs  
**Week 29:** Clocks and Regulators  
**Week 30:** Thermal Management  
**Week 31:** Network Drivers Basics  
**Week 32:** USB Drivers  
**Week 33:** Block Device Drivers  
**Week 34:** Filesystems Basics  
**Week 35:** Kernel Security (SELinux, Capabilities)  
**Week 36:** Real-Time Linux (PREEMPT_RT)  
**Week 37:** Integration Project and Assessment

---

## 📚 Required Hardware

- **Development Board:** Raspberry Pi 4 or BeagleBone Black
- **Camera Module:** Raspberry Pi Camera Module V2 or compatible MIPI CSI-2 camera
- **Debug Tools:** JTAG debugger, Serial console
- **Sensors:** I2C/SPI sensors for driver development
- **Storage:** SD card, USB storage

## 📖 Recommended Resources

**Books:**
- "Linux Device Drivers" by Corbet, Rubini, Kroah-Hartman
- "Linux Kernel Development" by Robert Love
- "Essential Linux Device Drivers" by Sreekrishnan Venkateswaran

**Online:**
- Linux kernel documentation
- V4L2 specification
- Kernel mailing lists

---

## ✅ Phase 2 Completion Criteria

Upon completing Phase 2, you should be able to:
- ✓ Develop various types of Linux device drivers
- ✓ Work with V4L2 subsystem and camera drivers
- ✓ Configure and use device trees
- ✓ Debug kernel-level code
- ✓ Implement power management
- ✓ Build embedded Linux systems
- ✓ Optimize driver performance
- ✓ Contribute to kernel development

**Next:** Proceed to Phase 3 - Camera Systems, SerDes & ISP Development
