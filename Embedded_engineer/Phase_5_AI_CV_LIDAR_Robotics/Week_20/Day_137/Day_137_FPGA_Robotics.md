# Day 137: FPGA for Robotics (Verilog Basics)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 20: Sim-to-Real & Hardware Acceleration

---

> **📝 Content Creator Instructions:**
> CPU is sequential. FPGA is concurrent.
> - **Focus:** Why use FPGAs in Robotics (BLDC Control, Encoder counting, Camera interfacing), Basics of Verilog, LUTs vs Flip-Flops, and generating a logic design.
> - **Code:** A Python script `generate_verilog.py` that outputs a valid Verilog `.v` file for a PID Controller, showing how Fixed-Point math is handled in hardware.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Microcontroller (Instruction Stream) and FPGA (Gate Array).
2.  **Explain** the clock cycle parallelism: A 100-step PID loop on CPU takes 100 cycles. On FPGA, it handles all steps in 1 cycle (Pipelined).
3.  **Read** basic Verilog: `module`, `input`, `output`, `always @(posedge clk)`.
4.  **Implement** a Quadrature Encoder Counter in hardware logic logic.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulation). If you have a KR260/PYNQ logic board, you can deploy it.

### Software Environment
```bash
pip install numpy
# Optional: Icarus Verilog (sudo apt install iverilog)
```

### Prior Knowledge
- Digital Logic (AND, OR, NOT).
- Binary Arithmetic (Two's Complement).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Bottleneck of von Neumann

CPUs fetch instruction $\to$ decode $\to$ execute.
*   **Latency:** Interrupts have jitter (~5 $\mu$s).
*   **FPGA:** Hardwired logic. Input $\to$ Logic $\to$ Output.
*   **Latency:** deterministic (~10 ns).
*   **Use Case:** Reading 6 encoders at 40MHz. CPU chokes. FPGA sleeps.

### 🔹 Part 2: Verilog Basics

It describes **Hardware**, not Software.
```verilog
always @(posedge clk) begin
    a <= b; // Non-blocking assignment (Parallel)
    c <= d; // Happens AT THE SAME TIME as a<=b
end
```
In C++, line 2 happens after line 1. In Verilog, they happen simultaneously.

### 🔹 Part 3: Integer Math only

FPGAs hate Floats (takes huge area). We use **Fixed Point**.
*   Float 3.14 $\to$ (Multiply by 256) $\to$ Int 803.
*   Perform math.
*   Divide output by 256.

---

## 💻 Implementation: PID in Silicon

We write a generator that creates a Verilog PID module.

### 🛠️ Project Structure
```text
day137_fpga/
├── src/
│   ├── generate_verilog.py
└── output/
    ├── pid_controller.v
```

### 👨‍💻 Verilog Generator (`src/generate_verilog.py`)

```python
import os

def generate_pid_verilog(kp, ki, kd, bit_width=16):
    code = f"""
module pid_controller (
    input wire clk,
    input wire rst,
    input wire signed [{bit_width-1}:0] target,
    input wire signed [{bit_width-1}:0] current,
    output reg signed [{bit_width-1}:0] motor_cmd
);

    // Parameters (Fixed Point 8.8)
    // Kp = {kp}, Ki = {ki}, Kd = {kd}
    // We assume input constants are pre-scaled or handled here.
    // For simplicity, let's keep them as small integers.
    
    parameter KP = {int(kp)};
    parameter KI = {int(ki)};
    parameter KD = {int(kd)};

    // Internal Registers
    reg signed [{bit_width-1}:0] prev_error;
    reg signed [31:0] integral; // Wider accumulator
    
    // Wires for calculation
    wire signed [{bit_width-1}:0] error;
    assign error = target - current;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            prev_error <= 0;
            integral <= 0;
            motor_cmd <= 0;
        end else begin
            // 1. Proportional
            // P_term = KP * error
            
            // 2. Integral
            integral <= integral + error;
            
            // 3. Derivative
            // D_term = KD * (error - prev_error)
            
            // Summation (With clamping logic implied for real designs)
            // Note: Mixing blocking (=) and non-blocking (<=) is tricky.
            // Pure register logic:
            
            motor_cmd <= (KP * error) + (KI * (integral >>> 4)) + (KD * (error - prev_error));
            
            // Update History
            prev_error <= error;
        end
    end

endmodule
"""
    return code.strip()

def main():
    # Design Specs
    kp = 10
    ki = 1
    kd = 5
    
    verilog_code = generate_pid_verilog(kp, ki, kd)
    
    os.makedirs("output", exist_ok=True)
    with open("output/pid_controller.v", "w") as f:
        f.write(verilog_code)
        
    print("Verilog Generated at output/pid_controller.v")
    print("-" * 20)
    print(verilog_code)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Counter"

### 1. Lab Objectives
- **Task:** Write a simple Verilog module for a Counter.
- **Code:**
```verilog
module counter(input clk, output reg [7:0] count);
    always @(posedge clk)
        count <= count + 1;
endmodule
```
- **Simulate:** If you have `iverilog`:
    *   `iverilog -o cnt counter.v`
    *   `vvp cnt`
- **Result:** It counts 0...255...0 very fast.

---

## 🚀 Project: "Quadrature Decoder"

**Goal:** Read an Encoder (A/B) signals.
1.  **Logic:**
    *   State Machine: Look at previous (A,B) and current (A,B).
    *   Lookup Table:
        *   00 -> 01: +1
        *   00 -> 10: -1
        *   00 -> 11: Error (Skipped step)
2.  **Verilog:**
```verilog
always @(posedge clk) begin
   A_old <= A;
   B_old <= B;
   if ({A_old, B_old, A, B} == 4'b0001) count <= count + 1;
   // ... cases
end
```
3.  **Benefit:** An FPGA handles 10 encoders at 10 MHz with 0% CPU load.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Timing Violation"
*   **Cause:** Logic path too long between Flip-Flops. Signal didn't arrive before next clock tick.
*   **Fix:** Pipelining. Break calculation into stages. `Current -> Stage1 -> Stage2 -> Output`.

#### 2. "Metastability"
*   **Cause:** Reading an asynchronous signal (Button press) directly into logic.
*   **Fix:** Use a Double Flip-Flop Synchronizer.

---

## ⚡ Optimization: High Level Synthesis (HLS)

Writing Verilog is hard.
*   **HLS:** Write C++ code, tools convert it to Verilog.
*   **Tools:** AMD Vitis HLS, Intel HLS.
*   **Use:** OpenCV on FPGA.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a LUT?
    *   **A:** Look-Up Table. Basically a small RAM that implements truth tables. "If Input is 001, Output is 1".
2.  **Q:** Blocking (=) vs Non-Blocking (<=)?
    *   **A:** `=` is immediate (like software). `<=` is deferred (happens at end of clock cycle). Always use `<=` for sequential logic (Flip-Flops).
3.  **Q:** Why is FPGA better for BLDC control?
    *   **A:** FOC (Field Oriented Control) requires calculating Sine/Cosine at 100kHz. FPGA IP cores (CORDIC) do this efficiently.

### Challenge Task
> **Task:** PWM Generator.
> 1. Create a `reg [7:0] counter`.
> 2. Create `input [7:0] duty`.
> 3. `assign pwm_out = (counter < duty) ? 1 : 0;`
> 4. Verify getting a PWM signal.

---

## 📚 Further Reading
- **Nandland:** Beginner FPGA tutorials.
- **Kria Robotics Stack (KRS):** ROS 2 hardware acceleration on Xilinx Kria SOMs.

---

**Day 137 Complete**
