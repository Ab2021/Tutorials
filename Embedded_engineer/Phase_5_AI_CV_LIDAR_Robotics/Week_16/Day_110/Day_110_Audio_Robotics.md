# Day 110: Audio for Robotics (Direction of Arrival)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 16: Advanced Sensors

---

> **📝 Content Creator Instructions:**
> Ears are just as important as Eyes.
> - **Focus:** Microphone Arrays, Cross-Correlation, TDOA (Time Difference of Arrival), and estimating Sound Source Localization (SSL).
> - **Code:** A Python node implementing GCC-PHAT (Generalized Cross Correlation - Phase Transform) to find the angle of a sound source using a simulated 2-mic setup.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** how phase difference maps to angle of arrival ($\theta = \arcsin(\frac{c \cdot \Delta t}{d})$).
2.  **Implement** Cross-Correlation to find the time lag ($\Delta t$) between two audio signals.
3.  **Process** `audio_common_msgs` or raw PyAudio streams in ROS 2.
4.  **Visualize** the sound vector in Rviz (as an arrow).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Respeaker Mic Array / MiniDSP / or simply Laptop Stereo Microphones.

### Software Environment
```bash
sudo apt install ros-humble-audio-common
pip install pyaudio scipy numpy
```

### Prior Knowledge
- Signal Processing (FFT).
- Speed of Sound ($c \approx 343 m/s$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Geometry of Sound

Imagine two microphones separated by distance $d$.
A sound wave arrives from angle $\theta$.
*   It hits Mic 1 at $t_1$.
*   It hits Mic 2 at $t_2$.
*   Path difference $\Delta x = d \sin(\theta)$.
*   Time difference $\Delta t = \frac{\Delta x}{c}$.
*   Therefore: $\theta = \arcsin(\frac{c \cdot \Delta t}{d})$.

### 🔹 Part 2: TDOA (Time Difference of Arrival)

How to measure $\Delta t$ at 44.1kHz?
*   **Simple:** Peak detection. (Unreliable in noise/reverb).
*   **Robust:** Cross-Correlation.
    *   $R_{xy}(\tau) = \text{IFFT}(X(f) \cdot Y^*(f))$.
    *   The peak of $R_{xy}$ occurs at the lag $\tau = \Delta t$.

### 🔹 Part 3: GCC-PHAT

Generalized Cross Correlation with Phase Transform.
*   Normalizes the magnitude of the spectrum.
*   Purely phases.
*   Extremely robust to reverberation (echoes).

---

## 💻 Implementation: Sound Compass

We will read from `PyAudio` (Stereo Input) and calculate $\theta$.

### 🛠️ Project Structure
```text
day110_audio/
├── src/
│   ├── sound_locator.py
└── launch/
    ├── audio.launch.py
```

### 👨‍💻 Locator Node (`src/sound_locator.py`)

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import pyaudio
import struct

class SoundLocator(Node):
    def __init__(self):
        super().__init__('sound_locator')
        self.pub_pose = self.create_publisher(PoseStamped, '/sound_direction', 10)
        
        # Parameters
        self.mic_dist = 0.10 # 10 cm between laptop mics (Approx)
        self.RATE = 44100
        self.CHUNK = 2048
        self.c = 343.0 # m/s
        
        # Audio Setup
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=2, # Stereo
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
                                  
        self.timer = self.create_wall_timer(0.1, self.process_audio)

    def gcc_phat(self, sig1, sig2):
        n = len(sig1)
        # FFT
        SIG1 = np.fft.rfft(sig1)
        SIG2 = np.fft.rfft(sig2)
        
        # Cross Spectrum
        R = SIG1 * np.conj(SIG2)
        
        # PHAT Weighting (Whitening)
        # Avoid divide by zero
        cc = R / (np.abs(R) + 1e-6)
        
        # IFFT
        r = np.fft.irfft(cc, n=n)
        
        # Finding Peak
        # Shift so 0 lag is at center
        r = np.roll(r, n//2)
        idx = np.argmax(r)
        shift = idx - n//2
        
        return shift

    def process_audio(self):
        try:
            raw_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            data_int16 = np.frombuffer(raw_data, dtype=np.int16)
            
            # De-interleave Stereo
            ch1 = data_int16[0::2]
            ch2 = data_int16[1::2]
            
            # GCC-PHAT
            lag_samples = self.gcc_phat(ch1, ch2)
            
            # Calculate Angle
            tau = lag_samples / self.RATE
            
            # Clip for asin domain [-1, 1]
            val = (self.c * tau) / self.mic_dist
            val = np.clip(val, -1.0, 1.0)
            
            angle = np.arcsin(val)
            
            # Publish
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            
            # Visualization: Quaternion from Yaw
            # Simple 2D Rotation
            msg.pose.orientation.z = np.sin(angle / 2.0)
            msg.pose.orientation.w = np.cos(angle / 2.0)
            
            # Only publish if signal is loud enough
            if np.max(np.abs(ch1)) > 1000:
                self.pub_pose.publish(msg)
                
        except Exception as e:
            self.get_logger().error(f"Audio Error: {e}")

    def __del__(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

def main():
    rclpy.init()
    rclpy.spin(SoundLocator())
```

---

## 🔬 Lab Exercise: "The Clap Test"

### 1. Lab Objectives
- **Setup:** Run node. Open Rviz. Add Arrow marker for `/sound_direction`.
- **Action:** Clap hands to the Left.
- **Observation:** Arrow points Left.
- **Action:** Clap hands to the Right.
- **Observation:** Arrow points Right.
- **Ambiguity:** Front/Back ambiguity. With 2 mics, 30 deg front is same delay as 30 deg back (Cone of Confusion). Solution: Use 3 mics in a triangle.

---

## 🚀 Project: "Voice Command Turn"

**Goal:** Robot turns to face you when you speak.
1.  **Wait:** For sound level > Threshold.
2.  **Estimate:** Angle $\theta$.
3.  **Command:** `cmd_vel` turn at $0.5$ rad/s until odometry turns $\theta$.
4.  **Acknowledge:** Play a beep. "I'm listening".
5.  **Integration:** Pass audio to Whisper (OpenAI) for speech-to-text.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No Audio Device"
*   **Cause:** Linux permissions. User not in `audio` group.
*   **Fix:** `sudo usermod -aG audio $USER`. Docker needs `--device /dev/snd`.

#### 2. "Jittery Angle"
*   **Cause:** Room reverb.
*   **Fix:** Average the angle over 5 frames. Or use a Kalman Filter on the angle.

---

## ⚡ Optimization: Hardware Arrays

Respeaker USB 4-Mic Array.
*   Has internal FPGA performing Beamforming and Noise Cancellation.
*   Outputs cleaned audio + parsed Angle via Tuning parameters (ODAS - Open Embeded Audition System).
*   **Ros Package:** `respeaker_ros`. Uses `libusb` to extract raw channel data.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Max frequency for TDOA?
    *   **A:** Spatial Aliasing. If wavelength $\lambda < 2d$, phase wraps around. For $d=10cm$, max freq $\approx 1700 Hz$.
2.  **Q:** Difference between Beamforming and TDOA?
    *   **A:** TDOA finds source. Beamforming *isolates* source (suppresses noise from other directions).
3.  **Q:** Why 3 mics?
    *   **A:** To solve Front/Back ambiguity. Planar arrays ($X, Y$) give $360^{\circ}$ coverage. Tetrahedral ($X, Y, Z$) gives spherical coverage (Elevation).

### Challenge Task
> **Task:** Follow the Whistle.
> 1. Bandpass filter audio (1000Hz - 2000Hz).
> 2. Ignore talking (low freq).
> 3. Only track high-pitch whistles.

---

## 📚 Further Reading
- **ODAS (Open Embeded Audition System):** Industry standard open source library for SSL / SST / Beamforming.
- **Respeaker Docs:** Hardware guide.

---

**Day 110 Complete**
