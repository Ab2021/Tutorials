---
collection: "sensor-fusion-iot-and-realtime-perception"
file_id: "02"
topic: "Sensor time synchronization and calibration"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Sensor Time Synchronization and Calibration

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) is the consulted inventory; this is a new course-topic draft, not a reconstruction of unavailable source material.

## 1. Contract

For each sensor, declare clock domain, timestamp meaning (capture, receive or process), unit, epoch, uncertainty, calibration frame, valid range and failure behavior. A timestamp is not evidence that two measurements describe the same physical instant.

## 2. Time alignment

For offset model `t_ref = a × t_sensor + b`, `b` is offset and `a` corrects drift. At a 100 ppm relative drift, clocks diverge by `100 × 10⁻⁶ × 60 = 0.006 s` in one minute. Estimate and monitor alignment under declared network/measurement conditions; do not assume constant latency.

## 3. Calibration

Intrinsic calibration maps raw measurement to sensor coordinates; extrinsic calibration transforms between frames. Preserve transform direction, units, reference frame, uncertainty, calibration target/method, validity interval and version. Recalibrate after mounting, temperature, firmware or timing changes when those affect the model.

## 4. Fusion checks

Reject or flag measurements outside age, uncertainty, frame, unit and plausibility bounds before fusion. Test synthetic known transforms, stationary references, clock perturbations, dropped/out-of-order messages and a degraded single-sensor mode. A fused estimate can be more confidently wrong when correlated failures are ignored.

## 5. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| Clock drift | residual grows with elapsed time | resync, estimate drift, enforce age bound |
| Frame inversion | mirrored/offset trajectory | explicit transform convention and oracle test |
| Unit mismatch | scale discontinuity | typed units and range checks |
| Stale sample | implausible motion/lag | age metadata and drop/degrade policy |
| Calibration drift | residual shift by environment | monitor reference residual; recalibrate |

## 6. Interview checks

1. **Why distinguish capture and receive time?** Transport delay can vary.
2. **What does extrinsic calibration provide?** Transformation between coordinate frames.
3. **Why retain calibration version?** Results need provenance and valid rollback.
4. **What is a safe fusion precondition?** Compatible time, frame, units and stated uncertainty.
5. **Why test clock perturbation?** Nominal sync can hide real drift/latency failures.

## 7. Quick Reference Card

Record clock model, timestamp semantics, bounds, frame/units, intrinsic/extrinsic versions, uncertainty, validity interval, stale/out-of-order policy, recalibration trigger and degraded behavior. Verify on deployed hardware and workloads.

**Limitations:** Drift arithmetic is illustrative. This is not medical, vehicle, robotics or safety certification guidance.
