# Module 35 â€” Edge AI, Embedded Systems and TinyML

> **Part:** [Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems](../INDEX.md)  
> **Prerequisites:** 05,17,23  
> **Purpose:** Build resource-bounded AI on microcontrollers and small computers with power, latency, memory, and update constraints.

## Topics and current teaching homes

| # | Topic | Teaching home | Status |
|---:|---|---|---|
| 1 | embedded compute | [01 Embedded Acquisition, Firmware and TinyML](Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) §§3–4, source laboratories | partial draft |
| 2 | TinyML lifecycle | [01](Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) §§7–8 and planned deployment checklist | partial draft; lifecycle checklist planned |
| 3 | sensor data pipelines | [01](Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) §§2–6 | partial draft |
| 4 | RTOS and scheduling | [01](Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) §§4, 8 and Laboratory 04 | partial draft |
| 5 | power and memory budgets | [01](Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) §§4.3, 9 and Laboratories 07/12 | partial draft |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Cortex-M4F device constraints, STM32F401RE, ADC/DMA, FreeRTOS, CMSIS-DSP/NN, UART, power, Cortex-A76/NEON and debugging](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-arm-firmware) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.