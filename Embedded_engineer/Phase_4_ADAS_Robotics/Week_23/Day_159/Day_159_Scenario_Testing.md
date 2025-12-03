# Day 159: Scenario-Based Testing
## Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation

---

> **📝 Day 159 Focus:**
> Driving isn't just "Lane Keeping". It's "Merging onto a highway at 60mph while a truck blocks your view". These are **Scenarios**. We use **OpenSCENARIO** (XML) to define them formally so simulators (CARLA/VTD) can execute them.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Understand** the hierarchy: Functional $\to$ Logical $\to$ Concrete Scenarios.
2.  **Read and Write** OpenSCENARIO 1.0 (XML) files.
3.  **Define** Actors, Maneuvers, and Trigger Conditions.
4.  **Parameterize** a scenario (e.g., vary speed from 10-30 m/s).
5.  **Execute** a scenario in a compatible engine (e.g., `scenario_runner`).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **XML:** Tags, Attributes.
-   **Day 145:** CARLA Simulator.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **CARLA:** `scenario_runner` (Python tool).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Scenario Levels (PEGASUS Project)

1.  **Functional:** "Car turns left across traffic." (Human language).
2.  **Logical:** "Car turns left. Oncoming traffic speed $v \in [30, 60]$ km/h. Gap $g \in [2, 5]$ s." (Parameter ranges).
3.  **Concrete:** "Car turns left. Oncoming speed = 45 km/h. Gap = 3.5 s." (Specific values for one test run).

### 🔹 Part 2: OpenSCENARIO Structure

-   **Entities:** Ego, NPC1, NPC2.
-   **Storyboard:** The script.
    -   **Init:** Start positions.
    -   **Story:**
        -   **Maneuver:** "Lane Change".
        -   **Event:** "Start changing lane".
        -   **Condition:** "When Ego speed > 50 km/h".

### 🔹 Part 3: Why Standardize?

-   If you write a scenario in OpenSCENARIO, you can run it in CARLA, VTD, IPG CarMaker, or AWS SimSpace Weaver.
-   It enables sharing test libraries across the industry.

---

## 💻 Implementation: "Cut-In" Scenario

**Scenario:**
-   Ego drives straight.
-   Adversary (NPC) overtakes and cuts in front of Ego.
-   Ego must brake.

### 🛠️ Setup
Create `week23_day159` and `cut_in.xosc`.

```bash
mkdir -p ~/ros2_ws/src/week23_day159
cd ~/ros2_ws/src/week23_day159
touch cut_in.xosc
```

### 👨‍💻 Code: OpenSCENARIO XML

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
    <FileHeader revMajor="1" revMinor="0" date="2023-10-27" description="Cut In Scenario" author="Antigravity"/>
    
    <ParameterDeclarations>
        <ParameterDeclaration name="CutInSpeed" parameterType="double" value="20.0"/>
        <ParameterDeclaration name="EgoSpeed" parameterType="double" value="15.0"/>
    </ParameterDeclarations>

    <CatalogLocations/>

    <RoadNetwork>
        <LogicFile filepath="Town04"/>
    </RoadNetwork>

    <Entities>
        <ScenarioObject name="hero">
            <Vehicle name="vehicle.tesla.model3" vehicleCategory="car">
                <BoundingBox>
                    <Center x="1.5" y="0.0" z="0.9"/>
                    <Dimensions width="2.1" length="4.5" height="1.8"/>
                </BoundingBox>
                <Performance maxSpeed="69.444" maxAcceleration="200" maxDeceleration="10.0"/>
                <Axles>
                    <FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.8" positionX="3.1" positionZ="0.3"/>
                    <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
                </Axles>
                <Properties/>
            </Vehicle>
        </ScenarioObject>
        <ScenarioObject name="adversary">
            <Vehicle name="vehicle.audi.tt" vehicleCategory="car">
                <!-- Properties omitted for brevity -->
            </Vehicle>
        </ScenarioObject>
    </Entities>

    <Storyboard>
        <Init>
            <Actions>
                <Private entityRef="hero">
                    <PrivateAction>
                        <TeleportAction>
                            <Position>
                                <LanePosition roadId="4" laneId="-1" offset="0.0" s="50.0"/>
                            </Position>
                        </TeleportAction>
                    </PrivateAction>
                    <PrivateAction>
                        <LongitudinalAction>
                            <SpeedAction>
                                <SpeedActionDynamics dynamicsShape="step" value="0" dynamicsDimension="time"/>
                                <SpeedActionTarget>
                                    <AbsoluteTargetSpeed value="$EgoSpeed"/>
                                </SpeedActionTarget>
                            </SpeedAction>
                        </LongitudinalAction>
                    </PrivateAction>
                </Private>
                <Private entityRef="adversary">
                    <PrivateAction>
                        <TeleportAction>
                            <Position>
                                <LanePosition roadId="4" laneId="-2" offset="0.0" s="40.0"/> <!-- Behind and Left -->
                            </Position>
                        </TeleportAction>
                    </PrivateAction>
                </Private>
            </Actions>
        </Init>

        <Story name="CutInStory">
            <Act name="CutInAct">
                <ManeuverGroup maximumExecutionCount="1" name="CutInManeuverGroup">
                    <Actors selectTriggeringEntities="false">
                        <EntityRef entityRef="adversary"/>
                    </Actors>
                    <Maneuver name="CutInManeuver">
                        <Event name="OvertakeAndCutIn" priority="overwrite">
                            <Action name="LaneChangeAction">
                                <PrivateAction>
                                    <LateralAction>
                                        <LaneChangeAction>
                                            <LaneChangeActionDynamics dynamicsShape="sinusoidal" value="5.0" dynamicsDimension="time"/>
                                            <LaneChangeTarget>
                                                <RelativeTargetLane entityRef="hero" value="0"/> <!-- Same lane as hero -->
                                            </LaneChangeTarget>
                                        </LaneChangeAction>
                                    </LateralAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="StartCondition" delay="0" conditionEdge="rising">
                                        <ByEntityCondition>
                                            <TriggeringEntities triggeringEntitiesRule="any">
                                                <EntityRef entityRef="adversary"/>
                                            </TriggeringEntities>
                                            <EntityCondition>
                                                <RelativeDistanceCondition entityRef="hero" relativeDistanceType="longitudinal" value="10.0" freespace="false" rule="greaterThan"/>
                                            </EntityCondition>
                                        </ByEntityCondition>
                                    </Condition>
                                </ConditionGroup>
                            </StartTrigger>
                        </Event>
                    </Maneuver>
                </ManeuverGroup>
            </Act>
        </Story>
        <StopTrigger/>
    </Storyboard>
</OpenSCENARIO>
```

---

## 🔬 Lab Exercise: Running the Scenario

### Lab Objectives
1.  **Install Scenario Runner:**
    ```bash
    git clone https://github.com/carla-simulator/scenario_runner.git
    pip install -r requirements.txt
    ```
2.  **Run:**
    ```bash
    python scenario_runner.py --openscenario ~/ros2_ws/src/week23_day159/cut_in.xosc
    ```
3.  **Observation:**
    -   CARLA launches.
    -   The Audi (Adversary) spawns behind the Tesla (Hero).
    -   The Audi speeds up.
    -   When the Audi is 10m ahead, it smoothly changes lanes into the Tesla's lane.
    -   **Pass/Fail:** If the Tesla crashes, the test fails.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. XML Syntax Errors
**Symptom:** Scenario runner crashes immediately.
**Cause:** Missing closing tag or typo in attribute name.
**Solution:** Use an XML validator or an OpenSCENARIO editor (e.g., Esmini).

#### 2. Trigger Never Fires
**Symptom:** Adversary drives forever in the left lane.
**Cause:** The Condition (`RelativeDistance > 10.0`) was never met (maybe it wasn't fast enough).
**Solution:** Check the initial speeds and positions. Ensure the physics allows the condition to happen.

---

## ⚡ Optimization & Best Practices

### 1. Parameter Sweeps
Don't just run one "Cut-in".
-   Write a Python script to generate 100 `.xosc` files.
-   Vary `CutInSpeed` from 10 to 40 m/s.
-   Vary `Distance` from 5 to 50 m.
-   Find the **Boundary** where the AV fails.

### 2. OpenSCENARIO 2.0
-   1.0 is XML (Verbose, hard to read).
-   2.0 is a Domain Specific Language (DSL) (Python-like).
-   Much more powerful for abstract scenarios.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is a "Trigger" in OpenSCENARIO?
    *   **A:** A condition that starts an Event (e.g., "Time > 5s" or "Distance < 10m").
2.  **Q:** Difference between `AbsoluteTargetSpeed` and `RelativeTargetSpeed`?
    *   **A:** Absolute sets a fixed value (e.g., 50 km/h). Relative sets speed based on another entity (e.g., Ego Speed + 10 km/h).
3.  **Q:** Why use Logical Scenarios?
    *   **A:** They cover a *range* of situations, allowing statistical validation, whereas Concrete scenarios only test one specific point.

### Challenge Task
**Task:** Pedestrian Crossing.
1.  Create a scenario where a pedestrian (`walker.pedestrian.0001`) waits at a crosswalk.
2.  Trigger: When Ego is 20m away.
3.  Action: Pedestrian walks across the road.
4.  Verify Ego stops.

---

## 📚 Further Reading & References
-   [ASAM OpenSCENARIO Standard](https://www.asam.net/standards/detail/openscenario/)
-   [CARLA Scenario Runner Docs](https://carla-scenariorunner.readthedocs.io/)

---

**Day 159 Complete** | Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation
