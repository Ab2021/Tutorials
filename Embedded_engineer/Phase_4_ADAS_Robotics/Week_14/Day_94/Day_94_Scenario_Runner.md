# Day 94: Scenario Runner (OpenSCENARIO)
## Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)

---

> **📝 Day 94 Focus:**
> Driving randomly (Day 93) is fun, but not rigorous. To validate an AV, we need **Scenarios**: "Cut-in from left", "Pedestrian crossing", "Red light violation". **Scenario Runner** allows us to script these events deterministically using the **OpenSCENARIO** standard.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** a scenario using the OpenSCENARIO (.xosc) XML format.
2.  **Execute** scenarios using the CARLA Scenario Runner tool.
3.  **Control** the environment: Weather (Rain, Fog) and Time of Day.
4.  **Script** dynamic behaviors: "Car B cuts in front of Ego when distance < 10m".
5.  **Evaluate** success/failure criteria (Collision, Lane Invasion).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **XML:** OpenSCENARIO is XML-heavy.
-   **Day 93:** Basic CARLA usage.

### Hardware Requirements
-   **GPU:** Required for CARLA.

### Software Stack
-   **Scenario Runner:** A separate Python tool for CARLA.
-   **OpenSCENARIO 1.0:** The standard.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is OpenSCENARIO?

A standard by ASAM (Association for Standardization of Automation and Measuring Systems).
It defines:
-   **Entities:** Ego vehicle, NPCs, Pedestrians.
-   **Storyboard:** The timeline of events.
-   **Init:** Initial positions and speeds.
-   **Story:**
    -   **Act:** A phase of the scenario.
    -   **Maneuver:** A specific action (e.g., Lane Change).
    -   **Event:** Triggered by a condition (e.g., `Distance < 20m`).
    -   **Action:** The atomic command (e.g., `SpeedAction`, `LaneChangeAction`).

### 🔹 Part 2: The Traffic Manager

Scenario Runner handles the *scripted* vehicles.
The **Traffic Manager (TM)** handles the *background* traffic.
-   TM runs in "Autopilot" mode but respects the Scenario's constraints.
-   You can set TM parameters: `% Speeding`, `% Ignore Lights`, `% Keep Right`.

### 🔹 Part 3: Weather Control

Testing perception requires diverse weather.
-   **Parameters:** Cloudiness, Precipitation, Puddles, Wind, Sun Azimuth/Altitude.
-   **Impact:** Rain creates reflections (confuses Camera). Fog reduces visibility (confuses Lidar).

---

## 💻 Implementation: "The Cut-In"

**Scenario:**
-   **Ego:** Driving straight at 30 km/h.
-   **Adversary:** Driving in the left lane.
-   **Trigger:** When Ego is 15m behind Adversary.
-   **Action:** Adversary changes lane to right (cutting off Ego).

### 🛠️ Setup
Clone Scenario Runner (if not present).
Create `week14_day94` and `CutIn.xosc`.

```bash
mkdir -p ~/ros2_ws/src/week14_day94
cd ~/ros2_ws/src/week14_day94
touch CutIn.xosc
```

### 👨‍💻 Code: OpenSCENARIO Definition (`CutIn.xosc`)

```xml
<?xml version="1.0"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="0" date="2023-10-27T10:00:00" description="Cut In Scenario" author="Antigravity"/>
  
  <ParameterDeclarations/>
  
  <CatalogLocations/>
  
  <RoadNetwork>
    <LogicFile filepath="Town04"/>
    <SceneGraphFile filepath=""/>
  </RoadNetwork>
  
  <Entities>
    <!-- Ego Vehicle -->
    <ScenarioObject name="hero">
      <Vehicle name="vehicle.tesla.model3" vehicleCategory="car">
        <ParameterDeclarations/>
        <Performance maxSpeed="69.444" maxAcceleration="200" maxDeceleration="10.0"/>
        <BoundingBox>
          <Center x="1.5" y="0.0" z="0.9"/>
          <Dimensions width="2.1" length="4.5" height="1.8"/>
        </BoundingBox>
        <Axles>
          <FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.8" positionX="3.1" positionZ="0.3"/>
          <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties/>
      </Vehicle>
    </ScenarioObject>
    
    <!-- Adversary Vehicle -->
    <ScenarioObject name="adversary">
      <Vehicle name="vehicle.audi.tt" vehicleCategory="car">
        <ParameterDeclarations/>
        <Performance maxSpeed="69.444" maxAcceleration="200" maxDeceleration="10.0"/>
        <BoundingBox>
          <Center x="1.5" y="0.0" z="0.9"/>
          <Dimensions width="2.1" length="4.5" height="1.8"/>
        </BoundingBox>
        <Axles>
          <FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.8" positionX="3.1" positionZ="0.3"/>
          <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties/>
      </Vehicle>
    </ScenarioObject>
  </Entities>
  
  <Storyboard>
    <Init>
      <Actions>
        <!-- Spawn Ego -->
        <Private entityRef="hero">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <LanePosition roadId="4" laneId="-1" offset="0" s="50"/>
              </Position>
            </TeleportAction>
          </PrivateAction>
          <PrivateAction>
            <LongitudinalAction>
              <SpeedAction>
                <SpeedActionDynamics dynamicsShape="step" value="0" dynamicsDimension="time"/>
                <SpeedActionTarget>
                  <AbsoluteTargetSpeed value="8.33"/> <!-- 30 km/h -->
                </SpeedActionTarget>
              </SpeedAction>
            </LongitudinalAction>
          </PrivateAction>
        </Private>
        
        <!-- Spawn Adversary (Left Lane) -->
        <Private entityRef="adversary">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <LanePosition roadId="4" laneId="-2" offset="0" s="100"/> <!-- Ahead of Ego -->
              </Position>
            </TeleportAction>
          </PrivateAction>
          <PrivateAction>
            <LongitudinalAction>
              <SpeedAction>
                <SpeedActionDynamics dynamicsShape="step" value="0" dynamicsDimension="time"/>
                <SpeedActionTarget>
                  <AbsoluteTargetSpeed value="8.33"/>
                </SpeedActionTarget>
              </SpeedAction>
            </LongitudinalAction>
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
            <Event name="CutInEvent" priority="overwrite">
              <Action name="CutInAction">
                <PrivateAction>
                  <LateralAction>
                    <LaneChangeAction>
                      <LaneChangeActionDynamics dynamicsShape="sinusoidal" value="3.0" dynamicsDimension="time"/>
                      <LaneChangeTarget>
                        <RelativeTargetLane entityRef="hero" value="0"/> <!-- Target Ego's Lane -->
                      </LaneChangeTarget>
                    </LaneChangeAction>
                  </LateralAction>
                </PrivateAction>
              </Action>
              
              <!-- Trigger Condition -->
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="DistanceCondition" delay="0" conditionEdge="rising">
                    <ByEntityCondition>
                      <TriggeringEntities triggeringEntitiesRule="any">
                        <EntityRef entityRef="hero"/>
                      </TriggeringEntities>
                      <EntityCondition>
                        <RelativeDistanceCondition entityRef="adversary" relativeDistanceType="longitudinal" value="15.0" freespace="false" rule="lessThan"/>
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
    
    <StopTrigger>
      <!-- Stop if Collision -->
      <ConditionGroup>
        <Condition name="CollisionCondition" delay="0" conditionEdge="rising">
          <ByEntityCondition>
            <TriggeringEntities triggeringEntitiesRule="any">
              <EntityRef entityRef="hero"/>
            </TriggeringEntities>
            <EntityCondition>
              <CollisionCondition>
                <EntityRef entityRef="adversary"/>
              </CollisionCondition>
            </EntityCondition>
          </ByEntityCondition>
        </Condition>
      </ConditionGroup>
    </StopTrigger>
  </Storyboard>
</OpenSCENARIO>
```

### 👨‍💻 Code: Python Wrapper (`run_scenario.py`)

Usually, we run this via CLI:
`python scenario_runner.py --openscenario CutIn.xosc`

But here is how to control Weather via Python API before running:

```python
import carla
import time

def set_weather():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    
    # Stormy Weather
    weather = carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=60.0,
        precipitation_deposits=80.0, # Puddles
        wind_intensity=50.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=10.0 # Sunset
    )
    
    world.set_weather(weather)
    print("Weather set to Stormy Sunset.")

if __name__ == "__main__":
    set_weather()
    print("Now run: python scenario_runner.py --openscenario CutIn.xosc")
```

---

## 🔬 Lab Exercise: The Crash Test

### Lab Objectives
1.  Set the weather (Python script).
2.  Run the scenario (Scenario Runner).
3.  **Observation:**
    -   Ego (Hero) drives straight.
    -   Adversary (Audi) drives in left lane.
    -   As Ego gets closer, Audi swerves right.
4.  **Outcome:**
    -   If your Ego agent (not defined here, usually a separate script) has **AEB (Automatic Emergency Braking)**, it will brake and avoid collision.
    -   If Ego is dumb (constant velocity), it will crash.
    -   Scenario Runner will report: `FAILURE: Collision detected`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Road ID not found"
**Symptom:** Scenario fails to load.
**Cause:** The `roadId` and `laneId` in `Init` section must match the map (`Town04`).
**Solution:** Open the map in `OpenDRIVE Viewer` to find valid IDs. Or use `Town04` which is standard for highway tests.

#### 2. Ego doesn't move
**Symptom:** Scenario starts, but Ego sits still.
**Cause:** OpenSCENARIO usually expects an external agent (ROS stack) to control the Ego.
**Solution:** Run `manual_control.py` (provided by CARLA) alongside Scenario Runner to drive the Ego yourself, or attach an Agent.

---

## ⚡ Optimization & Best Practices

### 1. Atomic Behaviors
Don't write one giant XML.
-   Use **Catalogs**: Define `VehicleCatalog.xosc`, `ManeuverCatalog.xosc`.
-   Import them. This makes scenarios reusable (e.g., "Cut-In" logic can be applied to "Town01" and "Town05").

### 2. Randomization
Deterministic is good for debugging. Random is good for validation.
-   Use **ParameterDeclarations** in XML.
-   Pass values at runtime: `--openscenario CutIn.xosc --additionalScenarioParams speed=50`.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between `Act` and `Maneuver`?
    *   **A:** Hierarchy: Story -> Act -> ManeuverGroup -> Maneuver -> Event -> Action. An Act is a high-level phase (e.g., "Highway Driving"). A Maneuver is specific (e.g., "Overtake").
2.  **Q:** How do I trigger an event based on time?
    *   **A:** Use `<SimulationTimeCondition value="10.0" rule="greaterThan"/>`.
3.  **Q:** Can I change the weather mid-scenario?
    *   **A:** Yes, using `<EnvironmentAction>`.

### Challenge Task
**Task:** The Red Light Runner.
1.  Create a scenario in `Town01` (Urban).
2.  Ego approaches an intersection with Green light.
3.  Adversary approaches from cross-street (Red light).
4.  Trigger: When Ego is 20m away.
5.  Action: Adversary crosses intersection (ignoring Red light).
6.  Goal: Ego must stop.

---

## 📚 Further Reading & References
-   [OpenSCENARIO 1.0 User Guide](https://www.asam.net/standards/detail/openscenario/)
-   [CARLA Scenario Runner Docs](https://carla-scenariorunner.readthedocs.io/en/latest/)

---

**Day 94 Complete** | Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)
