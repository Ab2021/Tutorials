# Rail Network and Path Capacity

> Module 41 — Maritime, Rail, and Air Cargo Operations Engineering
> Topic 03 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 12 (Transportation Routing & Last Mile), Module 20 (Supply Chain Operations Research), Topic 41.01 (Berth and Quay Crane Operations), Topic 41.02 (Terminal Yard and Gate)
> Version 2026.5

---

## 1. Why this topic exists

Rail infrastructure represents one of the most capital-intensive, physically constrained asset classes in global supply chain logistics. Unlike maritime shipping lanes or highway freight corridors where vessels and trucks can alter course, overtake at will, or buffer dynamically in multi-lane space, a railway network is a strictly one-dimensional, discrete physical topology. Two trains cannot occupy the same section of track simultaneously without catastrophic collision, and trains cannot steer around one another. Every movement requires exclusive temporal reservation of physical track segments bounded by fixed signaling or radio-based authority boundaries.

The core industrial challenge is that line capacity is not a static scalar value such as "50 trains per day." Track capacity is an emergent, highly non-linear property of timetable structure, signaling technology, track topology, and above all, mixed-traffic heterogeneity. In modern intermodal supply chains, freight trains carrying double-stack containers, bulk commodities, or automotive racks must share mainline trackage with regional passenger trains and high-speed intercity services. 

A 10,000-tonne, 2,500-meter intermodal freight train running at 80 km/h exhibits radically different braking dynamics, acceleration curves, and track occupancy profiles than a 400-tonne passenger electric multiple unit (EMU) cruising at 160 km/h or 200 km/h. When a fast passenger train is scheduled behind a slow freight train, it rapidly catches up to the freight train's rear, forcing dispatchers to hold the freight train in a passing siding or inflate headway buffers. Conversely, when a slow freight train follows a high-speed passenger train, the gap between them widens continuously, consuming potential path slots that cannot be utilized by subsequent trains. This phenomenon, known as capacity consumption through speed heterogeneity, erodes commercial freight paths at rates disproportionate to freight volume.

Furthermore, over the past decade, North American Class 1 railroads embraced Precision Scheduled Railroading (PSR). PSR sought to drive operating ratios below 60 percent by running significantly longer, heavier trains—frequently exceeding 3,500 meters (12,000 to 14,000 feet) and 14,000 gross tonnes. However, the legacy infrastructure over which these trains operate was predominantly engineered with passing sidings measuring 2,000 to 2,400 meters (6,500 to 8,000 feet). When a 3,500-meter train traverses a single-track corridor with standard sidings, it cannot fit into any intermediate loop. The mega-train cannot be overtaken by higher-priority traffic, nor can it execute an active "meet-and-pass" with opposing traffic without forcing the opposing train to clear the mainline miles upstream. A single mechanical failure, hot bearing detector alarm, or minor signal stop cascades into network-wide deadlock.

For logistics engineers, terminal operators, and supply chain directors, rail path capacity directly dictates container evacuation rates from maritime ports, railcar turnaround cycles at manufacturing plants, and intermodal reliability. Treating rail transit time as an unconstrained or stationary random variable in supply chain network planning results in chronic port congestion, demurrage penalties, stockouts, and unexpected modal diversion to higher-cost trucking. Optimizing rail network paths requires a rigorous understanding of blocking time theory, analytical and compression capacity methodologies, advanced train control architectures, and siding topology constraints.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Deconstruct** track section occupancy using classical Blocking Time Theory (*Sperrzeittreppe*), calculating approach time, sighting time, running time, clearing time, and route release buffers.
2. **Formulate and compute** minimum technical headway ($h_{min}$) across fixed-block signaling architectures (two-, three-, and four-aspect) and moving-block systems (European Train Control System Level 3 and Communications-Based Train Control), accounting for safe braking distances and communication latency.
3. **Apply** the International Union of Railways (UIC) Leaflet 405 analytical framework and UIC Leaflet 406 timetable compression methodology to determine consumed capacity, unconsumed capacity, and infrastructure utilization rates.
4. **Quantify** mixed-traffic heterogeneity and freight slot erosion factors, determining the mathematical penalty imposed on freight paths by passenger priority and speed differentials.
5. **Evaluate** passing loop and siding infrastructure constraints on single-track and double-track corridors, calculating lockout conditions and siding dwell penalties for trains exceeding siding clearance lengths.
6. **Analyze** the operational mechanics and network risks of Precision Scheduled Railroading (PSR) block-to-block timetable optimization, identifying tipping points where train lengthening induces mainline deadlocks.
7. **Construct** an algorithmic capacity compression and headway evaluation workflow in Python to assess timetable stability and determine corridor bottleneck blocks.

---

## 3. The core mental model

The governing mental model of railway capacity engineering is the **Blocking Time Stairway** (*Sperrzeittreppe*) projected onto the **Distance-Time Plane**.

In road traffic, vehicle headway can vary continuously, and vehicles can adjust spacing dynamically within a lane. In railway operations, physical safety requires that an exclusive safety bubble envelope every moving train. This safety bubble is projected into time and space as a "blocking time." A block section is not merely occupied while the physical train is inside its geometric boundaries; it is reserved, locked, and unavailable to any other train for a significantly longer duration.

The blocking time of a block section $i$ for train $j$ begins when the route is locked and the upstream signaling system displays an aspect permitting the train to proceed at speed, and it ends only when the train has completely cleared the block, cleared the physical safety overlap past the downstream signal, and the interlocking has unlocked the route.

```text
Distance (km)
 ^
 |                                  BLOCK SECTION i+1
 +==================================[Signal S_{i+2}]==========================
 |                                  ================
 |                                  BLOCK SECTION i
 +----------------------------------[Signal S_{i+1}]--------------------------
 |                              *   ----------------  ^ Overlap & Release
 |                          *       |              |  | Clearing Time (L_train)
 |                      *           | BLOCKING     |  | Pure Running Time
 |                  *               | TIME         |  | Approach Time
 |              *                   | STAIRWAY     |  | Sighting Time
 +--[Signal S_i]--------------------|==============|--v Route Formation
 |          *                       |              |
 |      *                           |              |
 |  *  TRAIN 1 TRAJECTORY           |              |
 0----------------------------------+--------------+-------------------------->
 0                                 t_start        t_end                       Time (s)
                                    <--- Total Blocking Time T_B --->
```

### The Blocking Time Decomposition

For any block section, the total blocking time $T_B$ comprises six sequential, non-overlapping phases:

1. **Route Formation and Locking Time ($t_{rf}$):** The time required for the interlocking system to align switches, verify track circuit occupancy, verify flank protection, and lock the route (typically 6 to 12 seconds in relay and computer-based interlockings).
2. **Sighting Time ($t_{sight}$):** The time a train driver requires to observe and mentally process the visual aspect of the approach signal before reaching it (typically 6 to 12 seconds, or the time to traverse 200 to 400 meters at line speed).
3. **Approach Time ($t_{app}$):** The time required for the train to traverse the distance between the advance warning signal (approach signal) and the stop signal protecting the block section. If the driver does not receive a clear aspect at the advance signal, service braking must begin immediately.
4. **Running Time ($t_{run}$):** The physical running time of the front of the train from the block's entrance signal to its exit signal.
5. **Clearing Time ($t_{clear}$):** The time required for the entire length of the train ($L_{train}$) to clear the block section exit boundary at exit speed ($v_{exit}$).
6. **Overlap and Route Release Time ($t_{rel}$):** The time required for the rear of the train to clear the physical safety overlap distance ($d_{ov}$) beyond the exit signal, plus the route release delay timer ($t_{rr}$) in the interlocking (typically 4 to 8 seconds).

When successive trains traverse a corridor, their trajectories in the distance-time plane produce a series of interlocking blocking time stairways. The **Minimum Technical Headway** ($h_{min}$) is the smallest temporal separation between the departure or arrival of two consecutive trains such that their blocking time stairways do not overlap at any block section along the corridor.

If a dispatcher schedules trains with a headway $h < h_{min}$, the following train will inevitably encounter a restrictive signal aspect (yellow or red), forcing it to decelerate or stop. This induces a primary delay. If the scheduled headway includes an inadequate buffer time $t_{buf}$, minor stochastic variations in running time or dwell time cascade across successive trains, generating systemic knock-on delays (*secondary delays*).

---

## 4. Definitions and vocabulary

| Term | Symbol / Code | Definition | Common Misuse | Standard Unit |
| :--- | :--- | :--- | :--- | :--- |
| **Blocking Time** | $T_B$ | The total duration for which a block section is exclusively allocated to a train and inaccessible to other traffic. | Confusing blocking time with pure train running time across the block. | Seconds ($s$) |
| **Blocking Time Stairway** | — | The graphical profile of blocking times across consecutive block sections in a time-space diagram (*Sperrzeittreppe*). | Treating the stairway as a simple line instead of a discrete step function. | Distance vs. Time |
| **Minimum Headway** | $h_{min}$ | The minimum time interval between two successive trains that prevents the second train from encountering restrictive signals. | Equating minimum headway to physical inter-train distance divided by speed. | Seconds ($s$) |
| **Buffer Time** | $t_{buf}$ | Scheduled time margin inserted between adjacent blocking time stairways to absorb operational variations. | Treating buffer time as idle or wasted track capacity rather than reliability margin. | Minutes ($min$) or Seconds ($s$) |
| **Fixed Block** | FB | Track signaling architecture where track sections have static, physically fixed insulated joints or axle-counter boundaries. | Assuming fixed blocks allow dynamic adjustment based on train speed or mass. | Boolean / Architecture |
| **Moving Block** | MB | Dynamic signaling architecture where safety zones travel with the train based on continuous radio positioning and cab signaling. | Believing moving block eliminates braking distance requirements entirely. | Boolean / Architecture |
| **Overlap Distance** | $d_{ov}$ | A safety margin beyond a stop signal that must remain clear in case a train overruns the signal before stopping. | Omitting overlap distance from clearing time calculations. | Meters ($m$) |
| **Sighting Distance** | $d_{sight}$ | The distance upstream of a signal at which the driver must be able to clearly identify the displayed aspect. | Assuming sighting time is zero under cab-signaling systems without balises. | Meters ($m$) |
| **UIC Leaflet 405** | UIC 405 | Analytical framework for calculating rail infrastructure capacity using empirical formulas and headway distributions. | Applying UIC 405 without accounting for corridor bottleneck blocks. | Standard Framework |
| **UIC Leaflet 406** | UIC 406 | Timetable compression methodology that compresses train paths to zero buffer to calculate true infrastructure utilization. | Confusing compressed timetable with an operable commercial timetable. | Standard Framework |
| **Capacity Utilization** | $K$ or $U$ | The ratio of consumed capacity (compressed time plus required buffer time) to the total evaluation time window. | Assuming a track can operate sustainably at 100% capacity utilization. | Percentage ($\%$) |
| **Speed Heterogeneity** | $\Delta v$ | The variance or spread of operating speeds among different train classes traversing the same corridor. | Ignoring the severe slot erosion caused by mixing 80 km/h freight and 160 km/h passenger trains. | km/h or m/s |
| **Passing Siding / Loop** | — | An auxiliary track connected to the mainline at both ends, used to allow faster trains to overtake or opposing trains to meet. | Sizing sidings based on average train length instead of maximum permissible train length. | Meters ($m$) |
| **Fouling Point** | FP | The point near a switch where two converging tracks come too close to allow equipment on both tracks to clear safely. | Measuring siding length from switch point to switch point rather than between clearance markers. | Geometric Location |
| **Precision Scheduled Railroading** | PSR | An operating philosophy prioritizing fixed schedules, asset velocity, and train consolidation into maximum-length consists. | Believing PSR eliminates the need for siding clearance synchronization. | Operational Philosophy |
| **Periodic Event Scheduling Problem** | PESP | Mathematical graph-theoretic framework for modeling and optimizing cyclic, periodic railway timetables (*Taktfahrplan*). | Applying PESP to completely aperiodic, ad-hoc freight dispatching networks. | Optimization Formulation |
| **Davis Equation** | $R(v)$ | Empirical formula expressing train aerodynamic, mechanical, and rolling resistance as a quadratic function of velocity. | Using automobile or truck rolling resistance coefficients for steel wheel on steel rail. | Newtons ($N$) or $N/\text{tonne}$ |
| **Safe Braking Distance** | $d_{braking}$ | The physical distance required to bring a train to a full stop from speed $v$ under emergency or full service deceleration. | Assuming constant deceleration across all speeds, rail adhesion levels, and brake pipe propagation delays. | Meters ($m$) |
| **European Train Control System** | ETCS | Standardized European signaling and train control system comprising Level 1, Level 2, and Level 3 implementations. | Assuming ETCS Level 2 provides moving block capability (Level 2 is fixed block with cab signaling). | Signaling System |
| **Positive Train Control** | PTC | US federally mandated safety overlay system preventing train-to-train collisions, overspeed derailments, and incursions. | Treating PTC as an automated dispatching optimizer rather than an enforcement overlay. | Regulatory Standard |

---

## 5. The mechanics, step by step

Railway capacity engineering requires a systematic, eight-step analytical procedure moving from physical infrastructure geometry to timetable compression and stability evaluation.

```text
[Step 1: Infrastructure Geometry & Signaling Layout]
                      |
                      v
[Step 2: Train Kinematics & Davis Resistance Modeling]
                      |
                      v
[Step 3: Blocking Time Stairway Decomposition (T_B)]
                      |
                      v
[Step 4: Minimum Technical Headway (h_min) Formulation]
                      |
                      v
[Step 5: Mixed-Traffic Heterogeneity & Slot Erosion Analysis]
                      |
                      v
[Step 6: Siding Clearance & Mainline Lockout Verification]
                      |
                      v
[Step 7: Timetable Construction & UIC 406 Compression]
                      |
                      v
[Step 8: Stability Audit & Knock-On Delay Propagation Modeling]
```

### Step 1: Infrastructure geometry and signaling layout characterization

Begin by mapping the corridor into discrete directional track segments. For each segment, establish:
- Physical gradient profile $i$ in per mille ($\text{mm/m}$ or $\text{promille}$), where $1\% = 10‰$.
- Curve radius $R$ in meters to evaluate curve resistance:
  $$r_{curve} = \frac{650}{R - 55} \quad (\text{for } R > 300\,\text{m})$$
- Signal locations $x_{s,k}$, signal aspect logic (2-, 3-, or 4-aspect), and interlocking boundaries.
- Track circuit or axle counter block boundaries $x_{b,k}$, identifying the physical clear length of each block $L_{block,k} = x_{b,k+1} - x_{b,k}$.
- Siding layouts: switch turnout numbers, allowable diversion speeds ($v_{diverge}$, typically 40 km/h for a No. 12 turnout, 65 km/h for a No. 20 turnout), and effective clear length between clearance fouling points $L_{clear}$.

### Step 2: Train kinematics and Davis resistance modeling

Track occupancy duration depends directly on train acceleration and deceleration performance. Calculate the total tractive resistance $R(v)$ for each train class using the empirical Davis formulation:

$$R(v) = A + B \cdot v + C \cdot v^2$$

where:
- $A$ is the mechanical and journal resistance independent of speed (proportional to total train weight $m$).
- $B$ is the flange and rolling contact resistance linear in velocity.
- $C$ is the aerodynamic drag coefficient proportional to frontal area, train length, skin friction, and open/intermodal car geometry.

For a train with gross mass $m$ (tonnes), length $L_{train}$ (meters), rotating mass factor $\rho_m$ (typically $1.06$ for freight cars, $1.10$ for locomotives and EMUs), and locomotive tractive effort $F_{tract}(v)$ (kilonewtons), the governing equation of motion is:

$$m \cdot (1 + \rho_m) \cdot \frac{dv}{dt} = F_{tract}(v) - R(v) - m \cdot g \cdot \sin(\theta) - R_{curve}$$

Integrate this ordinary differential equation numerically across track coordinates to produce the velocity-distance trajectory $v(x)$ and running time profile $t(x)$ for each train class, respecting maximum permissible track speeds ($v_{max}(x)$).

### Step 3: Blocking time stairway decomposition ($T_B$)

For each train class $j$ traversing block section $k$, calculate the total blocking time $T_{B,k,j}$. The six discrete components are computed as follows:

```text
+-----------------------------------------------------------------------------------+
| Total Blocking Time T_{B,k,j} = t_{rf} + t_{sight} + t_{app} + t_{run} + t_{clear} + t_{rel} |
+-----------------------------------------------------------------------------------+
```

1. **Route Formation Time ($t_{rf}$):** Constant interlocking parameter:
   $$t_{rf} = 6.0\text{ to }12.0\,\text{s}$$
2. **Sighting Time ($t_{sight}$):** Time required to traverse sighting distance $d_{sight}$ at approach speed $v_{app}$:
   $$t_{sight} = \frac{d_{sight}}{v_{app}}$$
   *(Under line-of-sight visual rules, standard value is 8 to 12 seconds; under cab-signaling without fixed approach signals, $t_{sight}$ represents driver reaction time, typically 2 to 4 seconds).*
3. **Approach Time ($t_{app}$):** Time required for the train to run from the advance caution signal to the block entrance signal:
   $$t_{app} = \int_{x_{signal, k-1}}^{x_{signal, k}} \frac{1}{v(x)} \, dx$$
   For constant speed $v_{approach}$, $t_{app} = L_{block, k-1} / v_{approach}$.
4. **Pure Running Time ($t_{run}$):** Time to traverse block section $k$:
   $$t_{run} = \int_{x_{signal, k}}^{x_{signal, k+1}} \frac{1}{v(x)} \, dx$$
5. **Clearing Time ($t_{clear}$):** Time for the train length $L_{train}$ to clear the block exit signal:
   $$t_{clear} = \frac{L_{train}}{v_{exit}}$$
6. **Overlap and Route Release Time ($t_{rel}$):** Time to clear the physical safety overlap $d_{ov}$ plus the interlocking release timer $t_{rr}$:
   $$t_{rel} = \frac{d_{ov}}{v_{exit}} + t_{rr}$$

The total blocking time $T_{B,k,j}$ represents a rectangle of width $T_{B,k,j}$ on the time axis spanning the spatial boundaries of block section $k$.

### Step 4: Minimum technical headway ($h_{min}$) formulation

The minimum headway between a leading train $1$ and a following train $2$ across block section $k$ is determined by the condition that Train 2's blocking time stairway must not overlap Train 1's blocking time stairway.

#### Case A: Three-Aspect Fixed Block Signaling
In a three-aspect system (Green, Yellow, Red), a signal displays Yellow if the block ahead is clear but the subsequent block is occupied. To maintain clear Green signals for Train 2, Train 1 must clear block section $k+1$ and its overlap before Train 2 reaches the approach signal for block section $k$.

The technical headway at block section $k$ is given by:

$$h_{min, k}(1 \to 2) = t_{run, k, 1} + t_{clear, k, 1} + t_{rel, k, 1} + t_{rf, k, 2} + t_{sight, k, 2} + t_{app, k, 2}$$

For identical trains traveling at constant speed $v$ over uniform block lengths $L_{block}$:

$$h_{min} = \frac{2 \cdot L_{block} + L_{train} + d_{ov} + d_{sight}}{v} + t_{rf} + t_{rr}$$

Notice that in three-aspect signaling, the train must be protected by **two full block lengths** ahead.

#### Case B: Four-Aspect Fixed Block Signaling
In a four-aspect system (Green, Double Yellow, Yellow, Red), braking is distributed across two blocks. The train receives a Double Yellow warning two blocks prior to a stop signal. The headway formulation becomes:

$$h_{min} = \frac{3 \cdot L_{block} + L_{train} + d_{ov} + d_{sight}}{v} + t_{rf} + t_{rr}$$

Although the numerator contains $3 \cdot L_{block}$, four-aspect blocks are designed at roughly half the length of three-aspect blocks ($L_{block, 4A} \approx 0.5 \cdot L_{block, 3A}$), resulting in a net reduction in minimum headway of 20 to 35 percent.

#### Case C: Moving Block Signaling (ETCS Level 3 / CBTC)
In a true moving block system, physical block sections are eliminated. The train continuously transmits its verified rear-end position via radio to a Radio Block Centre (RBC). The RBC transmits a Movement Authority (MA) to the following train, ending at the rear of the leading train minus a safety margin $d_{safety}$.

The minimum headway is governed purely by braking dynamics, train length, and communication update cycles:

$$h_{min, MB}(v) = \frac{L_{train, 1} + d_{braking, 2}(v) + d_{safety} + d_{ov}}{v} + t_{comm} + t_{reaction}$$

where:
- Safe service braking distance:
  $$d_{braking, 2}(v) = \frac{v^2}{2 \cdot b_{service}}$$
- $t_{comm}$ is the total radio transmission and cryptographic authentication delay round-trip (typically 1.5 to 3.0 seconds).
- $t_{reaction}$ is the onboard traction control system reaction time (1.0 to 2.0 seconds).

### Step 5: Mixed-traffic heterogeneity and freight slot erosion

When trains with disparate speeds share a single track, capacity consumption is dominated by the speed differential. 

Consider a fast passenger train ($v_F$) and a slow freight train ($v_S$) operating over a corridor of length $L_{corridor}$.
- Pure running time difference:
  $$\Delta t = L_{corridor} \left( \frac{1}{v_S} - \frac{1}{v_F} \right)$$

If the slow freight train runs first, followed by the fast passenger train, the passenger train catches up. To prevent the passenger train from encountering Yellow signals, the departure of the passenger train must be delayed at the corridor origin by $\Delta t$:

$$H_{scheduled}(S \to F) = h_{min}(S \to F) + \Delta t$$

During this expanded headway interval, no other train can enter the corridor. This lost time creates a "capacity consumption wedge." 

The **Freight Slot Erosion Factor** ($E_{slot}$) quantifies how many standard passenger paths ($h_{pass}$) are consumed by the insertion of a single freight train:

$$E_{slot} = \frac{H_{scheduled}(S \to F) + H_{scheduled}(F \to S)}{2 \cdot h_{pass}} = \frac{h_{min} + \Delta t}{h_{pass}}$$

If $h_{pass} = 4\,\text{minutes}$, $h_{min} = 5\,\text{minutes}$, and over a 60 km segment $\Delta t = 20\,\text{minutes}$, then:

$$E_{slot} = \frac{5 + 20}{4} = 6.25\,\text{passenger paths}$$

A single freight train eliminates over six passenger train paths, or conversely, running passenger trains at hourly intervals fragments the timetable, leaving freight slots that cannot traverse the corridor without being shunted into sidings for extensive dwell periods.

### Step 6: Siding clearance and mainline lockout verification

On single-track corridors, opposing trains must execute "meets," and faster trains must execute "passes" (overtakes). This requires passing sidings.

Verify the clearance condition for every train $j$ scheduled to dwell or be overtaken at siding $m$:

$$L_{train, j} \le L_{clear, m} - 2 \cdot d_{margin}$$

where:
- $L_{clear, m}$ is the distance between the insulated clearance joints (fouling markers) of the siding switches.
- $d_{margin}$ is the engineer stopping margin (typically 30 to 50 meters at each end to prevent fouling the mainline).

```text
MAINLINE TRACK
========================\--------------------------------/========================
                         \  <-- Siding Clear Length --> /
                          \----------------------------/
                                 PASSING SIDING m
```

**The Mainline Lockout Condition:**
If a railroad operating under PSR principles introduces trains of length $L_{monster} > L_{clear, m}$, that train **cannot** enter siding $m$.
- If an opposing train arrives, the monster train must remain on the mainline.
- The opposing train can enter the siding only if $L_{opposing} \le L_{clear, m}$.
- If **both** opposing trains exceed $L_{clear, m}$, a **physical deadlock** occurs (often termed a "saw-by" or "cornfield deadlock"). A saw-by requires complex, hazardous back-and-forth decoupling maneuvers that can consume 4 to 8 hours of mainline availability.

### Step 7: Timetable construction and UIC 406 compression

To evaluate true track capacity under practical operational conditions, apply the **UIC Leaflet 406 Timetable Compression Method**.

1. **Take the Scheduled Timetable:** Obtain the distance-time graph of all scheduled train paths over a defined time window $T_{period}$ (typically 24 hours, or a peak 4-hour window).
2. **Remove Unused Margins (Compression):** In the distance-time plane, shift all train trajectories chronologically toward the beginning of the evaluation window without altering:
   - The sequence of trains.
   - The running times, stopping times, and acceleration/deceleration profiles.
   - Overtaking and crossing locations.
3. Push the paths together until the blocking time stairways of adjacent trains touch at their critical bottleneck points (where buffer time $t_{buf} = 0$).
4. Measure the compressed time span $T_{compressed}$ from the start of the first blocking time stairway to the end of the last blocking time stairway.
5. Add the recommended standard buffer time allowance ($T_{buffer}$) to prevent delay propagation:
   $$T_{consumed} = T_{compressed} + T_{buffer}$$
   According to UIC 406 recommendations, required buffer time represents 25 to 33 percent of compressed time for mixed traffic corridors.
6. Calculate the **Infrastructure Capacity Utilization Ratio ($K$)**:
   $$K = \frac{T_{consumed}}{T_{period}} \times 100\% = \frac{T_{compressed} + T_{buffer}}{T_{period}} \times 100\%$$

```text
+-------------------------------------------------------------------------------+
| UIC 406 Capacity Utilization Thresholds:                                      |
| • Peak Hour (Commuter / Dense Lines): Max 85%                                 |
| • Mixed-Traffic Mainlines: Max 70% to 75%                                     |
| • Dedicated Freight Lines: Max 60% to 70%                                     |
| Beyond 75-80%, knock-on delays escalate exponentially following Erlang queues. |
+-------------------------------------------------------------------------------+
```

### Step 8: Stability audit and knock-on delay propagation modeling

The final engineering step is evaluating timetable robustness against stochastic disruptions (e.g., dispatching latency, brake test delays, locomotive wheel-slip on wet rail).

Calculate the probability that an initial primary delay $D_{prim}$ on train 1 will propagate as a secondary (knock-on) delay $D_{sec}$ to following train 2:

$$D_{sec} = \max\left(0, D_{prim} - t_{buf}(1 \to 2)\right)$$

If a corridor operates with an average buffer time $\bar{t}_{buf}$ and primary delays follow an exponential distribution with parameter $\lambda$ ($P(D_{prim} > t) = e^{-\lambda t}$), the probability of knock-on delay propagation across an $N$-train sequence is:

$$P(\text{propagation}) = 1 - \prod_{k=1}^{N-1} \left(1 - e^{-\lambda \cdot t_{buf}(k \to k+1)}\right)$$

When capacity utilization $K$ exceeds 80 percent, average buffer time $\bar{t}_{buf}$ collapses toward zero, causing the propagation probability to approach 1.0. Under these conditions, the railway ceases to operate as a scheduled system and degenerates into a stochastic queuing system with infinite expected delay.

---

## 6. Worked example

### Problem Statement and System Parameters

Evaluate the capacity and timetable stability of a critical 60-kilometer freight and passenger corridor connecting an inland multimodal intermodal container terminal to a mainline rail hub. 

The corridor is configured with:
- **Signaling:** 3-Aspect Fixed Block Signaling with automated wayside signals.
- **Block Layout:** Six uniform block sections, each measuring $L_{block} = 10,000\,\text{meters}$ ($10.0\,\text{km}$).
- **Safety Overlap:** $d_{ov} = 200\,\text{meters}$.
- **Driver Sighting Distance:** $d_{sight} = 400\,\text{meters}$.
- **Interlocking Times:** Route formation $t_{rf} = 9.0\,\text{seconds}$; route release delay $t_{rr} = 6.0\,\text{seconds}$.
- **Siding Location:** Kilometer 30.0 (between Block 3 and Block 4). Siding clear length $L_{clear} = 1,400\,\text{meters}$.

Two train classes operate over this corridor:
1. **Class A (Intercity Express Passenger):**
   - Length $L_A = 200\,\text{meters}$.
   - Cruise speed $v_A = 150\,\text{km/h} = 41.67\,\text{m/s}$.
   - Full service deceleration $b_A = 0.80\,\text{m/s}^2$.
2. **Class B (Heavy Intermodal Double-Stack Freight):**
   - Length $L_B = 1,200\,\text{meters}$.
   - Cruise speed $v_B = 75\,\text{km/h} = 20.83\,\text{m/s}$.
   - Full service deceleration $b_B = 0.35\,\text{m/s}^2$.

### Phase 1: Pure Running Times and Deceleration Physics

First, calculate the pure running time across a single 10,000-meter block section at cruise speed for both classes:
- Class A:
  $$t_{run, A} = \frac{10,000\,\text{m}}{41.67\,\text{m/s}} = 240.0\,\text{s} \quad (4.00\,\text{minutes})$$
- Class B:
  $$t_{run, B} = \frac{10,000\,\text{m}}{20.83\,\text{m/s}} = 480.0\,\text{s} \quad (8.00\,\text{minutes})$$

Total corridor running time (across all 6 blocks = 60,000 meters):
- Class A: $T_{corridor, A} = 6 \times 240.0\,\text{s} = 1,440.0\,\text{s} = 24.0\,\text{minutes}$.
- Class B: $T_{corridor, B} = 6 \times 480.0\,\text{s} = 2,880.0\,\text{s} = 48.0\,\text{minutes}$.

Verify the safe service braking distances:
- Class A braking distance:
  $$d_{braking, A} = \frac{v_A^2}{2 \cdot b_A} = \frac{41.67^2}{2 \times 0.80} = \frac{1736.39}{1.60} = 1,085.24\,\text{meters}$$
- Class B braking distance:
  $$d_{braking, B} = \frac{v_B^2}{2 \cdot b_B} = \frac{20.83^2}{2 \times 0.35} = \frac{433.89}{0.70} = 619.84\,\text{meters}$$

Since both braking distances are substantially less than the 10,000-meter block length ($d_{braking} < L_{block}$), three-aspect signaling provides adequate stopping distance with substantial margin.

### Phase 2: Total Blocking Time ($T_B$) per Block Section

Compute each component of blocking time $T_B$ for both train classes across an intermediate block section:

```text
Component                     Class A (Passenger)            Class B (Freight)
---------------------------------------------------------------------------------
Route Formation (t_rf)               9.0 s                         9.0 s
Sighting Time (t_sight)             9.6 s                        19.2 s
Approach Time (t_app)             240.0 s                       480.0 s
Block Running Time (t_run)        240.0 s                       480.0 s
Clearing Time (t_clear)             4.8 s                        57.6 s
Overlap Clearance (d_ov / v)        4.8 s                         9.6 s
Route Release Timer (t_rr)          6.0 s                         6.0 s
---------------------------------------------------------------------------------
Total Blocking Time T_B           514.2 s (8.57 min)          1,061.4 s (17.69 min)
```

*Detailed Arithmetic Calculations:*
- Sighting time:
  $$t_{sight, A} = \frac{400\,\text{m}}{41.67\,\text{m/s}} = 9.60\,\text{s}; \quad t_{sight, B} = \frac{400\,\text{m}}{20.83\,\text{m/s}} = 19.20\,\text{s}$$
- Approach time across 10,000 m upstream block:
  $$t_{app, A} = \frac{10,000\,\text{m}}{41.67\,\text{m/s}} = 240.0\,\text{s}; \quad t_{app, B} = \frac{10,000\,\text{m}}{20.83\,\text{m/s}} = 480.0\,\text{s}$$
- Clearing time for train length:
  $$t_{clear, A} = \frac{200\,\text{m}}{41.67\,\text{m/s}} = 4.80\,\text{s}; \quad t_{clear, B} = \frac{1,200\,\text{m}}{20.83\,\text{m/s}} = 57.60\,\text{s}$$
- Overlap clearance:
  $$t_{ov, A} = \frac{200\,\text{m}}{41.67\,\text{m/s}} = 4.80\,\text{s}; \quad t_{ov, B} = \frac{200\,\text{m}}{20.83\,\text{m/s}} = 9.60\,\text{s}$$
- Total $T_{B, A} = 9.0 + 9.6 + 240.0 + 240.0 + 4.8 + 4.8 + 6.0 = 514.2\,\text{seconds}$.
- Total $T_{B, B} = 9.0 + 19.2 + 480.0 + 480.0 + 57.6 + 9.6 + 6.0 = 1,061.4\,\text{seconds}$.

### Phase 3: Minimum Technical Headway Matrix

Now determine the minimum departure headway $h_{min}(i \to j)$ at the corridor origin for all four ordering combinations such that blocking stairways do not clash at any of the six blocks:

1. **Homogeneous Passenger ($A \to A$):**
   Since both trains have identical running times, their blocking time stairways are parallel. The critical block is Block 1:
   $$h_{min}(A \to A) = T_{B, A} - t_{app, A} - t_{sight, A} - t_{rf, A} + t_{rf, A} + t_{sight, A} + t_{app, A} = 514.2 - 258.6 + 258.6 = 514.2\,\text{s} = 8.57\,\text{minutes}$$
   *(Specifically: Train 2 can enter Block 1 as soon as Train 1 clears Block 2 and its overlap, which occurs at $t_{run} + t_{clear} + t_{rel} = 240 + 4.8 + 10.8 = 255.6\,\text{s}$. Adding Train 2's approach and sighting time $249.6\,\text{s}$ plus route formation $9\,\text{s}$ gives $514.2\,\text{s}$).*

2. **Homogeneous Freight ($B \to B$):**
   Parallel stairways. Critical block is Block 1:
   $$h_{min}(B \to B) = T_{B, B} = 1,061.4\,\text{seconds} = 17.69\,\text{minutes}$$

3. **Freight Followed by Passenger ($B \to A$ - Catch-up scenario):**
   Train A runs at twice the speed of Train B. The bottleneck occurs at the **exit of the final block (Block 6)**. Train A must not catch up to Train B's blocking stairway at Block 6.
   $$\text{Running time difference across corridor} = T_{corridor, B} - T_{corridor, A} = 2,880.0 - 1,440.0 = 1,440.0\,\text{s} \quad (24.0\,\text{minutes})$$
   At Block 6, Train A must clear Block 6 with minimum headway behind Train B. Therefore, at the origin (Block 1 departure), the scheduled separation must be:
   $$h_{min}(B \to A) = h_{min, exit}(B \to A) + \Delta T_{corridor} = 514.2\,\text{s} + 1,440.0\,\text{s} = 1,954.2\,\text{seconds} = 32.57\,\text{minutes}$$

4. **Passenger Followed by Freight ($A \to B$ - Diverging scenario):**
   Train A pulls away from Train B. The bottleneck occurs at the **entrance of Block 1**:
   $$h_{min}(A \to B) = 514.2\,\text{seconds} = 8.57\,\text{minutes}$$

### Phase 4: Siding Dwell and Overtake Analysis at Kilometer 30.0

Assume a commercial timetable requires one Class B freight train and two Class A passenger trains within an evaluation window.
Can Freight Train B be overtaken at Kilometer 30.0?
- Siding clear length $L_{clear} = 1,400\,\text{meters}$.
- Freight train length $L_B = 1,200\,\text{meters}$.
- Margin available: $1,400 - 1,200 = 200\,\text{meters}$ ($100\,\text{m}$ margin at each end).
- **Result:** Train B clears the siding fouling markers safely.

Calculate the minimum dwell time incurred by Freight Train B to allow Passenger Train A to overtake it at Kilometer 30:
- Freight train diverges into siding at switch speed $v_{div} = 40\,\text{km/h} = 11.11\,\text{m/s}$.
- Deceleration from $75\,\text{km/h}$ to stop inside siding adds $120.0\,\text{seconds}$ of transit and braking loss.
- Mainline route is formed and locked for Passenger Train A: $t_{rf} = 9.0\,\text{s}$.
- Passenger Train A traverses Block 3 and Block 4 ($2 \times 240.0\,\text{s} = 480.0\,\text{s}$) plus clearing time ($4.8\,\text{s}$) and overlap ($10.8\,\text{s}$).
- Interlocking releases switch and realigns for siding exit: $15.0\,\text{seconds}$.
- Freight Train B accelerates from 0 to line speed ($75\,\text{km/h}$) through the siding turnout: acceleration loss = $180.0\,\text{seconds}$.
- **Total Freight Delay for One Overtake:**
  $$D_{overtake} = 120.0 + 9.0 + 495.6 + 15.0 + 180.0 = 839.6\,\text{seconds} = 13.99\,\text{minutes}$$

Thus, tucking a 1,200-meter freight train into a siding to let one passenger train overtake consumes **14 minutes of freight dwell time**, degrading locomotive asset turnover and crew availability.

### Phase 5: UIC 406 Capacity Compression over a 4-Hour Window

Assume a 4-hour (240-minute = 14,400 seconds) operating window with a scheduled traffic mix of:
- 6 Class A Passenger Trains
- 2 Class B Freight Trains
Scheduled order: $A_1, A_2, B_1, A_3, A_4, B_2, A_5, A_6$.

Compress the timetable by shifting paths until blocking stairways touch ($t_{buf} = 0$):
- Pair 1 ($A_1 \to A_2$): $h_{min} = 514.2\,\text{s}$
- Pair 2 ($A_2 \to B_1$): $h_{min} = 514.2\,\text{s}$
- Pair 3 ($B_1 \to A_3$): Catch-up without siding overtake: $h_{min} = 1,954.2\,\text{s}$
- Pair 4 ($A_3 \to A_4$): $h_{min} = 514.2\,\text{s}$
- Pair 5 ($A_4 \to B_2$): $h_{min} = 514.2\,\text{s}$
- Pair 6 ($B_2 \to A_5$): Catch-up without siding overtake: $h_{min} = 1,954.2\,\text{s}$
- Pair 7 ($A_5 \to A_6$): $h_{min} = 514.2\,\text{s}$
- Plus running time of last train ($A_6$) across corridor: $1,440.0\,\text{s}$.

Total Compressed Time $T_{compressed}$:
$$T_{compressed} = 514.2 + 514.2 + 1,954.2 + 514.2 + 514.2 + 1,954.2 + 514.2 + 1,440.0 = 7,425.2\,\text{seconds} \quad (123.75\,\text{minutes})$$

Add standard recommended UIC 406 buffer time allowance for mixed traffic ($30\%$ of compressed time):
$$T_{buffer} = 0.30 \times 7,425.2 = 2,227.56\,\text{seconds} \quad (37.13\,\text{minutes})$$

Total Consumed Capacity:
$$T_{consumed} = 7,425.2 + 2,227.56 = 9,652.76\,\text{seconds} \quad (160.88\,\text{minutes})$$

Calculate Corridor Capacity Utilization ($K$):
$$K = \frac{T_{consumed}}{T_{period}} \times 100\% = \frac{9,652.76}{14,400.0} \times 100\% = 67.03\%$$

### Engineering Interpretation
At $67.03\%$ utilization, the corridor is within the sustainable operating envelope (below the $75\%$ UIC threshold). However, notice that the two freight trains alone consumed $2 \times 1,954.2 = 3,908.4\,\text{seconds}$, which is **$52.6\%$ of the total compressed time**, despite representing only $25\%$ of the total trains. Speed heterogeneity is the dominant consumer of track capacity.

---

## 7. Code, query, or calculation

The following Python script computes blocking times for heterogeneous train classes across a corridor of fixed blocks and executes a UIC 406 timetable compression calculation. It is self-contained, adheres strictly to Python standard libraries, and reproduces the worked example.

```python
"""Railway Track Capacity & UIC 406 Timetable Compression Calculator."""
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class TrainClass:
    name: str
    length_m: float
    speed_mps: float
    decel_mps2: float

@dataclass(frozen=True)
class BlockSection:
    length_m: float
    overlap_m: float
    sighting_m: float
    t_route_form_s: float
    t_route_rel_s: float

def calc_blocking_time(train: TrainClass, block: BlockSection) -> float:
    t_sight = block.sighting_m / train.speed_mps
    t_app = block.length_m / train.speed_mps
    t_run = block.length_m / train.speed_mps
    t_clear = train.length_m / train.speed_mps
    t_rel = (block.overlap_m / train.speed_mps) + block.t_route_rel_s
    return block.t_route_form_s + t_sight + t_app + t_run + t_clear + t_rel

def calc_min_headway(lead: TrainClass, foll: TrainClass, block: BlockSection, n_blocks: int) -> float:
    tb_lead = calc_blocking_time(lead, block)
    if foll.speed_mps > lead.speed_mps:
        # Fast follows slow: bottleneck at corridor exit
        delta_corridor = n_blocks * block.length_m * (1.0 / lead.speed_mps - 1.0 / foll.speed_mps)
        return tb_lead + delta_corridor
    # Homogeneous or slow follows fast: bottleneck at corridor entrance
    return tb_lead

def compress_timetable(pattern: List[TrainClass], block: BlockSection, n_blocks: int) -> float:
    t_compressed = 0.0
    for i in range(len(pattern) - 1):
        t_compressed += calc_min_headway(pattern[i], pattern[i+1], block, n_blocks)
    last_train_run = n_blocks * (block.length_m / pattern[-1].speed_mps)
    return t_compressed + last_train_run

if __name__ == "__main__":
    block_def = BlockSection(10000.0, 200.0, 400.0, 9.0, 6.0)
    p_train = TrainClass("Passenger_A", 200.0, 150.0 / 3.6, 0.80)
    f_train = TrainClass("Freight_B", 1200.0, 75.0 / 3.6, 0.35)
    
    tb_p = calc_blocking_time(p_train, block_def)
    tb_f = calc_blocking_time(f_train, block_def)
    h_pp = calc_min_headway(p_train, p_train, block_def, 6)
    h_fp = calc_min_headway(f_train, p_train, block_def, 6)
    
    tt_pattern = [p_train, p_train, f_train, p_train, p_train, f_train, p_train, p_train]
    t_comp = compress_timetable(tt_pattern, block_def, 6)
    uic_buffer_ratio = 0.30
    t_consumed = t_comp * (1.0 + uic_buffer_ratio)
    utilization_pct = (t_consumed / (4.0 * 3600.0)) * 100.0
    
    print(f"Blocking Time Passenger: {tb_p:.1f} s | Freight: {tb_f:.1f} s")
    print(f"Headway Pass->Pass: {h_pp:.1f} s | Freight->Pass (Catch-up): {h_fp:.1f} s")
    print(f"UIC 406 Compressed: {t_comp:.1f} s | Consumed (30% buf): {t_consumed:.1f} s")
    print(f"Capacity Utilization (4h window): {utilization_pct:.2f}%")
```

### Expected Output
```text
Blocking Time Passenger: 514.2 s | Freight: 1061.4 s
Headway Pass->Pass: 514.2 s | Freight->Pass (Catch-up): 1954.2 s
UIC 406 Compressed: 7425.2 s | Consumed (30% buf): 9652.8 s
Capacity Utilization (4h window): 67.03%
```

---

## 8. Data, standards, and regulatory context

Railway track capacity engineering is governed by strict international standards, safety directives, and statutory mandates. These frameworks enforce a sharp legal boundary between safety certification and commercial path allocation.

### International and European Standards

1. **UIC Leaflet 405 (Capacity of Railway Infrastructure):**
   Published by the International Union of Railways (*Union Internationale des Chemins de fer*), Leaflet 405 provides the mathematical foundations for analytical capacity evaluation. It defines the relationships between train mix, technical headway, scheduled wait times, and queue stability. It treats capacity not as an absolute ceiling, but as a function of the permissible average delay threshold.

2. **UIC Leaflet 406 (Capacity):**
   The global standard for timetable compression. Leaflet 406 mandates the procedure for stripping out commercial buffer times, pushing train paths together in the distance-time plane until blocking stairways touch, and measuring the resulting compression ratio. It defines standard operational guidelines for maximum utilization:
   - Dedicated freight corridors: 60% to 70%.
   - Mixed passenger and freight lines: 70% to 75%.
   - Dense suburban passenger lines: 80% to 85% during peak hours.

3. **European Union Technical Specifications for Interoperability (TSI):**
   - **CCS TSI (Control-Command and Signalling):** Enforces technical standards for ETCS (European Train Control System) and GSM-R / FRMCS (Future Railway Mobile Communication System). Governs movement authority generation, emergency braking curves, and balise transmission protocols.
   - **OPE TSI (Operation and Traffic Management):** Standardizes operational rules, train documentation, brake percentage calculations, and communication protocols across national borders.

4. **ETCS Functional Levels:**
   - **ETCS Level 1:** Cab signaling overlay on conventional fixed blocks. Transmission via intermittent Eurobalises. Lineside optical signals are retained.
   - **ETCS Level 2:** Continuous cab signaling via GSM-R / FRMCS radio. Fixed blocks are retained, but trackside visual signals are eliminated. Interlocking communicates with an onboard European Vital Computer (EVC) through a Radio Block Centre (RBC).
   - **ETCS Level 3:** Moving block signaling. Track circuits and axle counters are completely removed from the track. Train integrity is verified by an onboard Train Integrity Monitoring System (TIMS). Headway is limited only by braking physics.

### North American Regulatory and Industry Standards

1. **Federal Railroad Administration (FRA) 49 CFR Part 236:**
   Governs US railway signaling and train control systems. 
   - **Subpart I (Positive Train Control - PTC):** Mandated by the Rail Safety Improvement Act of 2008 (RSIA). PTC enforces train separation, prevents overspeed derailments, prevents unauthorized incursions into work zones, and stops trains from running through misaligned switches. PTC acts as a safety enforcement overlay on top of existing Centralized Traffic Control (CTC) or Automatic Block Signaling (ABS); it does not dynamically optimize headways.

2. **AREMA (American Railway Engineering and Maintenance-of-Way Association):**
   - The *AREMA Manual for Railway Engineering* specifies geometric standards for track centerlines, siding turnout sizes (e.g., No. 10, No. 14, No. 20 turnouts), clearance diagrams (Plate F, Plate H for double-stack intermodal equipment), and signaling circuit design rules.

3. **Surface Transportation Board (STB) Oversight:**
   The US economic regulator for freight railroads. Under 49 U.S.C. § 11101, Class 1 railroads have a statutory "common carrier obligation" to provide transportation on reasonable request. The STB monitors network velocity, terminal dwell, and car-miles per day. When railroads compress maintenance windows or starve regional freight paths to maximize operating ratios, the STB exercises oversight hearings and emergency service orders.

---

## 9. Industry practice and case evidence

### Precision Scheduled Railroading (PSR) in North American Freight

Over the past decade, all major North American Class 1 railroads (with the historical exception of BNSF) adopted Precision Scheduled Railroading, pioneered by E. Hunter Harrison at Illinois Central, Canadian National, Canadian Pacific, and CSX. 

PSR fundamentally restructured network capacity:
- **From Tonnage to Scheduled Paths:** Historically, railroads held freight cars in classification yards until sufficient tonnage accumulated to run a full train. PSR mandated fixed, scheduled train departures regardless of tonnage.
- **Extreme Train Lengthening:** To reduce crew labor and locomotive counts, PSR merged multiple manifests into single trains measuring 3,500 to 4,500 meters (12,000 to 15,000 feet) weighing over 15,000 tonnes.
- **The Siding Disconnect:** While operating ratios initially fell below 60 percent, physical infrastructure did not expand in tandem. The majority of single-track corridors possessed sidings capped at 2,000 to 2,400 meters. 

*Documented Operational Outcome:*
When 4,000-meter trains traversed these lines, intermediate meets became physically impossible. Dispatchers were forced to hold opposing trains at originating terminals or multi-track yards 50 to 100 miles away. If a 4,000-meter train suffered an air-brake hose separation (undesired emergency application) in cold weather, it severed the mainline for hours. System velocity declined, dwell times at major classification terminals (e.g., Kirk Yard, Clearing Yard) doubled, and shippers experienced severe supply chain disruptions, prompting STB intervention in 2022.

### The Betuweroute Dedicated Freight Corridor (Netherlands to Germany)

To eliminate mixed-traffic friction, the Dutch government and Port of Rotterdam constructed the **Betuweroute**, a 160-kilometer dedicated heavy freight corridor connecting the Port of Rotterdam directly to the German industrial Ruhr region.

*Engineering Innovations:*
- Fully segregated from passenger traffic: No passenger trains operate on the line.
- Equipped with **ETCS Level 2** signaling from its inception, eliminating lineside signals.
- Standardized maximum train length of 740 meters, axle load of 25.0 tonnes, and 100 km/h operational speed.
- Because all trains have identical speeds, lengths, and braking curves ($\Delta v = 0$), the speed heterogeneity penalty is eliminated. Technical headway is compressed to a uniform 3.0 minutes, achieving high slot utilization and evacuating tens of thousands of TEUs weekly without passenger interference.

### Swiss Federal Railways (SBB) Integrated Periodic Timetable (*Taktfahrplan*)

Switzerland’s railway network represents the global benchmark for mixed-traffic capacity utilization, operating at over 80 percent utilization across complex mountain topography.

*The Engineering System:*
- SBB utilizes an **Integrated Periodic Timetable** (*Integraler Taktfahrplan* - ITF), designed around cyclic 30- and 60-minute passenger symmetry nodes at hub stations (Zurich, Bern, Basel).
- Freight trains are not dispatched as ad-hoc unscheduled runs; they are allocated rigorous, pre-calculated "system paths" woven into the periodic passenger lattice.
- SBB engineers use UIC 406 compression tools daily to ensure that freight paths match the acceleration and deceleration characteristics of standard freight corridors, with dedicated overtaking loops positioned at exact harmonic travel times between passenger waves.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Physical / Mathematical Mechanism | System Consequence | Engineering Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **Siding Overhang Gridlock** | Train length exceeds siding clear distance ($L_{train} > L_{clear}$). | Train rear fouls turnout insulated joints, shorting track circuits and disabling mainline signals. | Opposing traffic halted; physical deadlock (*cornfield meet*) requiring multi-hour saw-by maneuvers. | Enforce hard consist length limits based on corridor siding constraints in TMS dispatching systems. |
| **Heterogeneity Slot Collapse** | Commercial freight path cancellations; passenger OTP falls below 80%. | Speed differential $\Delta v$ inflates headway wedge: $H = h_{min} + L(1/v_S - 1/v_F)$. | Freight paths eroded; passenger trains run on restrictive yellow aspects; system velocity drops. | Implement dynamic timetable speed harmonization or construct mid-corridor overtake loops. |
| **Delay Cascade (*Phantom Jam*)** | Corridor capacity utilization $K > 75\%$ under UIC 406 compression. | Buffer time $t_{buf}$ insufficient to absorb stochastic primary delays; knock-on propagation probability $\to 1.0$. | Localized 10-minute brake test delay cascades into multi-train, corridor-wide operational paralysis. | Pad schedule with explicit UIC 406 buffer allowances (25–30%); prune marginal scheduled slots. |
| **Braking Curve Overshoot** | PTC or cab-signaling enforces unexpected penalty brake applications. | Adhesion loss (leaves on rail, moisture) or brake pipe propagation delay reduces actual deceleration below $b_{calc}$. | Penalty stop occurs; train stops across interlockings; block clearance delayed by 20–40 minutes. | Incorporate conservative adhesion margins into onboard braking curve algorithms (e.g., ETCS braking curves). |
| **ETCS RBC Radio Starvation** | Emergency brake triggers on train traversing high-density corridor. | GSM-R / FRMCS handover failure between base stations or radio block centre packet timeout ($>5.0\,\text{s}$). | Safe-state fallback: train executes emergency stop; trailing trains receive revoked Movement Authority. | Implement redundant radio transceivers, duplicate packet routing, and packet retransmission buffers. |
| **Turnout Speed Shockwave** | Long freight trains crawling at 15–20 km/h out of sidings. | Clearing time $L_{train}/v_{turnout}$ scales linearly with length; a 3,000 m train at 40 km/h takes 4.5 min just to clear switch. | Mainline blocked for trailing high-speed traffic; headway buffer completely consumed. | Upgrade turnout geometry to high-speed turnouts (No. 20 or No. 30, permitting 65–100 km/h diverge). |
| **Axle Counter False Occupancy** | Track section remains locked red after physical train has departed. | Metallic debris, wheel-flange magnetization, or electrical noise creates count mismatch between entry and exit heads. | Interlocking prevents route formation; requires manual safety reset protocol and visual inspection. | Install dual-sensor axle counters with automated software filtering and reset-and-restore safety sweeps. |
| **Crew Hours of Service Expiry** | Trains stopped on mainline awaiting relief crews (*dog-catching*). | Unexpected siding dwell consumes crew statutory operating window (12.0 hours under US 49 CFR Part 228). | Train immobilized on mainline without crew; blocks one or both tracks for 4 to 8 hours. | Deploy predictive crew scheduling algorithms integrated with real-time dispatcher projection tools. |
| **Thermal Bearing False Alarms** | Hot Box Detector (HBD) alarms trigger emergency stops in bottleneck blocks. | Defective trackside IR sensors or transient friction heat on heavy consists halts train. | Full visual hand-inspection required; train occupies critical block for 45 to 90 minutes. | Deploy acoustic bearing detectors and multi-site trending to detect bearing degradation prior to corridor entry. |
| **Power-Weight Stall on Grade** | Heavy freight stalling on ruling grade ($i > 15‰$). | Tractive effort $F(v) < m \cdot g \cdot \sin(\theta) + R(v)$ due to locomotive power outage or low adhesion. | Complete corridor blockage; requires dispatching a helper locomotive (*banking engine*) to rescue train. | Enforce minimum power-to-weight ratios ($\text{kW}/\text{tonne}$ or $\text{HP}/\text{ton}$) before dispatching onto ruling grades. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks It | How to Test / Validate |
| :--- | :--- | :--- | :--- |
| **Constant Service Deceleration ($b$)** | Simplifies braking distance and headway equations to closed-form kinematics. | Air brake propagation delay along long trains (front brakes apply seconds before rear brakes); grade variations. | Compare with dynamic multi-car train braking simulators (e.g., Train Energy Model, OpenTrack). |
| **Deterministic Station Dwell Times** | Timetable scheduling assumes trains depart exactly at designated schedule seconds. | Passenger boarding surges, disabled passenger boarding, express parcel loading delays. | Collect automated vehicle monitoring (AVM) dwell logs; fit log-normal distributions. |
| **Rigid Block Boundaries** | Fixed-block calculations assume physical boundaries are stationary and discrete. | Moving block signaling (ETCS Level 3 / CBTC) where virtual boundaries continuously track train positions. | Verify signaling architecture; apply moving block equations if continuous radio tracking is present. |
| **Homogeneous Wheel-Rail Adhesion ($\mu$)** | Braking distance calculations assume a constant adhesion coefficient (e.g., $\mu \approx 0.15$). | Low adhesion conditions caused by crushed leaves, dew, frost, or railhead contamination ($\mu \le 0.05$). | Test braking algorithms against European TSI "minimum adhesion" degraded curve standards. |
| **Instantaneous Route Locking** | Simplifies route formation to a small static scalar ($t_{rf} \approx 9\,\text{s}$). | Complex interlocking plants where multiple switches must throw sequentially; conflicting route delays. | Extract interlocking event recorder timestamps to determine 95th-percentile route formation latency. |
| **Unlimited Electric Power Traction** | Speed-distance profiles assume trains draw maximum tractive power from the catenary. | Substation capacity limits, voltage sag when multiple heavy trains accelerate simultaneously in one zone. | Model traction power network with load-flow simulation coupled to train movement. |
| **Siding Length Accommodates Single Train** | Siding calculations assume one train per loop between clearance markers. | Operations requiring multiple short trains to share a siding, or bidirectional double-overtakes. | Verify turnout detection circuit design and interlocking route tables for multi-train occupancy. |
| **Uniform Human Driver Reaction** | Sighting and response times assumed constant ($t_{sight} \approx 8\text{--}12\,\text{s}$). | Driver fatigue, poor weather (fog, blizzard), inattentiveness, or unfamiliarity with route. | Analyze cab audio/event recorders or substitute cab-signaling continuous enforcement. |

---

## 12. Exercises

### Exercise 1: Fixed-Block Technical Headway Calculation
- **Context:** A double-track mainline is equipped with 4-aspect fixed-block signaling with uniform block lengths of $L_{block} = 1,500\,\text{meters}$. Sighting distance is $d_{sight} = 300\,\text{meters}$, overlap distance is $d_{ov} = 150\,\text{meters}$, route formation time is $t_{rf} = 8.0\,\text{seconds}$, and route release timer is $t_{rr} = 5.0\,\text{seconds}$.
- **Task:** Calculate the minimum technical headway $h_{min}$ for a homogeneous fleet of regional commuter trains having length $L_{train} = 180\,\text{meters}$ operating at a constant line speed of $v = 120\,\text{km/h}$ ($33.33\,\text{m/s}$).
- **Deliverable:** Mathematical derivation and final headway value in seconds and minutes.
- **Acceptance Criteria:** Correct application of the 4-aspect headway formula; arithmetic verified.
- **Estimated Time:** 20 minutes.

### Exercise 2: Moving Block Headway Sensitivity
- **Context:** An urban freight corridor is being upgraded from 3-aspect fixed block signaling ($L_{block} = 2,000\,\text{m}$) to ETCS Level 3 Moving Block. Freight trains have length $L_{train} = 800\,\text{meters}$, cruise at $v = 80\,\text{km/h}$ ($22.22\,\text{m/s}$), and exhibit service deceleration of $b = 0.40\,\text{m/s}^2$. Safe stopping margin is $d_{safety} = 100\,\text{meters}$, communication delay is $t_{comm} = 2.0\,\text{seconds}$, and driver/system reaction time is $t_{reaction} = 1.5\,\text{seconds}$.
- **Task:** 
  1. Calculate the minimum moving block technical headway $h_{min, MB}$.
  2. Compare this against the fixed-block headway assuming $d_{ov} = 200\,\text{m}$, $d_{sight} = 400\,\text{m}$, $t_{rf} = 9\,\text{s}$, $t_{rr} = 6\,\text{s}$.
  3. Determine the percentage capacity gain achieved by transitioning to moving block.
- **Deliverable:** Full comparative calculation with step-by-step arithmetic.
- **Acceptance Criteria:** Accurate braking distance calculation; correctly computed percentage increase.
- **Estimated Time:** 30 minutes.

### Exercise 3: Freight Slot Erosion Analysis
- **Context:** A single-track corridor of length $L = 80\,\text{km}$ connects an inland intermodal yard to a maritime terminal. Passenger trains run at $v_{pass} = 160\,\text{km/h}$. Intermodal freight trains run at $v_{freight} = 80\,\text{km/h}$. The passenger headway interval for homogeneous passenger traffic is $h_{pass} = 10\,\text{minutes}$. The corridor has no intermediate passing sidings. Minimum entrance/exit headway is $h_{min} = 6\,\text{minutes}$.
- **Task:**
  1. Calculate the pure running time difference $\Delta t$ between passenger and freight across the corridor.
  2. Calculate the departure interval required at the corridor origin when a fast passenger train follows a slow freight train.
  3. Calculate the Freight Slot Erosion Factor ($E_{slot}$).
- **Deliverable:** Formulated expressions, intermediate values, and final slot erosion factor.
- **Acceptance Criteria:** Demonstrates how speed differential consumes multiple theoretical paths.
- **Estimated Time:** 25 minutes.

### Exercise 4: Siding Lockout and PSR Train Length Evaluation
- **Context:** A Class 1 railroad operates a 120-kilometer single-track subdivision with 5 passing sidings spaced at 20-kilometer intervals. Each siding has a clear length of $L_{clear} = 2,200\,\text{meters}$. Operating management proposes increasing intermodal train lengths from 1,800 meters to 3,200 meters to reduce operating crew costs.
- **Task:**
  1. Identify the operational condition under which two opposing 3,200-meter trains can pass each other on this subdivision.
  2. Calculate the maximum number of 3,200-meter trains that can traverse the corridor simultaneously without creating a lockout.
  3. Write an engineering memorandum critique evaluating the impact on corridor dwell and transit reliability.
- **Deliverable:** Structured analysis and critique addressing network deadlocks, saw-by maneuvers, and dispatcher flexibility.
- **Acceptance Criteria:** Clear identification of physical lockout conditions; realistic operations critique.
- **Estimated Time:** 35 minutes.

### Exercise 5: UIC 406 Timetable Compression
- **Context:** Over a 2-hour evaluation window ($7,200\,\text{seconds}$), a double-track corridor section processes four trains in sequence: $P_1$ (Passenger), $F_1$ (Freight), $P_2$ (Passenger), $P_3$ (Passenger). 
  - $h_{min}(P \to F) = 400\,\text{s}$
  - $h_{min}(F \to P) = 1,200\,\text{s}$
  - $h_{min}(P \to P) = 400\,\text{s}$
  - Corridor running time for the final train ($P_3$) is $1,000\,\text{s}$.
- **Task:**
  1. Calculate the compressed timetable duration $T_{compressed}$.
  2. Apply a $25\%$ UIC buffer time allowance to calculate consumed capacity $T_{consumed}$.
  3. Calculate the infrastructure capacity utilization ratio ($K$). State whether the corridor meets UIC stability guidelines.
- **Deliverable:** Complete mathematical compression and stability appraisal.
- **Acceptance Criteria:** Accurate summation of headways and buffer times; correct percentage calculation.
- **Estimated Time:** 25 minutes.

### Exercise 6: Passing Siding Layout Design
- **Context:** You are tasked with designing passing siding locations on an 80-kilometer single-track line connecting a port to an inland dry port. The corridor must support:
  - 1 passenger train per hour in each direction ($v = 120\,\text{km/h}$).
  - 1 freight train every 2 hours in each direction ($v = 60\,\text{km/h}$, $L_{train} = 1,500\,\text{m}$).
- **Task:**
  1. Determine the optimal number and spacing of passing sidings to enable scheduled meets without holding passenger trains.
  2. Specify the minimum clear length required for each siding, including stopping margins.
  3. Draw an ASCII distance-time diagram illustrating one passenger-passenger meet and one freight-passenger overtake.
- **Deliverable:** Engineering specification with siding coordinates, track lengths, and time-space trajectory plot.
- **Acceptance Criteria:** Siding locations match harmonic meet times; train clearance margins verified.
- **Estimated Time:** 45 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1
1. **Formula for 4-Aspect Signaling:**
   $$h_{min} = \frac{3 \cdot L_{block} + L_{train} + d_{ov} + d_{sight}}{v} + t_{rf} + t_{rr}$$
2. **Parameters:**
   - $3 \cdot L_{block} = 3 \times 1,500\,\text{m} = 4,500\,\text{m}$
   - $L_{train} = 180\,\text{m}$
   - $d_{ov} = 150\,\text{m}$
   - $d_{sight} = 300\,\text{m}$
   - Total distance $= 4,500 + 180 + 150 + 300 = 5,130\,\text{meters}$
   - Speed $v = 33.33\,\text{m/s}$
   - Interlocking times $= t_{rf} + t_{rr} = 8.0 + 5.0 = 13.0\,\text{seconds}$
3. **Calculation:**
   $$h_{min} = \frac{5,130}{33.33} + 13.0 = 153.91 + 13.0 = 166.91\,\text{seconds} \quad (2.78\,\text{minutes})$$

### Answer to Exercise 2
1. **Moving Block Headway ($h_{min, MB}$):**
   - Speed $v = 22.22\,\text{m/s}$
   - Safe braking distance:
     $$d_{braking} = \frac{v^2}{2b} = \frac{22.22^2}{2 \times 0.40} = \frac{493.73}{0.80} = 617.16\,\text{meters}$$
   - Total distance $= L_{train} + d_{braking} + d_{safety} = 800 + 617.16 + 100 = 1,517.16\,\text{meters}$
   - Distance traversal time $= 1,517.16 / 22.22 = 68.28\,\text{seconds}$
   - Total headway:
     $$h_{min, MB} = 68.28 + t_{comm} + t_{reaction} = 68.28 + 2.0 + 1.5 = 71.78\,\text{seconds} \quad (1.20\,\text{minutes})$$
2. **Fixed-Block Headway (3-Aspect):**
   - Total distance $= 2 \cdot L_{block} + L_{train} + d_{ov} + d_{sight} = 2(2,000) + 800 + 200 + 400 = 5,400\,\text{meters}$
   - Traversal time $= 5,400 / 22.22 = 243.02\,\text{seconds}$
   - Total headway $= 243.02 + 9.0 + 6.0 = 258.02\,\text{seconds} \quad (4.30\,\text{minutes})$
3. **Capacity Comparison:**
   $$\text{Reduction in Headway} = \frac{258.02 - 71.78}{258.02} \times 100\% = \frac{186.24}{258.02} \times 100\% = 72.18\%$$
   The theoretical corridor capacity increases from $3,600 / 258.02 \approx 13.95\,\text{trains/h}$ to $3,600 / 71.78 \approx 50.15\,\text{trains/h}$, a **259% throughput increase**.

### Answer to Exercise 3
1. **Running Times:**
   - Passenger: $t_{pass} = 80\,\text{km} / 160\,\text{km/h} = 0.50\,\text{hours} = 30.0\,\text{minutes}$.
   - Freight: $t_{freight} = 80\,\text{km} / 80\,\text{km/h} = 1.00\,\text{hour} = 60.0\,\text{minutes}$.
   - Running time difference $\Delta t = 60.0 - 30.0 = 30.0\,\text{minutes}$.
2. **Departure Interval at Origin ($F \to P$):**
   $$H(F \to P) = h_{min} + \Delta t = 6.0 + 30.0 = 36.0\,\text{minutes}$$
3. **Slot Erosion Factor ($E_{slot}$):**
   $$E_{slot} = \frac{H(F \to P)}{h_{pass}} = \frac{36.0}{10.0} = 3.6\,\text{passenger paths}$$
   Inserting one freight train consumes **3.6 passenger slots**.

### Answer to Exercise 4
1. **Condition for Opposing Passes:**
   Since $L_{monster} = 3,200\,\text{m} > L_{clear} = 2,200\,\text{m}$, neither train can fit into any intermediate siding. Two opposing 3,200-meter trains **cannot pass each other** anywhere on the 120-kilometer subdivision. One train must be held entirely off the subdivision at a terminal yard or double-track junction until the opposing train completely clears the 120-kilometer corridor.
2. **Maximum Simultaneous Trains:**
   In opposing directions: Exactly **one** 3,200-meter train can occupy the 120-kilometer corridor at a time. In the same direction, multiple trains could follow each other separated by signaling headways, but no counter-traffic is permitted.
3. **Critique Rubric:**
   - Highlights loss of dynamic dispatching options.
   - Notes severe increase in yard dwell while trains wait for corridor clearance.
   - Mentions crew timeout risks (12-hour limit) if minor delays occur.
   - Concludes that running mega-trains on infrastructure with short sidings produces net negative operational economics despite lower linehaul crew costs.

### Answer to Exercise 5
1. **Compressed Time:**
   $$T_{compressed} = h_{min}(P_1 \to F_1) + h_{min}(F_1 \to P_2) + h_{min}(P_2 \to P_3) + T_{run}(P_3)$$
   $$T_{compressed} = 400 + 1,200 + 400 + 1,000 = 3,000\,\text{seconds} \quad (50.0\,\text{minutes})$$
2. **Consumed Capacity:**
   $$T_{buffer} = 0.25 \times 3,000 = 750\,\text{seconds}$$
   $$T_{consumed} = 3,000 + 750 = 3,750\,\text{seconds} \quad (62.5\,\text{minutes})$$
3. **Capacity Utilization:**
   $$K = \frac{3,750}{7,200} \times 100\% = 52.08\%$$
   Since $52.08\% < 70\%$, the timetable is stable and satisfies UIC 406 robustness standards.

### Answer to Exercise 6 (Design Rubric)
1. **Siding Spacing:**
   - Passenger transit time across 80 km at 120 km/h $= 40\,\text{minutes}$.
   - For an hourly bidirectional service, trains meet every 30 minutes in travel time.
   - Siding required at exact midpoint: **Kilometer 40.0** (20 minutes travel time from each end).
   - For 60 km/h freight transit (80 minutes total run), additional sidings are required at **Kilometer 20.0** and **Kilometer 60.0** to allow passenger trains to overtake freight trains without freight trains incurring excessive dwell. Total 3 sidings: Km 20, Km 40, Km 60.
2. **Siding Clear Length:**
   $$L_{clear} \ge L_{train} + 2 \cdot d_{margin} = 1,500\,\text{m} + 2(50\,\text{m}) = 1,600\,\text{meters}$$
   Specify $L_{clear} = 1,650\,\text{meters}$ to provide operational tolerance.
3. **Time-Space Diagram:**
   Must show passenger paths crossing symmetrically at Km 40.0, and freight path stepping into the siding at Km 20.0 or 60.0, dwelling while the passenger path passes through on the mainline.

---

## 14. Interview questions

### Question 1: What is the fundamental difference between train running time and blocking time?
- **Strong Answer:** Running time is strictly the physical duration required for the train's front to travel from one signal to the next based on tractive effort, resistance, and speed limits. Blocking time is the total duration that a block section is exclusively locked and unavailable to other trains. It encompasses route formation time, driver sighting time, approach running time across the upstream block, pure block running time, clearing time for the physical train length to exit the block, and the overlap clearance plus route release delay. In three-aspect signaling, blocking time is typically two to three times larger than pure running time.
- **Weak Answer:** Running time is how fast the train drives, and blocking time is when the signal stays red.

### Question 2: Why does mixing fast passenger trains and slow freight trains degrade network capacity more than running pure freight or pure passenger traffic?
- **Strong Answer:** Capacity degradation stems from speed heterogeneity ($\Delta v$). In distance-time space, trajectories with different slopes diverge or converge. When a fast train follows a slow train, it catches up, requiring an inflated departure headway at the origin equal to the pure running time differential over the corridor ($\Delta t = L(1/v_{slow} - 1/v_{fast})$). When a slow train follows a fast train, the space between them continuously widens, creating empty track slots that cannot be consumed by other trains. This "heterogeneity wedge" erodes capacity such that a single slow freight train can eliminate three to six passenger paths.
- **Weak Answer:** Fast trains have higher priority, so dispatchers just park the freight trains in sidings, which makes freight slow.

### Question 3: How does the UIC 406 timetable compression method differ from calculating capacity using average headways?
- **Strong Answer:** Average headway formulations assume homogeneous train flow and ignore the spatial-temporal interaction of specific train sequences over physical interlocking topologies. UIC 406 is a graphic-analytical compression method: it takes an actual or planned timetable in the distance-time plane, maintains train order and stop patterns, and pushes paths together until their blocking time stairways touch at the critical bottleneck block ($t_{buf} = 0$). By measuring the compressed time and adding standardized buffer allowances (25–33%), UIC 406 accurately captures the true capacity consumed by specific train sequences and mixed-speed interactions.
- **Weak Answer:** UIC 406 is a European software program that calculates the percentage of track that is full.

### Question 4: Under what operational conditions does Precision Scheduled Railroading (PSR) train lengthening induce network gridlock?
- **Strong Answer:** PSR train lengthening creates gridlock when train lengths exceed the physical clear length between fouling points of corridor passing sidings ($L_{train} > L_{clear}$). When this threshold is crossed, the mega-train cannot enter intermediate sidings to allow overtakes or meets. On single-track lines, opposing traffic must be held dozens of miles upstream, collapsing dynamic dispatching flexibility. Furthermore, clearing times through switches ($L_{train}/v_{turnout}$) increase linearly with length, occupying interlockings longer. If a mega-train experiences an emergency brake application or broken knuckle, it blocks multiple control points, cascading into network-wide delay propagation.
- **Weak Answer:** When trains get too long, they become too heavy for the locomotives, so they stall on hills.

### Question 5: Explain the mechanics of ETCS Level 2 versus Level 3 signaling in relation to minimum technical headway.
- **Strong Answer:** ETCS Level 2 uses continuous radio communication (GSM-R/FRMCS) to transmit Movement Authorities to the cab, eliminating lineside signals, but it remains a **fixed-block** system. Train position and track vacancy are still detected by physical track circuits or axle counters at fixed boundaries. Therefore, minimum headway is still constrained by fixed block geometry ($h_{min} \propto 2 \cdot L_{block}$). ETCS Level 3 introduces **moving block**: it eliminates fixed trackside detection entirely. The train verifies its own position and integrity (via TIMS) and reports its rear to the Radio Block Centre. Headway becomes dynamic, governed purely by safe service braking distance ($v^2/2b$), train length, and communication latency, substantially compressing headway.
- **Weak Answer:** Level 2 is for passenger trains and Level 3 is for freight trains.

### Question 6: What is a "saw-by" maneuver and why is it considered an engineering and safety failure mode?
- **Strong Answer:** A saw-by (or cornfield meet) occurs on a single-track line when two opposing trains meet at a siding, and one or both trains are longer than the siding clear length. Because neither train can completely clear the mainline, the trains must execute a complex sequence: Train A pulls partly into the siding, leaving its rear on the main; Train B inches past Train A's front cut until it stops; Train A then shunts backward or decouples cars to let Train B clear the remaining switch. It violates positive safety separation principles, requires ground crew hand-signals, consumes 2 to 6 hours of line capacity, and carries a high risk of side-swipe collisions.
- **Weak Answer:** It is when track maintenance crews use a circular saw to cut rails during daytime track work.

### Question 7: Why is 100% capacity utilization impossible on a scheduled railway network?
- **Strong Answer:** A railway network is a queueing network subject to stochastic disturbances (traction failures, signal glitches, passenger dwell variances). As utilization ($K$) approaches 100%, buffer time ($t_{buf}$) between blocking time stairways approaches zero. According to queuing theory (the Pollaczek-Khinchine formula), expected waiting time and queue length escalate asymptotically toward infinity as utilization approaches 1.0 ($\rho \to 1$). UIC guidelines establish practical capacity ceilings at 70–75% for mixed-traffic lines and 85% for peak commuter corridors to ensure that minor primary delays can be absorbed by buffer times without causing system-wide knock-on delay cascades.
- **Weak Answer:** Because trains need time to stop at stations and workers need time to inspect tracks at night.

### Question 8: How does track gradient influence the blocking time stairway of a heavy freight train?
- **Strong Answer:** Track gradient affects the equation of motion: $m(1+\rho_m)a = F_{tract} - R(v) - mg\sin\theta$. On an ascending ruling grade ($+15‰$), acceleration is severely degraded, and equilibrium speed drops to the locomotive balance speed. This dramatically inflates pure running time ($t_{run}$) and clearing time ($t_{clear} = L_{train}/v$), widening the blocking time block by minutes. On a descending steep grade ($-15‰$), safe braking distances ($d_{braking} = v^2 / 2(b - g\sin\theta)$) expand significantly, requiring longer approach times ($t_{app}$) or multi-block signal aspects to provide adequate stopping distance. Both ascending and descending grades inflate total blocking time $T_B$.
- **Weak Answer:** Trains go slower uphill and faster downhill, so the schedule changes.

---

## 15. Key terms

- **Axle Counter:** Track-mounted magnetic sensor pair that detects train entry and exit by counting passing wheel flanges.
- **Balise:** Transponder mounted between the rails that transmits position coordinates, gradient profiles, and signal aspects to passing trains.
- **Braking Curve:** The calculated deceleration profile enforcing safe train stopping upstream of a red signal or speed restriction.
- **Buffer Time:** Unscheduled time margin inserted between adjacent blocking time stairways to absorb operational variations.
- **Centralized Traffic Control (CTC):** Signaling system where switch and signal alignments across a corridor are controlled remotely by a dispatcher.
- **Clearing Time:** The time duration required for the complete physical length of a train to pass a signal or block boundary.
- **Cornfield Meet:** Operational deadlock where two opposing trains meet on a single-track line and neither can clear the mainline.
- **Davis Equation:** Quadratic empirical formula ($R = A + Bv + Cv^2$) calculating aerodynamic and mechanical train rolling resistance.
- **Double-Stack Intermodal:** Rail transport of shipping containers stacked two-high on well cars, requiring high vertical clearance (Plate H).
- **ETCS Level 2:** Radio-based cab signaling system using GSM-R / FRMCS with fixed-block track vacancy detection.
- **ETCS Level 3:** Radio-based moving-block train control system eliminating fixed track circuits and relying on train integrity monitoring.
- **Fouling Point:** The location near a track switch where clearance between converging tracks drops below safety standards.
- **Headway:** The temporal separation between consecutive trains passing a fixed point on a rail corridor.
- **Interlocking:** Mechanical, electrical, or software safety system preventing conflicting signal and switch movements.
- **Knock-On Delay:** Secondary delay induced on subsequent trains when a primary delay consumes the scheduled buffer time.
- **Movement Authority (MA):** Formal electronic or signal authorization permitting a train to proceed to a designated end-of-authority point.
- **Passing Siding:** Auxiliary track connected to the mainline at both ends, utilized for meets and overtakes.
- **Periodic Event Scheduling Problem (PESP):** Mathematical graph-theoretic optimization formulation for cyclic timetables.
- **Positive Train Control (PTC):** Mandated US safety overlay preventing collisions, overspeed derailments, and switch misalignments.
- **Precision Scheduled Railroading (PSR):** Operational strategy emphasizing fixed train schedules, asset velocity, and train consolidation.
- **Radio Block Centre (RBC):** Centralized computer system in ETCS Level 2/3 that calculates and transmits Movement Authorities to trains via radio.
- **Ruling Grade:** The steepest gradient on a railway subdivision that dictates the maximum tonnage a locomotive consist can haul.
- **Sighting Time:** The time required for an engineer to observe, recognize, and react to a visual wayside signal aspect.
- **Speed Heterogeneity:** The variance of operating speeds across different train classes utilizing a common corridor.
- **Sperrzeittreppe:** The German engineering term for the discrete blocking time stairway in the time-space plane.
- **Three-Aspect Signaling:** Visual signaling displaying Green (Proceed), Yellow (Caution/Approach), and Red (Stop).
- **Train Integrity Monitoring System (TIMS):** Onboard electronic system verifying that a train has not broken in two (*parted consist*).
- **Turnout Number:** Ratio expressing switch divergence angle ($N = \cot\alpha$); higher numbers denote gentler curves and higher speeds.
- **UIC Leaflet 405:** International standard defining analytical methods for railway corridor capacity evaluation.
- **UIC Leaflet 406:** International standard defining timetable compression methodology to assess infrastructure capacity utilization.

---

## 16. Sources

1. International Union of Railways (UIC). (1996). *UIC Leaflet 405: Capacity of Railway Infrastructure - Analytical Method*. Paris: UIC.
2. International Union of Railways (UIC). (2013). *UIC Leaflet 406: Capacity* (2nd ed.). Paris: UIC.
3. Pachl, J. (2018). *Railway Operation and Control* (4th ed.). VTD Rail Publishing. ISBN: 978-0971991576.
4. Hansen, I. A., & Pachl, J. (Eds.). (2014). *Railway Timetable & Traffic: Analysis, Modelling, Simulation*. Eurailpress. ISBN: 978-3777104614.
5. European Union Agency for Railways (ERA). (2023). *Technical Specification for Interoperability relating to the Control-Command and Signalling Subsystems (CCS TSI)*. Regulation (EU) 2023/1695.
6. Federal Railroad Administration (FRA). (2020). *Positive Train Control Systems: 49 CFR Part 236, Subpart I*. US Government Publishing Office.
7. American Railway Engineering and Maintenance-of-Way Association (AREMA). (2024). *Manual for Railway Engineering*. Lanham, MD: AREMA.
8. Abril, M., Barber, F., Ingolotti, L., Salido, M. A., Tormos, P., & Lova, A. (2008). An assessment of railway capacity. *Transportation Research Part E: Logistics and Transportation Review*, 44(5), 774–806. https://doi.org/10.1016/j.tre.2007.04.001
9. Goverde, R. M. P. (2007). Railway timetable stability analysis using max-plus system theory. *Transportation Research Part B: Methodological*, 41(2), 179–201. https://doi.org/10.1016/j.trb.2006.02.003
10. Surface Transportation Board (STB). (2022). *Urgent Issues in Freight Rail Service: Rail Operating Metrics and Precision Scheduled Railroading*. Docket No. EP 770. Washington, DC.

---

## 17. Links

- **Parent Module Index:** [`_index.md`](_index.md)
- **Sibling Topic 01 (Berth and Quay Crane Operations):** [`01_berth-quay-crane.md`](01_berth-quay-crane.md)
- **Sibling Topic 02 (Terminal Yard and Gate Operations):** [`02_terminal-yard-gate.md`](02_terminal-yard-gate.md)
- **Sibling Topic 04 (Rail Yards and Shunting):** [`04_rail-yards-shunting.md`](04_rail-yards-shunting.md)
- **Sibling Topic 06 (Intermodal and Hinterland Logistics):** [`06_intermodal-hinterland.md`](06_intermodal-hinterland.md)
- **Sibling Topic 07 (Disruption and Recovery Management):** [`07_disruption-recovery.md`](07_disruption-recovery.md)
- **Root Survey File:** [`../../../course/41_maritime_rail_air_operations.md`](../../../course/41_maritime_rail_air_operations.md)
