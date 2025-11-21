# Day 46: Mock Interview 2 - Ride Sharing (Uber/Lyft)

> **Phase**: 5 - Interview Mastery
> **Week**: 10 - Mock Interviews
> **Focus**: Spatio-Temporal Forecasting
> **Reading Time**: 60 mins

---

## 1. Problem Statement

"Design an ETA (Estimated Time of Arrival) prediction system for a ride-sharing app."

---

## 2. Step-by-Step Design

### Step 1: Requirements
*   **Accuracy**: Mean Absolute Error (MAE) < 1 minute.
*   **Latency**: < 10ms (Called millions of times for route optimization).

### Step 2: Data
*   **Route**: Sequence of GPS points.
*   **Traffic**: Real-time speeds on road segments.
*   **Context**: Weather, Day of Week, Event (Concert).

### Step 3: Modeling
*   **Segment Level**: Predict travel time for each road segment. Sum them up?
    *   *Issue*: Ignores intersection delays (Left turn vs Right turn).
*   **Route Level**: Deep Learning (Deepr).
    *   *Input*: Embedding of Start, End, Route Sequence.
    *   *Architecture*: LSTM or Transformer.

### Step 4: Features
*   **Geohash**: Encode lat/long into grid cells.
*   **Global Features**: "Is it raining?"
*   **Real-time**: "Average speed in this geohash last 5 mins".

---

## 3. Deep Dive Questions

**Interviewer**: "How do you handle the 'Left Turn' problem?"
**Candidate**: "Standard graph weights (distance/speed) fail here. We need 'Turn Costs'. I would model the graph nodes as (Edge_In, Edge_Out) pairs rather than just Intersections. Or use a Sequence model that sees the full path context."

**Interviewer**: "How do you correct for bias? Drivers drive faster than Google Maps predicts."
**Candidate**: "We can learn a user-specific embedding. `Predicted_Time = Base_Time * User_Aggression_Factor`. If User A consistently beats the ETA by 10%, the model learns a factor of 0.9 for them."

---

## 4. Evaluation
*   **Metric**: MAE (Mean Absolute Error), MAPE (Percentage Error).
*   **Business Impact**: Better ETAs = Better Matching = Lower Wait Times.

---

## 5. Further Reading
- [Deepr: Deep Neural Networks for ETA](https://arxiv.org/abs/1804.02828)
- [Uber Engineering Blog: ETA](https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/)
