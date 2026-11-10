# 🚗 Autonomous Navigation Using Deep Reinforcement Learning
### Vision-Based Driving with CNN + PPO Actor-Critic

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=900&color=60A5FA&center=true&vCenter=true&width=850&lines=Teaching+Machines+to+Navigate+Autonomously;CNN-Based+Visual+Perception;Proximal+Policy+Optimization;Learning+Steering+%26+Throttle+Through+Experience" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PPO-6C3483?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep%20Reinforcement%20Learning-1E3A8A?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Computer%20Vision-0891B2?style=for-the-badge" />
</p>

<p align="center">
  <b>An autonomous driving agent that learns steering and throttle control directly from RGB camera observations using a CNN-based Proximal Policy Optimization (PPO) Actor-Critic architecture.</b>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-learning-agent">Learning Agent</a> •
  <a href="#-training">Training</a> •
  <a href="#-results">Results</a> •
  <a href="#-setup">Setup</a>
</p>

---

## `~/overview` — Project Introduction

Autonomous driving requires an intelligent system to perceive its surroundings, understand the current driving situation, and make continuous control decisions.

Traditional approaches often depend on manually designed rules and predefined driving strategies. This project explores a different approach: allowing an agent to learn driving behavior through interaction with a simulated environment.

The system uses **Deep Reinforcement Learning (DRL)** to train an autonomous driving agent that processes raw RGB camera observations and learns to control steering and throttle.

A CNN extracts visual features from camera frames, while a PPO-based Actor-Critic architecture learns a policy for selecting driving actions.

### Project Highlights

| Component | Implementation |
|---|---|
| Learning approach | Deep Reinforcement Learning |
| Algorithm | Proximal Policy Optimization (PPO) |
| Policy architecture | Actor-Critic |
| Visual perception | Convolutional Neural Network |
| Input | Raw RGB camera observations |
| Control outputs | Steering and throttle |
| Reward objectives | Lane-centering, speed regulation, collision avoidance |
| Evaluation | Multiple track configurations |
| Generalization | Mountain-track scenario evaluation |
| Framework | Python, PyTorch |

### Key Objective

Develop an autonomous driving agent capable of learning navigation behavior from visual observations rather than relying entirely on manually programmed driving rules.

---

## `~/architecture` — System Architecture

The project follows a perception-to-action learning pipeline.

```mermaid
flowchart TD
    ENV["Driving Simulation Environment"]
    CAMERA["RGB Camera Observation"]
    PRE["Image Preprocessing"]
    CNN["CNN Feature Extractor"]

    subgraph PPO["PPO Actor-Critic Agent"]
        ACTOR["Actor Network"]
        CRITIC["Critic Network"]
    end

    ACTION["Steering + Throttle"]
    REWARD["Reward Calculation"]
    UPDATE["PPO Policy Update"]

    ENV --> CAMERA
    CAMERA --> PRE
    PRE --> CNN
    CNN --> ACTOR
    CNN --> CRITIC

    ACTOR --> ACTION
    ACTION --> ENV

    ENV --> REWARD
    REWARD --> UPDATE
    CRITIC --> UPDATE
    UPDATE --> ACTOR
    UPDATE --> CRITIC

    classDef environment fill:#172554,stroke:#60A5FA,color:#EFF6FF
    classDef perception fill:#164E63,stroke:#22D3EE,color:#ECFEFF
    classDef learning fill:#312E81,stroke:#A78BFA,color:#F5F3FF
    classDef control fill:#78350F,stroke:#F59E0B,color:#FFFBEB

    class ENV,CAMERA environment
    class PRE,CNN perception
    class ACTOR,CRITIC,UPDATE learning
    class ACTION,REWARD control
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| Simulation environment | Provides driving scenarios and vehicle interaction |
| RGB camera | Captures visual observations |
| Preprocessing | Prepares camera frames for neural network input |
| CNN | Extracts useful visual representations |
| Actor | Learns the driving policy |
| Critic | Estimates the value of the current state |
| PPO | Updates the policy using collected experience |
| Reward function | Encourages desirable driving behavior |

---

## `~/visual_perception` — Learning from Camera Images

Instead of relying only on manually supplied driving instructions, the agent receives visual information from its camera.

The CNN acts as the perception component of the learning system.

### Visual Perception Pipeline

```mermaid
flowchart LR
    A["RGB Camera Frame"] --> B["Image Preprocessing"]
    B --> C["CNN"]
    C --> D["Visual Feature Representation"]
    D --> E["Actor-Critic Network"]
    E --> F["Driving Decision"]

    classDef input fill:#172554,stroke:#60A5FA,color:#EFF6FF
    classDef vision fill:#164E63,stroke:#22D3EE,color:#ECFEFF
    classDef decision fill:#312E81,stroke:#A78BFA,color:#F5F3FF

    class A,B input
    class C,D vision
    class E,F decision
```

The CNN learns visual representations that help the policy make driving decisions.

These representations can help the agent learn patterns associated with:

- Road and lane positioning.
- Visual driving conditions.
- Track geometry.
- Navigation-relevant scene information.

*The exact CNN layer configuration and image dimensions should be documented from the implemented model.*

---

## `~/learning_agent` — PPO Actor-Critic

The learning agent uses **Proximal Policy Optimization (PPO)**, a policy-gradient reinforcement learning algorithm.

PPO is designed to improve policy learning while restricting excessively large policy updates.

### Actor Network

The Actor learns the driving policy.

It uses the visual representation to determine the control action, including steering and throttle.

### Critic Network

The Critic estimates how valuable the current state is under the learned policy.

Its value estimation helps calculate the advantage used during policy optimization.

### Actor-Critic Relationship

```mermaid
flowchart TD
    OBS["Visual Observation"]
    FEATURES["CNN Feature Extraction"]

    OBS --> FEATURES

    FEATURES --> ACTOR["Actor"]
    FEATURES --> CRITIC["Critic"]

    ACTOR --> POLICY["Driving Policy"]
    POLICY --> CONTROL["Steering and Throttle"]

    CRITIC --> VALUE["State Value Estimate"]

    CONTROL --> ENV["Environment Interaction"]
    ENV --> REWARD["Reward and Next Observation"]

    REWARD --> ADV["Advantage Estimation"]
    VALUE --> ADV

    ADV --> PPO["PPO Optimization"]
    PPO --> ACTOR
    PPO --> CRITIC

    classDef perception fill:#164E63,stroke:#22D3EE,color:#ECFEFF
    classDef network fill:#312E81,stroke:#A78BFA,color:#F5F3FF
    classDef action fill:#78350F,stroke:#F59E0B,color:#FFFBEB

    class OBS,FEATURES perception
    class ACTOR,CRITIC,VALUE,ADV,PPO network
    class POLICY,CONTROL,ENV,REWARD action
```

### Why PPO?

- Uses policy-gradient learning.
- Employs a clipped objective to limit excessively large policy updates.
- Uses an Actor-Critic structure.
- Supports learning continuous control policies.
- Can learn through repeated interaction with a simulation environment.

---

## `~/control_system` — Steering and Throttle

The agent learns to control vehicle movement through its policy.

| Control | Purpose |
|---|---|
| Steering | Controls the vehicle's turning direction |
| Throttle | Controls vehicle acceleration |

The objective is not simply to move forward. The agent must learn how its actions affect the vehicle's position and driving behavior over time.

The learned policy is evaluated across different track configurations.

---

## `~/reward_design` — Driving Objectives

Reinforcement learning depends on a reward signal that guides the agent toward desirable behavior.

This project incorporates reward objectives associated with:

| Objective | Learning purpose |
|---|---|
| Lane-centering | Encourage maintaining a suitable lane position |
| Speed regulation | Encourage appropriate vehicle speed |
| Collision avoidance | Discourage unsafe interactions |
| Navigation progress | Encourage forward movement through the track |

### Learning Loop

```mermaid
flowchart TD
    A["Observe Camera Frame"] --> B["Extract Visual Features"]
    B --> C["Select Driving Action"]
    C --> D["Execute Action"]
    D --> E["Receive Reward"]
    E --> F["Store Experience"]
    F --> G["Calculate Advantage"]
    G --> H["Update Actor-Critic with PPO"]
    H --> A

    classDef observation fill:#164E63,stroke:#22D3EE,color:#ECFEFF
    classDef learning fill:#312E81,stroke:#A78BFA,color:#F5F3FF
    classDef feedback fill:#78350F,stroke:#F59E0B,color:#FFFBEB

    class A,B observation
    class C,D,F,G,H learning
    class E feedback
```

The reward function influences what the agent learns. Its exact numerical weights and termination conditions should be taken from the training implementation.

---

## `~/training` — Training and Evaluation

The agent learns by repeatedly interacting with the driving simulation.

### Training Workflow

1. Initialize the simulation environment.
2. Capture an RGB camera observation.
3. Process the observation through the CNN.
4. Generate driving actions using the Actor.
5. Estimate state value using the Critic.
6. Execute the selected control actions.
7. Collect reward and subsequent observations.
8. Use collected experience for PPO optimization.
9. Evaluate the learned policy on different track configurations.

### Training Pipeline

```mermaid
flowchart TD
    INIT["Initialize Environment and Agent"]
    RESET["Reset Simulation"]
    OBS["Capture RGB Observation"]
    CNN["CNN Feature Extraction"]
    POLICY["Actor-Critic Inference"]
    ACTION["Steering and Throttle"]
    STEP["Environment Step"]
    EXPERIENCE["Collect Experience"]
    PPO["PPO Policy Optimization"]
    CHECK{"Episode Complete?"}
    EVAL["Evaluate Trained Agent"]

    INIT --> RESET
    RESET --> OBS
    OBS --> CNN
    CNN --> POLICY
    POLICY --> ACTION
    ACTION --> STEP
    STEP --> EXPERIENCE
    EXPERIENCE --> PPO
    PPO --> CHECK
    CHECK -->|No| OBS
    CHECK -->|Yes| EVAL

    classDef setup fill:#172554,stroke:#60A5FA,color:#EFF6FF
    classDef perception fill:#164E63,stroke:#22D3EE,color:#ECFEFF
    classDef learning fill:#312E81,stroke:#A78BFA,color:#F5F3FF
    classDef evaluation fill:#14532D,stroke:#4ADE80,color:#F0FDF4

    class INIT,RESET setup
    class OBS,CNN perception
    class POLICY,ACTION,STEP,EXPERIENCE,PPO learning
    class CHECK,EVAL evaluation
```

### Evaluation Scenarios

The agent was trained and evaluated across multiple track configurations.

The project specifically explored:

- Learning lane-following behavior.
- Regulating vehicle speed.
- Avoiding collisions.
- Testing navigation on different track layouts.
- Evaluating generalization to a mountain-track scenario.

---

## `~/results` — Observed Outcomes

The project demonstrated:

- End-to-end visual control using RGB camera input.
- CNN-based feature extraction for driving decisions.
- PPO Actor-Critic policy learning.
- Steering and throttle control through reinforcement learning.
- Full-lap navigation in simulation.
- Generalization to a mountain-track scenario.

### Outcome Summary

| Evaluation Area | Reported Outcome |
|---|---|
| Visual input | Raw RGB camera observations |
| Learning algorithm | CNN-based PPO Actor-Critic |
| Vehicle controls | Steering and throttle |
| Lane behavior | Lane-centering reward objective |
| Safety behavior | Collision-avoidance reward objective |
| Navigation | Full-lap navigation achieved |
| Generalization | Mountain-track scenario |

*Quantitative success rates, collision counts, lap times, and training curves are not included here because verified numerical measurements were not available in the project summary.*

---

## `~/technology` — Technology Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,pytorch,opencv,git,github" />
</p>

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | CNN |
| Reinforcement Learning | Proximal Policy Optimization |
| Neural Architecture | Actor-Critic |
| Input Modality | RGB Camera Images |
| Control | Steering and Throttle |
| Development | Git, GitHub |

---

## `~/engineering_decisions` — Design Rationale

| Design Choice | Reason |
|---|---|
| CNN-based perception | Extract visual features from camera images |
| PPO | Learn a policy through stable, constrained policy updates |
| Actor-Critic | Separate policy generation and value estimation |
| RGB observations | Learn directly from visual information |
| Reward-based training | Encourage lane-centering, speed regulation, and collision avoidance |
| Multiple track configurations | Evaluate behavior beyond a single track |

---

## `~/setup` — Getting Started

The exact simulation framework, folder structure, dependency versions, and training scripts should be matched to the repository before publishing executable setup instructions.

### Prerequisites

- Python
- PyTorch
- Required simulation environment
- Git

### 01 / Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 02 / Create a Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 03 / Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Install the appropriate PyTorch version for your hardware and Python environment if it is not included in the dependency file.

### 04 / Train the Agent

Run the actual training entry-point script from the repository:

```bash
python <TRAINING_SCRIPT>.py
```

### 05 / Evaluate the Agent

Run the implemented evaluation or inference script:

```bash
python <EVALUATION_SCRIPT>.py
```

> The commands above are templates. Replace the placeholders with the actual script names and simulation setup instructions from the repository.

---

## `~/roadmap` — Future Improvements

Potential future directions:

- [ ] Add quantitative evaluation metrics and training curves.
- [ ] Compare PPO against alternative reinforcement learning algorithms.
- [ ] Evaluate robustness under different lighting and visual conditions.
- [ ] Introduce domain randomization.
- [ ] Investigate sim-to-real transfer.
- [ ] Add detailed experiment tracking.
- [ ] Evaluate generalization across a larger set of unseen tracks.
- [ ] Investigate explainability of visual driving decisions.

*These are proposed enhancements, not claims about existing functionality.*

---

## `~/engineering_learnings` — What This Project Explores

This project brings together several important areas of intelligent systems engineering:

**Deep Learning**
- CNN-based visual feature extraction.
- Neural network representation learning.
- Actor-Critic network design.

**Reinforcement Learning**
- Policy-gradient methods.
- PPO optimization.
- Reward design.
- Learning through environment interaction.

**Autonomous Systems**
- Visual perception.
- Continuous vehicle control.
- Navigation under changing track conditions.
- Evaluation of learned behavior.

---

## `~/author` — Project Information

**Developer:** Sarthak Tanpure  
**Institution:** Pravara Rural Engineering College  
**Degree:** B.E. Information Technology  
**Project Domain:** Deep Reinforcement Learning · Computer Vision · Autonomous Systems

<p align="center">
  <a href="https://github.com/Sarthaktanpure">
    <img src="https://img.shields.io/badge/GitHub-Sarthaktanpure-181717?style=for-the-badge&logo=github" />
  </a>
  <a href="https://www.linkedin.com/in/sarthak-tanpure-a74b5133a/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin" />
  </a>
</p>

<p align="center">
  <b>From pixels to policies — teaching an agent to learn how to drive.</b>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F172A,50:1E3A8A,100:EA580C&height=120&section=footer" width="100%" />
</p>