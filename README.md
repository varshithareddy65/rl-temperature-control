# Reinforcement Learning-Based Industrial Temperature Control Using PPO

A simulation-based industrial temperature control system that compares a Proximal Policy Optimization (PPO) reinforcement learning agent with a tuned PID controller. The project models thermal dynamics, trains an adaptive control policy, and evaluates controller performance under changing setpoints, disturbances, and different heating capacities.

**Technologies:** Python, PyTorch, Stable-Baselines3, Gymnasium, NumPy, Pandas, Matplotlib, Streamlit.

> **Note:** This is a simulation project. It has not been validated on real industrial equipment.

## 1. Project Overview

Industrial temperature control systems must maintain desired temperatures while handling changing operating conditions and external disturbances.

Traditional PID controllers are widely used, but their performance depends on controller tuning and process characteristics.

This project explores reinforcement learning as an alternative control strategy.

The system:
- Simulates a first-order thermal process.
- Implements a tuned PID controller as a baseline.
- Develops a custom Gymnasium environment.
- Trains a PPO agent for adaptive temperature control.
- Compares controller performance across fixed and randomized test scenarios.
- Studies the trade-off between tracking accuracy and control smoothness.

## 2. System Architecture

```text
             Thermal Process Model
                      |
             Gymnasium Environment
                      |
              Observation Space
                      |
                 PPO Agent
                      |
                Heater Action
                      |
              Thermal Plant
                      |
             Temperature Feedback
                      |
              Performance Metrics
                      |
             PID vs PPO Evaluation
```

### Project Structure

```text
rl_temperature_control/
│
├── app.py
├── train.py
├── evaluate.py
├── compare_models.py
├── diagnose.py
├── backup_model.py
├── plot_random_summary.py
│
├── thermal_model.py
├── environment.py
├── ppo_agent.py
├── pid_controller.py
├── visualization.py
│
├── config.yaml
├── config_v2.yaml
├── config_v3.yaml
├── requirements.txt
│
├── models/
├── final_model/
├── logs/
├── results/
├── backups/
│
└── tests/
```

## 3. Thermal Process Model

The project uses a lumped first-order thermal model.

The governing equation is:

\[
C\frac{dT}{dt}
=
sP_{\max}u-\frac{T-T_{\mathrm{amb}}}{R}+d(t)
\]

Where:

| Parameter | Description | Value |
|---|---|---:|
| C | Thermal capacitance | 5000 J/K |
| R | Thermal resistance | 0.02 K/W |
| Pmax | Maximum heater power | 2000 W |
| τ = RC | Thermal time constant | 100 s |
| u | Heater control fraction | 0 to 1 |
| d(t) | External heat disturbance | ±300 W |

The model is discretized using an exact zero-order-hold update with a simulation time step of one second.

Sensor noise is included to make the control environment more realistic.

## 4. Reinforcement Learning Approach

### PPO Algorithm

Proximal Policy Optimization (PPO) is an on-policy actor-critic reinforcement learning algorithm.

The agent interacts with the simulated thermal environment, observes temperature-related information, and learns actions that improve temperature tracking.

**Observation space:**
- Temperature error
- Current temperature
- Target temperature
- Ambient temperature
- Temperature change
- Integral of temperature error
- Previous heater action

**Action space:**

A continuous action is mapped to the heater control range:

\[
u=\frac{a+1}{2}, \qquad a\in[-1,1]
\]

**Reward function:**

\[
r=-(0.1|e|+0.02u+w_{\mathrm{smooth}}|u-u_{\mathrm{prev}}|)
\]

An additional penalty is applied when the temperature exceeds the safety threshold.

The reward balances:
- Temperature tracking accuracy
- Heater energy usage
- Smoothness of control actions

### Reward Function Experiments

Three PPO configurations were evaluated:

| Version | Smoothness weight | Purpose |
|---|---:|---|
| v1 | 0.02 | Initial PPO model |
| v2 | 1.0 | Stronger smoothness penalty |
| v3 | 0.1 | Intermediate smoothness penalty |

The experiments showed that increasing the smoothness penalty too much can reduce control fluctuations but negatively affect tracking performance.

**Current best PPO candidate: v3.**

## 5. PID Controller

A tuned PID controller is implemented as the baseline.

\[
u=K_p e+K_i\int e\,dt-K_d\frac{dT}{dt}
\]

The controller includes:
- Output saturation
- Derivative filtering
- Conditional integration anti-windup
- Grid-search parameter tuning

The PID parameters were tuned using randomized scenarios and then kept fixed during evaluation.

## 6. Experimental Evaluation

Five experiments were conducted.

| Experiment | Test scenario |
|---|---|
| 1 | Constant target: 25°C to 50°C |
| 2 | Changing targets: 40°C → 55°C → 35°C |
| 3 | External heat-loss disturbance and ambient temperature change |
| 4 | Different initial temperatures: 20°C, 35°C and 55°C |
| 5 | Heating capacity variations: 0.6x, 1.0x and 1.4x |

The heating capacity tests at 0.6x and 1.4x also examine operation outside the PPO training range.

### Evaluation Metrics

- **IAE:** Integral of Absolute Error
- **RMSE:** Root Mean Square Error
- **Overshoot:** Temperature deviation beyond the target
- **Settling Time:** Time required to enter the target tolerance band
- **Steady-State Error:** Average absolute error during the final 50 seconds
- **Control Variation:** Total variation in heater control actions

## 7. Results and Performance Comparison

### Held-Out Random Scenarios

PID, v1, v2 and v3 were evaluated across the same 50 randomized scenarios.

Values below are mean results. Lower is better for all metrics shown.

| Metric | PID | PPO v1 | PPO v2 | PPO v3 |
|---|---:|---:|---:|---:|
| IAE | 475.32 | 437.23 | 629.60 | **429.94** |
| RMSE | 2.613 | 2.542 | 2.731 | **2.540** |
| Overshoot (%) | 10.22 | **2.59** | 26.64 | 4.93 |
| Settling Time (s) | 66.38 | **38.50** | 246.19 | 38.52 |
| Steady-State Error | **0.058** | 0.119 | 0.686 | 0.095 |
| Control Variation | **4.79** | 58.44 | 21.64 | 22.97 |

### Key Observations

1. PPO v3 achieved the lowest mean IAE and RMSE among the evaluated controllers.
2. PPO v1 achieved the lowest overshoot and marginally faster settling time.
3. PPO v2 showed poor tracking performance after increasing the smoothness penalty to 1.0.
4. PPO v3 reduced control variation considerably compared with v1.
5. PID maintained the smoothest control actions and the lowest steady-state error.

The results demonstrate a trade-off between temperature tracking performance and control smoothness.

The relatively large standard deviations across randomized scenarios also mean that the small differences between v1 and v3 should not be interpreted as statistically significant without further testing.

### Additional Fixed-Scenario Evaluation

The v3 model was also evaluated against PID across the five predefined experiments.

Notable results:

| Scenario | PID RMSE | PPO v3 RMSE |
|---|---:|---:|
| Constant target | 6.917 | 6.909 |
| Changing target | 5.950 | 6.006 |
| Disturbance | 0.576 | 0.135 |
| Initial temperature 55°C | 1.742 | 1.647 |
| Heating capacity 1.0x | 3.079 | 3.062 |

PPO v3 showed particularly strong disturbance response and performance under the higher initial temperature scenario.

## 8. Visualizations

The project generates the following plots:

- Constant target temperature response
- Changing setpoint response
- Disturbance rejection
- Different initial temperature responses
- Heating capacity comparison
- PID vs PPO held-out scenario metrics

The held-out comparison chart includes mean and standard deviation for all four controllers.

## 9. Installation and Execution

### Requirements

- Python 3.11 recommended
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- Pandas
- Matplotlib
- Streamlit

### Setup

```bash
git clone <your-github-repository-url>
cd rl_temperature_control

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Run Tests

```bash
python -m unittest discover -s tests -v
```

### Train PPO

```bash
python train.py
```

### Evaluate PPO

```bash
python evaluate.py --model final_model/ppo_v3.zip
```

### Compare Controllers

```bash
python compare_models.py \
  --models \
  v1=models/ppo_temperature.zip \
  v2=models/ppo_v2.zip \
  v3=models/ppo_v3.zip
```

### Generate Held-Out Comparison Chart

```bash
python plot_random_summary.py
```

### Launch Dashboard

```bash
streamlit run app.py
```

## 10. Limitations and Future Improvements

The current implementation uses a simplified simulated thermal process.

Limitations include:
- Assumes a single uniform temperature.
- Does not model spatial temperature gradients.
- Uses a simplified linear heat-loss model.
- Does not include realistic actuator delays or faults.
- Has not been validated on physical industrial equipment.
- PPO control actions exhibit more variation than PID.

Potential improvements:
- Test multiple random seeds.
- Explore filtered temperature-change observations.
- Improve control smoothness without sacrificing tracking accuracy.
- Investigate model predictive control.
- Evaluate robustness under model mismatch and sensor faults.
- Explore safe reinforcement learning and simulation-to-real transfer.

## 11. Conclusion

This project implements and evaluates a PPO-based temperature controller against a tuned PID baseline in a simulated industrial thermal process.

The experiments demonstrate that PPO can achieve competitive tracking performance across randomized scenarios and selected disturbances. Reward-function tuning significantly affects the balance between tracking accuracy and control smoothness.

The results also highlight the continuing importance of conventional PID control as a reliable baseline for industrial temperature regulation.

**The project provides practical experience in reinforcement learning, process modeling, control systems, simulation, and experimental performance evaluation.**