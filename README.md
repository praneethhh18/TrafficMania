# Smart Traffic - SUMO + RL Baseline

This repo wires up a minimal SUMO simulation and a simple DQN agent to learn traffic light control in a 3x3 grid. You already have `vehicle_detector.py` for perception; this adds simulation + control training to generate policies in a virtual world.

## Files
- `net/traffic_network.net.xml`: generated 3x3 grid network with traffic lights
- `routes/traffic_routes.rou.xml`: generated flows using `randomTrips.py`
- `traffic_config.sumocfg`: SUMO configuration tying net + routes
- `sumo_env.py`: Python TraCI environment wrapper for RL
- `train_dqn.py`: PyTorch DQN trainer (action space: keep/switch phase)
- `requirements.txt`: Python dependencies

## Prereqs
- SUMO installed and `SUMO_HOME` set (Windows example: `C:\\Program Files (x86)\\Eclipse\\Sumo`)
- Python 3.9+

## Setup (PowerShell)
```powershell
# (Optional) create venv
python -m venv .venv
. .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

## TL;DR: Three simple commands

1) Open the SUMO world (GUI):

```powershell
sumo-gui -c traffic_config.sumocfg --quit-on-end false
```

2) Train a small policy (no GUI, a few minutes):

```powershell
python train_dqn_advanced.py
```

3) Run the trained policy (with GUI):

```powershell
python run_dqn_policy.py --gui --model dqn_sumo_advanced.pt
```

## Regenerate network and routes (optional)

```powershell
# 3x3 grid, 2 lanes, auto TLS
netgenerate --grid --grid.number=3 --default.lanenumber=2 --default.speed=13.9 --tls.guess=1 --output-file net/traffic_network.net.xml

# create trips and routes for 10 simulated minutes
python "$env:SUMO_HOME/tools/randomTrips.py" -n net/traffic_network.net.xml -e 600 -l --binomial=2 --period 1.5 --route-file routes/traffic_routes.rou.xml --seed 42
```

## Run the sim in GUI

```powershell
sumo-gui -c traffic_config.sumocfg
```

## Quick sanity run (no GUI)

```powershell
# Smart Traffic RL (SUMO + DQN)

This project trains a traffic light controller in SUMO using a Double Dueling DQN with Prioritized Experience Replay (PER), safety shaping, and real-world constraints like ambulance priority and red-light rule enforcement.

## Highlights

- Advanced SUMO environment (`sumo_env_advanced.py`):
	- Safe switching with a minimum green and decision interval to prevent thrashing
	- Reward shaping: queue, wait, switch penalty, arrivals reward, collision penalties
	- Red-light violation detection with per-step penalties, optional violator logging
	- Ambulance priority: auto-detect emergency vehicles and switch to a serving phase with cooldown
- Trainer (`train_dqn_advanced.py`):
	- Double Dueling DQN + PER (importance sampling)
	- Epsilon schedule controls exploration
	- Checkpoints, best moving-average save, CSV and optional TensorBoard logging
- Runner (`run_dqn_policy.py`): headless or GUI evaluation

## Quick start

1) Install SUMO and set `SUMO_HOME` environment variable.
2) Create a Python venv and install requirements used in your environment (PyTorch, NumPy, etc.).
3) Train:

```powershell
# Final fine-tune example used in this repo
& "venv/Scripts/python.exe" .\train_dqn_advanced.py `
	--episodes 60 --max-steps 600 --no-plot `
	--log-dir ".\final_finetune" `
	--init-model .\dqn_positive_per_final.pt `
	--model-out .\dqn_final_finetune.pt `
	--alpha 0.02 --beta 0.005 --gamma-r 0.02 `
	--arrival-reward 4.0 --violation-penalty 5 --collision-penalty 30 `
	--decision-interval 5 --min-green 12 `
	--per --per-alpha 0.6 --per-beta-start 0.6 --per-beta-end 1.0 --per-beta-steps 200000 `
	--eps-start 0.25 --eps-end 0.03 --eps-decay-steps 35000 `
	--ckpt-every 10 --tensorboard
```

4) Evaluate (headless):

```powershell
& "venv/Scripts/python.exe" .\run_dqn_policy.py `
	--model .\dqn_final_finetune.pt `
	--max-steps 600 `
	--min-green 12 --decision-interval 5 `
	--alpha 0.02 --beta 0.005 --gamma-r 0.02 `
	--collision-penalty 30 --arrival-reward 4.0
```

5) Demo (GUI):

```powershell
& "venv/Scripts/python.exe" .\run_dqn_policy.py `
	--model .\dqn_final_finetune.pt `
	--gui --max-steps 300 `
	--min-green 12 --decision-interval 5 `
	--alpha 0.02 --beta 0.005 --gamma-r 0.02 `
	--collision-penalty 30 --arrival-reward 4.0
```

## Final results (this run)

- Headless (600 steps, arrival_reward=4.5):
	- total reward: -1937.42
	- avg_total_q: 1.18
	- avg_total_w: 8.36
	- collisions_sum: 0
	- arrivals_sum: 245
- GUI (300 steps, arrival_reward=4.5):
	- total reward: -1236.79
	- avg_total_q: 1.28
	- avg_total_w: 9.86
	- collisions_sum: 0
	- arrivals_sum: 108

Notes:
- Totals include queue/wait penalties, so they can remain negative even with improved behavior. We maintained zero collisions, rule enforcement penalties, ambulance priority, and reasonable queues/waits.

## Configuration tips

- Make totals less negative: raise `--arrival-reward` (e.g., 4.0–5.0). Keep an eye on queues.
- Harsher rule enforcement: raise `--violation-penalty` (e.g., 8–10) or enable escalation in code (`violation_reoffender_factor > 0`).
- Stability: keep `--decision-interval` and `--min-green` to avoid oscillations.
- Exploration: lower `--eps-start` and lengthen `--eps-decay-steps` when fine-tuning from a good model.

## Files

- `sumo_env_advanced.py`: SUMO environment with safety shaping, ambulance priority, and rule enforcement
- `train_dqn_advanced.py`: Dueling DQN + PER trainer
- `run_dqn_policy.py`: Evaluate in headless/GUI
- `routes/`, `net/`, `traffic_config*.sumocfg`: SUMO network and configuration (if present in your setup)

## Troubleshooting

- SUMO_HOME not set: install SUMO and set the environment variable before running.
- TensorBoard: use your env’s `tensorboard` executable; if `python -m tensorboard` fails, try `tensorboard --logdir <dir>`.
- Teleports: if you see “waited too long (wrong lane)”, refine your lane connections or TLS phases in the network.

## From simulation to real-world traffic management

This project is a simulation prototype designed to transfer to real-world traffic signal control to help maintain road safety and improve flow. A safe and practical path to deployment typically includes:

- Hardware and controller integration
	- Interface with existing signal controllers (e.g., via NTCIP or a vendor SDK). Keep the min-green and intergreen (yellow/all-red) enforced in hardware as a safety net.
	- Keep a conservative fallback plan (fixed-time or actuated) that can take over instantly.
- Perception and priority
	- Tie your camera/ANPR feeds to ambulance detection and call `notify_ambulance(lane_id)` when an emergency vehicle queues at red.
	- Maintain red-light violation logging for auditing and safety analytics; ensure compliance with privacy laws.
- Safety gate and shadow mode
	- Run in “shadow” first: the AI proposes decisions while the controller still follows the legacy plan; compare and audit outcomes.
	- Add a safety gate (e.g., block risky rapid toggles, enforce min-green, cap switch frequency, prevent phase jumps that violate clearance).
- Robust training and sim2real
	- Calibrate simulation with real counts/turning ratios; randomize demand and weather to improve generalization.
	- Periodically retrain offline on recorded data; roll out carefully with A/B or time-window trials.

Important: This software is not a substitute for certified traffic controllers. Real deployments require adherence to local regulations, rigorous testing, and sign-off from traffic engineering authorities.
