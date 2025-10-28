# Smart Traffic Management System — Requirements Mapping

This document maps the interview prompt (data → model → deployment) to what exists in this repo, what’s partially done, and what we added now.

## 1) Data Understanding

- What to collect: vehicle count, type, speed, time-of-day, camera ID, optional weather.
  - Status: Added `data_pipeline/collector.py` to extract counts/types/speeds from video/webcam using YOLOv8, with CSV output. Time-of-day and camera ID included; weather left as a field for future integration.

- Handle incomplete/noisy data:
  - Status: Implemented basic tracking + smoothing (centroid-based), confidence thresholds, and optional frame anonymization in `data_pipeline/privacy.py`. Notes included for adding plate/face detectors for stronger privacy.

## 2) Model Design

- Techniques: CV for detection (YOLOv8), lightweight tracking to estimate speed; RL (Double Dueling DQN) for signal timing.
  - Status: Complete. See `train_dqn_advanced.py`, `sumo_env_advanced.py`. Live dashboards via TensorBoard and live Matplotlib.

- Train & validate: SUMO environment with baselines and policy runner.
  - Status: Complete. Policies can be trained headless or with GUI; evaluation via `run_dqn_policy.py`.

## 3) System Implementation

- Integration with traffic lights/control systems:
  - Status: Complete for simulation through TraCI (SUMO). New `realtime_controller.py` demonstrates a controller loop that uses the trained policy to control a SUMO TLS in real-time; falls back to a heuristic or fixed plan if model missing.

- Real-time processing & reliability:
  - Status: Implemented decision intervals, min-green; controller has try/except with fallback to fixed-time plan; logs actions to CSV.

## 4) Ethical and Practical Considerations

- Privacy from video data:
  - Status: `data_pipeline/privacy.py` supports anonymization (full-frame blur option) and recommends stronger plate/face redaction for production. The collector defaults to not saving raw frames; only structured counts are saved.

- Robustness (weather/network failures):
  - Status: Controller has fallback control strategy on errors/timeouts; SUMO-side teleport prevents deadlocks.

## Gaps and Next Steps

- Weather integration: Add a simple weather fetcher (OpenWeatherMap) and enrich CSVs. Not included to avoid adding API keys.
- Advanced tracking: Replace centroid tracking with DeepSORT/ByteTrack for precise ID and speed. Requires extra dependencies.
- Real TLS adapter: For deployment on real intersections, implement an adapter to the field controller (e.g., NTCIP). Out of scope here but the controller design anticipates a pluggable backend.

## How to try

- Collect data from webcam:
  - `python data_pipeline/collector.py --source 0 --save-csv data/events.csv --display --anonymize`

- Train RL with dashboards:
  - `python train_dqn_advanced.py --episodes 50 --max-steps 600 --log-dir .\\tb_run --tensorboard`

- Run real-time SUMO control with trained model:
  - `python realtime_controller.py --gui --model dqn_sumo_advanced.pt --max-steps 600`
