import argparse
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from train_dqn_advanced import DuelingDQN, Device


class DecideRequest(BaseModel):
    observation: list  # [queues..., total_q, total_w, phase_idx, time_since_switch]
    # Optional external timing if available
    phase_idx: Optional[int] = None
    time_since_switch: Optional[float] = None


def load_policy(model_path: str, obs_size: int) -> DuelingDQN:
    policy = DuelingDQN(obs_size, 2).to(Device)
    state = torch.load(model_path, map_location=Device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def build_app(model_path: str, obs_size: int, min_green: int, decision_interval: int):
    app = FastAPI(title="Smart Traffic RL Policy", version="1.0")

    policy = load_policy(model_path, obs_size)

    # Simple stateful safety gate
    state = {
        'last_switch_step': 0,
        'step': 0,
    }

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(Device)}

    @app.post("/decide")
    def decide(req: DecideRequest):
        obs = np.array(req.observation, dtype=np.float32)
        if obs.shape[0] != obs_size:
            return {"error": f"obs_size mismatch: expected {obs_size}, got {obs.shape[0]}"}
        with torch.no_grad():
            q = policy(torch.tensor(obs, dtype=torch.float32, device=Device).unsqueeze(0))
            a = int(torch.argmax(q, dim=1).item())
        # Safety gate: min_green and decision interval
        step = state['step']
        time_since_switch = req.time_since_switch if req.time_since_switch is not None else float(obs[-1])
        can_switch_min = time_since_switch >= float(min_green)
        can_switch_interval = (decision_interval <= 0) or (step % max(1, decision_interval) == 0)
        eff_a = 1 if (a == 1 and can_switch_min and can_switch_interval) else 0
        if eff_a == 1:
            state['last_switch_step'] = step
        state['step'] = step + 1
        return {
            "action": a,
            "action_effective": eff_a,
            "can_switch_min_green": can_switch_min,
            "can_switch_interval": can_switch_interval,
        }

    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='dqn_sumo_advanced_curriculum.pt')
    p.add_argument('--obs-size', type=int, required=True, help='Observation vector length expected by the model')
    p.add_argument('--min-green', type=int, default=12)
    p.add_argument('--decision-interval', type=int, default=5)
    p.add_argument('--host', type=str, default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()

    app = build_app(args.model, args.obs_size, args.min_green, args.decision_interval)
    uvicorn.run(app, host=args.host, port=args.port)
