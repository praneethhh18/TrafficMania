import argparse
import torch
import numpy as np

from sumo_env_advanced import SumoTrafficEnvAdvanced
from train_dqn_advanced import DuelingDQN, Device


def run(model_path: str, gui: bool, max_steps: int, min_green: int = 8, alpha: float = 1.0, beta: float = 0.1, gamma_r: float = 0.2, decision_interval: int = 0, collision_penalty: float = 10.0, arrival_reward: float = 0.2, sumocfg: str = 'traffic_config.sumocfg', ambulance_requests_file: str = None):
    env = SumoTrafficEnvAdvanced(sumocfg=sumocfg, gui=gui, max_steps=max_steps, min_green=min_green, alpha=alpha, beta=beta, gamma=gamma_r, collision_penalty=collision_penalty, arrival_reward=arrival_reward, ambulance_requests_file=ambulance_requests_file)
    # Build network based on first observation size
    obs = env.reset()
    obs_size = len(obs)
    n_actions = 2
    policy = DuelingDQN(obs_size, n_actions).to(Device)
    try:
        policy.load_state_dict(torch.load(model_path, map_location=Device))
    except FileNotFoundError:
        print(f"Model file not found: {model_path}. Train first with train_dqn_advanced.py")
        env.close()
        return
    policy.eval()

    total_r = 0.0
    steps = 0
    collisions_sum = 0
    arrivals_sum = 0
    qs, ws = [], []
    s = obs
    t = 0
    while True:
        with torch.no_grad():
            q = policy(torch.tensor(s, dtype=torch.float32, device=Device).unsqueeze(0))
            a = int(torch.argmax(q, dim=1).item())
        # Optional gating to mirror training's decision interval
        eff_a = 1 if (a == 1 and decision_interval > 0 and (t % decision_interval == 0)) else (a if decision_interval == 0 else 0)
        ns, r, done, info = env.step_env(eff_a)
        s = ns
        total_r += r
        steps += 1
        if info is not None:
            collisions_sum += int(info.get('collisions', 0))
            arrivals_sum += int(info.get('arrivals', 0))
            if 'total_q' in info:
                qs.append(float(info['total_q']))
            if 'total_w' in info:
                ws.append(float(info['total_w']))
        t += 1
        if done:
            break
    env.close()
    avg_q = float(np.nanmean(qs)) if len(qs) else float('nan')
    avg_w = float(np.nanmean(ws)) if len(ws) else float('nan')
    print(f"Run finished. total reward={total_r:.2f}")
    print(f"Steps={steps} avg_total_q={avg_q:.2f} avg_total_w={avg_w:.2f} collisions_sum={collisions_sum} arrivals_sum={arrivals_sum}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dqn_sumo_advanced.pt')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--min-green', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma-r', type=float, default=0.2)
    parser.add_argument('--decision-interval', type=int, default=0)
    parser.add_argument('--collision-penalty', type=float, default=10.0)
    parser.add_argument('--arrival-reward', type=float, default=0.2)
    parser.add_argument('--sumocfg', type=str, default='traffic_config.sumocfg')
    parser.add_argument('--ambulance-requests-file', type=str, default=None, help='Optional text file path to ingest lane_ids for ambulance priority (one per line)')
    args = parser.parse_args()
    run(args.model, args.gui, args.max_steps, min_green=args.min_green, alpha=args.alpha, beta=args.beta, gamma_r=args.gamma_r, decision_interval=args.decision_interval, collision_penalty=args.collision_penalty, arrival_reward=args.arrival_reward, sumocfg=args.sumocfg, ambulance_requests_file=args.ambulance_requests_file)


if __name__ == '__main__':
    main()
