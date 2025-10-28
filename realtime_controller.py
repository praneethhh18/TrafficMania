import argparse
import torch

from sumo_env_advanced import SumoTrafficEnvAdvanced
from train_dqn_advanced import DuelingDQN, Device


def control_loop(model_path: str, gui: bool, max_steps: int, fallback_fixed: bool = True):
    env = SumoTrafficEnvAdvanced(gui=gui, max_steps=max_steps)
    obs = env.reset()
    obs_size = len(obs)
    n_actions = 2

    policy = None
    try:
        policy = DuelingDQN(obs_size, n_actions).to(Device)
        policy.load_state_dict(torch.load(model_path, map_location=Device))
        policy.eval()
    except Exception:
        if fallback_fixed:
            print('Model not available; falling back to simple fixed-time plan.')
        else:
            raise

    total_r = 0.0
    s = obs
    t = 0
    try:
        while True:
            a = 0
            if policy is not None:
                with torch.no_grad():
                    q = policy(torch.tensor(s, dtype=torch.float32, device=Device).unsqueeze(0))
                    a = int(torch.argmax(q, dim=1).item())
            else:
                # Fixed-time heuristic: switch every 12 steps
                a = 1 if (t % 12 == 0) else 0
            ns, r, done, info = env.step_env(a)
            s = ns
            t += 1
            total_r += r
            if done:
                break
    finally:
        env.close()
    print(f'Control loop finished. total reward={total_r:.2f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='dqn_sumo_advanced.pt')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--max-steps', type=int, default=600)
    p.add_argument('--no-fallback', action='store_true')
    args = p.parse_args()
    control_loop(args.model, args.gui, args.max_steps, fallback_fixed=(not args.no_fallback))


if __name__ == '__main__':
    main()
