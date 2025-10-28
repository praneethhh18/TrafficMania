import argparse
import numpy as np
from sumo_env_advanced import SumoTrafficEnvAdvanced


def run_episode(env: SumoTrafficEnvAdvanced, agent: str) -> float:
    obs = env.reset()
    total = 0.0
    prev_q = obs[-2]
    while True:
        if agent == 'random':
            action = np.random.randint(0, 2)
        elif agent == 'heuristic':
            # simple rule: if queue increases and we can switch, switch
            total_q = obs[-2]
            action = 1 if (total_q > prev_q and env._can_switch()) else 0
            prev_q = total_q
        else:
            action = 0
        obs, r, done, _ = env.step_env(action)
        total += r
        if done:
            break
    env.close()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='show SUMO GUI')
    parser.add_argument('--agent', default='heuristic', choices=['heuristic','random'])
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()

    env = SumoTrafficEnvAdvanced(gui=args.gui, max_steps=600)
    scores = []
    for ep in range(args.episodes):
        score = run_episode(env, args.agent)
        scores.append(score)
        print(f'Episode {ep+1}: reward={score:.2f}')
    print('Average reward:', np.mean(scores))

if __name__ == '__main__':
    main()
