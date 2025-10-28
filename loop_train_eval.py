import argparse
import os
import subprocess
import sys
import csv

PY = os.path.join('venv','Scripts','python.exe') if os.name == 'nt' else sys.executable


def run(cmd):
    print('>',' '.join(cmd))
    return subprocess.call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=600)
    p.add_argument('--log-base', type=str, default='iter_runs')
    p.add_argument('--decision-interval', type=int, default=4)
    p.add_argument('--min-green', type=int, default=10)
    p.add_argument('--alpha', type=float, default=1.2)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--gamma-r', type=float, default=0.3)
    p.add_argument('--model', type=str, default='dqn_sumo_advanced.pt')
    p.add_argument('--gui', action='store_true', help='Show GUI during evaluation runs')
    p.add_argument('--eval-first', action='store_true', help='Run SUMO GUI evaluation before training each round if a model exists')
    p.add_argument('--show-metrics', action='store_true', help='Open metrics.png after each evaluation or training when available')
    p.add_argument('--target-avg-reward', type=float, default=-2000.0, help='Stop when last-window average episode reward exceeds this')
    p.add_argument('--avg-window', type=int, default=5, help='Window size for average reward stopping criterion')
    args = p.parse_args()

    init = None
    for r in range(1, args.rounds+1):
        log_dir = os.path.join(args.log_base, f'round_{r}')
        os.makedirs(log_dir, exist_ok=True)
        # Optional evaluation first using current model
        if args.eval_first and init and os.path.exists(init):
            cmd_eval = [PY, 'run_dqn_policy.py', '--model', init]
            if args.gui:
                cmd_eval.append('--gui')
            code = run(cmd_eval)
            if code != 0:
                sys.exit(code)
            prev_metrics = os.path.join(args.log_base, f'round_{r-1}', 'metrics.png')
            if args.show_metrics and os.path.exists(prev_metrics):
                try:
                    if os.name == 'nt':
                        os.startfile(prev_metrics)
                except Exception:
                    pass
        # Train
        cmd_train = [PY, 'train_dqn_advanced.py',
                     '--episodes', str(args.episodes),
                     '--max-steps', str(args.max_steps),
                     '--log-dir', log_dir,
                     '--decision-interval', str(args.decision_interval),
                     '--min-green', str(args.min_green),
                     '--alpha', str(args.alpha),
                     '--beta', str(args.beta),
                     '--gamma-r', str(args.gamma_r),
                     '--model-out', args.model]
        if init:
            cmd_train += ['--init-model', init]
        code = run(cmd_train)
        if code != 0:
            sys.exit(code)
        init = args.model
        # Show training metrics if requested
        metrics_png = os.path.join(log_dir, 'metrics.png')
        if args.show_metrics and os.path.exists(metrics_png):
            try:
                if os.name == 'nt':
                    os.startfile(metrics_png)
            except Exception:
                pass
        # Check stopping criterion based on episode_log.csv
        ep_csv = os.path.join(log_dir, 'episode_log.csv')
        if os.path.exists(ep_csv):
            rewards = []
            try:
                with open(ep_csv, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if not row: continue
                        rewards.append(float(row[1]))
                if rewards:
                    window = min(args.avg_window, len(rewards))
                    avg_last = sum(rewards[-window:]) / window
                    print(f"Round {r}: avg reward over last {window} episodes = {avg_last:.2f}")
                    if avg_last >= args.target_avg_reward:
                        print("Target achieved. Stopping loop.")
                        break
            except Exception:
                pass
        # After training, run SUMO policy (optional GUI)
        cmd_eval = [PY, 'run_dqn_policy.py', '--model', args.model]
        if args.gui:
            cmd_eval.append('--gui')
        code = run(cmd_eval)
        if code != 0:
            sys.exit(code)
        print(f"Round {r} finished. See {log_dir} for CSV and metrics.png")


if __name__ == '__main__':
    main()
