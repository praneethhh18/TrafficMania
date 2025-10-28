import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import os
import sys
import subprocess

from sumo_env_advanced import SumoTrafficEnvAdvanced
def _reseed_routes(
    seed_base: int,
    end_time: int,
    period_cars: float,
    period_trucks: float,
    period_buses: float,
    period_bikes: float,
    net_path: str = 'net/traffic_network.net.xml',
    routes_dir: str = 'routes',
):
    """Regenerate route files with new seeds using SUMO's randomTrips.py.
    This replaces cars.rou.xml, trucks.rou.xml, buses.rou.xml, bikes.rou.xml.
    """
    try:
        sumo_home = os.environ.get('SUMO_HOME')
        if not sumo_home:
            print('[RESEED] SUMO_HOME not set; skipping reseed.')
            return
        tool = os.path.join(sumo_home, 'tools', 'randomTrips.py')
        py = sys.executable or 'python'
        os.makedirs(routes_dir, exist_ok=True)
        cmds = [
            [py, tool, '-n', net_path, '-e', str(end_time), '-l', '--period', str(period_cars), '-r', os.path.join(routes_dir, 'cars.rou.xml'), '--seed', str(seed_base+1), '--vtype', 'passenger', '--prefix', 'car_'],
            [py, tool, '-n', net_path, '-e', str(end_time), '-l', '--period', str(period_trucks), '-r', os.path.join(routes_dir, 'trucks.rou.xml'), '--seed', str(seed_base+2), '--vtype', 'truck', '--prefix', 'truck_'],
            [py, tool, '-n', net_path, '-e', str(end_time), '-l', '--period', str(period_buses), '-r', os.path.join(routes_dir, 'buses.rou.xml'), '--seed', str(seed_base+3), '--vtype', 'bus', '--prefix', 'bus_'],
            [py, tool, '-n', net_path, '-e', str(end_time), '-l', '--period', str(period_bikes), '-r', os.path.join(routes_dir, 'bikes.rou.xml'), '--seed', str(seed_base+4), '--vtype', 'motorbike', '--prefix', 'bike_'],
        ]
        for c in cmds:
            try:
                subprocess.run(c, check=True)
            except Exception as e:
                print(f"[RESEED] Failed command: {' '.join(c)} -> {e}")
        print(f"[RESEED] Regenerated routes with base seed {seed_base}")
    except Exception as e:
        print(f"[RESEED] Unexpected error: {e}")


Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DuelingDQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.adv(f)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

class Replay:
    def __init__(self, cap=50000):
        from collections import deque
        self.buf = deque(maxlen=cap)
    def push(self, s,a,r,ns,d):
        self.buf.append((s,a,r,ns,d))
    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s,a,r,ns,d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32, device=Device)
        a = torch.tensor(a, dtype=torch.int64, device=Device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=Device).unsqueeze(1)
        ns = torch.tensor(np.array(ns), dtype=torch.float32, device=Device)
        d = torch.tensor(d, dtype=torch.float32, device=Device).unsqueeze(1)
        return s,a,r,ns,d
    def __len__(self):
        return len(self.buf)


class PrioritizedReplay:
    """Simple Prioritized Experience Replay (PER) buffer.
    Sampling uses probability proportional to priority**alpha.
    Importance-sampling weights are computed with annealed beta.
    Implementation is O(N) for computing probabilities, which is acceptable for moderate buffers.
    """
    def __init__(self, cap=50000, alpha: float = 0.6, beta_start: float = 0.4, beta_end: float = 1.0, beta_steps: int = 100000, eps: float = 1e-3):
        self.cap = cap
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = max(1, beta_steps)
        self.beta = beta_start
        self.eps = eps
        self.data = []  # list of (s,a,r,ns,d)
        self.priorities = []  # list of float priorities
        self.pos = 0
        self._steps = 0

    def _current_beta(self):
        # Linear annealing of beta
        t = min(1.0, self._steps / float(self.beta_steps))
        return self.beta_start + t * (self.beta_end - self.beta_start)

    def push(self, s, a, r, ns, d):
        max_prio = max(self.priorities) if self.priorities else 1.0
        if len(self.data) < self.cap:
            self.data.append((s, a, r, ns, d))
            self.priorities.append(max_prio)
        else:
            self.data[self.pos] = (s, a, r, ns, d)
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.cap
        self._steps += 1

    def sample(self, bs):
        n = len(self.data)
        prios = np.array(self.priorities[:n], dtype=np.float64)
        if prios.ndim == 0:
            prios = np.array([float(prios)])
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.ones_like(probs) / float(len(probs))
        else:
            probs = probs / probs_sum
        idxs = np.random.choice(n, bs, p=probs)
        batch = [self.data[i] for i in idxs]
        s,a,r,ns,d = zip(*batch)
        S = torch.tensor(np.array(s), dtype=torch.float32, device=Device)
        A = torch.tensor(a, dtype=torch.int64, device=Device).unsqueeze(1)
        R = torch.tensor(r, dtype=torch.float32, device=Device).unsqueeze(1)
        NS = torch.tensor(np.array(ns), dtype=torch.float32, device=Device)
        D = torch.tensor(d, dtype=torch.float32, device=Device).unsqueeze(1)
        # IS weights
        beta = self._current_beta()
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = (n * probs[idxs]) ** (-beta)
        # normalize by max to keep weights in [0,1]
        weights = weights / (weights.max() + 1e-8)
        W = torch.tensor(weights, dtype=torch.float32, device=Device).unsqueeze(1)
        return S, A, R, NS, D, W, idxs

    def update_priorities(self, idxs, td_errors):
        # td_errors: torch tensor or np array
        errs = td_errors.detach().abs().view(-1).to('cpu').numpy() if isinstance(td_errors, torch.Tensor) else np.abs(td_errors).reshape(-1)
        for i, e in zip(idxs, errs):
            self.priorities[i] = float(e + self.eps)

    def __len__(self):
        return len(self.data)


def train(episodes=30, max_steps=300, gui=False, log_csv: Optional[str] = None, log_dir: Optional[str] = None, plot: bool = True, tensorboard: bool = False, tb_logdir: Optional[str] = None, live_plot: bool = False, decision_interval: int = 3, alpha: float = 1.0, beta: float = 0.1, gamma_r: float = 0.2, min_green: int = 8, init_model: Optional[str] = None, model_out: str = 'dqn_sumo_advanced.pt', collision_penalty: float = 10.0, arrival_reward: float = 0.2, violation_penalty: float = 50.0, violations_log: Optional[str] = None, lr: float = 1e-3, tau: float = 0.0, lr_step: int = 50, lr_gamma: float = 0.95, save_best: Optional[str] = None, best_window: int = 20, ckpt_every: int = 0, ckpt_dir: Optional[str] = None, reseed_every: int = 0, reseed_start: int = 0, reseed_seed_base: int = 500, period_cars: float = 6.0, period_trucks: float = 12.0, period_buses: float = 14.0, period_bikes: float = 8.0, sumocfg: str = 'traffic_config.sumocfg',
          use_per: bool = True, per_alpha: float = 0.6, per_beta_start: float = 0.4, per_beta_end: float = 1.0, per_beta_steps: int = 100000, per_eps: float = 1e-3,
          eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_steps: int = 3000,
          ambulance_requests_file: Optional[str] = None):
    env = SumoTrafficEnvAdvanced(sumocfg=sumocfg, gui=gui, max_steps=max_steps, min_green=min_green, alpha=alpha, beta=beta, gamma=gamma_r, collision_penalty=collision_penalty, arrival_reward=arrival_reward, violation_penalty=violation_penalty, violations_log=violations_log, ambulance_requests_file=ambulance_requests_file)
    obs = env.reset()
    obs_size = len(obs)
    n_actions = 2

    policy = DuelingDQN(obs_size, n_actions).to(Device)
    target = DuelingDQN(obs_size, n_actions).to(Device)
    # Initialize from checkpoint if provided
    if init_model and os.path.exists(init_model):
        try:
            state = torch.load(init_model, map_location=Device)
            policy.load_state_dict(state)
            print(f"Loaded initial weights from {init_model}")
        except Exception as e:
            print(f"Warning: failed to load init model {init_model}: {e}")
    target.load_state_dict(policy.state_dict())
    target.eval()

    opt = optim.Adam(policy.parameters(), lr=lr)
    scheduler = None
    if lr_step and lr_step > 0:
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
    # Replay buffer (optionally Prioritized)
    if use_per:
        buf = PrioritizedReplay(cap=30000, alpha=per_alpha, beta_start=per_beta_start, beta_end=per_beta_end, beta_steps=per_beta_steps, eps=per_eps)
    else:
        buf = Replay(30000)

    gamma = 0.99
    bs = 64
    eps_decay = max(1, int(eps_decay_steps))
    steps = 0

    import csv
    # Resolve logging directory
    if log_dir is None:
        if log_csv:
            # Use the directory of the provided CSV if any; else default to ./logs
            base = os.path.dirname(os.path.abspath(log_csv))
            log_dir = base if base else os.path.join(os.getcwd(), 'logs')
        else:
            log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Step-level CSV
    step_csv_path = log_csv if log_csv else os.path.join(log_dir, 'training_log.csv')
    log_f = open(step_csv_path, 'w', newline='')
    writer_csv = csv.writer(log_f)
    writer_csv.writerow(['episode','t','reward','total_q','total_w','phase','eps','collisions','arrivals'])

    # Episode-level CSV
    ep_csv_path = os.path.join(log_dir, 'episode_log.csv')
    ep_f = open(ep_csv_path, 'w', newline='')
    ep_writer_csv = csv.writer(ep_f)
    ep_writer_csv.writerow(['episode','episode_reward','avg_total_q','avg_total_w','switches','collisions_sum','arrivals_sum'])

    # For plotting at end and optional live dashboard
    rewards_hist, avg_q_hist, avg_w_hist, switch_hist = [], [], [], []
    best_ma = float('-inf')
    if save_best is None:
        # Default best model path inside log_dir
        save_best = os.path.join(log_dir, 'best_model.pt')
    if ckpt_dir is None and ckpt_every and ckpt_every > 0:
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    live_fig = None
    live_axes = None
    if live_plot:
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            live_fig = plt.figure(figsize=(10, 6))
            ax1 = live_fig.add_subplot(2,1,1)
            ax2 = live_fig.add_subplot(2,1,2)
            live_axes = (ax1, ax2)
        except Exception as e:
            print(f"Live plot disabled (matplotlib error): {e}")
            live_fig = None
            live_axes = None

    # TensorBoard (optional)
    tb_writer = None
    global_step = 0
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = tb_logdir if tb_logdir else os.path.join(log_dir, 'tb')
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir)
        except Exception as e:
            print(f"TensorBoard disabled (unable to initialize writer): {e}")
            tb_writer = None

    for ep in range(episodes):
        # Optional curriculum reseeding before environment reset
        if reseed_every and reseed_every > 0 and ep >= reseed_start and (ep % reseed_every == 0):
            _reseed_routes(
                seed_base=reseed_seed_base + ep,
                end_time=max_steps,
                period_cars=period_cars,
                period_trucks=period_trucks,
                period_buses=period_buses,
                period_bikes=period_bikes,
            )
        s = env.reset()
        ep_r = 0.0
        qs, ws = [], []
        switches = 0
        collisions_sum = 0
        arrivals_sum = 0
        for t in range(max_steps):
            eps = eps_end + (eps_start - eps_end) * math.exp(-steps/eps_decay)
            steps += 1
            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(s, dtype=torch.float32, device=Device).unsqueeze(0))
                    a = int(torch.argmax(q, dim=1).item())
            # Enforce decision interval: only allow switching at multiples of interval
            effective_a = 1 if (a == 1 and (t % max(1, decision_interval) == 0)) else 0
            ns, r, done, info = env.step_env(effective_a)
            buf.push(s,a,r,ns,done)
            s = ns
            ep_r += r
            if a == 1:
                switches += 1
            # Collect per-step metrics for episode averages
            qs.append(float(info.get('total_q', float('nan'))))
            ws.append(float(info.get('total_w', float('nan'))))

            if writer_csv is not None:
                coll = int(info.get('collisions', 0)) if info is not None else 0
                arr = int(info.get('arrivals', 0)) if info is not None else 0
                collisions_sum += coll
                arrivals_sum += arr
                writer_csv.writerow([ep, t, float(r), info.get('total_q', float('nan')), info.get('total_w', float('nan')), info.get('phase', -1), eps, coll, arr])

            # TensorBoard step logs
            if tb_writer is not None:
                tb_writer.add_scalar('step/reward', float(r), global_step)
                if info is not None:
                    if 'total_q' in info:
                        tb_writer.add_scalar('step/total_q', float(info['total_q']), global_step)
                    if 'total_w' in info:
                        tb_writer.add_scalar('step/total_w', float(info['total_w']), global_step)
                    if 'phase' in info:
                        tb_writer.add_scalar('step/phase', float(info['phase']), global_step)
                tb_writer.add_scalar('step/epsilon', float(eps), global_step)
                tb_writer.add_scalar('step/action_chosen', float(a), global_step)
                tb_writer.add_scalar('step/action_effective', float(effective_a), global_step)
                if info is not None:
                    if 'collisions' in info:
                        tb_writer.add_scalar('step/collisions', float(info['collisions']), global_step)
                    if 'arrivals' in info:
                        tb_writer.add_scalar('step/arrivals', float(info['arrivals']), global_step)
            global_step += 1

            if len(buf) >= bs:
                if isinstance(buf, PrioritizedReplay):
                    S,A,R,NS,D,W,idxs = buf.sample(bs)
                else:
                    S,A,R,NS,D = buf.sample(bs)
                q = policy(S).gather(1, A)
                with torch.no_grad():
                    # Double DQN: action from policy, value from target
                    next_actions = policy(NS).argmax(1, keepdim=True)
                    max_next = target(NS).gather(1, next_actions)
                    y = R + gamma * max_next * (1 - D)
                td_err = q - y
                if isinstance(buf, PrioritizedReplay):
                    # Weighted Huber loss
                    loss_unreduced = nn.functional.smooth_l1_loss(q, y, reduction='none')
                    loss = (W * loss_unreduced).mean()
                else:
                    loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                opt.step()
                if isinstance(buf, PrioritizedReplay):
                    buf.update_priorities(idxs, td_err)

            if steps % 1000 == 0:
                if tau and tau > 0.0:
                    # Soft update continuously
                    with torch.no_grad():
                        for tp, pp in zip(target.parameters(), policy.parameters()):
                            tp.data.copy_(tp.data * (1.0 - tau) + pp.data * tau)
                else:
                    # Periodic hard update
                    target.load_state_dict(policy.state_dict())

            if done:
                break
        # Compute episode aggregates
        avg_q = float(np.nanmean(qs)) if len(qs) else float('nan')
        avg_w = float(np.nanmean(ws)) if len(ws) else float('nan')
        ep_writer_csv.writerow([ep, ep_r, avg_q, avg_w, switches, collisions_sum, arrivals_sum])
        # TensorBoard episode logs
        if tb_writer is not None:
            tb_writer.add_scalar('episode/reward', float(ep_r), ep)
            if not (np.isnan(avg_q) or np.isinf(avg_q)):
                tb_writer.add_scalar('episode/avg_total_q', float(avg_q), ep)
            if not (np.isnan(avg_w) or np.isinf(avg_w)):
                tb_writer.add_scalar('episode/avg_total_w', float(avg_w), ep)
            tb_writer.add_scalar('episode/switches', float(switches), ep)
            tb_writer.add_scalar('episode/collisions_sum', float(collisions_sum), ep)
            tb_writer.add_scalar('episode/arrivals_sum', float(arrivals_sum), ep)
            if scheduler is not None:
                tb_writer.add_scalar('train/lr', float(scheduler.get_last_lr()[0]), ep)
        rewards_hist.append(ep_r)
        avg_q_hist.append(avg_q)
        avg_w_hist.append(avg_w)
        switch_hist.append(switches)
        print(f"Episode {ep+1}/{episodes} reward={ep_r:.2f} avg_q={avg_q:.2f} avg_w={avg_w:.2f} switches={switches}")

        # Best moving-average checkpoint
        if best_window and len(rewards_hist) >= max(1, best_window):
            import numpy as _np
            w = min(best_window, len(rewards_hist))
            ma = float(_np.mean(rewards_hist[-w:]))
            if ma > best_ma:
                best_ma = ma
                try:
                    torch.save(policy.state_dict(), save_best)
                    print(f"[CKPT] New best MA({w})={ma:.2f}. Saved best to {save_best}")
                except Exception as e:
                    print(f"[CKPT] Failed to save best model: {e}")

        # Periodic checkpoints
        if ckpt_every and ckpt_every > 0 and ((ep + 1) % ckpt_every == 0) and ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, f"ep_{ep+1:04d}.pt")
            try:
                torch.save(policy.state_dict(), ckpt_path)
                print(f"[CKPT] Saved checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"[CKPT] Failed to save checkpoint: {e}")

        # Step LR scheduler per episode
        if scheduler is not None:
            scheduler.step()

        # Live plot update
        if live_axes is not None:
            import matplotlib.pyplot as plt
            x = np.arange(1, len(rewards_hist)+1)
            ax1, ax2 = live_axes
            ax1.clear(); ax2.clear()
            # Reward with moving average (window 10)
            ax1.plot(x, rewards_hist, label='Episode Reward', alpha=0.6)
            if len(rewards_hist) >= 2:
                w = min(10, len(rewards_hist))
                ma = np.convolve(rewards_hist, np.ones(w)/w, mode='valid')
                ax1.plot(np.arange(w, len(rewards_hist)+1), ma, color='red', label=f'MA({w})')
            ax1.set_ylabel('Reward'); ax1.set_xlabel('Episode'); ax1.grid(True); ax1.legend()
            # Avg Queue per episode
            ax2.plot(x, avg_q_hist, color='orange', label='Avg Total Queue')
            ax2.set_ylabel('Avg Queue'); ax2.set_xlabel('Episode'); ax2.grid(True); ax2.legend()
            live_fig.tight_layout()
            plt.pause(0.001)
    env.close()
    torch.save(policy.state_dict(), model_out)
    print(f'Saved model to {model_out}')
    # Close CSVs
    log_f.close()
    ep_f.close()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    # Plot metrics if requested
    if plot:
        try:
            import matplotlib.pyplot as plt
            x = list(range(1, len(rewards_hist)+1))
            plt.figure(figsize=(12, 8))
            # Rewards
            ax1 = plt.subplot(3,1,1)
            ax1.plot(x, rewards_hist, label='Episode Reward')
            ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward'); ax1.grid(True)
            ax1.legend()
            # Avg queue
            ax2 = plt.subplot(3,1,2)
            ax2.plot(x, avg_q_hist, color='orange', label='Avg Total Queue')
            ax2.set_xlabel('Episode'); ax2.set_ylabel('Avg Queue'); ax2.grid(True)
            ax2.legend()
            # Avg wait
            ax3 = plt.subplot(3,1,3)
            ax3.plot(x, avg_w_hist, color='green', label='Avg Total Wait')
            ax3.set_xlabel('Episode'); ax3.set_ylabel('Avg Wait'); ax3.grid(True)
            ax3.legend()
            plt.tight_layout()
            out_png = os.path.join(log_dir, 'metrics.png')
            plt.savefig(out_png)
            print(f'Wrote plots to {out_png}')
        except Exception as e:
            print(f"Plotting skipped (install matplotlib to enable). Reason: {e}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--max-steps', type=int, default=600)
    p.add_argument('--gui', action='store_true')
    p.add_argument('--log-csv', type=str, default='training_log.csv')
    p.add_argument('--log-dir', type=str, default=None, help='Directory to store logs and plots (default: logs or dir of --log-csv)')
    p.add_argument('--no-plot', action='store_true', help='Disable plotting after training')
    p.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard live logging')
    p.add_argument('--tb-logdir', type=str, default=None, help='TensorBoard log directory (default: <log-dir>/tb)')
    p.add_argument('--live-plot', action='store_true', help='Show a live matplotlib dashboard during training')
    p.add_argument('--decision-interval', type=int, default=3, help='Only allow switching every N steps to reduce thrashing (default: 3)')
    p.add_argument('--alpha', type=float, default=1.0, help='Reward weight for total queue')
    p.add_argument('--beta', type=float, default=0.1, help='Reward weight for total waiting time')
    p.add_argument('--gamma-r', type=float, default=0.2, help='Reward penalty for switching')
    p.add_argument('--min-green', type=int, default=8, help='Minimum green steps before allowing a switch')
    p.add_argument('--init-model', type=str, default=None, help='Initialize policy from a saved model if provided')
    p.add_argument('--model-out', type=str, default='dqn_sumo_advanced.pt', help='Path to save the trained model')
    p.add_argument('--collision-penalty', type=float, default=10.0, help='Penalty multiplier per collision per step')
    p.add_argument('--arrival-reward', type=float, default=0.2, help='Positive reward per arrived vehicle per step')
    p.add_argument('--lr', type=float, default=1e-3, help='Optimizer learning rate')
    p.add_argument('--tau', type=float, default=0.0, help='Soft target update rate (0 to disable)')
    p.add_argument('--lr-step', type=int, default=50, help='LR scheduler step size in episodes (0 to disable)')
    p.add_argument('--lr-gamma', type=float, default=0.95, help='LR scheduler gamma (decay factor)')
    p.add_argument('--save-best', type=str, default=None, help='Path to save the best moving-average model (default: <log-dir>/best_model.pt)')
    p.add_argument('--best-window', type=int, default=20, help='Window size for best moving average reward')
    p.add_argument('--ckpt-every', type=int, default=0, help='Save periodic checkpoints every N episodes (0 to disable)')
    p.add_argument('--ckpt-dir', type=str, default=None, help='Directory to store periodic checkpoints (default: <log-dir>/checkpoints)')
    p.add_argument('--reseed-every', type=int, default=0, help='Regenerate routes every N episodes (0 to disable)')
    p.add_argument('--reseed-start', type=int, default=0, help='Episode index to start reseeding (inclusive)')
    p.add_argument('--reseed-seed-base', type=int, default=500, help='Base seed added to episode index for reseeding')
    p.add_argument('--period-cars', type=float, default=6.0, help='Cars generation period for reseeding')
    p.add_argument('--period-trucks', type=float, default=12.0, help='Trucks generation period for reseeding')
    p.add_argument('--period-buses', type=float, default=14.0, help='Buses generation period for reseeding')
    p.add_argument('--period-bikes', type=float, default=8.0, help='Bikes generation period for reseeding')
    p.add_argument('--violation-penalty', type=float, default=50.0, help='Penalty per red-light violator detected in a step')
    p.add_argument('--violations-log', type=str, default=None, help='Optional CSV path to append violator IDs (step,id)')
    p.add_argument('--sumocfg', type=str, default='traffic_config.sumocfg', help='SUMO config file to use')
    p.add_argument('--ambulance-requests-file', type=str, default=None, help='Optional text file path to ingest lane_ids for ambulance priority (one per line)')
    # PER options
    p.add_argument('--per', dest='use_per', action='store_true', help='Enable Prioritized Experience Replay')
    p.add_argument('--no-per', dest='use_per', action='store_false', help='Disable Prioritized Experience Replay')
    p.set_defaults(use_per=True)
    p.add_argument('--per-alpha', type=float, default=0.6, help='PER alpha (priority exponent)')
    p.add_argument('--per-beta-start', type=float, default=0.4, help='PER beta start (IS anneal start)')
    p.add_argument('--per-beta-end', type=float, default=1.0, help='PER beta end (IS anneal end)')
    p.add_argument('--per-beta-steps', type=int, default=100000, help='Steps over which to anneal beta')
    p.add_argument('--per-eps', type=float, default=1e-3, help='Small epsilon added to TD error for priority')
    # Epsilon schedule
    p.add_argument('--eps-start', type=float, default=1.0, help='Epsilon start for epsilon-greedy')
    p.add_argument('--eps-end', type=float, default=0.05, help='Epsilon end for epsilon-greedy')
    p.add_argument('--eps-decay-steps', type=int, default=3000, help='Steps to decay epsilon from start to end')
    args = p.parse_args()
    train(episodes=args.episodes, max_steps=args.max_steps, gui=args.gui, log_csv=args.log_csv, log_dir=args.log_dir, plot=(not args.no_plot), tensorboard=args.tensorboard, tb_logdir=args.tb_logdir, live_plot=args.live_plot, decision_interval=args.decision_interval, alpha=args.alpha, beta=args.beta, gamma_r=args.gamma_r, min_green=args.min_green, init_model=args.init_model, model_out=args.model_out, collision_penalty=args.collision_penalty, arrival_reward=args.arrival_reward, violation_penalty=args.violation_penalty, violations_log=args.violations_log, lr=args.lr, tau=args.tau, lr_step=args.lr_step, lr_gamma=args.lr_gamma, save_best=args.save_best, best_window=args.best_window, ckpt_every=args.ckpt_every, ckpt_dir=args.ckpt_dir, reseed_every=args.reseed_every, reseed_start=args.reseed_start, reseed_seed_base=args.reseed_seed_base, period_cars=args.period_cars, period_trucks=args.period_trucks, period_buses=args.period_buses, period_bikes=args.period_bikes, sumocfg=args.sumocfg,
          use_per=args.use_per, per_alpha=args.per_alpha, per_beta_start=args.per_beta_start, per_beta_end=args.per_beta_end, per_beta_steps=args.per_beta_steps, per_eps=args.per_eps,
        eps_start=args.eps_start, eps_end=args.eps_end, eps_decay_steps=args.eps_decay_steps, ambulance_requests_file=args.ambulance_requests_file)
