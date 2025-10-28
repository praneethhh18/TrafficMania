import os
import sys
import time
import random
import numpy as np
from typing import Tuple, Dict, Any

# Ensure SUMO is on sys.path via SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise RuntimeError('SUMO_HOME environment variable is not set. Install SUMO and set SUMO_HOME.')

import traci

class SumoTrafficEnv:
    """
    Minimal single-intersection environment for RL experiments using SUMO + TraCI.
    - Observation: lane-wise queue lengths (stopped vehicles) around the first traffic light.
    - Action space: 0 = keep current phase; 1 = switch to next phase (NS <-> EW).
    - Reward: negative total queue length (we want to minimize queues).
    """
    def __init__(self, sumocfg: str = 'traffic_config.sumocfg', gui: bool = False, max_steps: int = 600):
        self.sumocfg = sumocfg
        self.gui = gui
        self.max_steps = max_steps
        self.step = 0
        self.tls_id = None
        self.lanes = []
        self.last_switch = 0
        self.min_green = 5  # seconds

    def reset(self) -> np.ndarray:
        self.close()
        sumo_binary = 'sumo-gui' if self.gui else 'sumo'
        traci.start([sumo_binary, '-c', self.sumocfg, '--no-step-log=true', '--time-to-teleport', '-1'])
        self.step = 0
        # Pick the first traffic light in the network
        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError('No traffic lights found in the network.')
        self.tls_id = tls_ids[0]
        # Gather controlled lanes for observation
        self.lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        # Queue length = vehicles with speed < 0.1 on each lane
        obs = []
        for lane in self.lanes:
            q = traci.lane.getLastStepHaltingNumber(lane)
            obs.append(q)
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self) -> float:
        # Negative sum of queue lengths
        total_queue = 0.0
        for lane in self.lanes:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
        return -float(total_queue)

    def _can_switch(self) -> bool:
        return (self.step - self.last_switch) >= self.min_green

    def step_env(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # Action: 0 keep, 1 switch
        if action == 1 and self._can_switch():
            # Compute number of phases robustly from the current program logic
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
            num_phases = len(logics[0].phases) if logics and hasattr(logics[0], 'phases') else 2
            next_phase = (current_phase + 1) % max(1, num_phases)
            traci.trafficlight.setPhase(self.tls_id, next_phase)
            self.last_switch = self.step
        traci.simulationStep()
        self.step += 1
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.step >= self.max_steps or traci.simulation.getMinExpectedNumber() <= 0
        info = {}
        return obs, reward, done, info

    def close(self):
        if traci.isLoaded():
            traci.close()


def quick_sanity_run():
    env = SumoTrafficEnv(gui=False, max_steps=300)
    obs = env.reset()
    ep_reward = 0.0
    while True:
        # Simple heuristic: switch if NS queue is much larger than EW
        action = 0
        if obs.sum() > 0 and env._can_switch():
            action = random.choice([0, 1])
        obs, r, done, _ = env.step_env(action)
        ep_reward += r
        if done:
            break
    env.close()
    print('Sanity run finished, total reward:', ep_reward)

if __name__ == '__main__':
    quick_sanity_run()
