import os
import sys
import random
import warnings
from typing import Tuple, Dict, Any, List

import numpy as np

# Ensure SUMO tools on path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise RuntimeError('SUMO_HOME environment variable is not set. Install SUMO and set SUMO_HOME.')

import traci

# Silence specific deprecation warning from TraCI traffic light API name change
warnings.filterwarnings(
    "ignore",
    message=r".*deprecated function getAllProgramLogics.*",
    category=UserWarning,
)


class SumoTrafficEnvAdvanced:
    """
    Advanced SUMO + TraCI environment for traffic signal control.
    - Observation: [per-lane halting count ... , total_queue, total_wait, phase_idx, time_since_switch]
    - Actions: 0 keep phase, 1 switch to next phase (wraps safely, respects min_green)
    - Reward: -(alpha*queue + beta*wait) - gamma*switch_penalty
    """

    def __init__(self,
                 sumocfg: str = 'traffic_config.sumocfg',
                 gui: bool = False,
                 max_steps: int = 900,
                 min_green: int = 8,
                 alpha: float = 1.0,
                 beta: float = 0.1,
                 gamma: float = 0.2,
                 collision_penalty: float = 10.0,
                 arrival_reward: float = 0.2,
                 violation_penalty: float = 50.0,
                 violations_log: str = None,
                 # Penalty shaping for repeat violators (per-vehicle escalation)
                 violation_reoffender_factor: float = 0.0,
                 # Ambulance handling
                 ambulance_override_min_green: bool = True,
                 ambulance_vtype_keywords: List[str] = None,
                 auto_detect_ambulance: bool = True,
                 ambulance_cooldown_steps: int = 5,
                 ambulance_requests_file: str = None):
        self.sumocfg = sumocfg
        self.gui = gui
        self.max_steps = max_steps
        self.min_green = min_green
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.collision_penalty = collision_penalty
        self.arrival_reward = arrival_reward
        self.violation_penalty = violation_penalty
        self.violations_log = violations_log
        self.violation_reoffender_factor = violation_reoffender_factor
        self.ambulance_override_min_green = ambulance_override_min_green
        self.ambulance_vtype_keywords = ambulance_vtype_keywords or ['ambul', 'emerg']
        self.auto_detect_ambulance = auto_detect_ambulance
        self.ambulance_cooldown_steps = max(0, int(ambulance_cooldown_steps))
        self.ambulance_requests_file = ambulance_requests_file

        self.step = 0
        self.tls_id = None
        self.lanes = []
        self.num_phases = 2
        self.last_switch = 0
        # Vehicle tracking for violation detection
        self.prev_lane_vehicles = {}
        self.prev_phase_state = ''
        self.vehicle_violations = set()
        # violations detected in the last step (reset each step)
        self.last_step_violations = set()
        # track per-vehicle violation counts
        self.violator_counts = {}
        # Map lane -> list of phases where that lane is green
        self.lane_to_green_phases = {}
        # Ambulance requests (lane ids requested to be prioritized)
        self.ambulance_requests = set()
        self.last_ambulance_switch = -10**9

    def reset(self) -> np.ndarray:
        self.close()
        sumo_binary = 'sumo-gui' if self.gui else 'sumo'
        # Enable teleport for stuck vehicles to avoid permanent gridlock during training
        traci.start([
            sumo_binary,
            '-c', self.sumocfg,
            '--no-step-log=true',
            '--time-to-teleport', '60',
            # Ensure collisions are reported so we can penalize them
            '--collision.action', 'warn'
        ])
        self.step = 0
        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError('No traffic lights found in the network.')
        self.tls_id = tls_ids[0]
        self.lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
        self.num_phases = len(logics[0].phases) if logics and hasattr(logics[0], 'phases') else 2
        self.last_switch = 0
        # initialize previous lane vehicle sets
        self.prev_lane_vehicles = {l: set(traci.lane.getLastStepVehicleIDs(l)) for l in self.lanes}
        # store previous phase state string for violation detection
        try:
            self.prev_phase_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        except Exception:
            self.prev_phase_state = ''
        # build lane->green phase mapping
        try:
            self.lane_to_green_phases = {l: [] for l in self.lanes}
            logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
            if logics and hasattr(logics[0], 'phases'):
                for p_idx, phase in enumerate(logics[0].phases):
                    state = phase.state if hasattr(phase, 'state') else ''
                    # state string corresponds to controlled lanes order
                    for i, lane in enumerate(self.lanes):
                        if i < len(state) and state[i] in ('G', 'g'):
                            self.lane_to_green_phases[lane].append(p_idx)
        except Exception:
            self.lane_to_green_phases = {l: [] for l in self.lanes}
        return self._get_observation()

    def _lane_queue(self, lane: str) -> float:
        return float(traci.lane.getLastStepHaltingNumber(lane))

    def _lane_wait(self, lane: str) -> float:
        return float(traci.lane.getWaitingTime(lane))

    def _get_observation(self) -> np.ndarray:
        queues = [self._lane_queue(l) for l in self.lanes]
        waits = [self._lane_wait(l) for l in self.lanes]
        total_q = float(sum(queues))
        total_w = float(sum(waits))
        phase_idx = float(traci.trafficlight.getPhase(self.tls_id))
        time_since_switch = float(self.step - self.last_switch)
        obs = np.array(queues + [total_q, total_w, phase_idx, time_since_switch], dtype=np.float32)
        return obs

    def _compute_reward(self, switched: bool, collisions: int, arrivals: int) -> float:
        queues = [self._lane_queue(l) for l in self.lanes]
        waits = [self._lane_wait(l) for l in self.lanes]
        total_q = float(sum(queues))
        total_w = float(sum(waits))
        penalty = self.gamma if switched else 0.0
        # Encourage throughput (arrivals) and heavily penalize collisions
        # reward = -(alpha*queue + beta*wait + gamma*switch) - collision_penalty*collisions + arrival_reward*arrivals
        base = -(self.alpha * total_q + self.beta * total_w + penalty) - (self.collision_penalty * float(collisions)) + (self.arrival_reward * float(arrivals))
        # apply extra penalty for any detected red-light violations this step
        # use last step violations (per-step) for penalty
        num_viol = len(self.last_step_violations) if hasattr(self, 'last_step_violations') else 0
        if num_viol:
            base -= float(self.violation_penalty) * float(num_viol)
            # add escalation for repeat offenders if enabled
            if self.violation_reoffender_factor and self.violation_reoffender_factor > 0.0:
                extra = 0.0
                for v in self.last_step_violations:
                    c = float(self.violator_counts.get(v, 0))
                    extra += self.violation_reoffender_factor * c
                base -= extra
        return base

    def _can_switch(self) -> bool:
        return (self.step - self.last_switch) >= self.min_green

    def step_env(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        switched = False
        # Ingest external ambulance lane requests from a simple text file (one lane_id per line).
        # This enables vision-driven priority via an out-of-process observer.
        if self.ambulance_requests_file:
            try:
                if os.path.exists(self.ambulance_requests_file):
                    with open(self.ambulance_requests_file, 'r+') as f:
                        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                        # clear file after reading
                        f.seek(0); f.truncate()
                    for ln in lines:
                        if ln in self.lanes:
                            self.ambulance_requests.add(ln)
            except Exception:
                pass
        # Auto-detect ambulances each step and register their current lanes
        if self.auto_detect_ambulance:
            try:
                state_str = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            except Exception:
                state_str = ''
            try:
                for vid in traci.vehicle.getIDList():
                    try:
                        vtype = traci.vehicle.getTypeID(vid)
                        if any(k.lower() in vtype.lower() for k in self.ambulance_vtype_keywords):
                            lane_id = traci.vehicle.getLaneID(vid)
                            spd = 0.0
                            try:
                                spd = float(traci.vehicle.getSpeed(vid))
                            except Exception:
                                spd = 0.0
                            if lane_id:
                                # only request if lane currently red/yellow and vehicle is slow/standing
                                i = self.lanes.index(lane_id) if lane_id in self.lanes else -1
                                is_red = (i >= 0 and i < len(state_str) and state_str[i].lower() in ('r','y'))
                                if is_red and spd <= 1.0:
                                    self.ambulance_requests.add(lane_id)
                    except Exception:
                        continue
            except Exception:
                pass
        # If there is an ambulance request, try to prioritize its lane by switching to
        # a phase that gives green to the requested lane (if safe to switch).
        can_amb_switch = (self._can_switch() or self.ambulance_override_min_green)
        if self.ambulance_requests and can_amb_switch:
            # pick first requested lane and try to switch to a phase that gives it green
            req_lane = next(iter(self.ambulance_requests))
            target_phases = self.lane_to_green_phases.get(req_lane, [])
            try:
                current_phase = traci.trafficlight.getPhase(self.tls_id)
                if target_phases:
                    # Only force a switch if we're not already serving the ambulance lane
                    # and we haven't just forced a switch too recently (cooldown)
                    if (current_phase not in target_phases) and (self.step - self.last_ambulance_switch >= self.ambulance_cooldown_steps):
                        # Jump directly to the first target phase to clear the ambulance
                        traci.trafficlight.setPhase(self.tls_id, int(target_phases[0]))
                        self.last_switch = self.step
                        self.last_ambulance_switch = self.step
                        switched = True
                    # Clear handled request either way to avoid repeating
                    self.ambulance_requests.discard(req_lane)
            except Exception:
                pass
        elif action == 1 and self._can_switch():
            # End current phase now; SUMO will advance through proper yellow/all-red transitions
            traci.trafficlight.setPhaseDuration(self.tls_id, 0)
            self.last_switch = self.step
            switched = True

        # record previous lane vehicle sets and previous phase state before stepping
        try:
            prev_lane_sets = {l: set(traci.lane.getLastStepVehicleIDs(l)) for l in self.lanes}
        except Exception:
            prev_lane_sets = {l: set() for l in self.lanes}
        try:
            prev_phase_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        except Exception:
            prev_phase_state = self.prev_phase_state

        traci.simulationStep()
        # Gather safety/throughput metrics after the simulation step
        collisions = 0
        try:
            # SUMO >= 1.14
            collisions = int(traci.simulation.getCollidingVehiclesNumber())
        except Exception:
            try:
                collisions = len(traci.simulation.getCollidingVehiclesIDList())
            except Exception:
                collisions = 0
        arrivals = 0
        try:
            arrivals = int(traci.simulation.getArrivedNumber())
        except Exception:
            try:
                arrivals = len(traci.simulation.getArrivedIDList())
            except Exception:
                arrivals = 0
        self.step += 1
        obs = self._get_observation()
        reward = self._compute_reward(switched, collisions, arrivals)
        done = self.step >= self.max_steps or traci.simulation.getMinExpectedNumber() <= 0
        # Expose metrics for logging
        queues = [self._lane_queue(l) for l in self.lanes]
        waits = [self._lane_wait(l) for l in self.lanes]
        # detect red-light violations: vehicles that left a lane while that lane had red
        violators = set()
        try:
            curr_lane_sets = {l: set(traci.lane.getLastStepVehicleIDs(l)) for l in self.lanes}
            # prev_phase_state string maps to lanes by index; 'r' indicates red
            for i, lane in enumerate(self.lanes):
                prev_set = prev_lane_sets.get(lane, set())
                curr_set = curr_lane_sets.get(lane, set())
                left = prev_set - curr_set
                if left:
                    # if previous phase state for this lane was red, count as violation
                    if i < len(prev_phase_state) and prev_phase_state[i].lower() == 'r':
                        violators.update(left)
        except Exception:
            violators = set()
        # merge violators into global set for logging and penalty (track both current and cumulative)
        self.last_step_violations = violators
        if violators:
            self.vehicle_violations.update(violators)
            # increment per-vehicle violation counts
            for v in violators:
                self.violator_counts[v] = self.violator_counts.get(v, 0) + 1
            # write to log if requested
            if self.violations_log:
                try:
                    with open(self.violations_log, 'a') as f:
                        for v in violators:
                            f.write(f"{self.step},{v}\n")
                except Exception:
                    pass
        info: Dict[str, Any] = {
            'phase': traci.trafficlight.getPhase(self.tls_id),
            'switched': switched,
            'total_q': float(sum(queues)),
            'total_w': float(sum(waits)),
            'collisions': collisions,
            'arrivals': arrivals,
            'violators': list(violators),
            'ambulance_requests': list(self.ambulance_requests),
        }
        # update prev lane/phase trackers
        try:
            self.prev_lane_vehicles = {l: set(traci.lane.getLastStepVehicleIDs(l)) for l in self.lanes}
            self.prev_phase_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        except Exception:
            pass
        return obs, reward, done, info

    def notify_ambulance(self, lane_id: str):
        """External call (e.g., from vision system) to request priority for a given lane id."""
        if lane_id in self.lanes:
            self.ambulance_requests.add(lane_id)

    def close(self):
        if traci.isLoaded():
            traci.close()


def quick_demo(gui: bool = True):
    env = SumoTrafficEnvAdvanced(gui=gui, max_steps=300)
    obs = env.reset()
    total_r = 0.0
    while True:
        # heuristic: if total queue grows and we can switch, try switching
        total_q = float(obs[-2])
        action = 1 if total_q > 5 and env._can_switch() else 0
        obs, r, done, info = env.step_env(action)
        total_r += r
        if done:
            break
    env.close()
    print('Advanced demo finished. total reward=', total_r)


if __name__ == '__main__':
    # Launch GUI demo by default
    quick_demo(gui=True)
