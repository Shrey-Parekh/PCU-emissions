

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import sys

import numpy as np

# Add SUMO tools to path for libsumo
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)

# Use libsumo for in-process simulation (3-6x faster than TraCI)
# Falls back to TraCI if libsumo not available
try:
    import libsumo as traci
    USING_LIBSUMO = True
except ImportError:
    import traci
    USING_LIBSUMO = False

from .config import (
    VEHICLE_CLASSES,
    PEAK_HOUR_CONFIG,
    SUMO_CONFIG,
    SCENARIOS,
    OBS_FEATURES_PER_AGENT,
    INJECTION_CONFIG,
    REWARD_CONFIG,
)


@dataclass
class VehicleClass:
    """Vehicle class definition with PCU and arrival weight."""
    name: str
    pcu: float
    arrival_weight: float
    vtype: str
    service_rate: int


class MixedTrafficQueue:
    """Per-lane queue tracking with PCU calculation."""
    
    def __init__(self, lane_id: str):
        self.lane_id = lane_id
        self.vehicles: List[str] = []
        self.vehicle_types: Dict[str, str] = {}
    
    def update(self, vehicle_ids: List[str], traci_conn):
        """Update queue with current vehicles on lane."""
        self.vehicles = vehicle_ids
        self.vehicle_types = {}
        for veh_id in vehicle_ids:
            try:
                vtype = traci_conn.vehicle.getTypeID(veh_id)
                self.vehicle_types[veh_id] = vtype
            except traci.exceptions.TraCIException:
                pass
    
    def get_pcu(self) -> float:
        """Calculate total PCU for this lane."""
        total_pcu = 0.0
        for veh_id, vtype in self.vehicle_types.items():
            for vclass_name, vclass_data in VEHICLE_CLASSES.items():
                if vclass_data["vtype"] == vtype:
                    total_pcu += vclass_data["pcu"]
                    break
        return total_pcu
    
    def get_count(self) -> int:
        """Get raw vehicle count."""
        return len(self.vehicles)
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count per vehicle class."""
        counts = {vclass_data["vtype"]: 0 for vclass_data in VEHICLE_CLASSES.values()}
        for veh_id, vtype in self.vehicle_types.items():
            if vtype in counts:
                counts[vtype] += 1
        return counts


class PuneSUMOEnv:

    
    def __init__(self, config: Dict):
        """Initialize SUMO environment."""
        self.n_intersections = config.get("n_intersections", 9)
        self.n_agents = self.n_intersections  # Alias for consistency
        self.scenario = config.get("scenario", "uniform")
        self.render = config.get("render", False)
        self.seed = config.get("seed", 42)
        self.port = config.get("port", None)  # SUMO port for parallel runs
        self.max_steps = config.get("max_steps", 3600)
        self.use_global_reward = config.get("use_global_reward", True)

        # Multi-objective reward parameters. beta blends PCU-queue vs CO2.
        # co2_norm is calibrated via probe_co2.py; wrong values get flattened
        # by the loss-time clamp to [-1, 1] in train.py.
        self.beta = config.get("beta", REWARD_CONFIG.get("beta", 0.0))
        self.queue_norm = REWARD_CONFIG.get("reward_queue_norm", 30.0)
        self.co2_norm = config.get("co2_norm", REWARD_CONFIG.get("co2_norm", 8000.0))
        self.pressure_coef = REWARD_CONFIG.get("w_pressure_bonus", 0.2)
        # Per-episode CO2 accumulator (mg, network total), reset in reset().
        self.episode_co2 = 0.0
        
        self.sumo_config_file = SUMO_CONFIG["config_file"]
        self.step_length = SUMO_CONFIG["step_length"]
        self.min_green_steps = SUMO_CONFIG["min_green_steps"]
        self.clearance_steps = SUMO_CONFIG["clearance_steps"]
        self.lane_split_probability = SUMO_CONFIG["lane_split_probability"]
        self.lane_split_min_queue = SUMO_CONFIG["lane_split_min_queue"]
        
        self.tl_ids = [f"n{row}{col}" for row in range(3) for col in range(3)][:self.n_intersections]
        self.current_phases = [0] * self.n_intersections
        self.prev_phases = [0] * self.n_intersections  # Track previous phase for clearance transitions
        self.steps_since_switch = [0] * self.n_intersections
        self.queues: Dict[str, Dict[str, MixedTrafficQueue]] = {}
        self.current_step = 0
        self.traci_conn = None
        self.peak_config = PEAK_HOUR_CONFIG[self.scenario]
        
        # Initialize prev_pcu tracking for reward improvement signal
        self.prev_pcu = {}
        self._prev_network_pcu = None
        self._prev_queue_pcu = {}  # Track previous queue for delta reward
        
        # Initialize inflow tracking
        self._inflow_counts = {}
        
        # Track previous queue for derivative calculation
        self._prev_queue_ns = {}
        self._prev_queue_ew = {}
        
        # Track turning movements for paper documentation
        self.turning_counts = {
            "straight": 0,
            "right_turn": 0,
            "left_turn": 0,
            "u_turn": 0
        }
        
        self.vehicle_classes = [
            VehicleClass(
                name=name,
                pcu=data["pcu"],
                arrival_weight=data["arrival_weight"],
                vtype=data["vtype"],
                service_rate=data["service_rate"]
            )
            for name, data in VEHICLE_CLASSES.items()
        ]
        
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Return adjacency matrix for 3x3 grid topology."""
        adj = np.zeros((self.n_intersections, self.n_intersections), dtype=np.float32)
        for row in range(3):
            for col in range(2):
                idx = row * 3 + col
                adj[idx, idx + 1] = 1.0
                adj[idx + 1, idx] = 1.0
        for row in range(2):
            for col in range(3):
                idx = row * 3 + col
                adj[idx, idx + 3] = 1.0
                adj[idx + 3, idx] = 1.0
        return adj
    
    @property
    def observation_space(self):
        """Observation space definition for compatibility with gym interface."""
        try:
            from gym import spaces
        except ImportError:
            from gymnasium import spaces
        return spaces.Box(
            low=-1.0, high=10.0,
            shape=(24,),
            dtype=np.float32
        )
    
    def _get_neighbor_indices(self, intersection_idx: int) -> list:
        """Returns list of neighboring intersection indices for a given intersection.
        Based on 3x3 grid topology — each node connects to adjacent nodes only.
        Center node (4) has 4 neighbors, corner nodes have 2, edge nodes have 3."""
        # 3x3 grid adjacency
        # Node indices: 0=n00, 1=n01, 2=n02, 3=n10, 4=n11, 5=n12,
        #               6=n20, 7=n21, 8=n22
        adjacency = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        return adjacency.get(intersection_idx, [])
    
    def _start_sumo(self):
        """Start SUMO simulation with retry logic."""
        # libsumo doesn't support GUI mode
        if USING_LIBSUMO and self.render:
            print("Warning: libsumo doesn't support GUI. Disabling render mode.")
            self.render = False
        
        sumo_binary = "sumo-gui" if self.render else "sumo"
        sumo_cmd = [
            sumo_binary, "-c", self.sumo_config_file,
            "--step-length", str(self.step_length),
            "--no-warnings", "true", "--no-step-log", "true",
            "--time-to-teleport", "-1", "--seed", str(self.seed),
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.traci_conn is not None:
                    try:
                        self.traci_conn.close()
                    except:
                        pass
                
                # Use port if specified for parallel runs
                if self.port is not None:
                    traci.start(sumo_cmd, port=self.port)
                else:
                    traci.start(sumo_cmd)
                
                self.traci_conn = traci
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Failed to start SUMO on port {self.port}: {e}") from e
    
    def _initialize_queues(self):
        """Initialize queue tracking for all intersections."""
        self.queues = {}
        for tl_id in self.tl_ids:
            self.queues[tl_id] = {}
            try:
                controlled_lanes = self.traci_conn.trafficlight.getControlledLanes(tl_id)
                for lane_id in set(controlled_lanes):
                    self.queues[tl_id][lane_id] = MixedTrafficQueue(lane_id)
            except traci.exceptions.TraCIException:
                pass
    
    def _set_sumo_phase(self, tl_id: str, phase_idx: int):
        """Set SUMO traffic light phase."""
        try:
            # Get the number of controlled lanes for this traffic light
            controlled_lanes = self.traci_conn.trafficlight.getControlledLanes(tl_id)
            num_signals = len(controlled_lanes)
            
            # Create phase strings based on number of signals
            # Assuming equal split between NS and EW directions
            half = num_signals // 2
            
            if phase_idx == 0:  # NS_GREEN
                phase_str = "G" * half + "r" * (num_signals - half)
            elif phase_idx == 1:  # ALL_RED_CLEARANCE
                phase_str = "r" * num_signals
            elif phase_idx == 2:  # EW_GREEN
                phase_str = "r" * half + "G" * (num_signals - half)
            else:
                phase_str = "G" * num_signals  # Default to all green
            
            self.traci_conn.trafficlight.setRedYellowGreenState(tl_id, phase_str)
        except traci.exceptions.TraCIException:
            pass
    
    def _update_queues(self):
        """Update all queue states from SUMO."""
        for tl_id in self.tl_ids:
            if tl_id not in self.queues:
                continue
            for lane_id, queue in self.queues[tl_id].items():
                try:
                    vehicle_ids = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)
                    queue.update(vehicle_ids, self.traci_conn)
                except traci.exceptions.TraCIException:
                    pass
    
    def _get_intersection_queues(self, tl_id: str) -> Tuple[float, float, int, int, Dict[str, int], Dict[str, int]]:
        """Extract NS and EW queue information for an intersection.
        
        NS edges connect nodes with same column (row changes): e_nRC_nR'C where C==C'
        EW edges connect nodes with same row (column changes): e_nRC_nRC' where R==R'
        Lane suffix (_0, _1, etc.) is just lane index within edge, not direction.
        """
        ns_pcu = 0.0
        ew_pcu = 0.0
        ns_count = 0
        ew_count = 0
        ns_class_counts = {vclass_data["vtype"]: 0 for vclass_data in VEHICLE_CLASSES.values()}
        ew_class_counts = {vclass_data["vtype"]: 0 for vclass_data in VEHICLE_CLASSES.values()}
        
        if tl_id not in self.queues:
            return ns_pcu, ew_pcu, ns_count, ew_count, ns_class_counts, ew_class_counts
        
        for lane_id, queue in self.queues[tl_id].items():
            # Determine direction from edge name, not lane suffix
            # Edge format: e_nRC_nR'C'_lane
            # NS: same column (C == C'), e.g. e_n00_n10, e_n10_n20
            # EW: same row (R == R'), e.g. e_n00_n01, e_n01_n02
            is_ns = self._is_ns_lane(lane_id)
            pcu = queue.get_pcu()
            count = queue.get_count()
            class_counts = queue.get_class_counts()
            
            if is_ns:
                ns_pcu += pcu
                ns_count += count
                for vtype, cnt in class_counts.items():
                    ns_class_counts[vtype] += cnt
            else:
                ew_pcu += pcu
                ew_count += count
                for vtype, cnt in class_counts.items():
                    ew_class_counts[vtype] += cnt
        
        return ns_pcu, ew_pcu, ns_count, ew_count, ns_class_counts, ew_class_counts

    def get_raw_queue_pcu(self, intersection_idx: int) -> tuple[float, float]:
        """Return raw (ns_pcu, ew_pcu) for intersection by index. Public interface for external controllers."""
        tl_id = self.tl_ids[intersection_idx]
        ns_pcu, ew_pcu, _, _, _, _ = self._get_intersection_queues(tl_id)
        return ns_pcu, ew_pcu

    def _is_ns_lane(self, lane_id: str) -> bool:
        """Determine if a lane belongs to a NS edge.
        
        Edge name format: e_nRC_nR'C'_laneindex
        NS edge: column is same (C == C'), row differs
        EW edge: row is same (R == R'), column differs
        Falls back to False (EW) if parsing fails.
        """
        try:
            # Strip lane index suffix: e_n00_n10_0 -> e_n00_n10
            parts = lane_id.rsplit('_', 1)
            edge = parts[0] if len(parts) == 2 and parts[1].isdigit() else lane_id
            # edge format: e_nRC_nR'C'
            nodes = edge.split('_')
            # nodes = ['e', 'nRC', 'nR\'C\'']
            if len(nodes) < 3:
                return False
            src, dst = nodes[1], nodes[2]
            # src = nRC, dst = nR'C'
            if len(src) < 3 or len(dst) < 3:
                return False
            src_col = src[2]  # column index of source node
            dst_col = dst[2]  # column index of destination node
            return src_col == dst_col  # same column = NS direction
        except (IndexError, ValueError):
            return False
    
    def _apply_lane_splitting(self):
        """Simulate two-wheeler lane-splitting behavior."""
        for tl_id in self.tl_ids:
            if tl_id not in self.queues:
                continue
            for lane_id, queue in self.queues[tl_id].items():
                if queue.get_count() < self.lane_split_min_queue:
                    continue
                two_wheeler_vtype = VEHICLE_CLASSES["TWO_WHEELER"]["vtype"]
                for veh_id, vtype in queue.vehicle_types.items():
                    if vtype == two_wheeler_vtype:
                        if random.random() < self.lane_split_probability:
                            try:
                                current_speed = self.traci_conn.vehicle.getSpeed(veh_id)
                                self.traci_conn.vehicle.setSpeed(veh_id, min(current_speed + 2.0, 15.0))
                            except traci.exceptions.TraCIException:
                                pass
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self._start_sumo()
        self.current_step = 0
        self.current_phases = [0] * self.n_intersections
        self.prev_phases = [0] * self.n_intersections
        self.steps_since_switch = [0] * self.n_intersections
        self.turning_counts = {"straight": 0, "right_turn": 0, "left_turn": 0, "u_turn": 0}
        self.episode_co2 = 0.0

        # Initialize metrics tracking
        self.departed_vehicles = set()
        self.arrived_vehicles = set()
        self.vehicle_travel_times = {}

        # Initialize prev_pcu tracking for reward improvement signal
        self.prev_pcu = {}
        self._prev_network_pcu = None
        self._prev_queue_pcu = {}  # Track previous queue for delta reward

        # Initialize inflow tracking and queue derivatives
        self._inflow_counts = {i: {"NS": 0, "EW": 0} for i in range(self.n_intersections)}
        self._prev_queue_ns = {i: 0.0 for i in range(self.n_intersections)}
        self._prev_queue_ew = {i: 0.0 for i in range(self.n_intersections)}

        self._initialize_queues()
        for tl_id in self.tl_ids:
            self._set_sumo_phase(tl_id, 0)

        # Run 10 steps to warm up simulation
        for _ in range(10):
            try:
                self.traci_conn.simulationStep()
            except traci.exceptions.TraCIException:
                pass

        self._update_queues()
        return self._get_observation()


    
    def _get_observation(self) -> np.ndarray:
        """Get observation for all intersections.
        Returns (n_intersections, 24) array:
        0-14: self observation (unchanged)
        15-20: neighbor aggregate observations
        21: action mask (can_switch)
        22-23: NS and EW queue derivative (change from previous step)
        """
        obs = np.zeros((self.n_intersections, OBS_FEATURES_PER_AGENT), dtype=np.float32)
        scenario_flag = 0.0
        if self.scenario == "morning_peak":
            scenario_flag = 1.0
        elif self.scenario == "evening_peak":
            scenario_flag = 2.0
        
        MAX_PCU = 30.0  # Use reward_queue_norm as max PCU reference
        MAX_DERIVATIVE = 2.0  # maximum queue change per step
        
        for i, tl_id in enumerate(self.tl_ids):
            # Self features (0-14)
            ns_pcu, ew_pcu, ns_count, ew_count, ns_class_counts, ew_class_counts = self._get_intersection_queues(tl_id)
            obs[i, 0] = ns_count
            obs[i, 1] = ew_count
            obs[i, 2] = ns_pcu
            obs[i, 3] = ew_pcu
            obs[i, 4] = self.current_phases[i]
            obs[i, 5] = self.steps_since_switch[i]
            vtype_list = [vc["vtype"] for vc in VEHICLE_CLASSES.values()]
            for j, vtype in enumerate(vtype_list):
                obs[i, 6 + j] = ns_class_counts.get(vtype, 0)
                obs[i, 10 + j] = ew_class_counts.get(vtype, 0)
            obs[i, 14] = scenario_flag
            
            # Neighbor features (15-20)
            neighbors = self._get_neighbor_indices(i)
            # Filter neighbors to only include valid intersection indices
            neighbors = [n for n in neighbors if n < self.n_intersections]
            if len(neighbors) == 0:
                obs[i, 15:21] = 0.0
            else:
                neighbor_ns_pcu = []
                neighbor_ew_pcu = []
                neighbor_phases = []
                for n_idx in neighbors:
                    n_tl_id = self.tl_ids[n_idx]
                    n_ns_pcu, n_ew_pcu, _, _, _, _ = self._get_intersection_queues(n_tl_id)
                    neighbor_ns_pcu.append(n_ns_pcu)
                    neighbor_ew_pcu.append(n_ew_pcu)
                    neighbor_phases.append(self.current_phases[n_idx])
                
                obs[i, 15] = np.mean(neighbor_ns_pcu) / MAX_PCU  # mean NS PCU neighbors
                obs[i, 16] = np.mean(neighbor_ew_pcu) / MAX_PCU  # mean EW PCU neighbors
                obs[i, 17] = np.max(neighbor_ns_pcu) / MAX_PCU   # max NS PCU neighbors
                obs[i, 18] = np.max(neighbor_ew_pcu) / MAX_PCU   # max EW PCU neighbors
                obs[i, 19] = sum(1 for p in neighbor_phases if p == 0) / max(len(neighbors), 1)  # NS_GREEN count
                obs[i, 20] = sum(1 for p in neighbor_phases if p == 2) / max(len(neighbors), 1)  # EW_GREEN count
            
            # Action mask (21)
            can_switch = 1.0 if self.steps_since_switch[i] >= self.min_green_steps else 0.0
            obs[i, 21] = can_switch
            
            # Features 22-23: Queue derivative (change from previous step)
            # Positive = queue growing, negative = queue shrinking
            # This tells agent which direction is accumulating vehicles
            ns_derivative = ns_pcu - self._prev_queue_ns[i]
            ew_derivative = ew_pcu - self._prev_queue_ew[i]
            obs[i, 22] = np.clip(ns_derivative / MAX_DERIVATIVE, -1.0, 1.0)
            obs[i, 23] = np.clip(ew_derivative / MAX_DERIVATIVE, -1.0, 1.0)
            
            # Update previous queue for next step
            self._prev_queue_ns[i] = ns_pcu
            self._prev_queue_ew[i] = ew_pcu
        
        return obs
    
    def _inject_vehicles(self):
        """
        Dynamically injects vehicles using edge-based routes.
        Applies peak hour multipliers from PEAK_HOUR_CONFIG.
        Uses traci.vehicle.addFull() with edge lists instead of route IDs.
        """
        # Define routes as edge lists (not route IDs)
        straight_routes = [
            # NS routes
            (["e_n00_n10", "e_n10_n20"], "NS"),
            (["e_n20_n10", "e_n10_n00"], "NS"),
            (["e_n01_n11", "e_n11_n21"], "NS"),
            (["e_n21_n11", "e_n11_n01"], "NS"),
            (["e_n02_n12", "e_n12_n22"], "NS"),
            (["e_n22_n12", "e_n12_n02"], "NS"),
            # EW routes
            (["e_n00_n01", "e_n01_n02"], "EW"),
            (["e_n02_n01", "e_n01_n00"], "EW"),
            (["e_n10_n11", "e_n11_n12"], "EW"),
            (["e_n12_n11", "e_n11_n10"], "EW"),
            (["e_n20_n21", "e_n21_n22"], "EW"),
            (["e_n22_n21", "e_n21_n20"], "EW"),
        ]
        
        # Get multipliers
        ns_mult = self.peak_config["NS_multiplier"]
        ew_mult = self.peak_config["EW_multiplier"]

        # Optional per-scenario overrides (bus_heavy / high_saturation etc.)
        scenario_mix = self.peak_config.get("mix", None)
        demand_scale = self.peak_config.get("demand_scale", 1.0)

        # Inject vehicles
        for edges, direction in straight_routes:
            mult = ns_mult if direction == "NS" else ew_mult
            
            for vclass_name, vclass_data in VEHICLE_CLASSES.items():
                vtype = vclass_data["vtype"]
                if scenario_mix is not None:
                    weight = scenario_mix.get(vtype, vclass_data["arrival_weight"])
                else:
                    weight = vclass_data["arrival_weight"]

                # Calibrated base rate, scaled for higher-saturation scenarios
                base_rate = weight * INJECTION_CONFIG["base_rate"] * demand_scale
                rate = base_rate * mult
                num = np.random.poisson(rate)
                
                for i in range(num):
                    vid = f"{vtype}_s{self.current_step}_{i}_{edges[0]}"
                    try:
                        self.traci_conn.vehicle.addFull(
                            vehID=vid,
                            routeID="",
                            typeID=vtype,
                            depart="now",
                            departLane="random",
                            departPos="base",
                            departSpeed="random",
                            arrivalLane="current",
                            arrivalPos="max",
                            arrivalSpeed="current",
                            fromTaz="",
                            toTaz="",
                            line="",
                            personCapacity=0,
                            personNumber=0
                        )
                        self.traci_conn.vehicle.setRoute(vid, edges)
                        self.turning_counts["straight"] += 1
                    except traci.exceptions.TraCIException as e:
                        # Log first error only to avoid spam
                        if self.current_step == 0 and i == 0:
                            print(f"Warning: Failed to add vehicle {vid}: {e}")
                        pass

    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[float], bool, Dict]:
        """Execute one step in the environment."""
        # Track departed and arrived vehicles
        self._track_vehicle_metrics()
        
        # Inject vehicles
        self._inject_vehicles()
        
        self._apply_action(actions)
        self._apply_lane_splitting()
        
        try:
            self.traci_conn.simulationStep()
        except traci.exceptions.TraCIException:
            obs = self._get_observation()
            rewards = [0.0] * self.n_intersections
            info = self._get_info()
            return obs, rewards, True, info
        
        self.current_step += 1
        for i in range(self.n_intersections):
            self.steps_since_switch[i] += 1
        
        self._update_queues()

        # Accumulate network CO2 (mg) over all tracked lanes for this step
        step_co2 = sum(self._get_intersection_co2(tl_id) for tl_id in self.tl_ids)
        self.episode_co2 += step_co2

        obs = self._get_observation()
        rewards = self._calculate_rewards()
        done = self.current_step >= self.max_steps
        info = self._get_info()
        return obs, rewards, done, info
    
    def _apply_action(self, actions: List[int]):
        """Apply actions to traffic lights."""
        for i, (tl_id, action) in enumerate(zip(self.tl_ids, actions)):
            current_phase = self.current_phases[i]
            
            # Auto-transition out of clearance after clearance_steps
            if current_phase == 1 and self.steps_since_switch[i] >= self.clearance_steps:
                # Transition to opposite green phase from where we came
                if self.prev_phases[i] == 0:  # came from NS green
                    self.current_phases[i] = 2  # go to EW green
                else:  # came from EW green
                    self.current_phases[i] = 0  # go to NS green
                self.prev_phases[i] = self.current_phases[i]
                self.steps_since_switch[i] = 0
                self._set_sumo_phase(tl_id, self.current_phases[i])
                continue
            
            if action == 0:
                pass
            elif action == 1:
                if self.steps_since_switch[i] >= self.min_green_steps:
                    if current_phase == 0:  # NS green → clearance
                        self.prev_phases[i] = 0
                        self.current_phases[i] = 1
                        self.steps_since_switch[i] = 0
                        self._set_sumo_phase(tl_id, 1)
                    elif current_phase == 2:  # EW green → clearance
                        self.prev_phases[i] = 2
                        self.current_phases[i] = 1
                        self.steps_since_switch[i] = 0
                        self._set_sumo_phase(tl_id, 1)
    
    def _get_intersection_co2(self, tl_id: str) -> float:
        """Sum CO2 emission (mg/step) over an intersection's controlled lanes.

        Uses lane.getCO2Emission, which returns the CO2 emitted on the lane
        during the last simulation step (mg, since step_length = 1.0 s). Class
        differentiation comes from each vType's HBEFA4 emissionClass, so this is
        already class-specific CO2 — no extra per-class weighting is applied
        (that would double-count the class differences the emission model bakes in).
        """
        total_co2 = 0.0
        if tl_id not in self.queues:
            return total_co2
        for lane_id in self.queues[tl_id]:
            try:
                total_co2 += self.traci_conn.lane.getCO2Emission(lane_id)
            except traci.exceptions.TraCIException:
                pass
        return total_co2

    def _compute_reward(self, intersection_id: int) -> float:
        """
        Multi-objective reward: blends PCU-weighted queue with CO2 emissions.

          r = -(1-beta)*queue_penalty - beta*co2_penalty + (1-beta)*pressure_bonus

        beta=0 recovers the original congestion-only reward exactly. beta=1 is a
        pure emissions objective. Pressure shaping is a congestion heuristic, so
        it is scaled by (1-beta) to keep the beta=1 endpoint a clean
        emissions-optimal policy rather than a congestion-biased one.

        Both penalties are normalized to roughly [0, 1] so that after the
        loss-time clamp to [-1, 1] (train.py) neither term is silently flattened.
        CO2 is already class-differentiated via each vType's HBEFA4 emissionClass;
        no additional per-class weight is applied (would double-count).
        """
        tl_id = self.tl_ids[intersection_id]
        ns_pcu, ew_pcu, _, _, _, _ = self._get_intersection_queues(tl_id)

        total_pcu = ns_pcu + ew_pcu
        phase = self.current_phases[intersection_id]
        beta = self.beta

        # Term 1: PCU-weighted queue penalty (congestion objective)
        queue_penalty = total_pcu / self.queue_norm

        # Term 2: CO2 penalty (emissions objective), class-specific via HBEFA4
        co2_penalty = self._get_intersection_co2(tl_id) / self.co2_norm

        # Term 3: pressure shaping (congestion heuristic, faded out as beta->1)
        if phase == 0:        # NS_GREEN
            pressure = (ns_pcu - ew_pcu) / self.queue_norm
        elif phase == 2:      # EW_GREEN
            pressure = (ew_pcu - ns_pcu) / self.queue_norm
        else:                 # ALL_RED clearance
            pressure = 0.0
        pressure_bonus = self.pressure_coef * max(pressure, 0.0)

        reward = (
            -(1.0 - beta) * queue_penalty
            - beta * co2_penalty
            + (1.0 - beta) * pressure_bonus
        )

        return float(reward)

    
    def _calculate_rewards(self) -> List[float]:
        """Calculate rewards for all intersections.
        
        If use_global_reward is True, all agents receive the mean reward —
        cooperative signal that prevents agents optimising locally at the
        expense of network-wide throughput.
        """
        local_rewards = [self._compute_reward(i) for i in range(self.n_agents)]
        if self.use_global_reward:
            mean_r = float(np.mean(local_rewards))
            return [mean_r] * self.n_agents
        return local_rewards
    
    def _compute_avg_travel_time(self) -> float:
        """Calculate average travel time of completed vehicles."""
        if len(self.arrived_vehicles) == 0:
            return 0.0
        completed_times = [
            self.vehicle_travel_times[veh_id]
            for veh_id in self.arrived_vehicles
            if veh_id in self.vehicle_travel_times
        ]
        return sum(completed_times) / len(completed_times) if completed_times else 0.0
    
    def _compute_avg_queue_pcu(self) -> float:
        """Calculate average queue length in PCU across all intersections."""
        total_pcu = 0.0
        for tl_id in self.tl_ids:
            ns_pcu, ew_pcu, _, _, _, _ = self._get_intersection_queues(tl_id)
            total_pcu += ns_pcu + ew_pcu
        return total_pcu / self.n_intersections if self.n_intersections > 0 else 0.0
    
    def _compute_avg_waiting_time(self) -> float:
        """Calculate average waiting time across all lanes."""
        try:
            total_wait = 0.0
            lane_count = 0
            for tl_id in self.tl_ids:
                if tl_id not in self.queues:
                    continue
                for lane_id in self.queues[tl_id].keys():
                    try:
                        wait = self.traci_conn.lane.getWaitingTime(lane_id)
                        total_wait += wait
                        lane_count += 1
                    except:
                        pass
            return total_wait / lane_count if lane_count > 0 else 0.0
        except:
            return 0.0
    
    def _get_info(self) -> Dict:
        """Get additional information about the environment state."""
        total_pcu = 0.0
        total_vehicles = 0
        
        # Aggregate vehicle class counts across all intersections
        vehicle_class_counts = {
            "two_wheeler": 0,
            "auto_rickshaw": 0,
            "car": 0,
            "bus_truck": 0
        }
        
        for tl_id in self.tl_ids:
            ns_pcu, ew_pcu, ns_count, ew_count, ns_class_counts, ew_class_counts = self._get_intersection_queues(tl_id)
            total_pcu += ns_pcu + ew_pcu
            total_vehicles += ns_count + ew_count
            
            # Aggregate class counts from both NS and EW directions
            for vtype in vehicle_class_counts.keys():
                vehicle_class_counts[vtype] += ns_class_counts.get(vtype, 0) + ew_class_counts.get(vtype, 0)
        
        # Calculate average travel time
        avg_travel_time = self._compute_avg_travel_time()
        
        # Calculate average queue (raw vehicle count)
        avg_queue_raw = total_vehicles / self.n_intersections if self.n_intersections > 0 else 0.0
        
        return {
            "step": self.current_step,
            "total_pcu": total_pcu,
            "total_vehicles": total_vehicles,
            "avg_pcu_per_intersection": total_pcu / self.n_intersections,
            "avg_queue_pcu": total_pcu / self.n_intersections,  # Average PCU per intersection
            "avg_queue_raw": avg_queue_raw,  # Average raw vehicle count per intersection
            "avg_queue": avg_queue_raw,  # For compatibility with train.py
            "throughput": len(self.arrived_vehicles),
            "avg_travel_time": avg_travel_time,
            # Emissions (mg). episode_co2 is the running network total; co2_per_veh
            # normalizes by completed trips to suppress the fixed distance-driven
            # component and surface the controllable (idle/stop-go) slice.
            "episode_co2": self.episode_co2,
            "co2_per_veh": (self.episode_co2 / max(len(self.arrived_vehicles), 1)),
            "scenario": self.scenario,
            "vehicle_class_counts": vehicle_class_counts,
            "turning_counts": self.turning_counts.copy(),
            "queue_length": self._compute_avg_queue_pcu(),
            "waiting_time": self._compute_avg_waiting_time(),
        }
    
    def _track_vehicle_metrics(self) -> None:
        try:
            # Track newly departed vehicles
            current_departed = set(self.traci_conn.simulation.getDepartedIDList())
            for veh_id in current_departed:
                if veh_id not in self.departed_vehicles:
                    self.departed_vehicles.add(veh_id)
                    self.vehicle_travel_times[veh_id] = self.current_step
            
            # Track newly arrived vehicles and calculate travel time
            current_arrived = set(self.traci_conn.simulation.getArrivedIDList())
            for veh_id in current_arrived:
                if veh_id not in self.arrived_vehicles:
                    self.arrived_vehicles.add(veh_id)
                    if veh_id in self.vehicle_travel_times:
                        departure_time = self.vehicle_travel_times[veh_id]
                        travel_time = self.current_step - departure_time
                        self.vehicle_travel_times[veh_id] = travel_time
        except Exception:
            # If TraCI calls fail, skip metrics tracking for this step
            pass
    
    def close(self):

        if self.traci_conn is not None:
            try:
                self.traci_conn.close()
            except:
                pass
            self.traci_conn = None
    
    def __del__(self):
        self.close()
