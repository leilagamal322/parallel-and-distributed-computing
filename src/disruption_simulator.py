"""
Network/Service Disruption Simulator
Simulates various failure scenarios:
- Force timeouts (slow responses)
- Drop packets (connection failures)
- Temporary shutdowns (kill and restart services)

Used to test resilience and automatic recovery
"""

import subprocess
import time
import random
import sys
import signal
import os
from typing import List, Dict
import psutil

BASE_PORT = 50051


class DisruptionSimulator:
    def __init__(self, num_replicas: int):
        self.num_replicas = num_replicas
        self.processes: Dict[int, subprocess.Popen] = {}  # port -> process
        self.disruption_log = []
        
    def launch_replica(self, replica_id: int) -> bool:
        """Launch a single replica server"""
        port = BASE_PORT + replica_id
        
        try:
            # Check if port is already in use
            if port in self.processes and self.processes[port].poll() is None:
                print(f"[SIMULATOR] Replica on port {port} already running")
                return False
            
            # Launch server process
            p = subprocess.Popen(
                [sys.executable, "src/server.py", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.processes[port] = p
            print(f"[SIMULATOR] [OK] Launched replica {replica_id} on port {port} (PID: {p.pid})")
            return True
            
        except Exception as e:
            print(f"[SIMULATOR] [ERROR] Failed to launch replica {replica_id}: {e}")
            return False
    
    def launch_all_replicas(self):
        """Launch all replica servers"""
        print(f"\n[SIMULATOR] Launching {self.num_replicas} replicas...")
        
        for i in range(self.num_replicas):
            self.launch_replica(i)
            time.sleep(1.5)  # Avoid GPU initialization race
        
        print(f"[SIMULATOR] All {self.num_replicas} replicas launched\n")
    
    def kill_replica(self, replica_id: int) -> bool:
        """Kill a specific replica (simulate crash/shutdown)"""
        port = BASE_PORT + replica_id
        
        if port not in self.processes:
            print(f"[SIMULATOR] Replica {replica_id} not tracked")
            return False
        
        process = self.processes[port]
        
        if process.poll() is not None:
            print(f"[SIMULATOR] Replica {replica_id} already dead")
            return False
        
        try:
            # Terminate the process
            if os.name == 'nt':  # Windows
                process.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(0.5)
                if process.poll() is None:
                    process.kill()
            else:  # Unix
                process.terminate()
                time.sleep(0.5)
                if process.poll() is None:
                    process.kill()
            
            timestamp = time.strftime('%H:%M:%S')
            print(f"[SIMULATOR] {timestamp} | [KILLED] replica {replica_id} on port {port}")
            self.disruption_log.append({
                'time': timestamp,
                'action': 'KILL',
                'replica': replica_id,
                'port': port
            })
            return True
            
        except Exception as e:
            print(f"[SIMULATOR] Failed to kill replica {replica_id}: {e}")
            return False
    
    def restart_replica(self, replica_id: int) -> bool:
        """Restart a specific replica (simulate recovery)"""
        timestamp = time.strftime('%H:%M:%S')
        print(f"[SIMULATOR] {timestamp} | [RESTARTING] replica {replica_id}...")
        
        # Kill if still running
        self.kill_replica(replica_id)
        time.sleep(1)
        
        # Relaunch
        success = self.launch_replica(replica_id)
        
        if success:
            self.disruption_log.append({
                'time': timestamp,
                'action': 'RESTART',
                'replica': replica_id,
                'port': BASE_PORT + replica_id
            })
        
        return success
    
    def simulate_temporary_shutdown(self, replica_id: int, duration: float):
        """
        Simulate temporary shutdown: kill replica, wait, then restart
        This tests automatic recovery without human intervention
        """
        timestamp = time.strftime('%H:%M:%S')
        print(f"\n[SIMULATOR] {timestamp} | [TEMPORARY SHUTDOWN] replica {replica_id} for {duration}s")
        
        self.kill_replica(replica_id)
        
        print(f"[SIMULATOR] Waiting {duration}s before auto-recovery...")
        time.sleep(duration)
        
        print(f"[SIMULATOR] Auto-recovering replica {replica_id}...")
        self.restart_replica(replica_id)
    
    def simulate_packet_drop(self, replica_id: int):
        """
        Simulate packet drop by killing replica momentarily
        This causes connection failures
        """
        timestamp = time.strftime('%H:%M:%S')
        print(f"\n[SIMULATOR] {timestamp} | [PACKET DROP] replica {replica_id}")
        
        self.kill_replica(replica_id)
        time.sleep(0.5)  # Brief outage
        self.restart_replica(replica_id)
    
    def get_replica_status(self) -> Dict[int, str]:
        """Get status of all replicas"""
        status = {}
        for i in range(self.num_replicas):
            port = BASE_PORT + i
            if port in self.processes:
                process = self.processes[port]
                if process.poll() is None:
                    status[i] = "RUNNING"
                else:
                    status[i] = "DEAD"
            else:
                status[i] = "NOT_STARTED"
        return status
    
    def print_status(self):
        """Print current status of all replicas"""
        status = self.get_replica_status()
        print("\n[SIMULATOR] Current Status:")
        for replica_id, state in status.items():
            port = BASE_PORT + replica_id
            status_marker = "[OK]" if state == "RUNNING" else "[ERROR]"
            print(f"  {status_marker} Replica {replica_id} (port {port}): {state}")
        print()
    
    def run_disruption_scenario(self, duration: int = 60):
        """
        Run a comprehensive disruption scenario
        - Random temporary shutdowns
        - Random packet drops
        - Automatic recovery
        """
        print("\n" + "=" * 80)
        print("[SIMULATOR] Starting Disruption Scenario")
        print(f"[SIMULATOR] Duration: {duration} seconds")
        print(f"[SIMULATOR] Replicas: {self.num_replicas}")
        print("=" * 80 + "\n")
        
        start_time = time.time()
        last_disruption = start_time
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Introduce disruptions every 10-20 seconds
            if current_time - last_disruption > random.uniform(10, 20):
                disruption_type = random.choice(['shutdown', 'packet_drop', 'restart'])
                replica_id = random.randint(0, self.num_replicas - 1)
                
                if disruption_type == 'shutdown':
                    shutdown_duration = random.uniform(5, 15)
                    self.simulate_temporary_shutdown(replica_id, shutdown_duration)
                    
                elif disruption_type == 'packet_drop':
                    self.simulate_packet_drop(replica_id)
                    
                elif disruption_type == 'restart':
                    self.restart_replica(replica_id)
                
                last_disruption = current_time
                self.print_status()
            
            time.sleep(1)
        
        print("\n" + "=" * 80)
        print("[SIMULATOR] Disruption Scenario Complete")
        print(f"[SIMULATOR] Total disruptions: {len(self.disruption_log)}")
        print("=" * 80 + "\n")
    
    def cleanup(self):
        """Kill all replica processes"""
        print("\n[SIMULATOR] Cleaning up all replicas...")
        
        for i in range(self.num_replicas):
            self.kill_replica(i)
        
        print("[SIMULATOR] Cleanup complete")


def main():
    if len(sys.argv) < 2:
        print("Usage: python disruption_simulator.py <num_replicas> [duration]")
        print("Example: python disruption_simulator.py 3 60")
        sys.exit(1)
    
    num_replicas = int(sys.argv[1])
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    simulator = DisruptionSimulator(num_replicas)
    
    try:
        # Launch all replicas
        simulator.launch_all_replicas()
        time.sleep(3)  # Wait for all to be ready
        
        simulator.print_status()
        
        # Run disruption scenario
        simulator.run_disruption_scenario(duration)
        
    except KeyboardInterrupt:
        print("\n[SIMULATOR] Interrupted by user")
    
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    main()

