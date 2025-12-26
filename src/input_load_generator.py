"""
Input Load Generator for MPI Simulation
Simulates continuous incoming requests (streaming) with adjustable input rate.
Logs timestamps for all events.
"""

import time
import random
import logging
import argparse
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
from queue import Queue
import json


class RequestType(Enum):
    """Types of requests that can be generated"""
    ADD_PARTICLES = "add_particles"
    CHANGE_PARAMETERS = "change_parameters"
    FRAME_UPDATE = "frame_update"
    RENDER_REQUEST = "render_request"
    SEED_CHANGE = "seed_change"


@dataclass
class Request:
    """Represents a single request event"""
    request_id: int
    request_type: RequestType
    timestamp: float
    data: dict
    processing_time: Optional[float] = None


class InputLoadGenerator:
    """
    Generates continuous streaming requests with adjustable rate.
    Logs all events with timestamps.
    """
    
    def __init__(self, 
                 requests_per_second: float = 10.0,
                 duration_seconds: float = 60.0,
                 log_file: Optional[str] = None,
                 callback: Optional[Callable] = None):
        """
        Initialize the load generator.
        
        Args:
            requests_per_second: Rate of request generation (requests/sec)
            duration_seconds: Duration to run (must be >= 60 seconds)
            log_file: Optional file path to write logs
            callback: Optional callback function to handle requests
        """
        if duration_seconds < 60.0:
            raise ValueError("Duration must be at least 60 seconds")
            
        self.requests_per_second = requests_per_second
        self.duration_seconds = duration_seconds
        self.callback = callback
        self.request_queue = Queue()
        self.running = False
        self.request_counter = 0
        self.start_time = None
        self.end_time = None
        
        # Setup logging
        self.logger = logging.getLogger('InputLoadGenerator')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'by_type': {rt.value: 0 for rt in RequestType},
            'first_request_time': None,
            'last_request_time': None,
            'min_interval': float('inf'),
            'max_interval': 0.0,
            'avg_interval': 0.0
        }
        self.last_request_time = None
        
    def _generate_add_particles_request(self) -> Request:
        """Generate a request to add particles"""
        num_particles = random.randint(1, 50)
        x_range = (0, 1200)
        y_range = (0, 400)
        vx_range = (-2, 2)
        vy_range = (0, 2)
        
        data = {
            'num_particles': num_particles,
            'x_range': x_range,
            'y_range': y_range,
            'vx_range': vx_range,
            'vy_range': vy_range
        }
        
        return Request(
            request_id=self.request_counter,
            request_type=RequestType.ADD_PARTICLES,
            timestamp=time.perf_counter(),
            data=data
        )
    
    def _generate_change_parameters_request(self) -> Request:
        """Generate a request to change simulation parameters"""
        params = {}
        param_choices = {
            'gravity': (0.1, 1.0),
            'damping': (0.5, 0.95),
            'dt': (0.5, 2.0)
        }
        
        # Randomly select 1-3 parameters to change
        selected_params = random.sample(list(param_choices.keys()), 
                                       random.randint(1, len(param_choices)))
        
        for param in selected_params:
            min_val, max_val = param_choices[param]
            params[param] = random.uniform(min_val, max_val)
        
        data = {'parameters': params}
        
        return Request(
            request_id=self.request_counter,
            request_type=RequestType.CHANGE_PARAMETERS,
            timestamp=time.perf_counter(),
            data=data
        )
    
    def _generate_frame_update_request(self) -> Request:
        """Generate a frame update request"""
        data = {
            'frame_number': self.request_counter,
            'force_update': random.random() < 0.1  # 10% chance of forced update
        }
        
        return Request(
            request_id=self.request_counter,
            request_type=RequestType.FRAME_UPDATE,
            timestamp=time.perf_counter(),
            data=data
        )
    
    def _generate_render_request(self) -> Request:
        """Generate a render/view update request"""
        data = {
            'render_quality': random.choice(['low', 'medium', 'high']),
            'viewport': {
                'x': random.randint(0, 800),
                'y': random.randint(0, 600),
                'width': random.randint(200, 400),
                'height': random.randint(200, 400)
            }
        }
        
        return Request(
            request_id=self.request_counter,
            request_type=RequestType.RENDER_REQUEST,
            timestamp=time.perf_counter(),
            data=data
        )
    
    def _generate_seed_change_request(self) -> Request:
        """Generate a request to change random seed/parameters"""
        data = {
            'new_seed': random.randint(0, 2**31 - 1),
            'parameter_set': random.choice(['default', 'high_energy', 'low_energy', 'chaotic'])
        }
        
        return Request(
            request_id=self.request_counter,
            request_type=RequestType.SEED_CHANGE,
            timestamp=time.perf_counter(),
            data=data
        )
    
    def _generate_request(self) -> Request:
        """Generate a random request based on weighted probabilities"""
        self.request_counter += 1
        
        # Weighted distribution of request types
        weights = {
            RequestType.ADD_PARTICLES: 0.3,
            RequestType.CHANGE_PARAMETERS: 0.2,
            RequestType.FRAME_UPDATE: 0.25,
            RequestType.RENDER_REQUEST: 0.15,
            RequestType.SEED_CHANGE: 0.1
        }
        
        request_type = random.choices(
            list(weights.keys()),
            weights=list(weights.values())
        )[0]
        
        generators = {
            RequestType.ADD_PARTICLES: self._generate_add_particles_request,
            RequestType.CHANGE_PARAMETERS: self._generate_change_parameters_request,
            RequestType.FRAME_UPDATE: self._generate_frame_update_request,
            RequestType.RENDER_REQUEST: self._generate_render_request,
            RequestType.SEED_CHANGE: self._generate_seed_change_request
        }
        
        return generators[request_type]()
    
    def _log_request(self, request: Request):
        """Log a request event with timestamp"""
        elapsed = request.timestamp - self.start_time if self.start_time else 0.0
        
        log_msg = (
            f"REQUEST[{request.request_id:06d}] "
            f"Type={request.request_type.value} "
            f"Time={request.timestamp:.6f} "
            f"Elapsed={elapsed:.3f}s "
            f"Data={json.dumps(request.data, indent=None)}"
        )
        
        self.logger.info(log_msg)
        
        # Update statistics
        self.stats['total_requests'] += 1
        self.stats['by_type'][request.request_type.value] += 1
        
        if self.stats['first_request_time'] is None:
            self.stats['first_request_time'] = request.timestamp
        
        self.stats['last_request_time'] = request.timestamp
        
        if self.last_request_time is not None:
            interval = request.timestamp - self.last_request_time
            self.stats['min_interval'] = min(self.stats['min_interval'], interval)
            self.stats['max_interval'] = max(self.stats['max_interval'], interval)
            # Update running average
            n = self.stats['total_requests'] - 1
            self.stats['avg_interval'] = (
                (self.stats['avg_interval'] * (n - 1) + interval) / n
                if n > 0 else interval
            )
        
        self.last_request_time = request.timestamp
    
    def _generator_thread(self):
        """Thread that generates requests at the specified rate"""
        interval = 1.0 / self.requests_per_second
        
        self.logger.info(f"Starting request generation at {self.requests_per_second} req/s")
        self.logger.info(f"Target duration: {self.duration_seconds} seconds")
        
        while self.running:
            request = self._generate_request()
            self._log_request(request)
            
            # Add to queue
            self.request_queue.put(request)
            
            # Call callback if provided
            if self.callback:
                try:
                    self.callback(request)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
            
            # Sleep to maintain rate (with some jitter for realism)
            sleep_time = interval * random.uniform(0.8, 1.2)
            time.sleep(sleep_time)
            
            # Check if we've exceeded duration
            if self.start_time and (time.perf_counter() - self.start_time) >= self.duration_seconds:
                self.running = False
                break
    
    def run(self):
        """Run the load generator for the specified duration"""
        self.running = True
        self.start_time = time.perf_counter()
        self.end_time = self.start_time + self.duration_seconds
        
        self.logger.info("=" * 80)
        self.logger.info("INPUT LOAD GENERATOR STARTED")
        self.logger.info(f"Rate: {self.requests_per_second} requests/second")
        self.logger.info(f"Duration: {self.duration_seconds} seconds")
        self.logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        self.logger.info("=" * 80)
        
        try:
            self._generator_thread()
        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user")
            self.running = False
        
        actual_duration = time.perf_counter() - self.start_time
        self.logger.info("=" * 80)
        self.logger.info("INPUT LOAD GENERATOR STOPPED")
        self.logger.info(f"Actual duration: {actual_duration:.3f} seconds")
        self._print_statistics()
        self.logger.info("=" * 80)
    
    def _print_statistics(self):
        """Print statistics about generated requests"""
        self.logger.info("STATISTICS:")
        self.logger.info(f"  Total requests: {self.stats['total_requests']}")
        self.logger.info(f"  Requests by type:")
        for req_type, count in self.stats['by_type'].items():
            if count > 0:
                percentage = (count / self.stats['total_requests']) * 100
                self.logger.info(f"    {req_type}: {count} ({percentage:.1f}%)")
        
        if self.stats['total_requests'] > 1:
            self.logger.info(f"  Inter-request intervals:")
            self.logger.info(f"    Min: {self.stats['min_interval']*1000:.3f} ms")
            self.logger.info(f"    Max: {self.stats['max_interval']*1000:.3f} ms")
            self.logger.info(f"    Avg: {self.stats['avg_interval']*1000:.3f} ms")
            
            actual_rate = self.stats['total_requests'] / (
                self.stats['last_request_time'] - self.stats['first_request_time']
            ) if self.stats['last_request_time'] != self.stats['first_request_time'] else 0
            self.logger.info(f"  Actual rate: {actual_rate:.2f} requests/second")
    
    def stop(self):
        """Stop the load generator"""
        self.running = False
    
    def get_queue(self) -> Queue:
        """Get the request queue"""
        return self.request_queue


def example_callback(request: Request):
    """Example callback function to process requests"""
    print(f"Processing request {request.request_id}: {request.request_type.value}")


def main():
    """Main entry point for standalone execution"""
    parser = argparse.ArgumentParser(
        description='Input Load Generator for MPI Simulation'
    )
    parser.add_argument(
        '--rate', '-r',
        type=float,
        default=10.0,
        help='Requests per second (default: 10.0)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=60.0,
        help='Duration in seconds (minimum: 60.0, default: 60.0)'
    )
    parser.add_argument(
        '--log-file', '-l',
        type=str,
        default=None,
        help='Log file path (default: stdout only)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    if args.duration < 60.0:
        parser.error("Duration must be at least 60 seconds")
    
    if args.seed is not None:
        random.seed(args.seed)
    
    generator = InputLoadGenerator(
        requests_per_second=args.rate,
        duration_seconds=args.duration,
        log_file=args.log_file,
        callback=example_callback
    )
    
    generator.run()


if __name__ == "__main__":
    main()

