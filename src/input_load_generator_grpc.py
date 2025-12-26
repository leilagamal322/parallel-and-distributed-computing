import time
import random
import logging
import argparse
import threading
import grpc
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
from queue import Queue
import json

try:
    import particle_pb2
    import particle_pb2_grpc
except ImportError:
    print("Error: particle_pb2 modules not found. Make sure particle.proto is compiled.")
    print("Run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. particle.proto")
    raise


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
    grpc_response: Optional[dict] = None
    grpc_success: Optional[bool] = None
    grpc_error: Optional[str] = None


class GrpcLoadGenerator:
    
    
    def __init__(self, 
                 requests_per_second: float = 10.0,
                 duration_seconds: float = 60.0,
                 grpc_targets: List[str] = None,
                 log_file: Optional[str] = None,
                 timeout: float = 5.0):
       
        if duration_seconds < 60.0:
            raise ValueError("Duration must be at least 60 seconds")
            
        self.requests_per_second = requests_per_second
        self.duration_seconds = duration_seconds
        self.timeout = timeout
        self.request_queue = Queue()
        self.running = False
        self.request_counter = 0
        self.start_time = None
        self.end_time = None
        
        # Setup logging FIRST (before gRPC connections that use logger)
        self.logger = logging.getLogger('GrpcLoadGenerator')
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
        
        # Setup gRPC connections
        if grpc_targets is None:
            grpc_targets = ['localhost:50051']
        self.grpc_targets = grpc_targets
        self.channels = []
        self.stubs = []
        self._initialize_grpc_connections()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'by_type': {rt.value: {'total': 0, 'success': 0, 'failed': 0} for rt in RequestType},
            'first_request_time': None,
            'last_request_time': None,
            'min_interval': float('inf'),
            'max_interval': 0.0,
            'avg_interval': 0.0,
            'min_latency_ms': float('inf'),
            'max_latency_ms': 0.0,
            'total_latency_ms': 0.0,
            'grpc_errors': {}
        }
        self.last_request_time = None
        self.current_target_index = 0
        
    def _initialize_grpc_connections(self):
        """Initialize gRPC channels and stubs"""
        self.logger.info(f"Initializing gRPC connections to {len(self.grpc_targets)} target(s)")
        
        for target in self.grpc_targets:
            try:
                options = [
                    ('grpc.keepalive_time_ms', 10000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', 1),
                    ('grpc.http2.max_pings_without_data', 0),
                ]
                
                channel = grpc.insecure_channel(target, options=options)
                stub = particle_pb2_grpc.ParticleSimStub(channel)
                
                self.channels.append(channel)
                self.stubs.append((target, stub))
                
                self.logger.info(f"  Connected to {target}")
            except Exception as e:
                self.logger.error(f"  Failed to connect to {target}: {e}")
        
        if not self.stubs:
            raise RuntimeError("No gRPC connections established")
    
    def _get_stub(self):
        """Get the next stub using round-robin"""
        if not self.stubs:
            return None
        target, stub = self.stubs[self.current_target_index]
        self.current_target_index = (self.current_target_index + 1) % len(self.stubs)
        return target, stub
    
    def _map_request_to_steps(self, request: Request) -> int:
        """Map request type to number of simulation steps"""
        mapping = {
            RequestType.ADD_PARTICLES: random.randint(5, 15),
            RequestType.CHANGE_PARAMETERS: random.randint(1, 5),
            RequestType.FRAME_UPDATE: random.randint(1, 3),
            RequestType.RENDER_REQUEST: random.randint(1, 2),
            RequestType.SEED_CHANGE: random.randint(10, 20),
        }
        return mapping.get(request.request_type, 10)
    
    def _send_grpc_request(self, request: Request) -> Tuple[bool, Optional[dict], Optional[str]]:
        """Send request to gRPC service"""
        steps = self._map_request_to_steps(request)
        
        target, stub = self._get_stub()
        if stub is None:
            return False, None, "No gRPC stub available"
        
        try:
            grpc_request = particle_pb2.StepRequest(steps=steps)
            start_time = time.perf_counter()
            
            response = stub.Step(grpc_request, timeout=self.timeout)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            response_data = {
                'replica_id': response.replica_id,
                'latency_ms': response.latency_ms,
                'timestamp': response.timestamp,
                'measured_latency_ms': latency_ms
            }
            
            return True, response_data, None
            
        except grpc.RpcError as e:
            error_msg = f"gRPC error [{e.code()}]: {e.details()}"
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            return False, None, error_msg
    
    def _generate_add_particles_request(self) -> Request:
        """Generate a request to add particles"""
        num_particles = random.randint(1, 50)
        data = {
            'num_particles': num_particles,
            'x_range': [0, 1200],
            'y_range': [0, 400],
            'vx_range': [-2, 2],
            'vy_range': [0, 2]
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
            'force_update': random.random() < 0.1
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
        """Log a request event with timestamp and gRPC response"""
        elapsed = request.timestamp - self.start_time if self.start_time else 0.0
        
        log_parts = [
            f"REQUEST[{request.request_id:06d}]",
            f"Type={request.request_type.value}",
            f"Time={request.timestamp:.6f}",
            f"Elapsed={elapsed:.3f}s"
        ]
        
        if request.grpc_success:
            latency = request.grpc_response.get('measured_latency_ms', 0) if request.grpc_response else 0
            replica = request.grpc_response.get('replica_id', 'unknown') if request.grpc_response else 'unknown'
            log_parts.append(f"gRPC=SUCCESS Latency={latency:.2f}ms Replica={replica}")
        elif request.grpc_success is False:
            log_parts.append(f"gRPC=FAILED Error={request.grpc_error}")
        
        log_parts.append(f"Data={json.dumps(request.data, indent=None)}")
        
        self.logger.info(" ".join(log_parts))
        
        # Update statistics
        self.stats['total_requests'] += 1
        type_stats = self.stats['by_type'][request.request_type.value]
        type_stats['total'] += 1
        
        if request.grpc_success:
            self.stats['successful_requests'] += 1
            type_stats['success'] += 1
            
            if request.grpc_response:
                latency = request.grpc_response.get('measured_latency_ms', 0)
                if latency > 0:
                    self.stats['min_latency_ms'] = min(self.stats['min_latency_ms'], latency)
                    self.stats['max_latency_ms'] = max(self.stats['max_latency_ms'], latency)
                    self.stats['total_latency_ms'] += latency
        else:
            self.stats['failed_requests'] += 1
            type_stats['failed'] += 1
            if request.grpc_error:
                error_type = request.grpc_error.split(':')[0] if ':' in request.grpc_error else request.grpc_error
                self.stats['grpc_errors'][error_type] = self.stats['grpc_errors'].get(error_type, 0) + 1
        
        if self.stats['first_request_time'] is None:
            self.stats['first_request_time'] = request.timestamp
        
        self.stats['last_request_time'] = request.timestamp
        
        if self.last_request_time is not None:
            interval = request.timestamp - self.last_request_time
            self.stats['min_interval'] = min(self.stats['min_interval'], interval)
            self.stats['max_interval'] = max(self.stats['max_interval'], interval)
            n = self.stats['total_requests'] - 1
            self.stats['avg_interval'] = (
                (self.stats['avg_interval'] * (n - 1) + interval) / n
                if n > 0 else interval
            )
        
        self.last_request_time = request.timestamp
    
    def _generator_thread(self):
        """Thread that generates requests at the specified rate and sends them via gRPC"""
        interval = 1.0 / self.requests_per_second
        
        self.logger.info(f"Starting request generation at {self.requests_per_second} req/s")
        self.logger.info(f"Target duration: {self.duration_seconds} seconds")
        self.logger.info(f"gRPC timeout: {self.timeout} seconds")
        
        while self.running:
            request = self._generate_request()
            
            # Send to gRPC service
            success, response, error = self._send_grpc_request(request)
            request.grpc_success = success
            request.grpc_response = response
            request.grpc_error = error
            
            # Log the request
            self._log_request(request)
            
            # Add to queue
            self.request_queue.put(request)
            
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
        self.logger.info("gRPC INPUT LOAD GENERATOR STARTED")
        self.logger.info(f"Rate: {self.requests_per_second} requests/second")
        self.logger.info(f"Duration: {self.duration_seconds} seconds")
        self.logger.info(f"gRPC Targets: {', '.join(self.grpc_targets)}")
        self.logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        self.logger.info("=" * 80)
        
        try:
            self._generator_thread()
        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user")
            self.running = False
        
        actual_duration = time.perf_counter() - self.start_time
        
        # Cleanup gRPC connections
        for channel in self.channels:
            channel.close()
        
        self.logger.info("=" * 80)
        self.logger.info("gRPC INPUT LOAD GENERATOR STOPPED")
        self.logger.info(f"Actual duration: {actual_duration:.3f} seconds")
        self._print_statistics()
        self.logger.info("=" * 80)
    
    def _print_statistics(self):
        """Print statistics about generated requests"""
        self.logger.info("STATISTICS:")
        self.logger.info(f"  Total requests: {self.stats['total_requests']}")
        self.logger.info(f"  Successful gRPC requests: {self.stats['successful_requests']}")
        self.logger.info(f"  Failed gRPC requests: {self.stats['failed_requests']}")
        
        if self.stats['total_requests'] > 0:
            success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100
            self.logger.info(f"  Success rate: {success_rate:.2f}%")
        
        self.logger.info(f"  Requests by type:")
        for req_type, type_stats in self.stats['by_type'].items():
            if type_stats['total'] > 0:
                percentage = (type_stats['total'] / self.stats['total_requests']) * 100
                success_pct = (type_stats['success'] / type_stats['total']) * 100 if type_stats['total'] > 0 else 0
                self.logger.info(
                    f"    {req_type}: {type_stats['total']} ({percentage:.1f}%) "
                    f"[Success: {type_stats['success']} ({success_pct:.1f}%), "
                    f"Failed: {type_stats['failed']}]"
                )
        
        if self.stats['successful_requests'] > 0:
            avg_latency = self.stats['total_latency_ms'] / self.stats['successful_requests']
            self.logger.info(f"  gRPC Latency:")
            self.logger.info(f"    Min: {self.stats['min_latency_ms']:.3f} ms")
            self.logger.info(f"    Max: {self.stats['max_latency_ms']:.3f} ms")
            self.logger.info(f"    Avg: {avg_latency:.3f} ms")
        
        if self.stats['total_requests'] > 1:
            self.logger.info(f"  Inter-request intervals:")
            self.logger.info(f"    Min: {self.stats['min_interval']*1000:.3f} ms")
            self.logger.info(f"    Max: {self.stats['max_interval']*1000:.3f} ms")
            self.logger.info(f"    Avg: {self.stats['avg_interval']*1000:.3f} ms")
            
            actual_rate = self.stats['total_requests'] / (
                self.stats['last_request_time'] - self.stats['first_request_time']
            ) if self.stats['last_request_time'] != self.stats['first_request_time'] else 0
            self.logger.info(f"  Actual rate: {actual_rate:.2f} requests/second")
        
        if self.stats['grpc_errors']:
            self.logger.info(f"  gRPC Errors:")
            for error_type, count in self.stats['grpc_errors'].items():
                self.logger.info(f"    {error_type}: {count}")
    
    def stop(self):
        """Stop the load generator"""
        self.running = False
    
    def get_queue(self) -> Queue:
        """Get the request queue"""
        return self.request_queue


def main():
    """Main entry point for standalone execution"""
    parser = argparse.ArgumentParser(
        description='gRPC Input Load Generator for MPI Simulation'
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
        '--targets', '-t',
        type=str,
        nargs='+',
        default=['localhost:50051'],
        help='gRPC server targets (default: localhost:50051)'
    )
    parser.add_argument(
        '--log-file', '-l',
        type=str,
        default=None,
        help='Log file path (default: stdout only)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=5.0,
        help='gRPC request timeout in seconds (default: 5.0)'
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
    
    generator = GrpcLoadGenerator(
        requests_per_second=args.rate,
        duration_seconds=args.duration,
        grpc_targets=args.targets,
        log_file=args.log_file,
        timeout=args.timeout
    )
    
    generator.run()


if __name__ == "__main__":
    main()

