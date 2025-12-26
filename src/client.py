import grpc
import time
import sys
import particle_pb2
import particle_pb2_grpc

BASE_PORT = 50051
DURATION = 60      
TIMEOUT = 2.0      

def main(num_replicas):
    # 1. Dynamically generate target list based on input argument
    targets = [f"localhost:{BASE_PORT + i}" for i in range(num_replicas)]
    stubs = []
    
    print(f"[CLIENT] Configuration:")
    print(f" -> Looking for {num_replicas} replicas")
    print(f" -> Ports: {BASE_PORT} to {BASE_PORT + num_replicas - 1}")
    print(f" -> Duration: {DURATION} seconds")
    print("-" * 50)
    
    # Create lazy connections (no immediate connection check)
    for target in targets:
        channel = grpc.insecure_channel(target)
        stub = particle_pb2_grpc.ParticleSimStub(channel)
        stubs.append((target, stub))

    start_time = time.time()
    request_count = 0

    # 2. Main Request Loop
    while time.time() - start_time < DURATION:
        success = False
        
        # Round-robin or Failover attempt
        for target, stub in stubs:
            try:
                # Attempt request
                resp = stub.Step(
                    particle_pb2.StepRequest(steps=10), 
                    timeout=TIMEOUT
                )
                
                print(f"{time.strftime('%H:%M:%S')} | {resp.replica_id:<20} | {resp.latency_ms:.2f} ms | ✅ OK")
                success = True
                break # Request succeeded, stop trying other replicas

            except grpc.RpcError:
                print(f"[CLIENT] Failed to reach {target}, trying next replica...")
                # Replica is offline/busy, try the next one silently
                continue 
        
        # If loop finishes and success is still False, NO replicas worked
        if not success:
            print(f"[CLIENT] ⚠️ All {num_replicas} replicas are down. Retrying...")
            time.sleep(1)
        else:
            # Normal delay between requests
            time.sleep(0.5)
            
        request_count += 1

    print("-" * 50)
    print(f"[CLIENT] Test Complete. Total Requests: {request_count}")

if __name__ == "__main__":
    # Default to 3 if no argument is provided
    count = 3
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
        
    main(count)