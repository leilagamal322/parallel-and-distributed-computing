from mpi4py import MPI
import numpy as np
import time  

comm = MPI.COMM_WORLD       # initialize communicator that has all ranks
rank = comm.Get_rank()      # Rank id of each process
size = comm.Get_size()      # total number of processes

print(f"Rank {rank} started")   # confirming all processes launched correctly


# 1️ Bcast collective (timed)
if rank == 0:  # the root (rank 0) has the real value 42
    data = 42
else:
    data = None  # other ranks have no value

comm.Barrier()                     # sync before timing starts
t0 = time.perf_counter()

data = comm.bcast(data, root=0)    # rank 0 sends data to all ranks

t1 = time.perf_counter()

print(f"Rank {rank} received Bcast value = {data} (time={t1 - t0:.6f}s)")


# 2️⃣ Point-to-point (blocking send/receive)
comm.Barrier()
t0 = time.perf_counter()

if rank == 0:
    comm.send("Hello from rank 0", dest=1)
elif rank == 1:
    msg = comm.recv(source=0)
    print(f"Rank 1 received message: {msg}")

t1 = time.perf_counter()

if rank in [0, 1]:
    print(f"Rank {rank} blocking P2P time = {t1 - t0:.6f}s")


# 3️⃣ Non-blocking point-to-point (Isend, Irecv, Wait)
reqs = []  # list to store pending communications

comm.Barrier()
t0 = time.perf_counter()

if rank == 0:
    # immediately send and don't wait
    req = comm.isend(np.array([rank]), dest=1)
    reqs.append(req)

elif rank == 1:
    # receive without blocking the program
    req = comm.irecv(source=0)
    data_async = req.wait()  # wait for message to actually arrive
    print("Rank 1 got async:", data_async)

t1 = time.perf_counter()

if rank in [0, 1]:
    print(f"Rank {rank} non-blocking P2P time = {t1 - t0:.6f}s")


# 4️⃣ Scatter + Gather (two collectives)
# SCATTER
comm.Barrier()
if rank == 0:
    arr = np.arange(size * 2)      # 8 items total
    scatter_data = np.array_split(arr, size)  # split into 4 equal chunks
else:
    scatter_data = None

t0 = time.perf_counter()

recv_chunk = comm.scatter(scatter_data, root=0)

t1 = time.perf_counter()

print(f"Rank {rank} received SCATTER chunk: {recv_chunk} (time={t1 - t0:.6f}s)")


# GATHER
comm.Barrier()
value = rank  # each rank sends its ID

t0 = time.perf_counter()

all_values = comm.gather(value, root=0)

t1 = time.perf_counter()

if rank == 0:
    print("Gathered:", all_values, f"(time={t1 - t0:.6f}s)")
