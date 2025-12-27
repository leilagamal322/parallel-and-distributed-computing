from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import time
import grpc

import particle_pb2
import particle_pb2_grpc


# ---------------- gRPC setup ----------------
GRPC_TARGETS = [
    "localhost:50051",
    "localhost:50052",
    "localhost:50053",
]

channels = [grpc.insecure_channel(t) for t in GRPC_TARGETS]
stubs = [particle_pb2_grpc.ParticleSimStub(ch) for ch in channels]


def call_grpc(batch_id, num_records):
    """
    One gRPC call per micro-batch
    """
    stub = stubs[batch_id % len(stubs)]

    start = time.perf_counter()
    response = stub.Step(
        particle_pb2.StepRequest(steps=10),
        timeout=3.0
    )
    latency = (time.perf_counter() - start) * 1000

    print(
        f"[SPARK] batch={batch_id} "
        f"records={num_records} "
        f"replica={response.replica_id} "
        f"latency={latency:.2f}ms",
        flush=True
    )


def process_batch(batch_df, batch_id):
    """
    Called once per micro-batch
    """
    count = batch_df.count()
    if count == 0:
        return

    call_grpc(batch_id, count)


# ---------------- Spark job ----------------
spark = (
    SparkSession.builder
    .appName("SparkStructuredStreaming-gRPC")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Built-in streaming source (no Kafka needed)
stream_df = (
    spark.readStream
    .format("rate")
    .option("rowsPerSecond", 5)
    .load()
)

query = (
    stream_df
    .writeStream
    .foreachBatch(process_batch)
    .outputMode("update")
    .trigger(processingTime="5 seconds")  # micro-batch every 5s
    .start()
)

query.awaitTermination()
