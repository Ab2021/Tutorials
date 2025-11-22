import os
import glob

base_phase2 = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase2_Stream_Processing_Flink"
base_phase3 = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase3_Advanced_Architecture"

patterns_to_delete = [
    # Week 4 Duplicates
    os.path.join(base_phase2, "Week4_Stateful_Processing", "Day1_State_Management_*.md"),
    os.path.join(base_phase2, "Week4_Stateful_Processing", "Day2_State_Backends_*.md"),
    os.path.join(base_phase2, "Week4_Stateful_Processing", "Day3_Checkpointing_Fault_Tolerance_*.md"),
    os.path.join(base_phase2, "Week4_Stateful_Processing", "Day4_Savepoints_vs_Checkpoints_*.md"),
    os.path.join(base_phase2, "Week4_Stateful_Processing", "Day5_State_Evolution_*.md"),
    
    # Week 5 Duplicates
    os.path.join(base_phase2, "Week5_Advanced_Flink", "Day2_Complex_Event_Processing_*.md"),
    os.path.join(base_phase2, "Week5_Advanced_Flink", "Day3_Joins_in_Streaming_*.md"),
    os.path.join(base_phase2, "Week5_Advanced_Flink", "Day5_Deployment_Modes_*.md"),
    
    # Week 6 Duplicates
    os.path.join(base_phase3, "Week6_Streaming_Patterns", "Day2_Kappa_Architecture_*.md"),
    os.path.join(base_phase3, "Week6_Streaming_Patterns", "Day3_Stream_Enrichment_*.md"),
    os.path.join(base_phase3, "Week6_Streaming_Patterns", "Day4_DLQ_Error_Handling_*.md"),
    os.path.join(base_phase3, "Week6_Streaming_Patterns", "Day5_Idempotency_Transactions_*.md"),
]

print("🚀 Starting Cleanup of Duplicate Placeholders...")

deleted_count = 0
for pattern in patterns_to_delete:
    files = glob.glob(pattern)
    for f in files:
        try:
            # Double check size to be safe (only delete small files < 500 bytes)
            if os.path.getsize(f) < 500:
                os.remove(f)
                print(f"Deleted: {os.path.basename(f)}")
                deleted_count += 1
            else:
                print(f"SKIPPED (Too large): {os.path.basename(f)}")
        except Exception as e:
            print(f"Error deleting {f}: {e}")

print(f"✅ Cleanup Complete. Deleted {deleted_count} files.")
