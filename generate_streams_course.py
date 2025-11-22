import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda"

structure = {
    "Phase1_Foundations": {
        "Week1_Kafka_Architecture": [
            "Day1_Intro_Log_Abstraction",
            "Day2_Kafka_Architecture_Deep_Dive",
            "Day3_Topics_Partitions_Segments",
            "Day4_Producers_Consumers",
            "Day5_Reliability_Durability"
        ],
        "Week2_Redpanda_HighPerformance": [
            "Day1_Intro_Redpanda_Architecture",
            "Day2_Redpanda_vs_Kafka",
            "Day3_Schema_Registry_Serialization",
            "Day4_Admin_Operations_Tiered_Storage",
            "Day5_Advanced_Config_Tuning"
        ]
    },
    "Phase2_Stream_Processing_Flink": {
        "Week3_Flink_Fundamentals": [
            "Day1_Intro_Stream_Processing",
            "Day2_DataStream_API_Basics",
            "Day3_Time_Semantics_Watermarks",
            "Day4_Windowing_Strategies",
            "Day5_Triggers_Evictors"
        ],
        "Week4_Stateful_Processing": [
            "Day1_State_Management",
            "Day2_State_Backends",
            "Day3_Checkpointing_Fault_Tolerance",
            "Day4_Savepoints_vs_Checkpoints",
            "Day5_State_Evolution"
        ],
        "Week5_Advanced_Flink": [
            "Day1_Flink_SQL_Table_API",
            "Day2_Complex_Event_Processing",
            "Day3_Joins_in_Streaming",
            "Day4_Async_IO_Side_Outputs",
            "Day5_Deployment_Modes"
        ]
    },
    "Phase3_Advanced_Architecture": {
        "Week6_Streaming_Patterns": [
            "Day1_Event_Sourcing_CQRS",
            "Day2_Kappa_Architecture",
            "Day3_Stream_Enrichment",
            "Day4_DLQ_Error_Handling",
            "Day5_Idempotency_Transactions"
        ],
        "Week7_Reliability_Scalability": [
            "Day1_Backpressure_Handling",
            "Day2_Scaling_Streaming_Systems",
            "Day3_Geo_Replication",
            "Day4_Stream_Governance",
            "Day5_Security_in_Streaming"
        ]
    },
    "Phase4_Production_CaseStudies": {
        "Week8_Observability_Operations": [
            "Day1_Monitoring_Kafka_Redpanda",
            "Day2_Monitoring_Flink",
            "Day3_Alerting_SLOs",
            "Day4_Troubleshooting_Issues",
            "Day5_Capacity_Planning"
        ],
        "Week9_RealWorld_CaseStudies": [
            "Day1_Fraud_Detection",
            "Day2_IoT_Telemetry",
            "Day3_Clickstream_Analytics",
            "Day4_CDC_Debezium",
            "Day5_Log_Aggregation_SIEM"
        ],
        "Week10_Challenges_Trends": [
            "Day1_Handling_Skewed_Data",
            "Day2_Schema_Evolution_Challenges",
            "Day3_Late_Data_Correctness",
            "Day4_Streaming_Databases",
            "Day5_Unified_Batch_Stream"
        ]
    }
}

def create_file(path, title, content_type):
    content = f"# {title}\n\n"
    if content_type == "Core":
        content += "## Core Concepts & Theory\n\n### Theoretical Foundation\n[Detailed theoretical explanation of the concept...]\n\n### Architectural Reasoning\n[Why is this designed this way? What are the trade-offs?]\n\n### Key Components\n- Component 1\n- Component 2\n"
    elif content_type == "DeepDive":
        content += "## Deep Dive & Internals\n\n### Internal Mechanics\n[How it works under the hood...]\n\n### Advanced Reasoning\n[Complex scenarios and edge cases...]\n\n### Performance Implications\n[Latency, Throughput, Resource usage...]\n"
    elif content_type == "Interview":
        content += "## Interview Questions & Challenges\n\n### Common Interview Questions\n1. Question 1?\n2. Question 2?\n\n### Production Challenges\n- Challenge 1\n- Challenge 2\n\n### Troubleshooting Scenarios\n[Scenario description...]\n"
    
    with open(path, "w") as f:
        f.write(content)

print("🚀 Starting Course Generation...")

for phase, weeks in structure.items():
    phase_path = os.path.join(base_path, phase)
    os.makedirs(phase_path, exist_ok=True)
    
    for week, days in weeks.items():
        week_path = os.path.join(phase_path, week)
        os.makedirs(week_path, exist_ok=True)
        
        # Create Labs directory
        labs_path = os.path.join(week_path, "labs")
        os.makedirs(labs_path, exist_ok=True)
        
        # Create README for Labs
        with open(os.path.join(labs_path, "README.md"), "w") as f:
            f.write(f"# Labs for {week}\n\nThis directory contains hands-on labs for {week}.\n\n## Lab Index\n")
            for i in range(1, 16):
                f.write(f"- [Lab {i:02d}](lab_{i:02d}.md)\n")

        for day in days:
            # Create 3 files per day
            create_file(os.path.join(week_path, f"{day}_Core.md"), f"{day}: Core Concepts", "Core")
            create_file(os.path.join(week_path, f"{day}_DeepDive.md"), f"{day}: Deep Dive", "DeepDive")
            create_file(os.path.join(week_path, f"{day}_Interview.md"), f"{day}: Interview Prep", "Interview")

print("✅ Course Structure Generated Successfully!")
