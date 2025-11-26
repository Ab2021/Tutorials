# Lab: Day 24 - gRPC Python

## Goal
Build a high-performance RPC service. You will define a `Calculator` service in Protobuf and implement it in Python.

## Directory Structure
```
day24/
├── protos/
│   └── calculator.proto
├── server.py
├── client.py
└── requirements.txt
```

## Step 1: Requirements
```text
grpcio
grpcio-tools
```

## Step 2: The Proto (`protos/calculator.proto`)

```protobuf
syntax = "proto3";

package calculator;

service Calculator {
  rpc Add (AddRequest) returns (AddReply) {}
}

message AddRequest {
  int32 a = 1;
  int32 b = 2;
}

message AddReply {
  int32 result = 1;
}
```

## Step 3: Generate Code
Run this command to generate the Python stubs:
```bash
python -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/calculator.proto
```
*   Generates `calculator_pb2.py` (Messages) and `calculator_pb2_grpc.py` (Service).

## Step 4: The Server (`server.py`)

```python
from concurrent import futures
import grpc
import calculator_pb2
import calculator_pb2_grpc

class CalculatorServicer(calculator_pb2_grpc.CalculatorServicer):
    def Add(self, request, context):
        print(f"Received request: {request.a} + {request.b}")
        result = request.a + request.b
        return calculator_pb2.AddReply(result=result)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    calculator_pb2_grpc.add_CalculatorServicer_to_server(CalculatorServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("🚀 Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

## Step 5: The Client (`client.py`)

```python
import grpc
import calculator_pb2
import calculator_pb2_grpc

def run():
    # Connect to server
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = calculator_pb2_grpc.CalculatorStub(channel)
        
        # Call RPC
        response = stub.Add(calculator_pb2.AddRequest(a=10, b=20))
    
    print(f"✅ Result from Server: {response.result}")

if __name__ == '__main__':
    run()
```

## Step 6: Run It

1.  **Install**: `pip install -r requirements.txt`
2.  **Generate**: Run the `protoc` command from Step 3.
3.  **Server**: `python server.py`
4.  **Client**: `python client.py`

## Challenge
Add a **Streaming RPC**.
*   `rpc PrimeNumberDecomposition (PrimeRequest) returns (stream PrimeReply) {}`
*   Client sends `120`. Server streams `2, 2, 2, 3, 5`.
