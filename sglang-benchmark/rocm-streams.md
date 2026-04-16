# ROCm HIP Streams Reference

Sources:
- [HIP Programming Model - Execution Model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html#execution-model)
- [Asynchronous Concurrent Execution - Managing Streams](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html#managing-streams)

---

## Core Concepts

### Streams

Streams are FIFO command buffers tied to a device. Operations enqueued on a stream execute asynchronously and in-order within that stream. Multiple streams on the same device may execute concurrently but are not guaranteed to.

```cpp
hipStream_t streamA, streamB;
hipStreamCreate(&streamA);
hipStreamCreate(&streamB);

// hipStreamCreateWithFlags options:
//   hipStreamDefault      — blocking (default)
//   hipStreamNonBlocking  — non-blocking, allows concurrent execution
hipStreamCreateWithFlags(&streamA, hipStreamNonBlocking);
```

### Events (for cross-stream synchronization)

Events mark a point in a stream. Other streams can wait on an event to enforce ordering.

```cpp
hipEvent_t event;
hipEventCreate(&event);

// Record after kernelA on streamA
kernelA<<<blocks, threads, 0, streamA>>>(d_A, size);
hipEventRecord(event, streamA);

// streamB waits for event before launching kernelB
hipStreamWaitEvent(streamB, event, 0);
kernelB<<<blocks, threads, 0, streamB>>>(d_A, d_B, size);
```

### Synchronization

```cpp
hipStreamSynchronize(streamA);   // block host until streamA completes
hipDeviceSynchronize();          // block host until ALL device work completes
hipEventSynchronize(event);      // block host until event is recorded
```

---

## Concurrency Patterns

### 1. Overlap Data Transfer + Kernel Execution

Use separate streams for memcpy and compute. Requires pinned (page-locked) host memory for async transfers.

```cpp
hipMemcpyAsync(d_data, h_data, size, hipMemcpyHostToDevice, streamA);
kernel<<<grid, block, 0, streamB>>>(d_other, ...);
```

### 2. Concurrent Kernel Execution

Independent kernels on separate streams can run concurrently if GPU resources allow (registers, LDS, CU slots).

```cpp
kernelA<<<grid, block, 0, streamA>>>(d_A, size);
kernelB<<<grid, block, 0, streamB>>>(d_B, size);
```

Concurrent kernels are beneficial only when individual kernels don't fully saturate the GPU. Check traces to verify actual overlap.

### 3. Producer-Consumer with Events

```cpp
// Producer on streamA
kernelProduce<<<grid, block, 0, streamA>>>(d_out, size);
hipEventRecord(doneEvent, streamA);

// Consumer on streamB waits for producer
hipStreamWaitEvent(streamB, doneEvent, 0);
kernelConsume<<<grid, block, 0, streamB>>>(d_out, d_result, size);
```

---

## Relevance to SGLang MoE Dual-Stream

In the SGLang MoE dual-stream optimization for MI355X:
- **Stream 1**: Runs shared expert computation
- **Stream 2**: Runs routed expert (MoE gate + dispatch + expert compute)
- Events synchronize the merge point where shared + routed expert outputs are combined
- `--disable-shared-experts-fusion` flag in SGLang separates the ops so they can run on different HIP streams
- This works because shared experts and routed experts are independent until the final add

### Key Considerations for MI355X

- CDNA 4 wavefront size = 64 threads (vs 32 on NVIDIA)
- Multiple CUs can run concurrent kernels from different streams
- `hipStreamNonBlocking` avoids implicit synchronization with the default stream
- Profile with `rocprof` or PyTorch profiler to verify actual stream overlap

---

## HIP Graphs (Alternative to Streams)

For repeated patterns of kernels + dependencies, HIP graphs capture the workflow once and replay it, reducing launch overhead.

```cpp
hipGraph_t graph;
hipGraphExec_t graphExec;

hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
// ... enqueue kernels, memcpys ...
hipStreamEndCapture(stream, &graph);
hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// Replay many times
hipGraphLaunch(graphExec, stream);
```

SGLang uses CUDA/HIP graphs (cuda graph) by default for decode. Profile mode runs both with and without cuda graph (`--disable-cuda-graph`) to get layer-level structure from the no-graph trace.

---

## Quick API Reference

| Function | Purpose |
|----------|---------|
| `hipStreamCreate()` | Create default stream |
| `hipStreamCreateWithFlags()` | Create stream with flags (non-blocking, etc.) |
| `hipStreamCreateWithPriority()` | Create stream with priority |
| `hipStreamSynchronize()` | Block host until stream completes |
| `hipStreamWaitEvent()` | Make stream wait for an event |
| `hipStreamDestroy()` | Destroy stream |
| `hipEventCreate()` | Create event |
| `hipEventRecord()` | Record event on a stream |
| `hipEventSynchronize()` | Block host until event completes |
| `hipEventElapsedTime()` | Measure time between two events |
| `hipDeviceSynchronize()` | Block host until all device work completes |
| `hipMemcpyAsync()` | Async memcpy on a stream |
