# Error Handling and GPU Fallback

This document describes the error handling and graceful GPU fallback mechanisms in the PCR library.

## Overview

The PCR library provides robust error handling with:
- **GPU capability detection** at runtime
- **Graceful fallback** from GPU to CPU when GPU operations fail
- **Configurable error handling** modes (strict, fallback, auto)
- **Detailed error messages** with context and suggestions
- **Device information logging** for debugging

## GPU Capability Detection

### Runtime Detection Functions

The library provides several utility functions to detect GPU capabilities:

```cpp
#include "pcr/core/types.h"

// Check if CUDA is compiled into the library
bool is_compiled = cuda_is_compiled();

// Check if a CUDA device is available
bool has_gpu = cuda_device_available();

// Get number of CUDA devices
int num_devices = cuda_device_count();

// Get GPU device name
std::string name = cuda_device_name(0);  // Device 0

// Get GPU memory information
size_t free_bytes, total_bytes;
bool success = cuda_get_memory_info(&free_bytes, &total_bytes, 0);
if (success) {
    std::cout << "Free: " << (free_bytes / 1024 / 1024) << " MB\n";
    std::cout << "Total: " << (total_bytes / 1024 / 1024) << " MB\n";
}
```

### Example Detection Code

```cpp
if (!cuda_is_compiled()) {
    std::cout << "CUDA not compiled - using CPU mode\n";
} else if (!cuda_device_available()) {
    std::cout << "No GPU detected - using CPU mode\n";
} else {
    int count = cuda_device_count();
    std::cout << "Found " << count << " GPU(s)\n";
    for (int i = 0; i < count; ++i) {
        std::cout << "  GPU " << i << ": " << cuda_device_name(i) << "\n";
    }
}
```

## Execution Modes

### ExecutionMode Enum

```cpp
enum class ExecutionMode {
    CPU,   // Force CPU execution
    GPU,   // Request GPU execution (may fall back based on config)
    Auto   // Automatically choose based on GPU availability and data location
};
```

### Configuration Options

```cpp
PipelineConfig config;

// Execution mode
config.exec_mode = ExecutionMode::GPU;

// Fallback behavior
config.gpu_fallback_to_cpu = true;   // If true, fall back to CPU on GPU errors
config.gpu_require_strict = false;   // If true, fail instead of falling back

// GPU settings
config.cuda_device_id = 0;           // Which GPU to use
config.gpu_pool_size_bytes = 512 * 1024 * 1024;  // 512MB memory pool
```

## Fallback Behavior

### Fallback Configuration Matrix

| exec_mode | gpu_fallback_to_cpu | gpu_require_strict | Behavior |
|-----------|---------------------|-------------------|----------|
| CPU       | (ignored)           | (ignored)         | Always use CPU |
| GPU       | true                | false             | Try GPU, fall back to CPU on error (default) |
| GPU       | false               | false             | Try GPU, fail on error |
| GPU       | (any)               | true              | GPU required, fail if unavailable |
| Auto      | (ignored)           | (ignored)         | Use GPU if available, else CPU |

### Default Configuration

```cpp
PipelineConfig config;
config.exec_mode = ExecutionMode::Auto;      // Smart selection
config.gpu_fallback_to_cpu = true;           // Graceful fallback
config.gpu_require_strict = false;           // Allow fallback
```

This default configuration provides maximum compatibility - it will use GPU if available but seamlessly fall back to CPU if needed.

## Error Scenarios and Handling

### 1. No GPU Available

**Scenario:** CUDA compiled but no GPU detected

**Fallback Mode (default):**
```
Warning: No CUDA-capable GPU detected - falling back to CPU mode
Pipeline continues with CPU execution
```

**Strict Mode:**
```
Error: No CUDA-capable GPU detected - GPU mode requested but no GPU available
Pipeline creation fails, returns nullptr
```

### 2. Invalid Device ID

**Scenario:** Requested GPU device doesn't exist

**Fallback Mode:**
```
Warning: Failed to set CUDA device 999: invalid device ordinal - falling back to CPU mode
Pipeline continues with CPU execution
```

**Strict Mode:**
```
Error: Failed to set CUDA device 999: invalid device ordinal
Pipeline creation fails
```

### 3. GPU Out of Memory

**Scenario:** Not enough GPU memory to allocate resources

**Enhanced Error Message:**
```
Error: Failed to create GPU memory pool
GPU has 128 MB free / 6144 MB total
Try reducing batch size or using CPU mode
```

**Fallback Mode:** Falls back to CPU automatically

### 4. Host→Device Transfer Failure

**Scenario:** Cannot transfer point cloud to GPU

**Enhanced Error Message:**
```
Error: Failed to transfer 10000000 points to GPU device
GPU has 50 MB free / 6144 MB total
Try reducing batch size or using CPU mode
```

Provides actionable information about memory constraints.

## Logging and Diagnostics

### GPU Initialization Logging

When GPU mode is successfully initialized:

```
Info: Using GPU 0: NVIDIA GeForce RTX 2060 (5.0 GB free / 6.0 GB total)
```

### Auto Mode Selection

When Auto mode selects CPU:

```
Info: No CUDA-capable GPU detected - using CPU mode
```

### Fallback Warnings

When falling back from GPU to CPU:

```
Warning: Failed to create GPU memory pool - falling back to CPU mode
```

## Best Practices

### 1. Use Auto Mode for Maximum Compatibility

```cpp
config.exec_mode = ExecutionMode::Auto;
```

This automatically handles GPU detection and falls back gracefully.

### 2. Enable Fallback for Production

```cpp
config.gpu_fallback_to_cpu = true;  // Default
config.gpu_require_strict = false;  // Default
```

This ensures your application works even if GPU becomes unavailable.

### 3. Use Strict Mode for Debugging

```cpp
config.gpu_require_strict = true;
```

When debugging GPU-specific issues, strict mode helps identify problems early.

### 4. Check GPU Availability Before Large Jobs

```cpp
if (cuda_device_available()) {
    size_t free_mem, total_mem;
    cuda_get_memory_info(&free_mem, &total_mem);

    // Ensure we have enough memory
    size_t required = estimated_memory_usage();
    if (free_mem < required) {
        std::cout << "Warning: GPU may run out of memory\n";
        config.exec_mode = ExecutionMode::CPU;
    } else {
        config.exec_mode = ExecutionMode::GPU;
    }
} else {
    config.exec_mode = ExecutionMode::CPU;
}
```

### 5. Handle Pipeline Creation Failures

```cpp
auto pipeline = Pipeline::create(config);
if (!pipeline) {
    std::cerr << "Failed to create pipeline\n";
    // Handle error - maybe try with CPU mode
    config.exec_mode = ExecutionMode::CPU;
    pipeline = Pipeline::create(config);
}
```

### 6. Check Status Returns

```cpp
Status s = pipeline->ingest(*cloud);
if (!s.ok()) {
    std::cerr << "Error: " << s.message << "\n";

    // Check error code for specific handling
    if (s.code == StatusCode::CudaError) {
        // GPU error - maybe switch to CPU for next batch
    }
}
```

## Error Status Codes

### Common Status Codes

```cpp
enum class StatusCode : uint8_t {
    Success,
    InvalidArgument,   // Bad configuration or input
    OutOfMemory,       // Memory allocation failed
    CudaError,         // CUDA operation failed
    IoError,           // File I/O error
    NotImplemented,    // Feature not available
    // ... others
};
```

### Checking Status

```cpp
Status s = some_operation();

// Check if successful
if (s.ok()) {
    // Success
}

// Check specific error
if (s.code == StatusCode::CudaError) {
    std::cerr << "GPU error: " << s.message << "\n";
}

// Get error message
std::cerr << "Error: " << s.message << "\n";
```

## Testing Error Handling

### Running Error Handling Tests

```bash
./test_error_handling
```

This test suite verifies:
- ✓ CUDA compilation detection
- ✓ GPU device detection
- ✓ Device name and memory queries
- ✓ Fallback to CPU when no GPU
- ✓ Strict mode behavior
- ✓ Auto mode selection
- ✓ CPU mode always works
- ✓ Clear error messages for common mistakes
- ✓ Invalid device ID handling
- ✓ Configuration validation

### Test Coverage

**12 tests passing:**
- CUDA capability detection (4 tests)
- GPU fallback behavior (4 tests)
- Error message quality (3 tests)
- Configuration validation (2 tests)

## Examples

### Example 1: Safe Production Configuration

```cpp
PipelineConfig config;
config.exec_mode = ExecutionMode::Auto;      // Use GPU if available
config.gpu_fallback_to_cpu = true;           // Fall back on error
config.state_dir = "/data/tiles";

// This will work on any system
auto pipeline = Pipeline::create(config);
```

### Example 2: GPU Required for Performance

```cpp
PipelineConfig config;
config.exec_mode = ExecutionMode::GPU;
config.gpu_require_strict = true;            // Fail if no GPU

auto pipeline = Pipeline::create(config);
if (!pipeline) {
    std::cerr << "GPU not available - cannot process\n";
    return -1;
}
```

### Example 3: Adaptive Execution

```cpp
// Check GPU capability first
PipelineConfig config;

if (cuda_device_available()) {
    size_t free_mem, total_mem;
    cuda_get_memory_info(&free_mem, &total_mem);

    if (free_mem > 1024 * 1024 * 1024) {  // 1GB free
        config.exec_mode = ExecutionMode::GPU;
        std::cout << "Using GPU acceleration\n";
    } else {
        config.exec_mode = ExecutionMode::CPU;
        std::cout << "Insufficient GPU memory, using CPU\n";
    }
} else {
    config.exec_mode = ExecutionMode::CPU;
    std::cout << "No GPU detected, using CPU\n";
}

auto pipeline = Pipeline::create(config);
```

## Troubleshooting

### Issue: Pipeline always uses CPU even with GPU available

**Check:**
1. Is CUDA compiled? `cuda_is_compiled()`
2. Is GPU detected? `cuda_device_available()`
3. Is cloud on Host memory? (Auto mode only uses GPU for Device clouds)
4. Check exec_mode setting

**Solution:**
```cpp
config.exec_mode = ExecutionMode::GPU;  // Force GPU
```

### Issue: "No CUDA-capable GPU detected" error

**Check:**
1. Is a GPU installed? Run `nvidia-smi`
2. Is CUDA driver installed?
3. Is the GPU visible to CUDA? Run `nvidia-smi` or `deviceQuery`

**Solution:**
```cpp
config.exec_mode = ExecutionMode::CPU;  // Use CPU instead
// Or enable fallback
config.gpu_fallback_to_cpu = true;
```

### Issue: GPU out of memory errors

**Check:**
```cpp
size_t free_mem, total_mem;
cuda_get_memory_info(&free_mem, &total_mem);
std::cout << "Free GPU memory: " << (free_mem / 1024 / 1024) << " MB\n";
```

**Solutions:**
1. Reduce batch size (process fewer points per ingest)
2. Reduce GPU pool size: `config.gpu_pool_size_bytes = 256 * 1024 * 1024;`
3. Use CPU mode for this dataset
4. Close other GPU applications

### Issue: Inconsistent results between GPU and CPU

This should not happen - please file a bug report if you see this.

**Workaround:**
```cpp
config.exec_mode = ExecutionMode::CPU;  // Use CPU for consistency
```

## Summary

The PCR library provides comprehensive error handling with:

✓ **Automatic GPU detection** at runtime
✓ **Graceful fallback** from GPU to CPU
✓ **Configurable strictness** (fail vs fallback)
✓ **Detailed error messages** with context
✓ **Device information logging** for debugging
✓ **Comprehensive test coverage** (12 tests)

Default configuration is safe for all environments - it will use GPU if available but fall back to CPU automatically if needed.

---

**See Also:**
- `docs/GPU_AND_THREADING.md` - GPU acceleration guide
- `tests/cpp/test_error_handling.cpp` - Error handling tests
- `include/pcr/core/types.h` - GPU detection utilities
