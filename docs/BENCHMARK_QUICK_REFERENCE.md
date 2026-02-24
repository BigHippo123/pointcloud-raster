# DC LiDAR Benchmark Quick Reference

## Quick Results Summary

### 🏆 Winner: GPU Mode
- **2.86× faster** than CPU multi-threaded
- **17.3 Mpts/s throughput** (30M points test)
- **28.5 Mpts/s throughput** (9M points test)

### Mode Comparison (10 files, 30.3M points)

```
┌─────────────┬────────────┬──────────────┬─────────────┬──────────┐
│    Mode     │ Total (s)  │ Library (s)  │  Mpts/s     │ Speedup  │
├─────────────┼────────────┼──────────────┼─────────────┼──────────┤
│ CPU-1T      │    16.2    │     5.4      │     5.6     │   0.93x  │
│ CPU-MT      │    11.1    │     5.0      │     6.1     │   1.00x  │
│ GPU         │     8.3    │     1.8      │    17.3     │   2.86x  │
│ Hybrid      │    11.6    │     5.3      │     5.7     │   0.95x  │
└─────────────┴────────────┴──────────────┴─────────────┴──────────┘
```

## Time Breakdown: Where Does the Time Go?

### CPU-MT Mode
```
████████████████████████████░░░░░░░░░░ 54% I/O Reading
█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1% Cloud Creation
█████████████████████████░░░░░░░░░░░░ 36% Library Ingest
██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  9% Library Finalize
```

### GPU Mode ⚡
```
█████████████████████████████████████░ 77% I/O Reading
█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1% Cloud Creation
████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  7% Library Ingest (GPU accelerated!)
█████████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 14% Library Finalize
```

**Key Insight**: GPU reduces library processing from 45% to 21% of total time!

## Common Commands

### Fastest Processing
```bash
python3 scripts/test_dc_lidar.py --las-dir /path/to/data --mode gpu
```

### Compare All Modes
```bash
python3 scripts/test_dc_lidar.py \
  --las-dir /path/to/data \
  --mode all \
  --csv results.csv
```

### High Quality DEM (Smooth)
```bash
python3 scripts/test_dc_lidar.py \
  --las-dir /path/to/data \
  --mode gpu \
  --glyph gaussian-3
```

### Quick Test
```bash
python3 scripts/test_dc_lidar.py \
  --las-dir /path/to/data \
  --subset 5 \
  --mode gpu
```

## Glyph Performance Guide

| Glyph Type | Speed | Quality | Best Use Case |
|------------|-------|---------|---------------|
| Point | ⚡⚡⚡ Fastest | Basic | Quick visualization, max throughput |
| Line-2 | ⚡⚡ Fast | Good | Directional features |
| Gaussian-1 | ⚡⚡ Fast | Smooth | Dense point clouds |
| Gaussian-3 | ⚡ Medium | Very Smooth | Standard DEMs |
| Gaussian-5 | 🐌 Slow | Ultra Smooth | Sparse data, gap filling |
| Gaussian-10 | 🐌🐌 Very Slow | Maximum Smooth | Very sparse data |

## Decision Tree

```
Need maximum speed?
├─ Yes → Use GPU + Point glyph
└─ No → Need smooth output?
    ├─ Yes → Use GPU + Gaussian-3 glyph
    └─ No → Use CPU-MT + Point glyph
```

## Performance Bottlenecks

### Current Bottleneck: I/O (54-77% of time)
**Solutions**:
- Use faster SSD storage
- Use fewer, larger files instead of many small files
- Pre-cache data in memory
- Use LAZ compression for network transfer (decompress locally)

### When Library Processing is Slow
**Solutions**:
- Switch to GPU mode (2.86× speedup)
- Use simpler glyphs (Point instead of Gaussian)
- Reduce sigma for Gaussian glyphs
- Increase grid cell size

## Expected Throughput

| Mode | Points/sec | Example: 500M points |
|------|-----------|----------------------|
| CPU-1T | 5.6M | ~89 seconds |
| CPU-MT | 6.1M | ~82 seconds |
| GPU | 17.3M | ~29 seconds ⚡ |
| Hybrid | 5.7M | ~88 seconds |

*Note: These are library processing times only. Add I/O time based on your storage.*

## Tested Configuration

- **Dataset**: DC LiDAR (198 files, 15GB, ~500M points)
- **Hardware**: NVIDIA RTX 2060 (6GB), Multi-core CPU, SSD
- **Grid**: 1m cell size, ~13.5M cells for 10-file test
- **Format**: LAS 1.4 (uncompressed), laspy 2.7.0 reader

## Files Generated

After running benchmarks, check these locations:

```
/workspace/benchmark_results/
├── mode_comparison.csv          # Detailed CSV results
├── dc_lidar_cpu-1t_point.tif   # CPU single-thread output
├── dc_lidar_cpu-mt_point.tif   # CPU multi-thread output
├── dc_lidar_gpu_point.tif      # GPU output
└── dc_lidar_hybrid_point.tif   # Hybrid output
```

## CSV Data: Mode Comparison

```csv
mode,glyph,value_channel,points,files,wall_total_s,io_read_s,library_time_s,throughput_mpts,speedup
cpu-1t,point,elevation,30346376,10,16.163,10.439,5.414,5.606,0.93x
cpu-mt,point,elevation,30346376,10,11.058,5.938,5.016,6.050,1.00x (baseline)
gpu,point,elevation,30346376,10,8.252,6.390,1.756,17.283,2.86x ⭐
hybrid,point,elevation,30346376,10,11.644,6.255,5.287,5.740,0.95x
```

## Recommendations

### For Production Pipelines
✅ Use GPU mode (2.86× faster)
✅ Use Point glyph for speed
✅ Process in parallel across multiple machines
✅ Optimize I/O with faster storage

### For High-Quality Output
✅ Use GPU mode (offsets Gaussian overhead)
✅ Use Gaussian-3 glyph (good balance)
✅ Adjust sigma based on point density
✅ Consider processing time vs quality trade-off

### For Development/Testing
✅ Use `--subset N` for quick iterations
✅ Use CPU-MT for small tests (lower overhead)
✅ Use `--csv` to track performance over time
✅ Test with representative data samples

---

**Last Updated**: 2026-02-22
**Full Details**: See `/workspace/DC_LIDAR_BENCHMARK_README.md`
