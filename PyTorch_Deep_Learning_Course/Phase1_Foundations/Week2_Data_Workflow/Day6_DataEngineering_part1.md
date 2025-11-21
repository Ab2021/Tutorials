# Day 6: Data Engineering - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Multiprocessing Internals, Shared Memory, and WebDataset

## 1. How `num_workers` really works

When `num_workers > 0`:
1.  Main process spawns N worker processes.
2.  Each worker imports the script and re-initializes the `Dataset` object.
3.  Workers fetch data and put it into a `multiprocessing.Queue`.
4.  Main process reads from the Queue and puts tensors onto GPU.

**The Copy-on-Write (CoW) Trap**:
If your dataset loads a huge array in `__init__` (e.g., 50GB RAM), each worker might try to copy it.
Linux CoW usually prevents this, but if you modify the array (even refcounts), memory usage explodes ($50GB \times 4 = 200GB$).
**Fix**: Use `mmap` or load data in `__getitem__`.

## 2. Shared Memory

PyTorch tensors are special. When sent through a Queue, they are moved to **Shared Memory** (`/dev/shm`).
This avoids pickling/unpickling overhead.
**Error**: `RuntimeError: DataLoader worker (pid) is killed by signal: Bus error`.
**Cause**: Shared memory limit reached (Docker default is 64MB).
**Fix**: Increase `--shm-size` in Docker.

## 3. WebDataset (Sharding)

For ImageNet (1.2M images) or LAION (5B images), reading millions of small files is slow (inode lookup).
**WebDataset** stores data in TAR archives (shards).
*   `shard-0001.tar`: Contains 1000 images.
*   Sequential read speed = Disk bandwidth (500MB/s).

```python
import webdataset as wds

url = "http://storage.googleapis.com/imagenet/train-{0000..9999}.tar"
dataset = (
    wds.WebDataset(url)
    .shuffle(1000)
    .decode("pil")
    .to_tuple("jpg", "cls")
)
```

## 4. FFCV (Fast Forward Computer Vision)

A library optimized for training throughput.
*   Stores data in a custom `.beton` format (page-aligned).
*   Performs augmentation on GPU asynchronously.
*   Can saturate A100 GPUs (5000+ images/sec).

## 5. Persistent Workers

`persistent_workers=True` keeps workers alive between epochs.
Without this, workers are killed and re-spawned every epoch, causing a massive CPU spike and delay at the start of each epoch.
