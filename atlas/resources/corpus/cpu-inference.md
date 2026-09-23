# CPU inference tuning: the thread count that costs you four times the throughput

Running a small quantised language model on a CPU without a GPU is entirely
viable, but the default configuration is almost always wrong.

## The misconception

The intuitive setting is to use every core. llama.cpp and llama-cpp-python
default to cpu_count minus one worker threads. On a sixteen-core machine that
means fifteen threads for a model with a few hundred million parameters.

Measured throughput with the default setting is roughly four times *lower* than
with four threads.

## Why more threads is slower

Decoding a small quantised model is memory-bandwidth bound, not compute bound.
Each token requires streaming the entire weight matrix, and the weights do not
fit in L2 or L3 cache. With four threads the cache lines being read are shared
coherently and the prefetcher keeps up. With fifteen threads the same cache
lines are requested by many cores at once and the interconnect becomes the
bottleneck. The cores spend their time in cache-coherence stalls, not in
arithmetic. Adding threads adds contention, not work.

The practical rule: on CPUs, the optimum is a small constant, typically two to
four threads, regardless of core count. Measure it once on your hardware and
then pin it.

## Configuration

Expose the thread count as an explicit setting rather than deriving it from the
core count. This repository exposes ATLAS_CPU_THREADS, defaulting to four, and
clamps any explicit value to a positive integer. The resolved value is reported
by the health and statistics endpoints so that a deployment can be audited.

## Measurement protocol

To reproduce the measurement:

1. Fix the prompt and the maximum token count so the comparison is fair.
2. Run each thread count at least three times and report the median, because
   the variance across runs on a shared desktop is large.
3. Report tokens per second, not wall-clock time for the whole process, so
   model loading is excluded.
4. Repeat with a second model size. If the optimum moves sharply with model
   size, re-measure rather than extrapolating.

## Related guidance

Keep the context window as small as the task allows. Context memory grows
linearly with the window and competes with the weights for bandwidth. Prefer
quantised weights that fit in memory with headroom; swapping destroys
throughput far more thoroughly than any thread setting.
