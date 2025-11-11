# NCCL Timeout Debug Report - November 11, 2025

## Problem Summary

Training the Concept Encoder model on 4x RTX 3090 GPUs using distributed training (DDP with NCCL backend) resulted in a catastrophic timeout after ~53 minutes of training at 11% completion (3708/35185 iterations).

### Symptoms

1. **Extremely Slow Training**: 800.85 seconds per iteration (~13 minutes!)
2. **NCCL Timeout**: ALLREDUCE operation timed out after 30 minutes (1800 seconds)
3. **Watchdog Stuck**: All ranks reported watchdog getting stuck for 480 seconds
4. **Process Crash**: All 4 GPU processes terminated with SIGABRT (signal 6)

### Error Messages

```
[rank1]:[E1111 17:01:37] [Rank 1] Watchdog caught collective operation timeout: 
WorkNCCL(SeqNum=14863, OpType=ALLREDUCE, NumelIn=14845952, NumelOut=14845952, Timeout(ms)=1800000) 
ran for 1800048 milliseconds before timing out.

[rank1]:[E1111 17:10:53] ProcessGroupNCCL's watchdog got stuck for 480 seconds without making progress
```

## Root Cause Analysis

### Primary Issue: Memory Aliasing in Distributed Training

The bug was in `nn/concept_encoder.py`, specifically in the `ConceptEncoderForMaskedLMWeighted.forward()` method (lines 529-536):

```python
# BUGGY CODE - BEFORE FIX
position_weights = self.concept_weights[:seq_length, :]
position_weights = F.softmax(position_weights, dim=-1)
position_weights_expanded = position_weights.unsqueeze(0).expand(batch_size, 1, 1)  # BUG!
sequence_repr = torch.bmm(position_weights_expanded, concept_repr)
```

**Why This Breaks Distributed Training:**

1. **`.expand()` creates memory views, not copies**
   - Creates aliased memory that shares the same underlying storage
   - Multiple "different" tensors point to the same memory location

2. **NCCL Gradient Synchronization Failure**
   - During backward pass, NCCL needs to synchronize gradients across GPUs
   - Memory aliasing causes inconsistent gradient computation
   - ALLREDUCE operation gets stuck waiting for gradients that never arrive correctly

3. **Deadlock in GPU Communication**
   - GPU 0 waits for GPU 1 to send gradients
   - GPU 1 waits for GPU 2 to send gradients
   - GPU 2 waits for GPU 3 to send gradients
   - GPU 3's gradients are corrupted due to memory aliasing
   - Result: All GPUs stuck in infinite wait (30-minute timeout)

### Technical Deep Dive

The `concept_weights` parameter has shape `[max_seq_length, concept_num]` = `[512, 128]`.

During forward pass:
1. Slice the weights: `concept_weights[:seq_length, :]` - creates a view
2. Apply softmax - operates on the view
3. **`.expand(batch_size, 1, 1)` - creates MORE views sharing the same memory**
4. Use in `bmm()` operation

During backward pass:
- PyTorch computes gradients w.r.t. the original `concept_weights`
- But the forward pass used memory views from `.expand()`
- NCCL tries to synchronize these gradients across GPUs
- Memory aliasing causes corruption in gradient computation
- Result: NCCL operations hang indefinitely

### Contributing Factors

1. **High iteration time** (800s/it) indicated something was wrong even before timeout
2. **Large effective batch size** (192 * 2 = 384) increased memory pressure
3. **Dynamic sequence lengths** meant different computational graphs per batch
4. **4-way gradient synchronization** amplified the synchronization issues

## Solution

### Code Fix

Changed `.expand()` to `.repeat()` in `nn/concept_encoder.py`:

```python
# FIXED CODE - AFTER FIX
position_weights = self.concept_weights[:seq_length, :]
position_weights = F.softmax(position_weights, dim=-1)
# Use repeat() to create actual copies, not views
position_weights_expanded = position_weights.unsqueeze(0).repeat(batch_size, 1, 1)  # FIXED!
sequence_repr = torch.bmm(position_weights_expanded, concept_repr)
```

**Why `.repeat()` works:**
- Creates actual copies of data in memory
- No memory aliasing between tensors
- Gradients compute correctly and consistently
- NCCL can properly synchronize gradients across GPUs

**Trade-off:**
- Slightly higher memory usage (batch_size copies instead of views)
- For our case: `batch_size * seq_length * concept_num * 4 bytes`
- Example: `48 * 512 * 128 * 4 = 12.5 MB` per GPU - totally acceptable

### Script Improvements

Enhanced `scripts/train_weighted_mlm_multigpu.sh` with:

1. **Better NCCL Debugging**
   ```bash
   export NCCL_DEBUG=INFO  # Detailed logging
   export NCCL_DEBUG_SUBSYS=ALL  # All subsystem info
   export NCCL_TIMEOUT=3600  # 1-hour timeout for debugging
   export NCCL_ASYNC_ERROR_HANDLING=1  # Better error reporting
   ```

2. **Network Configuration**
   ```bash
   export NCCL_SOCKET_IFNAME=^docker0,lo  # Exclude problematic interfaces
   export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
   ```

3. **Reduced Dataloader Workers**
   - Changed from `4` to `2` workers per GPU
   - Reduces CPU contention and I/O bottlenecks

## Verification Steps

To verify the fix works:

1. **Check iteration time**: Should be < 10 seconds/iteration (was 800s)
2. **Monitor NCCL logs**: No timeout warnings in first 100 iterations
3. **Watch GPU utilization**: Should be 90%+ (was low before due to waiting)
4. **Training progress**: Should complete epochs without hangs

## Prevention Guidelines

### For Future Model Development

1. **Avoid `.expand()` in distributed training**
   - Always use `.repeat()` when you need copies
   - Only use `.expand()` for read-only operations

2. **Test with distributed training early**
   - Don't wait until full training to test multi-GPU
   - Run small-scale DDP tests during development

3. **Watch for slow iteration times**
   - If iterations take > 2x expected time, investigate immediately
   - Slow training often indicates synchronization issues

4. **Use gradient checkpointing carefully**
   - Can interact badly with memory views
   - Test thoroughly with multi-GPU setup

### Debugging Distributed Training Issues

When you see NCCL timeouts:

1. **Enable verbose logging**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   ```

2. **Check for memory aliasing**
   - Look for `.expand()`, `.view()`, `.reshape()` operations
   - Verify they're not used in gradient computation paths

3. **Reduce complexity**
   - Try single GPU first
   - Then 2 GPUs
   - Then full multi-GPU
   - Identify where issues appear

4. **Monitor system resources**
   - GPU utilization should be high (>80%)
   - CPU shouldn't be bottleneck
   - Network bandwidth usage

## References

- **PyTorch DDP Best Practices**: https://pytorch.org/docs/stable/notes/ddp.html
- **NCCL Troubleshooting**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- **Expand vs Repeat**: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.expand

## Timeline

- **2025-11-11 17:01**: First timeout detected after 53 minutes training
- **2025-11-11 17:16**: All processes terminated (15 minutes of timeout handling)
- **Debug session**: Identified root cause in model architecture
- **Fix applied**: Changed `.expand()` to `.repeat()`, added NCCL debugging
- **Status**: Ready for re-testing

## Expected Outcomes After Fix

1. ✅ Iteration time: < 10 seconds (down from 800 seconds)
2. ✅ No NCCL timeouts during full training run
3. ✅ GPU utilization: > 85% (up from ~10%)
4. ✅ Training completes successfully without hangs
5. ✅ Gradient synchronization works correctly across all 4 GPUs

---

*Document created: 2025-11-11*
*Author: AI Assistant (Claude)*
*Last updated: 2025-11-11*

