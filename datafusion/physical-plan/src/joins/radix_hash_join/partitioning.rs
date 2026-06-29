use crate::joins::radix_hash_join::sys_info::HardwareConfig;
use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use datafusion_common::hash_utils::{RandomState, create_hashes};

pub struct RadixPartitioner {
    pub passes: usize,
    pub radix_bits: usize,
}

impl RadixPartitioner {
    pub fn new(_config: &HardwareConfig) -> Self {
        // Calculate required bits and passes based on L2 cache and TLB entries
        // A common heuristic: we want the number of partitions (2^radix_bits) to be smaller than TLB_entries.
        // For simplicity and matching the task brief placeholder:
        Self {
            passes: 2, // Derived from config in a full implementation (e.g. log2(config.tlb_entries))
            radix_bits: 4, // Use 4 bits to limit the number of partitions in 2 passes to 256
        }
    }

    pub fn partition_batches(&self, batches: &[RecordBatch]) -> Result<Vec<RecordBatch>> {
        if batches.is_empty() {
            return Ok(vec![]);
        }

        let schema = batches[0].schema();
        let mut partitions = vec![batches.to_vec()];
        let random_state = RandomState::with_seed(0);

        for pass in 0..self.passes {
            let shift = pass * self.radix_bits;
            let num_sub_partitions = 1 << self.radix_bits;

            let mut next_partitions =
                Vec::with_capacity(partitions.len() * num_sub_partitions);

            for part_batches in partitions {
                if part_batches.is_empty() {
                    for _ in 0..num_sub_partitions {
                        next_partitions
                            .push(vec![RecordBatch::new_empty(schema.clone())]);
                    }
                    continue;
                }

                // 1. Histogram (Count) Phase: Compute local counts per thread
                let mut all_hashes = Vec::with_capacity(part_batches.len());
                for batch in &part_batches {
                    let mut hashes = vec![0; batch.num_rows()];
                    let arrays = batch.columns().to_vec();
                    create_hashes(&arrays, &random_state, &mut hashes)?;
                    all_hashes.push(hashes);
                }

                let mut histograms = vec![0; num_sub_partitions];
                for hashes in &all_hashes {
                    for &h in hashes {
                        let p = ((h >> shift) & ((1 << self.radix_bits) - 1)) as usize;
                        histograms[p] += 1;
                    }
                }

                // 2. Prefix sum: compute global offsets
                let mut offsets = vec![0; num_sub_partitions + 1];
                for i in 0..num_sub_partitions {
                    offsets[i + 1] = offsets[i] + histograms[i];
                }

                // 3. Scatter Phase: Write tuples to output buffers aligned to cache_line_size
                // Note: Arrow's interleave does not natively align per-tuple to hardware cache lines.
                // In a specialized implementation, we would scatter fixed-width rows into
                // cache-aligned buffers. Here we use columnar interleave to build the final arrays.
                let mut partition_row_indices = vec![Vec::new(); num_sub_partitions];
                for (i, p_indices) in partition_row_indices.iter_mut().enumerate() {
                    p_indices.reserve(histograms[i]);
                }

                for (batch_idx, hashes) in all_hashes.iter().enumerate() {
                    for (row_idx, &h) in hashes.iter().enumerate() {
                        let p = ((h >> shift) & ((1 << self.radix_bits) - 1)) as usize;
                        partition_row_indices[p].push((batch_idx, row_idx));
                    }
                }

                let num_columns = schema.fields().len();

                for p in 0..num_sub_partitions {
                    let indices = &partition_row_indices[p];
                    if indices.is_empty() {
                        next_partitions
                            .push(vec![RecordBatch::new_empty(schema.clone())]);
                        continue;
                    }

                    let mut partitioned_columns = Vec::with_capacity(num_columns);
                    for col_idx in 0..num_columns {
                        let arrays: Vec<&dyn arrow::array::Array> = part_batches
                            .iter()
                            .map(|b| b.column(col_idx).as_ref())
                            .collect();

                        let interleaved = arrow::compute::interleave(&arrays, indices)?;
                        partitioned_columns.push(interleaved);
                    }

                    let partition_batch =
                        RecordBatch::try_new(schema.clone(), partitioned_columns)?;
                    next_partitions.push(vec![partition_batch]);
                }
            }

            partitions = next_partitions;
        }

        // Flatten the final partitions into a single Vec<RecordBatch>
        let result = partitions.into_iter().flatten().collect();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joins::radix_hash_join::sys_info::HardwareConfig;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_radix_partitioner() {
        let config = HardwareConfig {
            l2_cache_size: 256 * 1024,
            tlb_entries: 64,
            cache_line_size: 64,
            num_cpus: 4,
        };

        let partitioner = RadixPartitioner::new(&config);

        // We set radix_bits to 4 and passes to 2 in the placeholder implementation.
        assert_eq!(partitioner.passes, 2);
        assert_eq!(partitioner.radix_bits, 4);

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        let array = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();

        let batches = vec![batch];
        let partitioned = partitioner.partition_batches(&batches).unwrap();

        // Total partitions expected: (1 << 4) ^ 2 = 256
        assert_eq!(partitioned.len(), 256);

        let total_rows: usize = partitioned.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 8);
    }
}
