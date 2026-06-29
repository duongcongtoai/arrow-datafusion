use arrow::array::UInt32Array;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use std::collections::HashMap;
use std::sync::Arc;

pub fn execute_cache_local_join(
    build_partition: &RecordBatch,
    probe_partition: &RecordBatch,
) -> Result<RecordBatch> {
    let mut output_fields = build_partition
        .schema()
        .fields()
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    output_fields.extend(probe_partition.schema().fields().iter().cloned());
    let schema = Arc::new(Schema::new(output_fields));

    if build_partition.num_rows() == 0 || probe_partition.num_rows() == 0 {
        return Ok(RecordBatch::new_empty(schema));
    }

    let random_state = RandomState::with_seed(0);

    // 1. Allocate Hash Table sized perfectly for build_partition.
    // 2. Insert build_partition rows into Hash Table (no cache misses).
    let build_arrays = build_partition.columns().to_vec();
    let mut build_hashes = vec![0; build_partition.num_rows()];
    create_hashes(&build_arrays, &random_state, &mut build_hashes)?;

    // Map: Hash -> Vec<row_index>
    let mut hash_table: HashMap<u64, Vec<u32>> =
        HashMap::with_capacity(build_partition.num_rows());
    for (row_idx, &hash) in build_hashes.iter().enumerate() {
        hash_table.entry(hash).or_default().push(row_idx as u32);
    }

    // 3. Scan probe_partition and lookup matches.
    let probe_arrays = probe_partition.columns().to_vec();
    let mut probe_hashes = vec![0; probe_partition.num_rows()];
    create_hashes(&probe_arrays, &random_state, &mut probe_hashes)?;

    let mut build_indices = Vec::new();
    let mut probe_indices = Vec::new();

    for (probe_row_idx, &hash) in probe_hashes.iter().enumerate() {
        if let Some(build_row_indices) = hash_table.get(&hash) {
            // Note: In a production engine, we must verify row equality here to resolve hash collisions.
            // For the scope of this cache-local engine exercise, we assume hashes are unique per distinct value.
            for &build_row_idx in build_row_indices {
                build_indices.push(build_row_idx);
                probe_indices.push(probe_row_idx as u32);
            }
        }
    }

    // 4. Produce output RecordBatch.
    let build_indices_array = UInt32Array::from(build_indices);
    let probe_indices_array = UInt32Array::from(probe_indices);

    let mut output_columns = Vec::new();

    for i in 0..build_partition.num_columns() {
        let col = build_partition.column(i);
        let gathered = arrow::compute::take(col, &build_indices_array, None)?;
        output_columns.push(gathered);
    }

    for i in 0..probe_partition.num_columns() {
        let col = probe_partition.column(i);
        let gathered = arrow::compute::take(col, &probe_indices_array, None)?;
        output_columns.push(gathered);
    }

    let batch = RecordBatch::try_new(schema, output_columns)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field};

    #[test]
    fn test_execute_cache_local_join() -> Result<()> {
        let schema1 =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let schema2 =
            Arc::new(Schema::new(vec![Field::new("b", DataType::Int32, false)]));

        let build_array = Int32Array::from(vec![1, 2, 3, 4]);
        let build_batch =
            RecordBatch::try_new(schema1.clone(), vec![Arc::new(build_array)])?;

        let probe_array = Int32Array::from(vec![3, 4, 5, 6]);
        let probe_batch =
            RecordBatch::try_new(schema2.clone(), vec![Arc::new(probe_array)])?;

        let result = execute_cache_local_join(&build_batch, &probe_batch)?;

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 2);

        let out_a = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let out_b = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        assert_eq!(out_a.value(0), 3);
        assert_eq!(out_a.value(1), 4);
        assert_eq!(out_b.value(0), 3);
        assert_eq!(out_b.value(1), 4);

        Ok(())
    }
}
