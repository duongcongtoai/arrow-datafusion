use std::sync::Arc;

use crate::execution_plan::{Boundedness, EmissionType};
use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion_common::Result;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::Partitioning;

#[derive(Debug)]
pub struct RadixHashJoinExec {
    pub left: Arc<dyn ExecutionPlan>,
    pub right: Arc<dyn ExecutionPlan>,
    cache: Arc<PlanProperties>,
}

impl RadixHashJoinExec {
    pub fn new(left: Arc<dyn ExecutionPlan>, right: Arc<dyn ExecutionPlan>) -> Self {
        // Dummy PlanProperties for the mock
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(left.schema().clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self { left, right, cache }
    }
}

impl DisplayAs for RadixHashJoinExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "RadixHashJoinExec")
            }
            DisplayFormatType::TreeRender => {
                write!(f, "RadixHashJoinExec")
            }
        }
    }
}

impl ExecutionPlan for RadixHashJoinExec {
    fn name(&self) -> &str {
        "RadixHashJoinExec"
    }

    fn schema(&self) -> arrow::datatypes::SchemaRef {
        self.left.schema()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(RadixHashJoinExec::new(
            children[0].clone(),
            children[1].clone(),
        )))
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Err(datafusion_common::DataFusionError::NotImplemented(
            "RadixHashJoinExec::execute is a mock".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::empty::EmptyExec;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_radix_hash_join_exec_creation() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let left = Arc::new(EmptyExec::new(schema.clone()));
        let right = Arc::new(EmptyExec::new(schema.clone()));

        let join = RadixHashJoinExec::new(left.clone(), right.clone());

        assert_eq!(join.children().len(), 2);
    }
}
