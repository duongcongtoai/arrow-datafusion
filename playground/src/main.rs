#![allow(unused_imports)]

use std::env;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::UInt32Array;
use futures::StreamExt;
use tokio::time::sleep;

use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionState;
use datafusion::logical_expr::BuiltinScalarFunction::Coalesce;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::optimize_children;
use datafusion::physical_optimizer::join_selection::JoinSelection;
use datafusion::physical_plan;
use datafusion::physical_plan::{ExecutionPlan, Partitioning, PhysicalExpr};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::common::collect;
use datafusion::physical_plan::file_format::CsvExec;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::prelude::{CsvReadOptions, JoinType, SessionContext};
use datafusion::test_util::UnboundedExec;
use datafusion_common::from_slice::FromSlice;
use datafusion_common::Result;
use datafusion_physical_expr::expressions::Column;

use crate::test::optimize_test;

mod test;
mod view_test;

#[tokio::main]
pub async fn main() -> Result<()> {
    let sql = std::env::args().nth(1).expect("no sql given");
    let ctx = SessionContext::new();
    let (original, optimized) = test::optimize_test(&ctx, &sql).await?;
    println!("optimized: {optimized:?}");
    println!("original: {original:?}");
    Ok(())
}


pub async fn new_plan_with_partition(ctx: &SessionContext, tblname: &str, partition: usize) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let csv_opt = CsvReadOptions::new().schema(schema.as_ref());
    ctx.register_csv(tblname, "/home/toai/proj/rust/arrow-datafusion/playground/students.csv", csv_opt).await.unwrap();
    let sql = format!(r#"
    select * from {tblname}
    "#);
    let plan = ctx.state().create_logical_plan(&sql).await?;
    let exeplan = ctx.state().create_physical_plan(&plan).await?;
    let on: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(Column::new("id", 0))];
    let paritioned_exe = RepartitionExec::try_new(exeplan, Partitioning::Hash(on, partition))?;
    Ok(Arc::new(paritioned_exe))
}








