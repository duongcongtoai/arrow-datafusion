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
use datafusion::execution::options::ReadOptions;
use datafusion::logical_expr::BuiltinScalarFunction::Coalesce;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::optimize_children;
use datafusion::physical_optimizer::join_selection::JoinSelection;
use datafusion::physical_plan;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::common::collect;
use datafusion::physical_plan::file_format::CsvExec;
use datafusion::physical_plan::joins::utils::JoinFilter;
use datafusion::physical_plan::joins::{
    HashJoinExec, PartitionMode, SymmetricHashJoinExec,
};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning, PhysicalExpr};
use datafusion::prelude::{CsvReadOptions, JoinType, SessionConfig, SessionContext};
use datafusion::test_util::UnboundedExec;
use datafusion_common::from_slice::FromSlice;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::Result;
use datafusion_physical_expr::create_physical_expr;
use datafusion_physical_expr::expressions::Column;

use crate::test::optimize_test;

mod test;
mod view_test;

#[tokio::main]
pub async fn main() -> Result<()> {
    // let sql = std::env::args().nth(1).expect("no sql given");
    let session_config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::with_config(session_config);
    test_unbounded_symetric_hashjoin(&ctx).await?;
    Ok(())
}
pub async fn test_unbounded_symetric_hashjoin(ctx: &SessionContext) -> Result<()> {
    new_plan_with_partition(ctx, "first", 1, true).await?;
    new_plan_with_partition(ctx, "second", 1, true).await?;
    let sql = r#"
    select * from first inner join second on first.id = second.id and 
    first.value > second.value - 10 and first.value < second.value + 10
    "#;
    let plan = ctx.state().create_logical_plan(sql).await?;
    let phys = ctx.state().create_physical_plan(&plan).await?;
    let pretty = physical_plan::displayable(phys.as_ref())
        .indent()
        .to_string();
    println!("{pretty}");
    Ok(())
}

pub async fn new_plan_with_partition(
    ctx: &SessionContext,
    tblname: &str,
    partition: usize,
    unbounded: bool,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let csv_opt = CsvReadOptions::new()
        .schema(schema.as_ref())
        .mark_infinite(unbounded);
    register_csv_with_sort(
        ctx,
        tblname,
        "/home/toai/proj/rust/arrow-datafusion/playground/students.csv",
        csv_opt,
        "value",
    )
    .await
    .unwrap();
    let sql = format!(
        r#"
    select * from {tblname}
    "#
    );
    let plan = ctx.state().create_logical_plan(&sql).await?;
    let exeplan = ctx.state().create_physical_plan(&plan).await?;
    let on: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(Column::new("id", 0))];
    let paritioned_exe =
        RepartitionExec::try_new(exeplan, Partitioning::Hash(on, partition))?;
    Ok(Arc::new(paritioned_exe))
}

async fn register_csv_with_sort(
    ctx: &SessionContext,
    name: &str,
    table_path: &str,
    options: CsvReadOptions<'_>,
    sort_col: &str,
) -> Result<()> {
    let file_sort_order = [datafusion_expr::col(sort_col)]
        .into_iter()
        .map(|e| {
            let ascending = true;
            let nulls_first = false;
            e.sort(ascending, nulls_first)
        })
        .collect::<Vec<_>>();

    let listing_options = options
        .to_listing_options(&ctx.copied_config())
        .with_file_sort_order(Some(file_sort_order));
    ctx.register_listing_table(
        name,
        table_path,
        listing_options,
        options.schema.map(|s| Arc::new(s.to_owned())),
        None,
    )
    .await?;

    Ok(())
}
