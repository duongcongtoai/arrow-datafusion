#![allow(unused_imports)]

use std::io::Read;
use std::sync::Arc;
use std::time::Duration;
use std::{env, io};

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
use datafusion::optimizer::push_down_filter::PushDownFilter;
use datafusion::optimizer::push_down_limit::PushDownLimit;
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
use datafusion_expr::LogicalPlanBuilder;
use datafusion_physical_expr::create_physical_expr;
use datafusion_physical_expr::expressions::Column;

use crate::test::optimize_test;
use crate::util::{new_logical_plan_with_partition, new_plan_with_partition, runquery};
use crate::window::sample_window_func_plan;

mod shj;
mod test;
mod util;
mod view_test;
mod window;

#[tokio::main]
pub async fn main() -> Result<()> {
    // let sql = std::env::args().nth(1).expect("no sql given");
    let session_config = SessionConfig::new().with_batch_size(3);
    let ctx = SessionContext::with_config(session_config);
    let args: Vec<String> = env::args().collect();
    let mut cmd = String::new();
    std::io::stdin().read_line(&mut cmd)?;
    match cmd.as_ref() {
        "window" => sample_window_func_plan(&ctx).await?,
        "abc" => predicate_pushdown(&ctx).await?,
        _ => {
            println!("write a query");
            let mut query = String::new();
            std::io::stdin().read_to_string(&mut query)?;
            println!("{}", query.trim());

            runquery(&ctx, query).await?
        }
    };
    Ok(())
}

pub async fn predicate_pushdown(ctx: &SessionContext) -> Result<()> {
    let logicalp = new_logical_plan_with_partition(ctx, "students", false).await?;
    println!("{logicalp:?}");
    Ok(())
}
