use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;

use crate::util::new_plan_with_partition;

pub async fn create_view_plan() -> datafusion_common::Result<()> {
    let ctx = SessionContext::new();
    new_plan_with_partition(&ctx, "first", 1, false).await?;
    let sql = r#"
        CREATE TABLE second AS
SELECT * FROM first WHERE 1=0;
    "#;
    let plan = ctx.state().create_logical_plan(sql).await?;
    if let LogicalPlan::CreateMemoryTable(memtable) = plan {
        println!("{}", memtable.name.to_string());
    }
    Ok(())
}
