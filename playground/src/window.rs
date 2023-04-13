use datafusion::prelude::SessionContext;
use datafusion_common::Result;

use crate::runquery;
use crate::util::new_plan_with_partition;

pub async fn sample_window_func_plan(ctx: &SessionContext) -> Result<()> {
    new_plan_with_partition(ctx, "first", 1, false).await?;
    let sql = r#"
    select id, 
    AVG(value) OVER (PARTITION BY class) as avg_class_value
    from first
    "#;
    runquery(ctx, sql.to_string()).await
}
