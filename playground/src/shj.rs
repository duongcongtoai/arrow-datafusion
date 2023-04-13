use crate::util::new_plan_with_partition;
use datafusion::physical_plan;
use datafusion::prelude::SessionContext;
use datafusion_common::Result;

pub async fn test_unbounded_symetric_hashjoin(ctx: &SessionContext) -> Result<()> {
    new_plan_with_partition(ctx, "first", 1, true).await?;
    new_plan_with_partition(ctx, "second", 1, true).await?;
    let sql = r#"
    select * from first inner join second on first.id = second.id and 
    first.value > second.value - 10 and first.value < second.value + 10
    "#;
    let plan = ctx.state().create_logical_plan(sql).await?;
    println!("{plan:?}");
    let phys = ctx.state().create_physical_plan(&plan).await?;
    let pretty = physical_plan::displayable(phys.as_ref())
        .indent()
        .to_string();
    println!("{pretty}");
    Ok(())
}
