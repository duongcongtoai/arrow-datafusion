use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use chrono::{DateTime, NaiveDateTime, Utc};

use datafusion::catalog::schema::SchemaProvider;
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::{AggregateUDF, LogicalPlan, ScalarUDF, TableSource};
use datafusion::optimizer::optimizer::Optimizer;
use datafusion::optimizer::{OptimizerContext, OptimizerRule};
use datafusion::prelude::SessionContext;
use datafusion::sql::planner::{ContextProvider, SqlToRel};
use datafusion::sql::sqlparser::ast::Statement;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, Result, TableReference};
use datafusion_sql::sqlparser::dialect::GenericDialect;

pub async fn optimize_test(
    ctx: &SessionContext,
    sql: &str,
) -> Result<(LogicalPlan, LogicalPlan)> {
    let dialect = GenericDialect {}; // or AnsiDialect, or your own dialect ...
    let ast: Vec<Statement> = Parser::parse_sql(&dialect, sql).unwrap();
    let statement = &ast[0];

    // create a logical query plan
    let schema_provider = FixedTableProvider::default();
    let sql_to_rel = SqlToRel::new(&schema_provider);
    let plan = sql_to_rel.sql_statement_to_plan(statement.clone()).unwrap();

    // hard code the return value of now()
    let ts = NaiveDateTime::from_timestamp_opt(1666615693, 0).unwrap();
    let now_time = DateTime::<Utc>::from_utc(ts, Utc);
    let config = OptimizerContext::new()
        .with_skip_failing_rules(false)
        .with_query_execution_start_time(now_time);
    let optimizer = Optimizer::new();
    // optimize the logical plan
    let optimized = optimizer.optimize(&plan, &config, &observe)?;
    Ok((plan, optimized))
}

fn observe(_plan: &LogicalPlan, _rule: &dyn OptimizerRule) {}

#[derive(Default)]
pub struct FixedTableProvider {
    options: ConfigOptions,
}

impl ContextProvider for FixedTableProvider {
    fn get_table_provider(&self, name: TableReference) -> Result<Arc<dyn TableSource>> {
        let table_name = name.table();
        if table_name.starts_with("test") {
            let schema = Schema::new_with_metadata(
                vec![
                    Field::new("col_int32", DataType::Int32, true),
                    Field::new("col_uint32", DataType::UInt32, true),
                    Field::new("col_utf8", DataType::Utf8, true),
                    Field::new("col_date32", DataType::Date32, true),
                    Field::new("col_date64", DataType::Date64, true),
                    // timestamp with no timezone
                    Field::new(
                        "col_ts_nano_none",
                        DataType::Timestamp(TimeUnit::Nanosecond, None),
                        true,
                    ),
                    // timestamp with UTC timezone
                    Field::new(
                        "col_ts_nano_utc",
                        DataType::Timestamp(TimeUnit::Nanosecond, Some("+00:00".into())),
                        true,
                    ),
                ],
                HashMap::new(),
            );

            Ok(Arc::new(MyTableSource {
                schema: Arc::new(schema),
            }))
        } else {
            Err(DataFusionError::Plan("table does not exist".to_string()))
        }
    }

    fn get_function_meta(&self, _name: &str) -> Option<Arc<ScalarUDF>> {
        None
    }

    fn get_aggregate_meta(&self, _name: &str) -> Option<Arc<AggregateUDF>> {
        None
    }

    fn get_variable_type(&self, _variable_names: &[String]) -> Option<DataType> {
        None
    }

    fn options(&self) -> &ConfigOptions {
        &self.options
    }
}

struct MyTableSource {
    schema: SchemaRef,
}

impl TableSource for MyTableSource {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
