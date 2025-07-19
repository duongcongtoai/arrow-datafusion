// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`DependentJoinRewriter`] converts correlated subqueries to `DependentJoin`

use crate::rewrite_dependent_join::DependentJoinRewriter;
use crate::{ApplyOrder, OptimizerConfig, OptimizerRule};
use std::ops::Deref;
use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{internal_datafusion_err, internal_err, Column, Result};
use datafusion_expr::expr::{
    self, Exists, InSubquery, WindowFunction, WindowFunctionParams,
};
use datafusion_expr::utils::conjunction;
use datafusion_expr::{
    binary_expr, col, lit, not, when, Aggregate, BinaryExpr, CorrelatedColumnInfo,
    DependentJoin, Expr, FetchType, Join, JoinType, LogicalPlan, LogicalPlanBuilder,
    Operator, Projection, SkipType, WindowFrame, WindowFunctionDefinition,
};

use datafusion_functions_window::row_number::row_number_udwf;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;

#[derive(Clone, Debug)]
pub struct DependentJoinDecorrelator {
    // immutable, defined when this object is constructed
    domains: IndexSet<(Column, DataType)>,
    // for each domain column, the corresponding column in delim_get
    correlated_column_to_delim_column: IndexMap<Column, Column>,
    is_initial: bool,

    // top-most subquery DecorrelateDependentJoin has depth 1 and so on
    // TODO: for now it has no usage
    // depth: usize,

    // all correlated columns in current depth and downward (if any)
    correlated_columns: IndexSet<CorrelatedColumnInfo>,
    // check if we have to replace any COUNT aggregates into "CASE WHEN X IS NULL THEN 0 ELSE COUNT END"
    // store a mapping between a expr and its original index in the loglan output
    replacement_map: IndexMap<String, Expr>,
    // if during the top down traversal, we observe any operator that requires
    // joining all rows from the lhs with nullable rows on the rhs
    any_join: bool,
    delim_scan_id: usize,
    dscan_cols: Vec<Column>,
}

// normal join, but remove redundant columns
// i.e if we join two table with equi joins left=right
// only take the matching table on the right;
fn natural_join(
    mut builder: LogicalPlanBuilder,
    right: LogicalPlan,
    join_type: JoinType,
    delim_join_conditions: Vec<(Column, Column)>,
) -> Result<LogicalPlanBuilder> {
    let mut exclude_cols = IndexSet::new();
    let join_exprs: Vec<_> = delim_join_conditions
        .iter()
        .map(|(lhs, rhs)| {
            exclude_cols.insert(rhs);
            binary_expr(
                Expr::Column(lhs.clone()),
                Operator::IsNotDistinctFrom,
                Expr::Column(rhs.clone()),
            )
        })
        .collect();
    let require_dedup = !join_exprs.is_empty();

    builder = builder.delim_join(
        right,
        join_type,
        (Vec::<Column>::new(), Vec::<Column>::new()),
        conjunction(join_exprs).or(Some(lit(true))),
    )?;
    if require_dedup {
        let remain_cols = builder.schema().columns().into_iter().filter_map(|c| {
            if exclude_cols.contains(&c) {
                None
            } else {
                Some(Expr::Column(c))
            }
        });
        builder.project(remain_cols)
    } else {
        Ok(builder)
    }
}

impl DependentJoinDecorrelator {
    fn new_root() -> Self {
        Self {
            domains: IndexSet::new(),
            correlated_column_to_delim_column: IndexMap::new(),
            is_initial: true,
            correlated_columns: IndexSet::new(),
            replacement_map: IndexMap::new(),
            any_join: true,
            delim_scan_id: 0,
            dscan_cols: vec![],
        }
    }

    fn new(
        node: &DependentJoin,
        // correlated_columns: &Vec<(usize, Column, DataType)>,
        correlated_columns_from_parent: &IndexSet<CorrelatedColumnInfo>,
        is_initial: bool,
        any_join: bool,
        delim_scan_id: usize,
        depth: usize,
    ) -> Self {
        // the correlated_columns may contains columns referenced by lower depth, filter them out
        // let domain_columns_from_current_node =
        //     node.correlated_columns.iter().filter_map(|info| {
        //         if depth == info.depth {
        //             Some(info)
        //         } else {
        //             None
        //         }
        //     });

        // TODO: it's better if dependentjoin node store all outer ref on RHS itself
        let all_outer_ref = node.right.all_out_ref_exprs();

        let domains = node
            .correlated_columns
            .iter()
            .chain(correlated_columns_from_parent)
            .filter_map(|info| {
                if all_outer_ref.contains(&Expr::OuterReferenceColumn(
                    info.data_type.clone(),
                    info.col.clone(),
                )) {
                    Some((info.col.clone(), info.data_type.clone()))
                } else {
                    None
                }
            })
            .unique()
            .collect();

        // some where below this dependent join node, another dependent join node
        // may exists, we still need to keep the correlated columns for its own sake.
        // note that even if there may exists some DependentJoins node below, their depth
        // may be equivalent to the current node, and some columns inside `correlated_columns_from_parent`
        // should be propagated down to them (and hence the condition `info.depth >= depth`)
        let mut merged_correlated_columns = correlated_columns_from_parent.clone();
        merged_correlated_columns.retain(|info| info.depth >= depth);
        merged_correlated_columns.extend(node.correlated_columns.iter().cloned());

        //  println!("\n\ndomains:{:?}\ncorrelated_columns:{:?}\n correlated_columns_from_parent:{:?}\n\n", &domains, &merged_correlated_columns, &correlated_columns_from_parent);

        Self {
            domains,
            correlated_column_to_delim_column: IndexMap::new(),
            is_initial,
            correlated_columns: merged_correlated_columns,
            replacement_map: IndexMap::new(),
            any_join,
            delim_scan_id,
            dscan_cols: vec![],
        }
    }

    #[allow(dead_code)]
    fn subquery_dependent_filter(expr: &Expr) -> bool {
        match expr {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                if *op == Operator::And {
                    if Self::subquery_dependent_filter(left)
                        || Self::subquery_dependent_filter(right)
                    {
                        return true;
                    }
                }
            }
            Expr::InSubquery(_) | Expr::ScalarSubquery(_) | Expr::Exists(_) => {
                return true;
            }
            _ => {}
        };
        false
    }
    // fn has_correlated_exprs(node: DependentJoin) -> Result<bool> {}

    fn decorrelate_independent(&mut self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        let mut decorrelator = DependentJoinDecorrelator::new_root();
        decorrelator.decorrelate_plan(plan.clone())
    }

    fn decorrelate(
        &mut self,
        node: &DependentJoin,
        parent_propagate_nulls: bool,
        lateral_depth: usize,
    ) -> Result<LogicalPlan> {
        let perform_delim = true;
        let left = node.left.as_ref();
        let has_correlated_expr = detect_correlated_expressions_in_dependent_join(
            node,
            &self.correlated_columns,
        )?;

        let new_left = if !self.is_initial {
            let new_left = if !has_correlated_expr {
                // self.decorrelate_plan(left.clone())?
                // TODO: fix me
                self.decorrelate_independent(left)?
            } else {
                self.push_down_dependent_join(
                    left,
                    parent_propagate_nulls,
                    lateral_depth,
                )?
            };

            // TODO: duckdb does this redundant rewrite for no reason???
            // let mut new_plan = Self::rewrite_outer_ref_columns(
            //     new_left,
            //     &self.correlated_column_to_delim_column,
            //     false,
            // )?;

            let new_plan = Self::rewrite_outer_ref_columns(
                new_left,
                &self.correlated_column_to_delim_column,
                true,
            )?;
            new_plan
        } else {
            self.decorrelate_plan(left.clone())?
        };
        let lateral_depth = 0;
        // let propagate_null_values = node.propagate_null_value();
        let _propagate_null_values = true;
        let mut decorrelator = DependentJoinDecorrelator::new(
            node,
            &self.correlated_columns,
            false,
            false,
            self.delim_scan_id,
            node.subquery_depth,
        );

        let right = decorrelator.push_down_dependent_join(
            &node.right,
            parent_propagate_nulls,
            lateral_depth,
        )?;

        let (join_condition, join_type, post_join_expr) = self.delim_join_conditions(
            &decorrelator,
            node,
            new_left.schema().columns(),
            right.schema().columns(),
            perform_delim,
        )?;

        let mut builder = LogicalPlanBuilder::new(new_left).join(
            right,
            join_type,
            (Vec::<Column>::new(), Vec::<Column>::new()),
            Some(join_condition),
        )?;

        if let Some(subquery_proj_expr) = post_join_expr {
            let new_exprs: Vec<Expr> = builder
                .schema()
                .columns()
                .into_iter()
                // remove any "mark" columns output by the markjoin
                .filter_map(|c| {
                    if c.name == "mark" {
                        None
                    } else {
                        Some(Expr::Column(c))
                    }
                })
                .chain(std::iter::once(subquery_proj_expr))
                .collect();
            builder = builder.project(new_exprs)?;
        }

        self.merge_child(&decorrelator);
        return builder.build();
    }

    fn merge_child(&mut self, child: &Self) {
        self.delim_scan_id = child.delim_scan_id;
        // TODO:
        // for future reference
        // current implementation assume delim scan columns are always taken from the
        // left plan, thus no need to merge correlated map from children (which is always from the RHS)
        // for entry in child.correlated_column_to_delim_column.iter() {
        //     if let Some(old_dscan) =
        //         self.correlated_column_to_delim_column.get_mut(entry.0)
        //     {
        //         if let Some(e) = self.dscan_cols.iter_mut().find(|c| *c == old_dscan) {
        //             *e = entry.1.clone();
        //         }
        //         *old_dscan = entry.1.clone();
        //     }
        // }
    }

    // TODO: support lateral join
    // convert dependent join into delim join
    fn delim_join_conditions(
        &self,
        right_side_decorrelator: &DependentJoinDecorrelator,
        node: &DependentJoin, // this node is on the right side
        left_columns: Vec<Column>,
        right_columns: Vec<Column>,
        perform_delim: bool,
    ) -> Result<(Expr, JoinType, Option<Expr>)> {
        if node.lateral_join_condition.is_some() {
            unimplemented!()
        }

        let _col_count = if perform_delim {
            node.correlated_columns.len()
        } else {
            unimplemented!()
        };
        let mut join_conditions = vec![];
        // if this is set, a new expr will be added to the parent projection
        // after delimJoin
        // this is because some expr cannot be evaluated during the join, for example
        // binary_expr(subquery_1,subquery_2)
        // this will result into 2 consecutive delim_join
        // project(binary_expr(result_subquery_1, result_subquery_2))
        //  delim_join on subquery1
        //   delim_join on subquery2
        let mut extra_expr_after_join = None;
        let mut join_type = JoinType::Inner;
        if let Some(ref expr) = node.subquery_expr {
            match expr {
                Expr::ScalarSubquery(_) => {
                    // TODO: support JoinType::Single
                    // That works similar to left outer join
                    // But having extra check that only for each entry on the LHS
                    // only at most 1 parter on the RHS matches
                    join_type = JoinType::Left;

                    // The reason we does not make this as a condition inside the delim join
                    // is because the evaluation of scalar_subquery expr may be needed
                    // somewhere above
                    extra_expr_after_join = Some(
                        Expr::Column(right_columns.first().unwrap().clone())
                            .alias(format!("{}_output", node.subquery_name)),
                    );
                }
                Expr::Exists(Exists { negated, .. }) => {
                    join_type = JoinType::LeftMark;
                    if *negated {
                        extra_expr_after_join = Some(
                            not(col("mark"))
                                .alias(format!("{}_output", node.subquery_name)),
                        );
                    } else {
                        extra_expr_after_join = Some(
                            col("mark").alias(format!("{}_output", node.subquery_name)),
                        );
                    }
                }
                Expr::InSubquery(InSubquery { expr, negated, .. }) => {
                    // TODO: looks like there is a comment that
                    // markjoin does not support fully null semantic for ANY/IN subquery
                    join_type = JoinType::LeftMark;
                    extra_expr_after_join =
                        Some(col("mark").alias(format!("{}_output", node.subquery_name)));
                    let op = if *negated {
                        Operator::NotEq
                    } else {
                        Operator::Eq
                    };
                    join_conditions.push(binary_expr(
                        expr.deref().clone(),
                        op,
                        Expr::Column(right_columns.first().unwrap().clone()),
                    ));
                }
                _ => {
                    unreachable!()
                }
            }
        }

        // TODO: Revisit this delim join condition
        for (right_domain, _) in right_side_decorrelator.domains.iter() {
            let right_delim_col = right_side_decorrelator
                .correlated_column_to_delim_column
                .get(right_domain)
                .ok_or(internal_datafusion_err!(
                    "correlated map on right side does not have entry for its own domain"
                ))?;

            let left_delim_col = if let Some(left_delim_col) =
                self.correlated_column_to_delim_column.get(right_domain)
            {
                left_delim_col.clone()
            } else if left_columns.contains(right_domain) {
                // check if domains on the right is actually a column provided directly
                // by the left side
                right_domain.clone()
            } else {
                continue;
            };

            join_conditions.push(binary_expr(
                Expr::Column(left_delim_col),
                Operator::IsNotDistinctFrom,
                Expr::Column(right_delim_col.clone()),
            ));
        }
        Ok((
            conjunction(join_conditions).or(Some(lit(true))).unwrap(),
            join_type,
            extra_expr_after_join,
        ))
    }

    fn rewrite_current_plan_outer_ref_columns(
        plan: LogicalPlan,
        correlated_map: &IndexMap<Column, Column>,
    ) -> Result<LogicalPlan> {
        // replace correlated column in dependent with delimget's column
        let new_plan = if let LogicalPlan::DependentJoin(DependentJoin { .. }) = plan {
            return internal_err!(
                "logical error, this function should not be called if one of the plan is still dependent join node");
        } else {
            plan
        };

        new_plan
            .map_expressions(|e| {
                e.transform(|e| {
                    if let Expr::OuterReferenceColumn(_, outer_col) = &e {
                        if let Some(delim_col) = correlated_map.get(outer_col) {
                            return Ok(Transformed::yes(Expr::Column(delim_col.clone())));
                        }else{
                            return internal_err!("correlated map does not detect for outer reference of column {}",outer_col);
                        }
                    }
                    Ok(Transformed::no(e))
                })
            })?
            .data
            .recompute_schema()
    }

    fn rewrite_outer_ref_columns(
        plan: LogicalPlan,
        correlated_map: &IndexMap<Column, Column>,
        recursive: bool,
    ) -> Result<LogicalPlan> {
        // TODO: take depth into consideration
        let new_plan = if recursive {
            plan.transform_down(|child| {
                Ok(Transformed::yes(
                    Self::rewrite_current_plan_outer_ref_columns(child, correlated_map)?,
                ))
            })?
            .data
            .recompute_schema()?
        } else {
            plan
        };

        Self::rewrite_current_plan_outer_ref_columns(new_plan, correlated_map)
    }

    fn rewrite_into_delim_column(
        correlated_map: &IndexMap<Column, Column>,
        original: &Column,
    ) -> Result<Column> {
        correlated_map
            .get(original)
            .ok_or(internal_datafusion_err!(
                "correlated map does not have entry for {}",
                original
            ))
            .cloned()
    }

    fn build_delim_scan(&mut self) -> Result<LogicalPlan> {
        // Clear last dscan info every time we build new dscan.
        self.dscan_cols.clear();

        // Collect all correlated columns of different outer table.
        let mut domains_by_table: IndexMap<String, Vec<(Column, DataType)>> =
            IndexMap::new();

        for domain in &self.domains {
            let table_ref = domain
                .0
                .relation
                .clone()
                .ok_or(internal_datafusion_err!(
                    "ta.leRef should exists in correlatd column"
                ))?
                .clone();
            let domains = domains_by_table.entry(table_ref.to_string()).or_default();
            if !domains.iter().any(|existing| &existing == &domain) {
                domains.push(domain.clone());
            }
        }

        // Collect all D from different tables.
        let mut delim_scans = vec![];
        for (table_ref, table_domains) in domains_by_table {
            self.delim_scan_id += 1;
            let delim_scan_name =
                format!("{0}_dscan_{1}", table_ref.clone(), self.delim_scan_id);

            table_domains.iter().for_each(|c| {
                let field_name = c.0.flat_name().replace(".", "_");
                let dscan_col = Column::from_qualified_name(format!(
                    "{}.{field_name}",
                    delim_scan_name
                ));
                self.correlated_column_to_delim_column
                    .insert(c.0.clone(), dscan_col.clone());
                self.dscan_cols.push(dscan_col);
            });

            delim_scans.push(
                LogicalPlanBuilder::delim_get(&table_domains)?
                    .alias(&delim_scan_name)?
                    .build()?,
            );
        }

        // Join all delim_scans together.
        let final_delim_scan = if delim_scans.len() == 1 {
            delim_scans.into_iter().next().unwrap()
        } else {
            let mut iter = delim_scans.into_iter();
            let first = iter
                .next()
                .ok_or_else(|| internal_datafusion_err!("Empty delim_scans vector"))?;
            iter.try_fold(first, |acc, delim_scan| {
                LogicalPlanBuilder::new(acc)
                    .join(
                        delim_scan,
                        JoinType::Inner,
                        (Vec::<Column>::new(), Vec::<Column>::new()),
                        None,
                    )?
                    .build()
            })?
        };

        final_delim_scan.recompute_schema()
    }

    fn rewrite_expr_from_replacement_map(
        replacement: &IndexMap<String, Expr>,
        plan: LogicalPlan,
    ) -> Result<LogicalPlan> {
        // TODO: not sure if rewrite should stop once found replacement expr
        plan.transform_down(|p| {
            if let LogicalPlan::DependentJoin(_) = &p {
                return internal_err!(
                    "calling rewrite_correlated_exprs while some of \
                    the plan is still dependent join plan"
                );
            }
            if let LogicalPlan::Projection(_proj) = &p {
                p.map_expressions(|e| {
                    e.transform(|e| {
                        if let Some(to_replace) = replacement.get(&e.to_string()) {
                            Ok(Transformed::yes(to_replace.clone()))
                        } else {
                            Ok(Transformed::no(e))
                        }
                    })
                })
            } else {
                Ok(Transformed::no(p))
                // unimplemented!()
            }
        })?
        .data
        .recompute_schema()
    }

    // on recursive rewrite, make sure to update any correlated_column
    // TODO: make all of the delim join natural join
    fn push_down_dependent_join_internal(
        &mut self,
        node: &LogicalPlan,
        parent_propagate_nulls: bool,
        lateral_depth: usize,
    ) -> Result<LogicalPlan> {
        let mut has_correlated_expr = false;
        // TODO: is there any way to do this more efficiently
        // TODO: this lookup must be associated with a list of correlated_columns
        // (from current DecorrelateDependentJoin context and its parent)
        // and check if the correlated expr (if any) exists in the correlated_columns
        detect_correlated_expressions(
            node,
            &self.correlated_columns,
            &mut has_correlated_expr,
        )?;

        if !has_correlated_expr {
            match node {
                // exit projection
                LogicalPlan::Projection(old_proj) => {
                    let mut proj = old_proj.clone();

                    let left = self.decorrelate_plan(proj.input.deref().clone())?;
                    if self.domains.is_empty() {
                        proj.input = Arc::new(left);
                        return LogicalPlan::Projection(proj).recompute_schema();
                    } else {
                        let delim_scan = self.build_delim_scan()?;
                        let cross_join = LogicalPlanBuilder::new(left)
                            .join(
                                delim_scan,
                                JoinType::Inner,
                                (Vec::<Column>::new(), Vec::<Column>::new()),
                                None,
                            )?
                            .build()?;

                        for domain_col in self.domains.iter() {
                            proj.expr.push(col(Self::rewrite_into_delim_column(
                                &self.correlated_column_to_delim_column,
                                &domain_col.0,
                            )?));
                        }

                        let proj = Projection::try_new(proj.expr, cross_join.into())?;

                        return Self::rewrite_outer_ref_columns(
                            LogicalPlan::Projection(proj),
                            &self.correlated_column_to_delim_column,
                            false,
                        );
                    }
                }
                LogicalPlan::RecursiveQuery(_) => {
                    // duckdb support this
                    unimplemented!("")
                }
                any => {
                    let left = self.decorrelate_plan(any.clone())?;
                    if self.domains.is_empty() {
                        return Ok(left);
                    } else {
                        let delim_scan = self.build_delim_scan()?;
                        let cross_join = natural_join(
                            LogicalPlanBuilder::new(left),
                            delim_scan,
                            JoinType::Inner,
                            vec![],
                        )?
                        .build()?;
                        return Ok(cross_join);
                    }
                }
            }
        }
        match node {
            LogicalPlan::Projection(old_proj) => {
                let mut proj = old_proj.clone();
                // for (auto &expr : plan->expressions) {
                // 	parent_propagate_null_values &= expr->PropagatesNullValues();
                // }
                // bool child_is_dependent_join = plan->children[0]->type == LogicalOperatorType::LOGICAL_DEPENDENT_JOIN;
                // parent_propagate_null_values &= !child_is_dependent_join;
                let new_input = self.push_down_dependent_join(
                    proj.input.as_ref(),
                    parent_propagate_nulls,
                    lateral_depth,
                )?;
                for domain_col in self.domains.iter() {
                    proj.expr.push(col(Self::rewrite_into_delim_column(
                        &self.correlated_column_to_delim_column,
                        &domain_col.0,
                    )?));
                }
                let proj = Projection::try_new(proj.expr, new_input.into())?;
                return Self::rewrite_outer_ref_columns(
                    LogicalPlan::Projection(proj),
                    &self.correlated_column_to_delim_column,
                    false,
                );
            }
            LogicalPlan::Filter(old_filter) => {
                // todo: define if any join is need
                let new_input = self.push_down_dependent_join(
                    old_filter.input.as_ref(),
                    parent_propagate_nulls,
                    lateral_depth,
                )?;
                let mut filter = old_filter.clone();
                filter.input = Arc::new(new_input);
                let new_plan = Self::rewrite_outer_ref_columns(
                    LogicalPlan::Filter(filter),
                    &self.correlated_column_to_delim_column,
                    false,
                )?;

                return Ok(new_plan);
            }
            LogicalPlan::Aggregate(old_agg) => {
                let delim_scan_above_agg = self.build_delim_scan()?;
                let new_input = self.push_down_dependent_join_internal(
                    old_agg.input.as_ref(),
                    parent_propagate_nulls,
                    lateral_depth,
                )?;
                // to differentiate between the delim scan above the aggregate
                // i.e
                // Delim -> Above agg
                //   Agg
                //     Join
                //       Delim -> Delim below agg
                //       Filter
                //       ..
                // let delim_scan_under_agg_rela = self.delim_scan_relation_name();

                let mut new_agg = old_agg.clone();
                new_agg.input = Arc::new(new_input);
                let new_plan = Self::rewrite_outer_ref_columns(
                    LogicalPlan::Aggregate(new_agg),
                    &self.correlated_column_to_delim_column,
                    false,
                )?;

                let (agg_expr, mut group_expr, input) = match new_plan {
                    LogicalPlan::Aggregate(Aggregate {
                        aggr_expr,
                        group_expr,
                        input,
                        ..
                    }) => (aggr_expr, group_expr, input),
                    _ => {
                        unreachable!()
                    }
                };
                // TODO: only false in case one of the correlated columns are of type
                // List or a struct with a subfield of type List
                let _perform_delim = true;
                // let new_group_count = if perform_delim { self.domains.len() } else { 1 };
                // TODO: support grouping set
                // select count(*)
                let mut extra_group_columns = vec![];
                for c in self.domains.iter() {
                    let delim_col = Self::rewrite_into_delim_column(
                        &self.correlated_column_to_delim_column,
                        &c.0,
                    )?;
                    group_expr.push(col(delim_col.clone()));
                    extra_group_columns.push(delim_col);
                }
                // perform a join of this agg (group by correlated columns added)
                // with the same delimScan of the set same of correlated columns
                // for now ungorup_join is always true
                // let ungroup_join = agg.group_expr.len() == new_group_count;
                let ungroup_join = true;
                if ungroup_join {
                    let mut join_type = JoinType::Inner;
                    if self.any_join || !parent_propagate_nulls {
                        join_type = JoinType::Left;
                    }

                    let mut delim_conditions = vec![];
                    for (lhs, rhs) in extra_group_columns
                        .iter()
                        .zip(delim_scan_above_agg.schema().columns().iter())
                    {
                        delim_conditions.push((lhs.clone(), rhs.clone()));
                    }

                    for agg_expr in agg_expr.iter() {
                        match agg_expr {
                            Expr::AggregateFunction(expr::AggregateFunction {
                                func,
                                ..
                            }) => {
                                // Transformed::yes(Expr::Literal(ScalarValue::Int64(Some(0))))
                                if func.name() == "count" {
                                    let expr_name = agg_expr.to_string();
                                    let expr_to_replace =
                                        when(agg_expr.clone().is_null(), lit(0))
                                            .otherwise(agg_expr.clone())?;
                                    self.replacement_map
                                        .insert(expr_name, expr_to_replace);
                                    continue;
                                }
                            }
                            _ => {}
                        }
                    }

                    let new_agg = Aggregate::try_new(input, group_expr, agg_expr)?;
                    let agg_output_cols = new_agg
                        .schema
                        .columns()
                        .into_iter()
                        .map(|c| Expr::Column(c));
                    let builder =
                        LogicalPlanBuilder::new(LogicalPlan::Aggregate(new_agg))
                            // TODO: a hack to ensure aggregated expr are ordered first in the output
                            .project(agg_output_cols.rev())?;

                    natural_join(
                        builder,
                        delim_scan_above_agg,
                        join_type,
                        delim_conditions,
                    )?
                    .build()
                } else {
                    unimplemented!()
                }
            }
            LogicalPlan::DependentJoin(djoin) => {
                return self.decorrelate(djoin, parent_propagate_nulls, lateral_depth);
            }
            LogicalPlan::Join(old_join) => {
                let mut left_has_correlation = false;
                detect_correlated_expressions(
                    old_join.left.as_ref(),
                    &self.correlated_columns, // TODO: self.correlated_columns is more correct?
                    &mut left_has_correlation,
                )?;
                let mut right_has_correlation = false;
                detect_correlated_expressions(
                    old_join.right.as_ref(),
                    &self.correlated_columns, // TODO: self.correlated_columns is more correct?
                    &mut right_has_correlation,
                )?;

                // Cross projuct, push into both sides of the plan.
                if old_join.is_cross_product() {
                    if !right_has_correlation {
                        // Only left has correlation, push into left.
                        let new_left = self.push_down_dependent_join_internal(
                            old_join.left.as_ref(),
                            parent_propagate_nulls,
                            lateral_depth,
                        )?;
                        let new_right =
                            self.decorrelate_independent(old_join.right.as_ref())?;

                        return self.join_without_correlation(
                            new_left,
                            new_right,
                            old_join.clone(),
                        );
                    } else if !left_has_correlation {
                        // Only right has correlation, push into right.
                        let new_right = self.push_down_dependent_join_internal(
                            old_join.right.as_ref(),
                            parent_propagate_nulls,
                            lateral_depth,
                        )?;
                        let new_left =
                            self.decorrelate_independent(old_join.left.as_ref())?;

                        return self.join_without_correlation(
                            new_left,
                            new_right,
                            old_join.clone(),
                        );
                    }

                    // Both sides have correlation, turn into an inner join.
                    let new_left = self.push_down_dependent_join_internal(
                        old_join.left.as_ref(),
                        parent_propagate_nulls,
                        lateral_depth,
                    )?;
                    let new_right = self.push_down_dependent_join_internal(
                        old_join.right.as_ref(),
                        parent_propagate_nulls,
                        lateral_depth,
                    )?;

                    // Add the correlated columns to th join conditions.
                    return self.join_with_correlation(
                        new_left,
                        new_right,
                        old_join.clone(),
                    );
                }

                // If it's a comparison join.
                match old_join.join_type {
                    JoinType::Inner => {
                        if !right_has_correlation {
                            // Only left has correlation, push info left.
                            let new_left = self.push_down_dependent_join_internal(
                                old_join.left.as_ref(),
                                parent_propagate_nulls,
                                lateral_depth,
                            )?;
                            let new_right =
                                self.decorrelate_independent(old_join.right.as_ref())?;

                            return self.join_without_correlation(
                                new_left,
                                new_right,
                                old_join.clone(),
                            );
                        }

                        if !left_has_correlation {
                            // Only right has correlation, push into right.
                            let new_right = self.push_down_dependent_join_internal(
                                old_join.right.as_ref(),
                                parent_propagate_nulls,
                                lateral_depth,
                            )?;
                            let new_left =
                                self.decorrelate_independent(old_join.left.as_ref())?;

                            return self.join_without_correlation(
                                new_left,
                                new_right,
                                old_join.clone(),
                            );
                        }
                    }
                    JoinType::Left => {
                        if !right_has_correlation {
                            // Only left has correlation, push info left.
                            let new_left = self.push_down_dependent_join_internal(
                                old_join.left.as_ref(),
                                parent_propagate_nulls,
                                lateral_depth,
                            )?;
                            let new_right =
                                self.decorrelate_independent(old_join.right.as_ref())?;

                            return self.join_without_correlation(
                                new_left,
                                new_right,
                                old_join.clone(),
                            );
                        }
                    }
                    JoinType::Right => {
                        if !left_has_correlation {
                            // Only right has correlation, push into right.
                            let new_right = self.push_down_dependent_join_internal(
                                old_join.right.as_ref(),
                                parent_propagate_nulls,
                                lateral_depth,
                            )?;
                            let new_left =
                                self.decorrelate_independent(old_join.left.as_ref())?;

                            return self.join_without_correlation(
                                new_left,
                                new_right,
                                old_join.clone(),
                            );
                        }
                    }
                    JoinType::LeftMark => {
                        // Push the child into the RHS.
                        let new_left = self.push_down_dependent_join_internal(
                            old_join.left.as_ref(),
                            parent_propagate_nulls,
                            lateral_depth,
                        )?;
                        let new_right =
                            self.decorrelate_independent(old_join.right.as_ref())?;

                        let new_join = self.join_without_correlation(
                            new_left,
                            new_right,
                            old_join.clone(),
                        )?;

                        return Self::rewrite_outer_ref_columns(
                            new_join,
                            &self.correlated_column_to_delim_column,
                            false,
                        );
                    }
                    _ => return internal_err!("unreachable"),
                }

                // Both sides have correlation, push into both sides.
                let new_left = self.push_down_dependent_join_internal(
                    old_join.left.as_ref(),
                    parent_propagate_nulls,
                    lateral_depth,
                )?;
                let left_dscan_cols = self.dscan_cols.clone();

                let new_right = self.push_down_dependent_join_internal(
                    old_join.right.as_ref(),
                    parent_propagate_nulls,
                    lateral_depth,
                )?;
                let right_dscan_cols = self.dscan_cols.clone();

                // NOTE: For OUTER JOINS it matters what the correlated column map is after the join:
                // for the LEFT OUTER JOIN: we want the LEFT side to be the base map after we push,
                // because the RIGHT might contains NULL values.
                if old_join.join_type == JoinType::Left {
                    self.dscan_cols = left_dscan_cols.clone();
                }

                // Add the correlated columns to the join conditions.
                let new_join = self.join_with_delim_scan(
                    new_left,
                    new_right,
                    old_join.clone(),
                    &left_dscan_cols,
                    &right_dscan_cols,
                )?;

                // Then we replace any correlated expressions with the corresponding entry in the
                // correlated_map.
                return Self::rewrite_outer_ref_columns(
                    new_join,
                    &self.correlated_column_to_delim_column,
                    false,
                );
            }
            LogicalPlan::Limit(old_limit) => {
                let mut sort = None;

                // Check if the direct child of this LIMIT node is an ORDER BY node, if so, keep is
                // separate. This is done for an optimization to avoid having to compute the total
                // order.
                let new_input = if let LogicalPlan::Sort(child) = old_limit.input.as_ref()
                {
                    sort = Some(old_limit.input.as_ref().clone());
                    self.push_down_dependent_join_internal(
                        &child.input,
                        parent_propagate_nulls,
                        lateral_depth,
                    )?
                } else {
                    self.push_down_dependent_join_internal(
                        &old_limit.input,
                        parent_propagate_nulls,
                        lateral_depth,
                    )?
                };

                let new_input_cols = new_input.schema().columns().clone();

                // We push a row_number() OVER (PARTITION BY [correlated columns])
                // TODO: take perform delim into consideration
                let mut partition_by = vec![];
                let partition_count = self.domains.len();
                for i in 0..partition_count {
                    if let Some(corr_col) = self.domains.get_index(i) {
                        let delim_col = Self::rewrite_into_delim_column(
                            &self.correlated_column_to_delim_column,
                            &corr_col.0,
                        )?;
                        partition_by.push(Expr::Column(delim_col));
                    }
                }

                let order_by = if let Some(LogicalPlan::Sort(sort)) = &sort {
                    // Optimization: if there is an ORDER BY node followed by a LIMIT rather than
                    // computing the entire order, we push the ORDER BY expressions into the
                    // row_num computation. This way the order only needs to be computed per
                    // partition.
                    sort.expr.clone()
                } else {
                    vec![]
                };

                // Create row_number() window function.
                let row_number_expr = Expr::WindowFunction(Box::new(WindowFunction {
                    fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
                    params: WindowFunctionParams {
                        args: vec![],
                        partition_by,
                        order_by,
                        window_frame: WindowFrame::new(Some(false)),
                        null_treatment: None,
                    },
                }))
                .alias("row_number");
                let mut window_exprs = vec![];
                window_exprs.push(row_number_expr);

                let window = LogicalPlanBuilder::new(new_input)
                    .window(window_exprs)?
                    .build()?;

                // Add filter based on row_number
                // the filter we add is "row_number > offset AND row_number <= offset + limit"
                let mut filter_conditions = vec![];

                if let FetchType::Literal(Some(fetch)) = old_limit.get_fetch_type()? {
                    let upper_bound =
                        if let SkipType::Literal(skip) = old_limit.get_skip_type()? {
                            // Both offset and limit specified - upper bound is offset + limit.
                            fetch + skip
                        } else {
                            // No offset - upper bound is not only the limit.
                            fetch
                        };

                    filter_conditions
                        .push(col("row_number").lt_eq(lit(upper_bound as i64)));
                }

                // We only need to add "row_number >= offset + 1" if offset is bigger than 0.

                if let SkipType::Literal(skip) = old_limit.get_skip_type()? {
                    if skip > 0 {
                        filter_conditions.push(col("row_number").gt(lit(skip as i64)));
                    }
                }

                let mut result_plan = window;
                if !filter_conditions.is_empty() {
                    let filter_expr = filter_conditions
                        .into_iter()
                        .reduce(|acc, expr| acc.and(expr))
                        .unwrap();

                    result_plan = LogicalPlanBuilder::new(result_plan)
                        .filter(filter_expr)?
                        .build()?;
                }

                // Project away the row_number column, keeping only original columns
                let final_exprs = new_input_cols
                    .iter()
                    .map(|c| col(c.clone()))
                    .collect::<Vec<_>>();
                result_plan = LogicalPlanBuilder::new(result_plan)
                    .project(final_exprs)?
                    .build()?;

                return Ok(result_plan);
            }
            LogicalPlan::Distinct(old_distinct) => {
                // Push down into child.
                let new_input = self.push_down_dependent_join(
                    old_distinct.input().as_ref(),
                    parent_propagate_nulls,
                    lateral_depth,
                )?;
                // Add all correlated columns to the DISTINCT targets.
                let mut distinct_exprs = old_distinct
                    .input()
                    .schema()
                    .columns()
                    .into_iter()
                    .map(|c| col(c.clone()))
                    .collect::<Vec<_>>();

                // Add correlated columns as additional columns for grouping
                for domain_col in self.domains.iter() {
                    let delim_col = Self::rewrite_into_delim_column(
                        &self.correlated_column_to_delim_column,
                        &domain_col.0,
                    )?;
                    distinct_exprs.push(col(delim_col));
                }

                // Create new distinct plan with additional correlated columns
                let distinct = LogicalPlanBuilder::new(new_input)
                    .distinct_on(distinct_exprs, vec![], None)?
                    .build()?;

                return Ok(distinct);
            }
            LogicalPlan::Sort(old_sort) => {
                let new_input = self.push_down_dependent_join(
                    old_sort.input.as_ref(),
                    parent_propagate_nulls,
                    lateral_depth,
                )?;
                let mut sort = old_sort.clone();
                sort.input = Arc::new(new_input);
                Ok(LogicalPlan::Sort(sort))
            }
            LogicalPlan::TableScan(old_table_scan) => {
                let delim_scan = self.build_delim_scan()?;

                // Add correlated columns to the table scan output
                let mut projection_exprs: Vec<Expr> = old_table_scan
                    .projected_schema
                    .columns()
                    .into_iter()
                    .map(|c| Expr::Column(c))
                    .collect();

                // Add delim columns to projection
                for domain_col in self.domains.iter() {
                    let delim_col = Self::rewrite_into_delim_column(
                        &self.correlated_column_to_delim_column,
                        &domain_col.0,
                    )?;
                    projection_exprs.push(Expr::Column(delim_col));
                }

                // Cross join with delim scan and project
                let cross_join = LogicalPlanBuilder::new(LogicalPlan::TableScan(
                    old_table_scan.clone(),
                ))
                .join(
                    delim_scan,
                    JoinType::Inner,
                    (Vec::<Column>::new(), Vec::<Column>::new()),
                    None,
                )?
                .project(projection_exprs)?
                .build()?;

                // Rewrite correlated expressions
                Self::rewrite_outer_ref_columns(
                    cross_join,
                    &self.correlated_column_to_delim_column,
                    false,
                )
            }
            LogicalPlan::Window(old_window) => {
                // Push into children.
                let new_input = self.push_down_dependent_join_internal(
                    &old_window.input,
                    parent_propagate_nulls,
                    lateral_depth,
                )?;

                // Create new window expressions with updated partition clauses
                let mut new_window_exprs = old_window.window_expr.clone();

                // Add correlated columns to PARTITION BY clauses in each window expression
                for window_expr in &mut new_window_exprs {
                    // Handle both direct window functions and aliased window functions
                    let window_func = match window_expr {
                        Expr::WindowFunction(ref mut window_func) => window_func,
                        Expr::Alias(alias) => {
                            if let Expr::WindowFunction(ref mut window_func) =
                                alias.expr.as_mut()
                            {
                                window_func
                            } else {
                                continue; // Skip if alias doesn't contain a window function
                            }
                        }
                        _ => continue, // Skip non-window expressions
                    };

                    // Add correlated columns to the partition by clause
                    for domain_col in self.domains.iter() {
                        let delim_col = Self::rewrite_into_delim_column(
                            &self.correlated_column_to_delim_column,
                            &domain_col.0,
                        )?;
                        window_func
                            .params
                            .partition_by
                            .push(Expr::Column(delim_col));
                    }
                }

                // Create new window plan with updated expressions and input
                let mut window = old_window.clone();
                window.input = Arc::new(new_input);
                window.window_expr = new_window_exprs;

                // We replace any correlated expressions with the corresponding entry in the
                // correlated_map.
                Self::rewrite_outer_ref_columns(
                    LogicalPlan::Window(window),
                    &self.correlated_column_to_delim_column,
                    false,
                )
            }
            plan_ => {
                unimplemented!("implement pushdown dependent join for node {plan_}")
            }
        }
    }

    fn push_down_dependent_join(
        &mut self,
        node: &LogicalPlan,
        parent_propagate_nulls: bool,
        lateral_depth: usize,
    ) -> Result<LogicalPlan> {
        let mut new_plan = self.push_down_dependent_join_internal(
            node,
            parent_propagate_nulls,
            lateral_depth,
        )?;
        if !self.replacement_map.is_empty() {
            new_plan =
                Self::rewrite_expr_from_replacement_map(&self.replacement_map, new_plan)?;
        }

        // let projected_expr = new_plan.schema().columns().into_iter().map(|c| {
        //     if let Some(alt_expr) = self.replacement_map.swap_remove(&c.name) {
        //         return alt_expr;
        //     }
        //     Expr::Column(c.clone())
        // });
        // new_plan = LogicalPlanBuilder::new(new_plan)
        //     .project(projected_expr)?
        //     .build()?;
        Ok(new_plan)
    }

    fn decorrelate_plan(&mut self, node: LogicalPlan) -> Result<LogicalPlan> {
        match node {
            LogicalPlan::DependentJoin(mut djoin) => {
                self.decorrelate(&mut djoin, true, 0)
            }
            _ => Ok(node
                .map_children(|n| Ok(Transformed::yes(self.decorrelate_plan(n)?)))?
                .data),
        }
    }

    fn join_without_correlation(
        &mut self,
        left: LogicalPlan,
        right: LogicalPlan,
        join: Join,
    ) -> Result<LogicalPlan> {
        let new_join = LogicalPlan::Join(Join::try_new(
            Arc::new(left),
            Arc::new(right),
            join.on,
            join.filter,
            join.join_type,
            join.join_constraint,
            join.null_equality,
        )?);

        Self::rewrite_outer_ref_columns(
            new_join,
            &self.correlated_column_to_delim_column,
            false,
        )
    }

    fn join_with_correlation(
        &mut self,
        left: LogicalPlan,
        right: LogicalPlan,
        join: Join,
    ) -> Result<LogicalPlan> {
        let mut join_conditions = vec![];
        if let Some(filter) = join.filter {
            join_conditions.push(filter);
        }

        for col_pair in &self.correlated_column_to_delim_column {
            join_conditions.push(binary_expr(
                Expr::Column(col_pair.0.clone()),
                Operator::IsNotDistinctFrom,
                Expr::Column(col_pair.1.clone()),
            ));
        }

        let new_join = LogicalPlan::Join(Join::try_new(
            Arc::new(left),
            Arc::new(right),
            join.on,
            conjunction(join_conditions).or(Some(lit(true))),
            join.join_type,
            join.join_constraint,
            join.null_equality,
        )?);

        Self::rewrite_outer_ref_columns(
            new_join,
            &self.correlated_column_to_delim_column,
            false,
        )
    }

    fn join_with_delim_scan(
        &mut self,
        left: LogicalPlan,
        right: LogicalPlan,
        join: Join,
        left_dscan_cols: &Vec<Column>,
        right_dscan_cols: &Vec<Column>,
    ) -> Result<LogicalPlan> {
        let mut join_conditions = vec![];
        if let Some(filter) = join.filter {
            join_conditions.push(filter);
        }

        // Ensure left_dscan_cols and right_dscan_cols have the same length
        if left_dscan_cols.len() != right_dscan_cols.len() {
            return Err(internal_datafusion_err!(
                "Mismatched dscan columns length: left_dscan_cols has {} elements, right_dscan_cols has {} elements",
                left_dscan_cols.len(),
                right_dscan_cols.len()
            ));
        }

        for (left_delim_col, right_delim_col) in
            left_dscan_cols.iter().zip(right_dscan_cols.iter())
        {
            join_conditions.push(binary_expr(
                Expr::Column(left_delim_col.clone()),
                Operator::IsNotDistinctFrom,
                Expr::Column(right_delim_col.clone()),
            ));
        }

        let new_join = LogicalPlan::Join(Join::try_new(
            Arc::new(left),
            Arc::new(right),
            join.on,
            conjunction(join_conditions).or(Some(lit(true))),
            join.join_type,
            join.join_constraint,
            join.null_equality,
        )?);

        Self::rewrite_outer_ref_columns(
            new_join,
            &self.correlated_column_to_delim_column,
            false,
        )
    }
}

// TODO: take lateral into consideration
fn detect_correlated_expressions_in_dependent_join(
    plan: &DependentJoin,
    correlated_columns: &IndexSet<CorrelatedColumnInfo>,
) -> Result<bool> {
    let is_correlated = &mut false;
    detect_correlated_expressions(&plan.left, correlated_columns, is_correlated)?;
    if *is_correlated {
        return Ok(true);
    }
    detect_correlated_expressions(&plan.right, correlated_columns, is_correlated)?;
    Ok(*is_correlated)
}

// TODO: take lateral into consideration
fn detect_correlated_expressions(
    plan: &LogicalPlan,
    correlated_columns: &IndexSet<CorrelatedColumnInfo>,
    has_correlated_expressions: &mut bool,
) -> Result<()> {
    plan.apply(|child| match child {
        any_plan => {
            for e in any_plan.all_out_ref_exprs().iter() {
                if let Expr::OuterReferenceColumn(data_type, col) = e {
                    if correlated_columns
                        .iter()
                        .any(|c| &c.col == col && &c.data_type == data_type)
                    {
                        *has_correlated_expressions = true;
                        return Ok(TreeNodeRecursion::Stop);
                    }
                }
            }
            Ok(TreeNodeRecursion::Continue)
        }
    })?;

    Ok(())
}

/// Optimizer rule for rewriting any arbitrary subqueries
#[allow(dead_code)]
#[derive(Debug)]
pub struct DecorrelateDependentJoin {}

impl DecorrelateDependentJoin {
    pub fn new() -> Self {
        return DecorrelateDependentJoin {};
    }
}

impl OptimizerRule for DecorrelateDependentJoin {
    fn supports_rewrite(&self) -> bool {
        true
    }

    // There will be 2 rewrites going on
    // - Convert all subqueries (maybe including lateral join in the future) to temporary
    // LogicalPlan node called DependentJoin
    // - Decorrelate DependentJoin following top-down approach recursively
    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let mut transformer =
            DependentJoinRewriter::new(Arc::clone(config.alias_generator()));
        let rewrite_result = transformer.rewrite_subqueries_into_dependent_joins(plan)?;

        if rewrite_result.transformed {
            println!("dependent join plan\n{}", rewrite_result.data);
            let mut decorrelator = DependentJoinDecorrelator::new_root();
            return Ok(Transformed::yes(
                decorrelator.decorrelate_plan(rewrite_result.data)?,
            ));
        }
        Ok(rewrite_result)
    }

    fn name(&self) -> &str {
        "decorrelate_subquery"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        None
    }
}

#[cfg(test)]
mod tests {

    use crate::decorrelate_dependent_join::DecorrelateDependentJoin;
    use crate::test::{test_table_scan_with_name, test_table_with_columns};
    use crate::Optimizer;
    use crate::{
        assert_optimized_plan_eq_display_indent_snapshot, OptimizerConfig,
        OptimizerContext, OptimizerRule,
    };
    use arrow::datatypes::DataType as ArrowDataType;
    use datafusion_common::{Column, Result};
    use datafusion_expr::expr::{WindowFunction, WindowFunctionParams};
    use datafusion_expr::{
        exists, expr_fn::col, in_subquery, lit, out_ref_col, scalar_subquery, Expr,
        LogicalPlan, LogicalPlanBuilder,
    };
    use datafusion_expr::{JoinType, WindowFrame, WindowFunctionDefinition};
    use datafusion_functions_aggregate::{count::count, sum::sum};
    use datafusion_functions_window::row_number::row_number_udwf;
    use std::sync::Arc;
    fn print_optimize_tree(plan: &LogicalPlan) {
        let rule: Arc<dyn OptimizerRule + Send + Sync> =
            Arc::new(DecorrelateDependentJoin::new());
        let optimizer = Optimizer::with_rules(vec![rule]);
        let optimized_plan = optimizer
            .optimize(plan.clone(), &OptimizerContext::new(), |_, _| {})
            .expect("failed to optimize plan");
        println!("{}", optimized_plan.display_tree());
    }

    macro_rules! assert_decorrelate {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            print_optimize_tree(&$plan);
            let rule: Arc<dyn crate::OptimizerRule + Send + Sync> = Arc::new(DecorrelateDependentJoin::new());
            assert_optimized_plan_eq_display_indent_snapshot!(
                rule,
                $plan,
                @ $expected,
            )?;
        }};
    }

    #[test]
    fn exit_projection_and_projection_expr_contains_uncorrlelated_complex_subquery(
    ) -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;

        let sq2 = Arc::new(
            LogicalPlanBuilder::from(t3.clone())
                .filter(col("t3.b").eq(out_ref_col(ArrowDataType::UInt32, "t2.a")))?
                .build()?,
        );
        let sq1 = Arc::new(
            LogicalPlanBuilder::from(t2.clone())
                .filter(exists(sq2))?
                // this is where exit_projection happens
                .project(vec![col("t2.a")])?
                .filter(col("t2.a").eq(out_ref_col(ArrowDataType::UInt32, "t1.a")))?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .project(vec![col("t1.a"), exists(sq1)])?
            .build()?;
        // dependent join plan
        // Projection: t1.a, __exists_sq_2_output AS EXISTS
        //   DependentJoin on [t1.a lvl 1] with expr EXISTS (<subquery>) depth 1
        //     TableScan: t1
        //     Filter: t2.a = outer_ref(t1.a)
        //       Projection: t2.a
        //         Projection: t2.a, t2.b, t2.c
        //           Filter: __exists_sq_1_output
        //             DependentJoin on [t2.a lvl 2] with expr EXISTS (<subquery>) depth 2
        //               TableScan: t2
        //               Filter: t3.b = outer_ref(t2.a)
        //                 TableScan: t3

        assert_decorrelate!(plan, @r"
        Projection: t1.a, __exists_sq_2_output AS EXISTS [a:UInt32, EXISTS:Boolean]
          Projection: t1.a, t1.b, t1.c, mark AS __exists_sq_2_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_2_output:Boolean]
            LeftMark Join(ComparisonJoin):  Filter: t1.a IS NOT DISTINCT FROM t1_dscan_2.t1_a [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
              TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
              Filter: t2.a = t1_dscan_2.t1_a [a:UInt32, t1_a:UInt32;N]
                Projection: t2.a, t1_dscan_2.t1_a [a:UInt32, t1_a:UInt32;N]
                  Cross Join(ComparisonJoin):  [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N]
                    Projection: t2.a, t2.b, t2.c [a:UInt32, b:UInt32, c:UInt32]
                      Filter: __exists_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean]
                        Projection: t2.a, t2.b, t2.c, mark AS __exists_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean]
                          LeftMark Join(ComparisonJoin):  Filter: t2.a IS NOT DISTINCT FROM t2_dscan_1.t2_a [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                            Filter: t3.b = t2_dscan_1.t2_a [a:UInt32, b:UInt32, c:UInt32, t2_a:UInt32;N]
                              Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t2_a:UInt32;N]
                                TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                                SubqueryAlias: t2_dscan_1 [t2_a:UInt32;N]
                                  DelimGet: t2.a [t2_a:UInt32;N]
                    SubqueryAlias: t1_dscan_2 [t1_a:UInt32;N]
                      DelimGet: t1.a [t1_a:UInt32;N]
        ");
        Ok(())
    }

    #[test]
    fn correlated_subquery_nested_in_uncorrelated_subquery() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;

        let sq2 = Arc::new(
            LogicalPlanBuilder::from(t3.clone())
                .filter(col("t3.b").eq(out_ref_col(ArrowDataType::UInt32, "t2.b")))?
                .build()?,
        );
        let sq1 = Arc::new(
            LogicalPlanBuilder::from(t2.clone())
                .filter(exists(sq2))?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(exists(sq1))?
            .build()?;
        // Filter: EXISTS (<subquery>)
        //   Subquery:
        //     Filter: EXISTS (<subquery>)
        //       Subquery:
        //         Filter: t3.b = outer_ref(t2.b)
        //           TableScan: t3
        //       TableScan: t2
        //   TableScan: t1

        // dependent join plan
        // Projection: t1.a, t1.b, t1.c
        //   Filter: __exists_sq_2_output
        //     DependentJoin on [] with expr EXISTS (<subquery>) depth 1
        //       TableScan: t1
        //       Projection: t2.a, t2.b, t2.c
        //         Filter: __exists_sq_1_output
        //           DependentJoin on [t2.b lvl 2] with expr EXISTS (<subquery>) depth 2
        //             TableScan: t2
        //             Filter: t3.b = outer_ref(t2.b)
        //               TableScan: t3

        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: __exists_sq_2_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_2_output:Boolean]
            Projection: t1.a, t1.b, t1.c, t2.mark AS __exists_sq_2_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_2_output:Boolean]
              LeftMark Join(ComparisonJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                Projection: t2.a, t2.b, t2.c [a:UInt32, b:UInt32, c:UInt32]
                  Filter: __exists_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean]
                    Projection: t2.a, t2.b, t2.c, mark AS __exists_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean]
                      LeftMark Join(ComparisonJoin):  Filter: t2.b IS NOT DISTINCT FROM t2_dscan_1.t2_b [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                        TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                        Filter: t3.b = t2_dscan_1.t2_b [a:UInt32, b:UInt32, c:UInt32, t2_b:UInt32;N]
                          Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t2_b:UInt32;N]
                            TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                            SubqueryAlias: t2_dscan_1 [t2_b:UInt32;N]
                              DelimGet: t2.b [t2_b:UInt32;N]
        ");
        Ok(())
    }
    #[test]
    fn two_dependent_joins_at_the_same_depth_of_1() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let sq1 = Arc::new(
            LogicalPlanBuilder::from(t2.clone())
                .filter(col("t2.b").eq(out_ref_col(ArrowDataType::UInt32, "t1.b")))?
                .build()?,
        );
        let sq2 = Arc::new(
            LogicalPlanBuilder::from(t2.clone())
                .filter(col("t2.c").eq(out_ref_col(ArrowDataType::UInt32, "t1.c")))?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(exists(sq1).and(exists(sq2)))?
            .build()?;

        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: __exists_sq_1_output AND __exists_sq_2_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean, __exists_sq_2_output:Boolean]
            Projection: t1.a, t1.b, t1.c, __exists_sq_1_output, mark AS __exists_sq_2_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean, __exists_sq_2_output:Boolean]
              LeftMark Join(ComparisonJoin):  Filter: t1.c IS NOT DISTINCT FROM t1_dscan_2.t1_c [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean, mark:Boolean]
                Projection: t1.a, t1.b, t1.c, mark AS __exists_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean]
                  LeftMark Join(ComparisonJoin):  Filter: t1.b IS NOT DISTINCT FROM t1_dscan_1.t1_b [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                    TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                    Filter: t2.b = t1_dscan_1.t1_b [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N]
                      Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N]
                        TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                        SubqueryAlias: t1_dscan_1 [t1_b:UInt32;N]
                          DelimGet: t1.b [t1_b:UInt32;N]
                Filter: t2.c = t1_dscan_2.t1_c [a:UInt32, b:UInt32, c:UInt32, t1_c:UInt32;N]
                  Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_c:UInt32;N]
                    TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                    SubqueryAlias: t1_dscan_2 [t1_c:UInt32;N]
                      DelimGet: t1.c [t1_c:UInt32;N]
        ");
        Ok(())
    }

    #[test]
    fn two_correlated_dependent_joins_at_the_same_depth_of_2() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;

        let sq2a = Arc::new(
            LogicalPlanBuilder::from(t3.clone())
                .filter(col("t3.b").eq(out_ref_col(ArrowDataType::UInt32, "t1.b")))?
                .build()?,
        );
        let sq2b = Arc::new(
            LogicalPlanBuilder::from(t3.clone())
                .filter(col("t3.c").eq(out_ref_col(ArrowDataType::UInt32, "t1.c")))?
                .build()?,
        );

        let sq1 = Arc::new(
            LogicalPlanBuilder::from(t2.clone())
                .filter(exists(sq2a).and(exists(sq2b)))?
                .build()?,
        );
        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(exists(sq1))?
            .build()?;
        // Original plan
        // Filter: EXISTS (<subquery>)
        //   Subquery:
        //     Filter: EXISTS (<subquery>) AND EXISTS (<subquery>)
        //       Subquery:
        //         Filter: T3.b = outer_ref(T1.b)
        //           TableScan: T3
        //       Subquery:
        //         Filter: T3.c = outer_ref(T1.c)
        //           TableScan: T3
        //       TableScan: T2
        //   TableScan: T1

        // Dependent join
        // Projection: T1.a, T1.b, T1.c
        //   Filter: __exists_sq_3.output
        //     DependentJoin on [T1.b lvl 2, T1.c lvl 2] with expr EXISTS (<subquery>) depth 1
        //       TableScan: T1
        //       Projection: T2.a, T2.b, T2.c
        //         Filter: __exists_sq_1.output AND __exists_sq_2.output
        //           DependentJoin on [] with expr EXISTS (<subquery>) depth 2
        //             DependentJoin on [] with expr EXISTS (<subquery>) depth 2
        //               TableScan: T2
        //               Filter: T3.b = outer_ref(T1.b)
        //                 TableScan: T3
        //             Filter: T3.c = outer_ref(T1.c)
        //               TableScan: T3

        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: __exists_sq_3_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_3_output:Boolean]
            Projection: t1.a, t1.b, t1.c, mark AS __exists_sq_3_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_3_output:Boolean]
              LeftMark Join(ComparisonJoin):  Filter: t1.b IS NOT DISTINCT FROM t1_dscan_1.t1_b AND t1.c IS NOT DISTINCT FROM t1_dscan_1.t1_c [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                Projection: t2.a, t2.b, t2.c, t1_dscan_1.t1_b, t1_dscan_1.t1_c [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N, t1_c:UInt32;N]
                  Filter: __exists_sq_1_output AND __exists_sq_2_output [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N, t1_c:UInt32;N, __exists_sq_1_output:Boolean, __exists_sq_2_output:Boolean]
                    Projection: t2.a, t2.b, t2.c, t1_dscan_1.t1_b, t1_dscan_1.t1_c, __exists_sq_1_output, mark AS __exists_sq_2_output [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N, t1_c:UInt32;N, __exists_sq_1_output:Boolean, __exists_sq_2_output:Boolean]
                      LeftMark Join(ComparisonJoin):  Filter: t1_dscan_1.t1_c IS NOT DISTINCT FROM t1_dscan_3.t1_c [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N, t1_c:UInt32;N, __exists_sq_1_output:Boolean, mark:Boolean]
                        Projection: t2.a, t2.b, t2.c, t1_dscan_1.t1_b, t1_dscan_1.t1_c, mark AS __exists_sq_1_output [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N, t1_c:UInt32;N, __exists_sq_1_output:Boolean]
                          LeftMark Join(ComparisonJoin):  Filter: t1_dscan_1.t1_b IS NOT DISTINCT FROM t1_dscan_2.t1_b [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N, t1_c:UInt32;N, mark:Boolean]
                            Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N, t1_c:UInt32;N]
                              TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                              SubqueryAlias: t1_dscan_1 [t1_b:UInt32;N, t1_c:UInt32;N]
                                DelimGet: t1.b, t1.c [t1_b:UInt32;N, t1_c:UInt32;N]
                            Filter: t3.b = t1_dscan_2.t1_b [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N]
                              Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_b:UInt32;N]
                                TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                                SubqueryAlias: t1_dscan_2 [t1_b:UInt32;N]
                                  DelimGet: t1.b [t1_b:UInt32;N]
                        Filter: t3.c = t1_dscan_3.t1_c [a:UInt32, b:UInt32, c:UInt32, t1_c:UInt32;N]
                          Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_c:UInt32;N]
                            TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                            SubqueryAlias: t1_dscan_3 [t1_c:UInt32;N]
                              DelimGet: t1.c [t1_c:UInt32;N]
        ");
        Ok(())
    }

    // Given a plan with 2 level of subquery
    // This test the fact that correlated columns from the top
    // are propagated to the very bottom subquery
    #[test]
    fn correlated_column_ref_from_parent() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let t3 = test_table_scan_with_name("t3")?;
        let scalar_sq_level2 = Arc::new(
            LogicalPlanBuilder::from(t3)
                .filter(col("t3.a").eq(out_ref_col(ArrowDataType::UInt32, "t1.a")))?
                .aggregate(Vec::<Expr>::new(), vec![count(col("t3.a"))])?
                .build()?,
        );
        let scalar_sq_level1 = Arc::new(
            LogicalPlanBuilder::from(t2.clone())
                .filter(
                    col("t2.c")
                        .eq(out_ref_col(ArrowDataType::UInt32, "t1.c"))
                        .and(scalar_subquery(scalar_sq_level2).eq(lit(1))),
                )?
                .aggregate(Vec::<Expr>::new(), vec![count(col("t2.a"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(scalar_subquery(scalar_sq_level1).eq(col("t1.a")))?
            .build()?;

        // Projection: T1.a, T1.b, T1.c
        //   Filter: __scalar_sq_2.output = T1.a
        //     DependentJoin on [T1.a lvl 2, T1.c lvl 1] with expr (<subquery>) depth 1
        //       TableScan: T1
        //       Aggregate: groupBy=[[]], aggr=[[count(T2.a)]]
        //         Projection: T2.a, T2.b, T2.c
        //           Filter: T2.c = outer_ref(T1.c) AND __scalar_sq_1.output = Int32(1)
        //             DependentJoin on [] with expr (<subquery>) depth 2
        //               TableScan: T2
        //               Aggregate: groupBy=[[]], aggr=[[count(T3.a)]]
        //                 Filter: T3.a = outer_ref(T1.a)
        //                   TableScan: T3
        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: __scalar_sq_2_output = t1.a [a:UInt32, b:UInt32, c:UInt32, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32;N, t1_c:UInt32;N, t1_a:UInt32;N, __scalar_sq_2_output:Int32;N]
            Projection: t1.a, t1.b, t1.c, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_c, t1_dscan_2.t1_a, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END AS __scalar_sq_2_output [a:UInt32, b:UInt32, c:UInt32, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32;N, t1_c:UInt32;N, t1_a:UInt32;N, __scalar_sq_2_output:Int32;N]
              Left Join(ComparisonJoin):  Filter: t1.a IS NOT DISTINCT FROM t1_dscan_2.t1_a AND t1.c IS NOT DISTINCT FROM t1_dscan_2.t1_c [a:UInt32, b:UInt32, c:UInt32, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32;N, t1_c:UInt32;N, t1_a:UInt32;N]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                Projection: CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_c, t1_dscan_2.t1_a [CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32, t1_c:UInt32;N, t1_a:UInt32;N]
                  Inner Join(DelimJoin):  Filter: t1_dscan_2.t1_a IS NOT DISTINCT FROM t1_dscan_1.t1_a AND t1_dscan_2.t1_c IS NOT DISTINCT FROM t1_dscan_1.t1_c [count(t2.a):Int64, t1_c:UInt32;N, t1_a:UInt32;N, t1_a:UInt32;N, t1_c:UInt32;N]
                    Projection: CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_c, t1_dscan_2.t1_a [count(t2.a):Int64, t1_c:UInt32;N, t1_a:UInt32;N]
                      Aggregate: groupBy=[[t1_dscan_2.t1_a, t1_dscan_2.t1_c]], aggr=[[count(t2.a)]] [t1_a:UInt32;N, t1_c:UInt32;N, count(t2.a):Int64]
                        Projection: t2.a, t2.b, t2.c, t1_dscan_2.t1_a, t1_dscan_2.t1_c [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N]
                          Filter: t2.c = t1_dscan_2.t1_c AND __scalar_sq_1_output = Int32(1) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END:Int32;N, t1_a:UInt32;N, __scalar_sq_1_output:Int32;N]
                            Projection: t2.a, t2.b, t2.c, t1_dscan_2.t1_a, t1_dscan_2.t1_c, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END, t1_dscan_4.t1_a, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END AS __scalar_sq_1_output [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END:Int32;N, t1_a:UInt32;N, __scalar_sq_1_output:Int32;N]
                              Left Join(ComparisonJoin):  Filter: t1_dscan_2.t1_a IS NOT DISTINCT FROM t1_dscan_4.t1_a [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END:Int32;N, t1_a:UInt32;N]
                                Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N]
                                  TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                                  SubqueryAlias: t1_dscan_2 [t1_a:UInt32;N, t1_c:UInt32;N]
                                    DelimGet: t1.a, t1.c [t1_a:UInt32;N, t1_c:UInt32;N]
                                Projection: CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END, t1_dscan_4.t1_a [CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END:Int32, t1_a:UInt32;N]
                                  Inner Join(DelimJoin):  Filter: t1_dscan_4.t1_a IS NOT DISTINCT FROM t1_dscan_3.t1_a [count(t3.a):Int64, t1_a:UInt32;N, t1_a:UInt32;N]
                                    Projection: CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END, t1_dscan_4.t1_a [count(t3.a):Int64, t1_a:UInt32;N]
                                      Aggregate: groupBy=[[t1_dscan_4.t1_a]], aggr=[[count(t3.a)]] [t1_a:UInt32;N, count(t3.a):Int64]
                                        Filter: t3.a = t1_dscan_4.t1_a [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N]
                                          Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N]
                                            TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                                            SubqueryAlias: t1_dscan_4 [t1_a:UInt32;N]
                                              DelimGet: t1.a [t1_a:UInt32;N]
                                    SubqueryAlias: t1_dscan_3 [t1_a:UInt32;N]
                                      DelimGet: t1.a [t1_a:UInt32;N]
                    SubqueryAlias: t1_dscan_1 [t1_a:UInt32;N, t1_c:UInt32;N]
                      DelimGet: t1.a, t1.c [t1_a:UInt32;N, t1_c:UInt32;N]
        ");
        Ok(())
    }

    #[test]
    fn decorrelated_two_nested_subqueries() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let t3 = test_table_scan_with_name("t3")?;
        let scalar_sq_level2 = Arc::new(
            LogicalPlanBuilder::from(t3)
                .filter(
                    col("t3.a")
                        .eq(out_ref_col(ArrowDataType::UInt32, "t1.a"))
                        .and(col("t3.b").eq(out_ref_col(ArrowDataType::UInt32, "t2.b"))),
                )?
                .aggregate(Vec::<Expr>::new(), vec![count(col("t3.a"))])?
                .build()?,
        );
        let scalar_sq_level1 = Arc::new(
            LogicalPlanBuilder::from(t2.clone())
                .filter(
                    col("t2.c")
                        .eq(out_ref_col(ArrowDataType::UInt32, "t1.c"))
                        .and(scalar_subquery(scalar_sq_level2).eq(lit(1))),
                )?
                .aggregate(Vec::<Expr>::new(), vec![count(col("t2.a"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(
                col("t1.a")
                    .gt(lit(1))
                    .and(scalar_subquery(scalar_sq_level1).eq(col("t1.a"))),
            )?
            .build()?;
        println!("{plan}");

        // Projection: T1.a, T1.b, T1.c
        //   Filter: T1.a > Int32(1) AND __scalar_sq_2.output = T1.a
        //     DependentJoin on [T1.a lvl 2, T1.c lvl 1] with expr (<subquery>) depth 1
        //       TableScan: T1
        //       Aggregate: groupBy=[[]], aggr=[[count(T2.a)]]
        //         Projection: T2.a, T2.b, T2.c
        //           Filter: T2.c = outer_ref(T1.c) AND __scalar_sq_1.output = Int32(1)
        //             DependentJoin on [T2.b lvl 2] with expr (<subquery>) depth 2
        //               TableScan: T2
        //               Aggregate: groupBy=[[]], aggr=[[count(T3.a)]]
        //                 Filter: T3.a = outer_ref(T1.a) AND T3.b = outer_ref(T2.b)
        //                   TableScan: T3

        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: t1.a > Int32(1) AND __scalar_sq_2_output = t1.a [a:UInt32, b:UInt32, c:UInt32, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32;N, t1_c:UInt32;N, t1_a:UInt32;N, __scalar_sq_2_output:Int32;N]
            Projection: t1.a, t1.b, t1.c, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_c, t1_dscan_2.t1_a, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END AS __scalar_sq_2_output [a:UInt32, b:UInt32, c:UInt32, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32;N, t1_c:UInt32;N, t1_a:UInt32;N, __scalar_sq_2_output:Int32;N]
              Left Join(ComparisonJoin):  Filter: t1.a IS NOT DISTINCT FROM t1_dscan_2.t1_a AND t1.c IS NOT DISTINCT FROM t1_dscan_2.t1_c [a:UInt32, b:UInt32, c:UInt32, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32;N, t1_c:UInt32;N, t1_a:UInt32;N]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                Projection: CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_c, t1_dscan_2.t1_a [CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32, t1_c:UInt32;N, t1_a:UInt32;N]
                  Inner Join(DelimJoin):  Filter: t1_dscan_2.t1_a IS NOT DISTINCT FROM t1_dscan_1.t1_a AND t1_dscan_2.t1_c IS NOT DISTINCT FROM t1_dscan_1.t1_c [count(t2.a):Int64, t1_c:UInt32;N, t1_a:UInt32;N, t1_a:UInt32;N, t1_c:UInt32;N]
                    Projection: CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_c, t1_dscan_2.t1_a [count(t2.a):Int64, t1_c:UInt32;N, t1_a:UInt32;N]
                      Aggregate: groupBy=[[t1_dscan_2.t1_a, t1_dscan_2.t1_c]], aggr=[[count(t2.a)]] [t1_a:UInt32;N, t1_c:UInt32;N, count(t2.a):Int64]
                        Projection: t2.a, t2.b, t2.c, t1_dscan_2.t1_a, t1_dscan_2.t1_c [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N]
                          Filter: t2.c = t1_dscan_2.t1_c AND __scalar_sq_1_output = Int32(1) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END:Int32;N, t1_a:UInt32;N, t2_b:UInt32;N, __scalar_sq_1_output:Int32;N]
                            Projection: t2.a, t2.b, t2.c, t1_dscan_2.t1_a, t1_dscan_2.t1_c, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END, t1_dscan_6.t1_a, t2_dscan_5.t2_b, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END AS __scalar_sq_1_output [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END:Int32;N, t1_a:UInt32;N, t2_b:UInt32;N, __scalar_sq_1_output:Int32;N]
                              Left Join(ComparisonJoin):  Filter: t2.b IS NOT DISTINCT FROM t2_dscan_5.t2_b AND t1_dscan_2.t1_a IS NOT DISTINCT FROM t1_dscan_6.t1_a [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N, CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END:Int32;N, t1_a:UInt32;N, t2_b:UInt32;N]
                                Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N]
                                  TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                                  SubqueryAlias: t1_dscan_2 [t1_a:UInt32;N, t1_c:UInt32;N]
                                    DelimGet: t1.a, t1.c [t1_a:UInt32;N, t1_c:UInt32;N]
                                Projection: CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END, t1_dscan_6.t1_a, t2_dscan_5.t2_b [CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END:Int32, t1_a:UInt32;N, t2_b:UInt32;N]
                                  Inner Join(DelimJoin):  Filter: t2_dscan_5.t2_b IS NOT DISTINCT FROM t2_dscan_3.t2_b AND t1_dscan_6.t1_a IS NOT DISTINCT FROM t1_dscan_4.t1_a [count(t3.a):Int64, t1_a:UInt32;N, t2_b:UInt32;N, t2_b:UInt32;N, t1_a:UInt32;N]
                                    Projection: CASE WHEN count(t3.a) IS NULL THEN Int32(0) ELSE count(t3.a) END, t1_dscan_6.t1_a, t2_dscan_5.t2_b [count(t3.a):Int64, t1_a:UInt32;N, t2_b:UInt32;N]
                                      Aggregate: groupBy=[[t2_dscan_5.t2_b, t1_dscan_6.t1_a]], aggr=[[count(t3.a)]] [t2_b:UInt32;N, t1_a:UInt32;N, count(t3.a):Int64]
                                        Filter: t3.a = t1_dscan_6.t1_a AND t3.b = t2_dscan_5.t2_b [a:UInt32, b:UInt32, c:UInt32, t2_b:UInt32;N, t1_a:UInt32;N]
                                          Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t2_b:UInt32;N, t1_a:UInt32;N]
                                            TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                                            Cross Join(ComparisonJoin):  [t2_b:UInt32;N, t1_a:UInt32;N]
                                              SubqueryAlias: t2_dscan_5 [t2_b:UInt32;N]
                                                DelimGet: t2.b [t2_b:UInt32;N]
                                              SubqueryAlias: t1_dscan_6 [t1_a:UInt32;N]
                                                DelimGet: t1.a [t1_a:UInt32;N]
                                    Cross Join(ComparisonJoin):  [t2_b:UInt32;N, t1_a:UInt32;N]
                                      SubqueryAlias: t2_dscan_3 [t2_b:UInt32;N]
                                        DelimGet: t2.b [t2_b:UInt32;N]
                                      SubqueryAlias: t1_dscan_4 [t1_a:UInt32;N]
                                        DelimGet: t1.a [t1_a:UInt32;N]
                    SubqueryAlias: t1_dscan_1 [t1_a:UInt32;N, t1_c:UInt32;N]
                      DelimGet: t1.a, t1.c [t1_a:UInt32;N, t1_c:UInt32;N]
        ");
        Ok(())
    }

    #[test]
    fn decorrelate_join_in_subquery_with_count_depth_1() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let sq_level1 = Arc::new(
            LogicalPlanBuilder::from(t2)
                .filter(
                    col("t2.a")
                        .eq(out_ref_col(ArrowDataType::UInt32, "t1.a"))
                        .and(out_ref_col(ArrowDataType::UInt32, "t1.a").gt(col("t2.c")))
                        .and(col("t2.b").eq(lit(1)))
                        .and(out_ref_col(ArrowDataType::UInt32, "t1.b").eq(col("t2.b"))),
                )?
                .aggregate(Vec::<Expr>::new(), vec![count(col("t2.a"))])?
                // TODO: if uncomment this the test fail
                // .project(vec![count(col("t2.a")).alias("count_a")])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(
                col("t1.a")
                    .gt(lit(1))
                    .and(in_subquery(col("t1.c"), sq_level1)),
            )?
            .build()?;
        // Projection: T1.a, T1.b, T1.c
        //   Filter: T1.a > Int32(1) AND __in_sq_1.output
        //     DependentJoin on [T1.a lvl 1, T1.b lvl 1] with expr T1.c IN (<subquery>) depth 1
        //       TableScan: T1
        //       Aggregate: groupBy=[[]], aggr=[[count(T2.a)]]
        //         Filter: T2.a = outer_ref(T1.a) AND outer_ref(T1.a) > T2.c AND T2.b = Int32(1) AND outer_ref(T1.b) = T2.b
        //           TableScan: T2

        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: t1.a > Int32(1) AND __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
            Projection: t1.a, t1.b, t1.c, t1_dscan_2.mark AS __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
              LeftMark Join(ComparisonJoin):  Filter: t1.c = CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END AND t1.a IS NOT DISTINCT FROM t1_dscan_2.t1_a AND t1.b IS NOT DISTINCT FROM t1_dscan_2.t1_b [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                Projection: CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_b, t1_dscan_2.t1_a [CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32, t1_b:UInt32;N, t1_a:UInt32;N]
                  Inner Join(DelimJoin):  Filter: t1_dscan_2.t1_a IS NOT DISTINCT FROM t1_dscan_1.t1_a AND t1_dscan_2.t1_b IS NOT DISTINCT FROM t1_dscan_1.t1_b [count(t2.a):Int64, t1_b:UInt32;N, t1_a:UInt32;N, t1_a:UInt32;N, t1_b:UInt32;N]
                    Projection: CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_b, t1_dscan_2.t1_a [count(t2.a):Int64, t1_b:UInt32;N, t1_a:UInt32;N]
                      Aggregate: groupBy=[[t1_dscan_2.t1_a, t1_dscan_2.t1_b]], aggr=[[count(t2.a)]] [t1_a:UInt32;N, t1_b:UInt32;N, count(t2.a):Int64]
                        Filter: t2.a = t1_dscan_2.t1_a AND t1_dscan_2.t1_a > t2.c AND t2.b = Int32(1) AND t1_dscan_2.t1_b = t2.b [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_b:UInt32;N]
                          Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_b:UInt32;N]
                            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                            SubqueryAlias: t1_dscan_2 [t1_a:UInt32;N, t1_b:UInt32;N]
                              DelimGet: t1.a, t1.b [t1_a:UInt32;N, t1_b:UInt32;N]
                    SubqueryAlias: t1_dscan_1 [t1_a:UInt32;N, t1_b:UInt32;N]
                      DelimGet: t1.a, t1.b [t1_a:UInt32;N, t1_b:UInt32;N]
        ");
        Ok(())
    }

    #[test]
    fn one_correlated_subquery_and_one_uncorrelated_subquery_at_the_same_level(
    ) -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let in_sq_level1 = Arc::new(
            LogicalPlanBuilder::from(t2.clone())
                .filter(col("t2.c").eq(out_ref_col(ArrowDataType::Int32, "t1.c")))?
                .project(vec![col("t2.a")])?
                .build()?,
        );
        let exist_sq_level1 = Arc::new(
            LogicalPlanBuilder::from(t2)
                .filter(col("t2.a").and(col("t2.b").eq(lit(1))))?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(
                col("t1.a")
                    .gt(lit(1))
                    .and(exists(exist_sq_level1))
                    .and(in_subquery(col("t1.b"), in_sq_level1)),
            )?
            .build()?;
        // Original
        // Filter: T1.a > Int32(1) AND EXISTS (<subquery>) AND T1.b IN (<subquery>)
        //   Subquery:
        //     Filter: T2.a AND T2.b = Int32(1)
        //       TableScan: T2
        //   Subquery:
        //     Projection: T2.a
        //       Filter: T2.c = outer_ref(T1.c)
        //         TableScan: T2
        //   TableScan: T1
        // dependent join plan
        // Projection: T1.a, T1.b, T1.c
        //   Filter: T1.a > Int32(1) AND __exists_sq_1_output AND __in_sq_2_output
        //     DependentJoin on [T1.c lvl 1] with expr T1.b IN (<subquery>) depth 1
        //       DependentJoin on [] with expr EXISTS (<subquery>) depth 1
        //         TableScan: T1
        //         Filter: T2.a AND T2.b = Int32(1)
        //           TableScan: T2
        //       Projection: T2.a
        //         Filter: T2.c = outer_ref(T1.c)
        //           TableScan: T2

        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: t1.a > Int32(1) AND __exists_sq_1_output AND __in_sq_2_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean, __in_sq_2_output:Boolean]
            Projection: t1.a, t1.b, t1.c, __exists_sq_1_output, mark AS __in_sq_2_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean, __in_sq_2_output:Boolean]
              LeftMark Join(ComparisonJoin):  Filter: t1.b = t2.a AND t1.c IS NOT DISTINCT FROM t1_dscan_1.t1_c [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean, mark:Boolean]
                Projection: t1.a, t1.b, t1.c, t2.mark AS __exists_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __exists_sq_1_output:Boolean]
                  LeftMark Join(ComparisonJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                    TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                    Filter: t2.a AND t2.b = Int32(1) [a:UInt32, b:UInt32, c:UInt32]
                      TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                Projection: t2.a, t1_dscan_1.t1_c [a:UInt32, t1_c:Int32;N]
                  Filter: t2.c = t1_dscan_1.t1_c [a:UInt32, b:UInt32, c:UInt32, t1_c:Int32;N]
                    Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_c:Int32;N]
                      TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                      SubqueryAlias: t1_dscan_1 [t1_c:Int32;N]
                        DelimGet: t1.c [t1_c:Int32;N]
        ");
        Ok(())
    }

    #[test]
    fn decorrelate_with_in_subquery_has_dependent_column() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let sq_level1 = Arc::new(
            LogicalPlanBuilder::from(t2)
                .filter(
                    col("t2.a")
                        .eq(out_ref_col(ArrowDataType::UInt32, "t1.a"))
                        .and(out_ref_col(ArrowDataType::UInt32, "t1.a").gt(col("t2.c")))
                        .and(col("t2.b").eq(lit(1)))
                        .and(out_ref_col(ArrowDataType::UInt32, "t1.b").eq(col("t2.b"))),
                )?
                .project(vec![col("t2.b")])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(
                col("t1.a")
                    .gt(lit(1))
                    .and(in_subquery(col("t1.c"), sq_level1)),
            )?
            .build()?;
        let dec = DecorrelateDependentJoin::new();
        let ctx: Box<dyn OptimizerConfig> = Box::new(OptimizerContext::new());
        let plan = dec.rewrite(plan, ctx.as_ref())?.data;

        // Projection: T1.a, T1.b, T1.c
        //   Filter: T1.a > Int32(1) AND __in_sq_1.output
        //     DependentJoin on [T1.a lvl 1, T1.b lvl 1] with expr T1.c IN (<subquery>) depth 1
        //       TableScan: T1
        //       Projection: T2.b
        //         Filter: T2.a = outer_ref(T1.a) AND outer_ref(T1.a) > T2.c AND T2.b = Int32(1) AND outer_ref(T1.b) = T2.b
        //           TableScan: T2
        assert_decorrelate!(plan,       @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: t1.a > Int32(1) AND __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
            Projection: t1.a, t1.b, t1.c, mark AS __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
              LeftMark Join(ComparisonJoin):  Filter: t1.c = t2.b AND t1.a IS NOT DISTINCT FROM t1_dscan_1.t1_a AND t1.b IS NOT DISTINCT FROM t1_dscan_1.t1_b [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                Projection: t2.b, t1_dscan_1.t1_a, t1_dscan_1.t1_b [b:UInt32, t1_a:UInt32;N, t1_b:UInt32;N]
                  Filter: t2.a = t1_dscan_1.t1_a AND t1_dscan_1.t1_a > t2.c AND t2.b = Int32(1) AND t1_dscan_1.t1_b = t2.b [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_b:UInt32;N]
                    Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_b:UInt32;N]
                      TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                      SubqueryAlias: t1_dscan_1 [t1_a:UInt32;N, t1_b:UInt32;N]
                        DelimGet: t1.a, t1.b [t1_a:UInt32;N, t1_b:UInt32;N]
        ");

        Ok(())
    }

    // This query is inside the paper
    #[test]
    fn decorrelate_two_different_outer_tables() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let t3 = test_table_scan_with_name("t3")?;

        // two outer columns are referenced here T1.a and T1.c
        // even if T2 never reference T1.c
        // such reference from scalar_sq_level2 will enforce
        // the delim scan at T2 includes T1.c
        let scalar_sq_level2 = Arc::new(
            LogicalPlanBuilder::from(t3)
                .filter(
                    col("t3.b")
                        .eq(out_ref_col(ArrowDataType::UInt32, "t2.b"))
                        .and(col("t3.a").eq(out_ref_col(ArrowDataType::UInt32, "t1.a")))
                        .and(col("t3.c").eq(out_ref_col(ArrowDataType::UInt32, "t1.c"))),
                )?
                .aggregate(Vec::<Expr>::new(), vec![sum(col("t3.a"))])?
                .build()?,
        );

        let scalar_sq_level1 = Arc::new(
            LogicalPlanBuilder::from(t2.clone())
                .filter(
                    col("t2.a")
                        .eq(out_ref_col(ArrowDataType::UInt32, "t1.a"))
                        .and(scalar_subquery(scalar_sq_level2).gt(lit(300000))),
                )?
                .aggregate(Vec::<Expr>::new(), vec![count(col("t2.a"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(
                col("t1.c")
                    .eq(lit(123))
                    .and(scalar_subquery(scalar_sq_level1).gt(lit(5))),
            )?
            .build()?;
        println!("{}", plan.display_indent_schema());

        //  Subquery: [count(t2.a):Int64]
        //    Aggregate: groupBy=[[]], aggr=[[count(t2.a)]] [count(t2.a):Int64]
        //      Filter: t2.a = outer_ref(t1.a) AND (<subquery>) > Int32(300000) [a:UInt32, b:UInt32, c:UInt32]
        //        Subquery: [sum(t3.a):UInt64;N]
        //          Aggregate: groupBy=[[]], aggr=[[sum(t3.a)]] [sum(t3.a):UInt64;N]
        //            Filter: t3.b = outer_ref(t2.b) AND t3.a = outer_ref(t1.a) AND t3.c = outer_ref(t1.c) [a:UInt32, b:UInt32, c:UInt32]
        //              TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
        //        TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        //  TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]

        // Projection: t1.a, t1.b, t1.c
        //   Filter: t1.c = Int32(123) AND __scalar_sq_2.output > Int32(5)
        //     DependentJoin on [t1.a lvl 1, t1.a lvl 2, t1.c lvl 2] with expr (<subquery>) depth 1
        //       TableScan: t1
        //       Aggregate: groupBy=[[]], aggr=[[count(t2.a)]]
        //         Projection: t2.a, t2.b, t2.c
        //           Filter: t2.a = outer_ref(t1.a) AND __scalar_sq_1.output > Int32(300000)
        //             DependentJoin on [t2.b lvl 2] with expr (<subquery>) depth 2
        //               TableScan: t2
        //               Aggregate: groupBy=[[]], aggr=[[sum(t3.a)]]
        //                 Filter: t3.b = outer_ref(t2.b) AND t3.a = outer_ref(t1.a) AND t3.c = outer_ref(t1.c)
        //                   TableScan: t3
        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: t1.c = Int32(123) AND __scalar_sq_2_output > Int32(5) [a:UInt32, b:UInt32, c:UInt32, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32;N, t1_c:UInt32;N, t1_a:UInt32;N, __scalar_sq_2_output:Int32;N]
            Projection: t1.a, t1.b, t1.c, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_c, t1_dscan_2.t1_a, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END AS __scalar_sq_2_output [a:UInt32, b:UInt32, c:UInt32, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32;N, t1_c:UInt32;N, t1_a:UInt32;N, __scalar_sq_2_output:Int32;N]
              Left Join(ComparisonJoin):  Filter: t1.a IS NOT DISTINCT FROM t1_dscan_2.t1_a AND t1.c IS NOT DISTINCT FROM t1_dscan_2.t1_c [a:UInt32, b:UInt32, c:UInt32, CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32;N, t1_c:UInt32;N, t1_a:UInt32;N]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                Projection: CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_c, t1_dscan_2.t1_a [CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END:Int32, t1_c:UInt32;N, t1_a:UInt32;N]
                  Inner Join(DelimJoin):  Filter: t1_dscan_2.t1_a IS NOT DISTINCT FROM t1_dscan_1.t1_a AND t1_dscan_2.t1_c IS NOT DISTINCT FROM t1_dscan_1.t1_c [count(t2.a):Int64, t1_c:UInt32;N, t1_a:UInt32;N, t1_a:UInt32;N, t1_c:UInt32;N]
                    Projection: CASE WHEN count(t2.a) IS NULL THEN Int32(0) ELSE count(t2.a) END, t1_dscan_2.t1_c, t1_dscan_2.t1_a [count(t2.a):Int64, t1_c:UInt32;N, t1_a:UInt32;N]
                      Aggregate: groupBy=[[t1_dscan_2.t1_a, t1_dscan_2.t1_c]], aggr=[[count(t2.a)]] [t1_a:UInt32;N, t1_c:UInt32;N, count(t2.a):Int64]
                        Projection: t2.a, t2.b, t2.c, t1_dscan_2.t1_a, t1_dscan_2.t1_c [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N]
                          Filter: t2.a = t1_dscan_2.t1_a AND __scalar_sq_1_output > Int32(300000) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N, sum(t3.a):UInt64;N, t1_c:UInt32;N, t1_a:UInt32;N, t2_b:UInt32;N, __scalar_sq_1_output:UInt64;N]
                            Projection: t2.a, t2.b, t2.c, t1_dscan_2.t1_a, t1_dscan_2.t1_c, sum(t3.a), t1_dscan_6.t1_c, t1_dscan_6.t1_a, t2_dscan_5.t2_b, sum(t3.a) AS __scalar_sq_1_output [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N, sum(t3.a):UInt64;N, t1_c:UInt32;N, t1_a:UInt32;N, t2_b:UInt32;N, __scalar_sq_1_output:UInt64;N]
                              Left Join(ComparisonJoin):  Filter: t2.b IS NOT DISTINCT FROM t2_dscan_5.t2_b AND t1_dscan_2.t1_a IS NOT DISTINCT FROM t1_dscan_6.t1_a AND t1_dscan_2.t1_c IS NOT DISTINCT FROM t1_dscan_6.t1_c [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N, sum(t3.a):UInt64;N, t1_c:UInt32;N, t1_a:UInt32;N, t2_b:UInt32;N]
                                Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_c:UInt32;N]
                                  TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                                  SubqueryAlias: t1_dscan_2 [t1_a:UInt32;N, t1_c:UInt32;N]
                                    DelimGet: t1.a, t1.c [t1_a:UInt32;N, t1_c:UInt32;N]
                                Projection: sum(t3.a), t1_dscan_6.t1_c, t1_dscan_6.t1_a, t2_dscan_5.t2_b [sum(t3.a):UInt64;N, t1_c:UInt32;N, t1_a:UInt32;N, t2_b:UInt32;N]
                                  Inner Join(DelimJoin):  Filter: t2_dscan_5.t2_b IS NOT DISTINCT FROM t2_dscan_3.t2_b AND t1_dscan_6.t1_a IS NOT DISTINCT FROM t1_dscan_4.t1_a AND t1_dscan_6.t1_c IS NOT DISTINCT FROM t1_dscan_4.t1_c [sum(t3.a):UInt64;N, t1_c:UInt32;N, t1_a:UInt32;N, t2_b:UInt32;N, t2_b:UInt32;N, t1_a:UInt32;N, t1_c:UInt32;N]
                                    Projection: sum(t3.a), t1_dscan_6.t1_c, t1_dscan_6.t1_a, t2_dscan_5.t2_b [sum(t3.a):UInt64;N, t1_c:UInt32;N, t1_a:UInt32;N, t2_b:UInt32;N]
                                      Aggregate: groupBy=[[t2_dscan_5.t2_b, t1_dscan_6.t1_a, t1_dscan_6.t1_c]], aggr=[[sum(t3.a)]] [t2_b:UInt32;N, t1_a:UInt32;N, t1_c:UInt32;N, sum(t3.a):UInt64;N]
                                        Filter: t3.b = t2_dscan_5.t2_b AND t3.a = t1_dscan_6.t1_a AND t3.c = t1_dscan_6.t1_c [a:UInt32, b:UInt32, c:UInt32, t2_b:UInt32;N, t1_a:UInt32;N, t1_c:UInt32;N]
                                          Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t2_b:UInt32;N, t1_a:UInt32;N, t1_c:UInt32;N]
                                            TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                                            Cross Join(ComparisonJoin):  [t2_b:UInt32;N, t1_a:UInt32;N, t1_c:UInt32;N]
                                              SubqueryAlias: t2_dscan_5 [t2_b:UInt32;N]
                                                DelimGet: t2.b [t2_b:UInt32;N]
                                              SubqueryAlias: t1_dscan_6 [t1_a:UInt32;N, t1_c:UInt32;N]
                                                DelimGet: t1.a, t1.c [t1_a:UInt32;N, t1_c:UInt32;N]
                                    Cross Join(ComparisonJoin):  [t2_b:UInt32;N, t1_a:UInt32;N, t1_c:UInt32;N]
                                      SubqueryAlias: t2_dscan_3 [t2_b:UInt32;N]
                                        DelimGet: t2.b [t2_b:UInt32;N]
                                      SubqueryAlias: t1_dscan_4 [t1_a:UInt32;N, t1_c:UInt32;N]
                                        DelimGet: t1.a, t1.c [t1_a:UInt32;N, t1_c:UInt32;N]
                    SubqueryAlias: t1_dscan_1 [t1_a:UInt32;N, t1_c:UInt32;N]
                      DelimGet: t1.a, t1.c [t1_a:UInt32;N, t1_c:UInt32;N]
        ");
        Ok(())
    }

    #[test]
    fn decorrelate_full_join_with_both_side_correlated() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;
        let t4 = test_table_scan_with_name("t4")?;
        // to avoid column conflict
        let t4_alias = LogicalPlanBuilder::from(t4.clone())
            .alias("t4.alias")?
            .build()?;
        let t2_join_t4 = LogicalPlanBuilder::from(t2.clone())
            .join(
                t4.clone(),
                JoinType::Inner,
                (vec!["t2.b"], vec!["t4.b"]),
                Some(col("t2.a").eq(out_ref_col(ArrowDataType::UInt32, "t1.a"))),
            )?
            .build()?;
        let t3_join_t4 = LogicalPlanBuilder::from(t3.clone())
            .join(
                t4_alias.clone(),
                JoinType::Inner,
                (vec!["t3.b"], vec!["t4.alias.b"]),
                Some(col("t3.a").eq(out_ref_col(ArrowDataType::UInt32, "t1.a"))),
            )?
            .build()?;

        let sq_level1 = Arc::new(
            LogicalPlanBuilder::from(t2_join_t4)
                .join(
                    t3_join_t4,
                    JoinType::Full,
                    (Vec::<Column>::new(), Vec::<Column>::new()),
                    Some(col("t2.c").eq(out_ref_col(ArrowDataType::UInt32, "t1.a"))),
                )?
                .project(vec![count(col("t2.a"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(
                col("t1.a")
                    .gt(lit(1))
                    .and(in_subquery(col("t1.c"), sq_level1)),
            )?
            .build()?;
        println!("{plan}");

        // Projection: T1.a, T1.b, T1.c [a:UInt32, b:UInt32, c:UInt32]
        //   Filter: T1.a > Int32(1) AND __in_sq_1.output [a:UInt32, b:UInt32, c:UInt32, output:Boolean]
        //     DependentJoin on [T1.a lvl 1, T1.b lvl 1] with expr T1.c IN (<subquery>) depth 1 [a:UInt32, b:UInt32, c:UInt32, output:Boolean]
        //       TableScan: T1 [a:UInt32, b:UInt32, c:UInt32]
        //       Projection: T2.b [b:UInt32]
        //         Inner Join(ComparisonJoin):  Filter: T2.a = outer_ref(T1.a) AND outer_ref(T1.a) > T2.c AND T2.b = Int32(1) AND outer_ref(T1.b) = T2.b AND T2.a = T3.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
        //           TableScan: T2 [a:UInt32, b:UInt32, c:UInt32]
        //           TableScan: T3 [a:UInt32, b:UInt32, c:UInt32]

        //TODO: invariants are not allowing correlated full join
        // assert_decorrelate!(plan, @r"
        // Projection: T1.a, T1.b, T1.c [a:UInt32, b:UInt32, c:UInt32]
        //   Filter: T1.a > Int32(1) AND __in_sq_1.output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1.output:Boolean]
        //     Projection: T1.a, T1.b, T1.c, mark AS __in_sq_1.output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1.output:Boolean]
        //       LeftMark Join(ComparisonJoin):  Filter: T1.c = T2.b AND T1.a IS NOT DISTINCT FROM delim_scan_1.outer_table_a AND T1.b IS NOT DISTINCT FROM delim_scan_1.outer_table_b [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
        //         TableScan: T1 [a:UInt32, b:UInt32, c:UInt32]
        //         Projection: T2.b, outer_table_dscan_1.outer_table_a, outer_table_dscan_1.outer_table_b [b:UInt32, outer_table_a:UInt32;N, outer_table_b:UInt32;N]
        //           Inner Join(ComparisonJoin):  Filter: T2.a = outer_table_dscan_1.outer_table_a AND outer_table_dscan_1.outer_table_a > T2.c AND T2.b = Int32(1) AND outer_table_dscan_1.outer_table_b = T2.b AND T2.a = T3.a [a:UInt32, b:UInt32, c:UInt32, outer_table_a:UInt32;N, outer_table_b:UInt32;N, a:UInt32, b:UInt32, c:UInt32]
        //             Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, outer_table_a:UInt32;N, outer_table_b:UInt32;N]
        //               TableScan: T2 [a:UInt32, b:UInt32, c:UInt32]
        //               SubqueryAlias: outer_table_dscan_1 [outer_table_a:UInt32;N, outer_table_b:UInt32;N]
        //                 DelimGet: T1.a, T1.b [outer_table_a:UInt32;N, outer_table_b:UInt32;N]
        //             TableScan: T3 [a:UInt32, b:UInt32, c:UInt32]
        // ");

        Ok(())
    }

    #[test]
    fn decorrelate_inner_join_left() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;

        let sq_level1 = Arc::new(
            LogicalPlanBuilder::from(t2)
                .join(
                    t3,
                    JoinType::Inner,
                    (Vec::<Column>::new(), Vec::<Column>::new()),
                    Some(
                        col("t2.a")
                            .eq(out_ref_col(ArrowDataType::UInt32, "t1.a"))
                            .and(
                                out_ref_col(ArrowDataType::UInt32, "t1.a")
                                    .gt(col("t2.c")),
                            )
                            .and(col("t2.b").eq(lit(1)))
                            .and(
                                out_ref_col(ArrowDataType::UInt32, "t1.b")
                                    .eq(col("t2.b")),
                            )
                            .and(col("t2.a").eq(col("t3.a"))),
                    ),
                )?
                .project(vec![col("t2.b")])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(
                col("t1.a")
                    .gt(lit(1))
                    .and(in_subquery(col("t1.c"), sq_level1)),
            )?
            .build()?;
        println!("{plan}");

        // Projection: T1.a, T1.b, T1.c [a:UInt32, b:UInt32, c:UInt32]
        //   Filter: T1.a > Int32(1) AND __in_sq_1.output [a:UInt32, b:UInt32, c:UInt32, output:Boolean]
        //     DependentJoin on [T1.a lvl 1, T1.b lvl 1] with expr T1.c IN (<subquery>) depth 1 [a:UInt32, b:UInt32, c:UInt32, output:Boolean]
        //       TableScan: T1 [a:UInt32, b:UInt32, c:UInt32]
        //       Projection: T2.b [b:UInt32]
        //         Inner Join(ComparisonJoin):  Filter: T2.a = outer_ref(T1.a) AND outer_ref(T1.a) > T2.c AND T2.b = Int32(1) AND outer_ref(T1.b) = T2.b AND T2.a = T3.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
        //           TableScan: T2 [a:UInt32, b:UInt32, c:UInt32]
        //           TableScan: T3 [a:UInt32, b:UInt32, c:UInt32]

        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: t1.a > Int32(1) AND __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
            Projection: t1.a, t1.b, t1.c, mark AS __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
              LeftMark Join(ComparisonJoin):  Filter: t1.c = t2.b AND t1.a IS NOT DISTINCT FROM t1_dscan_1.t1_a AND t1.b IS NOT DISTINCT FROM t1_dscan_1.t1_b [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                Projection: t2.b, t1_dscan_1.t1_a, t1_dscan_1.t1_b [b:UInt32, t1_a:UInt32;N, t1_b:UInt32;N]
                  Inner Join(ComparisonJoin):  Filter: t2.a = t1_dscan_1.t1_a AND t1_dscan_1.t1_a > t2.c AND t2.b = Int32(1) AND t1_dscan_1.t1_b = t2.b AND t2.a = t3.a [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_b:UInt32;N, a:UInt32, b:UInt32, c:UInt32]
                    Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, t1_b:UInt32;N]
                      TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
                      SubqueryAlias: t1_dscan_1 [t1_a:UInt32;N, t1_b:UInt32;N]
                        DelimGet: t1.a, t1.b [t1_a:UInt32;N, t1_b:UInt32;N]
                    TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
        ");

        Ok(())
    }

    #[test]
    fn decorrelate_in_subquery_with_sort_limit() -> Result<()> {
        let t1 = test_table_scan_with_name("customers")?;
        let inner_table = test_table_scan_with_name("orders")?;

        let in_subquery_plan = Arc::new(
            LogicalPlanBuilder::from(inner_table)
                .filter(
                    col("orders.a")
                        .eq(out_ref_col(ArrowDataType::UInt32, "customers.a"))
                        .and(col("orders.b").eq(lit(1))), // status = 'completed' simplified as b = 1
                )?
                .sort(vec![col("orders.c").sort(false, true)])? // ORDER BY order_amount DESC
                .limit(0, Some(3))? // LIMIT 3
                .project(vec![col("orders.c")])?
                .build()?,
        );

        // Outer query
        let plan = LogicalPlanBuilder::from(t1.clone())
            .filter(
                col("customers.a")
                    .gt(lit(100))
                    .and(in_subquery(col("customers.a"), in_subquery_plan)),
            )?
            .build()?;

        // Projection: customers.a, customers.b, customers.c
        //       Filter: customers.a > Int32(100) AND __in_sq_1.output
        //         DependentJoin on [customers.a lvl 1] with expr customers.a IN (<subquery>) depth 1
        //           TableScan: customers
        //           Projection: orders.c
        //             Limit: skip=0, fetch=3
        //               Sort: orders.c DESC NULLS FIRST
        //                 Filter: orders.a = outer_ref(customers.a) AND orders.b = Int32(1)
        //                   TableScan: orders

        assert_decorrelate!(plan, @r"
        Projection: customers.a, customers.b, customers.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: customers.a > Int32(100) AND __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
            Projection: customers.a, customers.b, customers.c, mark AS __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
              LeftMark Join(ComparisonJoin):  Filter: customers.a = orders.c AND customers.a IS NOT DISTINCT FROM customers_dscan_1.customers_a [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                TableScan: customers [a:UInt32, b:UInt32, c:UInt32]
                Projection: orders.c, customers_dscan_1.customers_a [c:UInt32, customers_a:UInt32;N]
                  Projection: orders.a, orders.b, orders.c, customers_dscan_1.customers_a [a:UInt32, b:UInt32, c:UInt32, customers_a:UInt32;N]
                    Filter: row_number <= Int64(3) [a:UInt32, b:UInt32, c:UInt32, customers_a:UInt32;N, row_number:UInt64]
                      WindowAggr: windowExpr=[[row_number() PARTITION BY [customers_dscan_1.customers_a] ORDER BY [orders.c DESC NULLS FIRST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW AS row_number]] [a:UInt32, b:UInt32, c:UInt32, customers_a:UInt32;N, row_number:UInt64]
                        Filter: orders.a = customers_dscan_1.customers_a AND orders.b = Int32(1) [a:UInt32, b:UInt32, c:UInt32, customers_a:UInt32;N]
                          Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, customers_a:UInt32;N]
                            TableScan: orders [a:UInt32, b:UInt32, c:UInt32]
                            SubqueryAlias: customers_dscan_1 [customers_a:UInt32;N]
                              DelimGet: customers.a [customers_a:UInt32;N]
        ");

        Ok(())
    }

    #[test]
    fn decorrelate_subquery_with_window_function() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let inner_table = test_table_scan_with_name("inner_table")?;

        // Create a subquery with window function
        let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
            fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
            params: WindowFunctionParams {
                args: vec![],
                partition_by: vec![col("inner_table.b")],
                order_by: vec![col("inner_table.c").sort(false, true)],
                window_frame: WindowFrame::new(Some(false)),
                null_treatment: None,
            },
        }))
        .alias("row_num");

        let subquery = Arc::new(
            LogicalPlanBuilder::from(inner_table)
                .filter(
                    col("inner_table.a").eq(out_ref_col(ArrowDataType::UInt32, "t1.a")),
                )?
                .window(vec![window_expr])?
                .filter(col("row_num").eq(lit(1)))?
                .project(vec![col("inner_table.b")])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(t1)
            .filter(
                col("t1.a")
                    .gt(lit(1))
                    .and(in_subquery(col("t1.c"), subquery)),
            )?
            .build()?;

        // Projection: T1.a, T1.b, T1.c
        //   Filter: T1.a > Int32(1) AND __in_sq_1.output
        //     DependentJoin on [T1.a lvl 1] with expr T1.c IN (<subquery>) depth 1
        //       TableScan: T1
        //       Projection: inner_table.b
        //         Filter: row_num = Int32(1)
        //           WindowAggr: windowExpr=[[row_number() PARTITION BY [inner_table.b] ORDER BY [inner_table.c DESC NULLS FIRST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW AS row_num]]
        //             Filter: inner_table.a = outer_ref(T1.a)
        //               TableScan: inner_table

        assert_decorrelate!(plan, @r"
        Projection: t1.a, t1.b, t1.c [a:UInt32, b:UInt32, c:UInt32]
          Filter: t1.a > Int32(1) AND __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
            Projection: t1.a, t1.b, t1.c, mark AS __in_sq_1_output [a:UInt32, b:UInt32, c:UInt32, __in_sq_1_output:Boolean]
              LeftMark Join(ComparisonJoin):  Filter: t1.c = inner_table.b AND t1.a IS NOT DISTINCT FROM t1_dscan_1.t1_a [a:UInt32, b:UInt32, c:UInt32, mark:Boolean]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                Projection: inner_table.b, t1_dscan_1.t1_a [b:UInt32, t1_a:UInt32;N]
                  Filter: row_num = Int32(1) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, row_num:UInt64]
                    WindowAggr: windowExpr=[[row_number() PARTITION BY [inner_table.b, t1_dscan_1.t1_a] ORDER BY [inner_table.c DESC NULLS FIRST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW AS row_num]] [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N, row_num:UInt64]
                      Filter: inner_table.a = t1_dscan_1.t1_a [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N]
                        Inner Join(DelimJoin):  Filter: Boolean(true) [a:UInt32, b:UInt32, c:UInt32, t1_a:UInt32;N]
                          TableScan: inner_table [a:UInt32, b:UInt32, c:UInt32]
                          SubqueryAlias: t1_dscan_1 [t1_a:UInt32;N]
                            DelimGet: t1.a [t1_a:UInt32;N]
        ");

        Ok(())
    }

    #[test]
    fn decorrelate_scalar_subquery_with_alias_in_select() -> Result<()> {
        // Test case for: SELECT t1_id, (SELECT sum(t2_int) FROM t2 WHERE t2.t2_id = t1.t1_id) as t2_sum from t1

        // Create test tables
        let t1 = test_table_with_columns(
            "t1",
            &[
                ("t1_id", ArrowDataType::UInt32),
                ("t1_name", ArrowDataType::Utf8),
                ("t1_int", ArrowDataType::Int32),
            ],
        )?;

        let t2 = test_table_with_columns(
            "t2",
            &[
                ("t2_id", ArrowDataType::UInt32),
                ("t2_int", ArrowDataType::Int32),
                ("t2_value", ArrowDataType::Utf8),
            ],
        )?;

        // Create the scalar subquery: SELECT sum(t2_int) FROM t2 WHERE t2.t2_id = t1.t1_id
        let scalar_sq = Arc::new(
            LogicalPlanBuilder::from(t2)
                .filter(
                    col("t2.t2_id").eq(out_ref_col(ArrowDataType::UInt32, "t1.t1_id")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![sum(col("t2_int"))])?
                .build()?,
        );

        // Create the main query plan: SELECT t1_id, (subquery) as t2_sum FROM t1
        let plan = LogicalPlanBuilder::from(t1)
            .project(vec![
                col("t1_id"),
                scalar_subquery(scalar_sq).alias("t2_sum"),
            ])?
            .build()?;

        // Projection: t1.t1_id, __scalar_sq_1.output AS t2_sum [t1_id:UInt32, t2_sum:Int64]
        //   DependentJoin on [t1.t1_id lvl 1] with expr (<subquery>) depth 1 [t1_id:UInt32, t1_name:Utf8, t1_int:Int32, output:Int64]
        //     TableScan: t1 [t1_id:UInt32, t1_name:Utf8, t1_int:Int32]
        //     Aggregate: groupBy=[[]], aggr=[[sum(t2.t2_int)]] [sum(t2.t2_int):Int64;N]
        //       Filter: t2.t2_id = outer_ref(t1.t1_id) [t2_id:UInt32, t2_int:Int32, t2_value:Utf8]
        //         TableScan: t2 [t2_id:UInt32, t2_int:Int32, t2_value:Utf8]

        assert_decorrelate!(plan, @r"
        Projection: t1.t1_id, __scalar_sq_1_output AS t2_sum [t1_id:UInt32, t2_sum:Int64]
          Projection: t1.t1_id, t1.t1_name, t1.t1_int, sum(t2.t2_int), t1_dscan_2.t1_t1_id, sum(t2.t2_int) AS __scalar_sq_1_output [t1_id:UInt32, t1_name:Utf8, t1_int:Int32, sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N, __scalar_sq_1_output:Int64;N]
            Left Join(ComparisonJoin):  Filter: t1.t1_id IS NOT DISTINCT FROM t1_dscan_2.t1_t1_id [t1_id:UInt32, t1_name:Utf8, t1_int:Int32, sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
              TableScan: t1 [t1_id:UInt32, t1_name:Utf8, t1_int:Int32]
              Projection: sum(t2.t2_int), t1_dscan_2.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
                Inner Join(DelimJoin):  Filter: t1_dscan_2.t1_t1_id IS NOT DISTINCT FROM t1_dscan_1.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N, t1_t1_id:UInt32;N]
                  Projection: sum(t2.t2_int), t1_dscan_2.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
                    Aggregate: groupBy=[[t1_dscan_2.t1_t1_id]], aggr=[[sum(t2.t2_int)]] [t1_t1_id:UInt32;N, sum(t2.t2_int):Int64;N]
                      Filter: t2.t2_id = t1_dscan_2.t1_t1_id [t2_id:UInt32, t2_int:Int32, t2_value:Utf8, t1_t1_id:UInt32;N]
                        Inner Join(DelimJoin):  Filter: Boolean(true) [t2_id:UInt32, t2_int:Int32, t2_value:Utf8, t1_t1_id:UInt32;N]
                          TableScan: t2 [t2_id:UInt32, t2_int:Int32, t2_value:Utf8]
                          SubqueryAlias: t1_dscan_2 [t1_t1_id:UInt32;N]
                            DelimGet: t1.t1_id [t1_t1_id:UInt32;N]
                  SubqueryAlias: t1_dscan_1 [t1_t1_id:UInt32;N]
                    DelimGet: t1.t1_id [t1_t1_id:UInt32;N]
        ");

        Ok(())
    }

    // TODO: need a mechanic to deduplicate subqueries, similar to CTE
    #[test]
    fn decorrelate_subqueries_without_alias_in_select() -> Result<()> {
        // Create test tables
        let t1 = test_table_with_columns(
            "t1",
            &[
                ("t1_id", ArrowDataType::UInt32),
                ("t1_name", ArrowDataType::Utf8),
                ("t1_int", ArrowDataType::Int32),
            ],
        )?;

        let t2 = test_table_with_columns(
            "t2",
            &[
                ("t2_id", ArrowDataType::UInt32),
                ("t2_int", ArrowDataType::Int32),
                ("t2_value", ArrowDataType::Utf8),
            ],
        )?;

        // Create the scalar subquery: SELECT sum(t2_int) FROM t2 WHERE t2.t2_id = t1.t1_id
        let scalar_sq = Arc::new(
            LogicalPlanBuilder::from(t2)
                .filter(
                    col("t2.t2_id").eq(out_ref_col(ArrowDataType::UInt32, "t1.t1_id")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![sum(col("t2_int"))])?
                .build()?,
        );

        // Create the main query plan: SELECT t1_id, (subquery) as t2_sum FROM t1
        let plan = LogicalPlanBuilder::from(t1)
            .project(vec![
                col("t1_id"),
                scalar_subquery(scalar_sq.clone()).alias("t2_sum"),
            ])?
            .project(vec![
                col("t1_id"),
                in_subquery(col("t1_id"), scalar_sq.clone()),
            ])?
            .project(vec![col("t1_id"), exists(scalar_sq)])?
            .build()?;
        // dependent join plan
        // Projection: t1.t1_id, __exists_sq_3_output AS EXISTS
        //   DependentJoin on [t1.t1_id lvl 1] with expr EXISTS (<subquery>) depth 1
        //     Projection: t1.t1_id, __in_sq_2_output
        //       DependentJoin on [t1.t1_id lvl 2] with expr t1.t1_id IN (<subquery>) depth 2
        //         Projection: t1.t1_id, __scalar_sq_1_output AS t2_sum
        //           DependentJoin on [t1.t1_id lvl 3] with expr (<subquery>) depth 3
        //             TableScan: t1
        //             Aggregate: groupBy=[[]], aggr=[[sum(t2.t2_int)]]
        //               Filter: t2.t2_id = outer_ref(t1.t1_id)
        //                 TableScan: t2
        //         Aggregate: groupBy=[[]], aggr=[[sum(t2.t2_int)]]
        //           Filter: t2.t2_id = outer_ref(t1.t1_id)
        //             TableScan: t2
        //     Aggregate: groupBy=[[]], aggr=[[sum(t2.t2_int)]]
        //       Filter: t2.t2_id = outer_ref(t1.t1_id)
        //         TableScan: t2

        assert_decorrelate!(plan, @r"
        Projection: t1.t1_id, __exists_sq_3_output AS EXISTS [t1_id:UInt32, EXISTS:Boolean]
          Projection: t1.t1_id, __in_sq_2_output, t1_dscan_6.mark AS __exists_sq_3_output [t1_id:UInt32, __in_sq_2_output:Boolean, __exists_sq_3_output:Boolean]
            LeftMark Join(ComparisonJoin):  Filter: t1.t1_id IS NOT DISTINCT FROM t1_dscan_6.t1_t1_id [t1_id:UInt32, __in_sq_2_output:Boolean, mark:Boolean]
              Projection: t1.t1_id, __in_sq_2_output [t1_id:UInt32, __in_sq_2_output:Boolean]
                Projection: t1.t1_id, t2_sum, t1_dscan_4.mark AS __in_sq_2_output [t1_id:UInt32, t2_sum:Int64, __in_sq_2_output:Boolean]
                  LeftMark Join(ComparisonJoin):  Filter: t1.t1_id = sum(t2.t2_int) AND t1.t1_id IS NOT DISTINCT FROM t1_dscan_4.t1_t1_id [t1_id:UInt32, t2_sum:Int64, mark:Boolean]
                    Projection: t1.t1_id, __scalar_sq_1_output AS t2_sum [t1_id:UInt32, t2_sum:Int64]
                      Projection: t1.t1_id, t1.t1_name, t1.t1_int, sum(t2.t2_int), t1_dscan_2.t1_t1_id, sum(t2.t2_int) AS __scalar_sq_1_output [t1_id:UInt32, t1_name:Utf8, t1_int:Int32, sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N, __scalar_sq_1_output:Int64;N]
                        Left Join(ComparisonJoin):  Filter: t1.t1_id IS NOT DISTINCT FROM t1_dscan_2.t1_t1_id [t1_id:UInt32, t1_name:Utf8, t1_int:Int32, sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
                          TableScan: t1 [t1_id:UInt32, t1_name:Utf8, t1_int:Int32]
                          Projection: sum(t2.t2_int), t1_dscan_2.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
                            Inner Join(DelimJoin):  Filter: t1_dscan_2.t1_t1_id IS NOT DISTINCT FROM t1_dscan_1.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N, t1_t1_id:UInt32;N]
                              Projection: sum(t2.t2_int), t1_dscan_2.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
                                Aggregate: groupBy=[[t1_dscan_2.t1_t1_id]], aggr=[[sum(t2.t2_int)]] [t1_t1_id:UInt32;N, sum(t2.t2_int):Int64;N]
                                  Filter: t2.t2_id = t1_dscan_2.t1_t1_id [t2_id:UInt32, t2_int:Int32, t2_value:Utf8, t1_t1_id:UInt32;N]
                                    Inner Join(DelimJoin):  Filter: Boolean(true) [t2_id:UInt32, t2_int:Int32, t2_value:Utf8, t1_t1_id:UInt32;N]
                                      TableScan: t2 [t2_id:UInt32, t2_int:Int32, t2_value:Utf8]
                                      SubqueryAlias: t1_dscan_2 [t1_t1_id:UInt32;N]
                                        DelimGet: t1.t1_id [t1_t1_id:UInt32;N]
                              SubqueryAlias: t1_dscan_1 [t1_t1_id:UInt32;N]
                                DelimGet: t1.t1_id [t1_t1_id:UInt32;N]
                    Projection: sum(t2.t2_int), t1_dscan_4.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
                      Inner Join(DelimJoin):  Filter: t1_dscan_4.t1_t1_id IS NOT DISTINCT FROM t1_dscan_3.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N, t1_t1_id:UInt32;N]
                        Projection: sum(t2.t2_int), t1_dscan_4.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
                          Aggregate: groupBy=[[t1_dscan_4.t1_t1_id]], aggr=[[sum(t2.t2_int)]] [t1_t1_id:UInt32;N, sum(t2.t2_int):Int64;N]
                            Filter: t2.t2_id = t1_dscan_4.t1_t1_id [t2_id:UInt32, t2_int:Int32, t2_value:Utf8, t1_t1_id:UInt32;N]
                              Inner Join(DelimJoin):  Filter: Boolean(true) [t2_id:UInt32, t2_int:Int32, t2_value:Utf8, t1_t1_id:UInt32;N]
                                TableScan: t2 [t2_id:UInt32, t2_int:Int32, t2_value:Utf8]
                                SubqueryAlias: t1_dscan_4 [t1_t1_id:UInt32;N]
                                  DelimGet: t1.t1_id [t1_t1_id:UInt32;N]
                        SubqueryAlias: t1_dscan_3 [t1_t1_id:UInt32;N]
                          DelimGet: t1.t1_id [t1_t1_id:UInt32;N]
              Projection: sum(t2.t2_int), t1_dscan_6.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
                Inner Join(DelimJoin):  Filter: t1_dscan_6.t1_t1_id IS NOT DISTINCT FROM t1_dscan_5.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N, t1_t1_id:UInt32;N]
                  Projection: sum(t2.t2_int), t1_dscan_6.t1_t1_id [sum(t2.t2_int):Int64;N, t1_t1_id:UInt32;N]
                    Aggregate: groupBy=[[t1_dscan_6.t1_t1_id]], aggr=[[sum(t2.t2_int)]] [t1_t1_id:UInt32;N, sum(t2.t2_int):Int64;N]
                      Filter: t2.t2_id = t1_dscan_6.t1_t1_id [t2_id:UInt32, t2_int:Int32, t2_value:Utf8, t1_t1_id:UInt32;N]
                        Inner Join(DelimJoin):  Filter: Boolean(true) [t2_id:UInt32, t2_int:Int32, t2_value:Utf8, t1_t1_id:UInt32;N]
                          TableScan: t2 [t2_id:UInt32, t2_int:Int32, t2_value:Utf8]
                          SubqueryAlias: t1_dscan_6 [t1_t1_id:UInt32;N]
                            DelimGet: t1.t1_id [t1_t1_id:UInt32;N]
                  SubqueryAlias: t1_dscan_5 [t1_t1_id:UInt32;N]
                    DelimGet: t1.t1_id [t1_t1_id:UInt32;N]
        ");

        Ok(())
    }
}
