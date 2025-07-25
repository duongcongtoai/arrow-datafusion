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

// This code is based on the DuckDB’s implementation:
// <https://github.com/duckdb/duckdb/blob/main/src/include/duckdb/common/render_tree.hpp>

//! This module provides functionality for rendering an execution plan as a tree structure.
//! It helps in visualizing how different operations in a query are connected and organized.

use core::fmt;
use std::{cmp, fmt::Formatter, sync::Arc};

pub use datafusion_common::display::{RenderTree, RenderTreeNode};
use datafusion_common::HashMap;

use crate::{DisplayFormatType, ExecutionPlan};

pub(crate) fn tree_render<'a>(
    plan: &dyn ExecutionPlan,
    max_width: usize,
) -> impl fmt::Display + 'a {
    let (width, height) = get_tree_width_height(plan);
    let mut result = RenderTree::new(width, height, max_width);
    create_tree_recursive(&mut result, plan, 0, 0);
    result
}

/// Calculates the required dimensions of the tree.
/// This ensures we allocate enough space for the entire tree structure.
///
/// # Arguments
/// * `plan` - The execution plan to measure
///
/// # Returns
/// * A tuple of (width, height) representing the dimensions needed for the tree
fn get_tree_width_height(plan: &dyn ExecutionPlan) -> (usize, usize) {
    let children = plan.children();

    // Leaf nodes take up 1x1 space
    if children.is_empty() {
        return (1, 1);
    }

    let mut width = 0;
    let mut height = 0;

    for child in children {
        let (child_width, child_height) = get_tree_width_height(child.as_ref());
        width += child_width;
        height = cmp::max(height, child_height);
    }

    height += 1;

    (width, height)
}

fn fmt_display(plan: &dyn ExecutionPlan) -> impl fmt::Display + '_ {
    struct Wrapper<'a> {
        plan: &'a dyn ExecutionPlan,
    }

    impl fmt::Display for Wrapper<'_> {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            self.plan.fmt_as(DisplayFormatType::TreeRender, f)?;
            Ok(())
        }
    }

    Wrapper { plan }
}

/// Recursively builds the render tree structure.
/// Traverses the execution plan and creates corresponding render nodes while
/// maintaining proper positioning and parent-child relationships.
///
/// # Arguments
/// * `result` - The render tree being constructed
/// * `plan` - Current execution plan node being processed
/// * `x` - Horizontal position in the tree
/// * `y` - Vertical position in the tree
///
/// # Returns
/// * The width of the subtree rooted at the current node
fn create_tree_recursive(
    result: &mut RenderTree,
    plan: &dyn ExecutionPlan,
    x: usize,
    y: usize,
) -> usize {
    let display_info = fmt_display(plan).to_string();
    let mut extra_info = HashMap::new();

    // Parse the key-value pairs from the formatted string.
    // See DisplayFormatType::TreeRender for details
    for line in display_info.lines() {
        if let Some((key, value)) = line.split_once('=') {
            extra_info.insert(key.to_string(), value.to_string());
        } else {
            extra_info.insert(line.to_string(), "".to_string());
        }
    }

    let mut node = RenderTreeNode::new(plan.name().to_string(), extra_info);

    let children = plan.children();

    if children.is_empty() {
        result.set_node(x, y, Arc::new(node));
        return 1;
    }

    let mut width = 0;
    for child in children {
        let child_x = x + width;
        let child_y = y + 1;
        node.add_child_position();
        width += create_tree_recursive(result, child.as_ref(), child_x, child_y);
    }

    result.set_node(x, y, Arc::new(node));

    width
}
