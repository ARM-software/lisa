/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{collections::BTreeMap, sync::Arc, vec::Vec};
use core::cell::{Cell, OnceCell};

use itertools::Itertools;

use crate::error::AllErrors;

pub struct Cursor<'a, T> {
    graph: &'a Graph<T>,
    idx: usize,
}

#[derive(Clone, Copy)]
enum LinkType {
    Child,
    Parent,
}

impl LinkType {
    #[inline]
    fn from_direction(direction: TraversalDirection) -> Self {
        match direction {
            TraversalDirection::FromRoots => LinkType::Child,
            TraversalDirection::FromLeaves => LinkType::Parent,
        }
    }
}

impl<'a, T> Cursor<'a, T> {
    pub fn value(&self) -> &'a T {
        &self.graph.nodes[self.idx]
    }

    pub fn children(&self) -> impl Iterator<Item = Self> {
        self.linked(LinkType::Child)
    }

    pub fn parents(&self) -> impl Iterator<Item = Self> {
        self.linked(LinkType::Parent)
    }

    #[inline]
    fn linked(&self, typ: LinkType) -> impl Iterator<Item = Self> {
        let graph = self.graph;
        graph
            .links
            .linked(typ, self.idx)
            .map(|idx| graph.cursor(idx))
    }
}

#[derive(Clone, Copy)]
pub enum TraversalDirection {
    FromRoots,
    FromLeaves,
}

pub struct MapState<T, U> {
    // Allow stealing the old value to compute new
    old: Cell<Option<T>>,
    // Allows sharing &U once initialized, unlike what would happen if it was behind a Cell
    // or RefCell
    new: OnceCell<U>,
}

pub trait DfsTraversal<T, U> {
    fn visit(&mut self, cursor: Cursor<'_, MapState<T, U>>, old: T);

    fn direction(&self) -> TraversalDirection;

    #[inline]
    fn visit_linked(&mut self, cursor: &Cursor<MapState<T, U>>) {
        let link_type = LinkType::from_direction(self.direction());
        for linked in cursor.linked(link_type) {
            self._visit(linked);
        }
    }

    #[inline]
    fn _visit(&mut self, cursor: Cursor<'_, MapState<T, U>>) {
        let state = cursor.value();
        let old = state.old.take();
        // If None, that means we already visited that node or we are currently visiting it.
        if let Some(old) = old {
            self.visit(cursor, old);
        }
    }

    fn traverse(&mut self, graph: Graph<T>) -> Graph<U> {
        let graph = graph.map(|x| MapState {
            old: Cell::new(Some(x)),
            new: OnceCell::new(),
        });

        {
            let nodes: &mut dyn Iterator<Item = _> = match self.direction() {
                TraversalDirection::FromRoots => &mut graph.roots(),
                TraversalDirection::FromLeaves => &mut graph.leaves(),
            };
            for root in nodes {
                self._visit(root);
            }
        }

        // We can unwrap() here as we processed the entire tree.
        graph.map(|value| value.new.into_inner().unwrap())
    }
}

pub struct DfsPreTraversal<F> {
    f: F,
    direction: TraversalDirection,
}

impl<F> DfsPreTraversal<F> {
    pub fn new(direction: TraversalDirection, f: F) -> Self {
        DfsPreTraversal { direction, f }
    }
}

impl<T, U, F> DfsTraversal<T, U> for DfsPreTraversal<F>
where
    F: FnMut(T) -> U,
{
    #[inline]
    fn visit(&mut self, cursor: Cursor<'_, MapState<T, U>>, old: T) {
        cursor.value().new.get_or_init(|| (self.f)(old));
        self.visit_linked(&cursor);
    }

    #[inline]
    fn direction(&self) -> TraversalDirection {
        self.direction
    }
}

pub struct DfsPostTraversal<F> {
    f: F,
    direction: TraversalDirection,
}

impl<F> DfsPostTraversal<F> {
    pub fn new(direction: TraversalDirection, f: F) -> Self {
        DfsPostTraversal { direction, f }
    }
}

impl<T, U, F> DfsTraversal<T, U> for DfsPostTraversal<F>
where
    F: FnMut(T, &mut dyn Iterator<Item = &U>) -> U,
{
    #[inline]
    fn visit(&mut self, cursor: Cursor<'_, MapState<T, U>>, old: T) {
        self.visit_linked(&cursor);
        cursor.value().new.get_or_init(|| {
            let mut linked = cursor
                .linked(LinkType::from_direction(self.direction))
                // We can unwrap() here as we have visited the linked nodes
                // already. If the value is still None, that could only be because
                // we have a loop in the graph, which is a bug.
                .map(|linked| linked.value().new.get().unwrap());
            (self.f)(old, &mut linked)
        });
    }

    #[inline]
    fn direction(&self) -> TraversalDirection {
        self.direction
    }
}

struct Links {
    children: Vec<Vec<usize>>,
    parents: Vec<Vec<usize>>,
}

impl Links {
    pub fn roots(&self) -> impl Iterator<Item = usize> {
        self.parents
            .iter()
            .enumerate()
            .filter_map(
                |(idx, parents)| {
                    if parents.is_empty() { Some(idx) } else { None }
                },
            )
    }

    pub fn leaves(&self) -> impl Iterator<Item = usize> {
        self.children
            .iter()
            .enumerate()
            .filter_map(
                |(idx, children)| {
                    if children.is_empty() { Some(idx) } else { None }
                },
            )
    }

    #[inline]
    fn linked(&self, typ: LinkType, idx: usize) -> impl Iterator<Item = usize> {
        match typ {
            LinkType::Child => &self.children[idx],
            LinkType::Parent => &self.parents[idx],
        }
        .iter()
        .copied()
    }
}

#[derive(Clone)]
pub struct Graph<T> {
    nodes: Vec<T>,
    links: Arc<Links>,
}

impl<T> Graph<T> {
    pub fn new<Spec, NodeId, ChildrenId>(nodes: Spec) -> Self
    where
        NodeId: Ord,
        Spec: IntoIterator<Item = (NodeId, ChildrenId, T)>,
        ChildrenId: IntoIterator<Item = NodeId>,
    {
        let mut idx_map = BTreeMap::new();
        let mut nodes_value = Vec::new();
        let mut nodes_children_id = Vec::new();

        for (idx, (id, children_id, value)) in nodes.into_iter().enumerate() {
            idx_map.insert(id, idx);
            nodes_children_id.push(children_id);
            nodes_value.push(value);
        }

        let mut children_idx = Vec::new();
        let mut parents_idx = Vec::new();
        children_idx.resize(nodes_value.len(), Vec::new());
        parents_idx.resize(nodes_value.len(), Vec::new());

        for (node_idx, (node_children_id, node_children_idx)) in nodes_children_id
            .into_iter()
            .zip(&mut children_idx)
            .enumerate()
        {
            for child_id in node_children_id {
                match idx_map.get(&child_id) {
                    Some(child_idx) => {
                        let child_idx = *child_idx;
                        node_children_idx.push(child_idx);
                        parents_idx[child_idx].push(node_idx);
                    }
                    None => panic!("Could not find child node for parent node {node_idx}"),
                }
            }
        }

        Graph {
            nodes: nodes_value,
            links: Arc::new(Links {
                parents: parents_idx,
                children: children_idx,
            }),
        }
    }

    #[inline]
    pub fn map<U, F>(self, f: F) -> Graph<U>
    where
        F: FnMut(T) -> U,
    {
        Graph {
            links: self.links,
            nodes: self.nodes.into_iter().map(f).collect(),
        }
    }

    #[inline]
    pub fn map_ref<U, F>(&self, f: F) -> Graph<U>
    where
        F: FnMut(&T) -> U,
    {
        Graph {
            links: self.links.clone(),
            nodes: self.nodes.iter().map(f).collect(),
        }
    }

    pub fn dfs_map<U, Traversal>(self, mut traversal: Traversal) -> Graph<U>
    where
        Traversal: DfsTraversal<T, U>,
    {
        traversal.traverse(self)
    }

    fn cursor(&self, idx: usize) -> Cursor<'_, T> {
        Cursor { graph: self, idx }
    }

    pub fn roots(&self) -> impl Iterator<Item = Cursor<'_, T>> {
        self.links.roots().map(|idx| self.cursor(idx))
    }

    pub fn leaves(&self) -> impl Iterator<Item = Cursor<'_, T>> {
        self.links.leaves().map(|idx| self.cursor(idx))
    }

    // FIXME: remove it if implementation is not finished
    pub fn children_of<F>(self, filter: F) -> Self
    where
        F: Fn(&T) -> bool,
    {
        let select = |value, parents: &mut dyn Iterator<Item = &(bool, _)>| {
            let selected = parents
                .collect::<Vec<_>>()
                .into_iter()
                .any(|(selected, _)| *selected)
                || filter(&value);
            (selected, value)
        };

        let graph = self.dfs_map(DfsPostTraversal::new(
            TraversalDirection::FromLeaves,
            select,
        ));

        let filter_links = |links: &Vec<Vec<usize>>| {
            links
                .iter()
                .map(|nodes_idx| {
                    nodes_idx
                        .iter()
                        .copied()
                        .filter(|idx: &usize| {
                            let (selected, _) = &graph.nodes[*idx];
                            *selected
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };

        let parents_idx = filter_links(&graph.links.parents);
        let children_idx = filter_links(&graph.links.children);

        let links = Arc::new(Links {
            parents: parents_idx,
            children: children_idx,
        });

        // graph.nodes.into_iter().map(|(selected, value)|).collect(),
        // for node in graph.nodes.into_iter() {

        // }

        // Graph {
        // links: self.links,
        // nodes: self.nodes.into_iter().map(f).collect(),
        // }

        // for node in self.

        // for node in self.roots() {
        // if filter(node.value())
        // }
        todo!()
    }
}

impl<T, E> From<Graph<Result<T, E>>> for Result<Graph<T>, AllErrors<E>> {
    fn from(graph: Graph<Result<T, E>>) -> Result<Graph<T>, AllErrors<E>> {
        let (err, ok): (Vec<E>, Vec<T>) = graph.nodes.into_iter().partition_map(Into::into);
        if err.is_empty() {
            Ok(Graph {
                nodes: ok,
                links: graph.links,
            })
        } else {
            Err(err.into_iter().collect())
        }
    }
}
