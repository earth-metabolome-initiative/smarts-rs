use alloc::{boxed::Box, collections::BTreeMap, vec, vec::Vec};
use core::mem;
use thiserror::Error;

use crate::{
    query::{
        AtomExpr, AtomId, AtomPrimitive, BondExpr, BondExprTree, BondId, BondPrimitive,
        BracketExpr, BracketExprTree, ComponentGroupId, ComponentId, QueryAtom, QueryBond,
        QueryMol,
    },
    QueryValidationError,
};

/// One path segment inside a bracket or bond expression tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprPathSegment {
    /// Descend through a unary negation node.
    Not,
    /// Descend into one child of an n-ary boolean node.
    Child(usize),
}

/// Stable path addressing one node inside a boolean expression tree.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExprPath(pub Vec<ExprPathSegment>);

/// Structured error for editable SMARTS graph operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EditError {
    /// One referenced atom id does not exist in the current graph.
    #[error("atom id {0} is out of range")]
    InvalidAtomId(AtomId),
    /// One referenced bond id does not exist in the current graph.
    #[error("bond id {0} is out of range")]
    InvalidBondId(BondId),
    /// One referenced component id does not exist in the current graph.
    #[error("component id {0} is out of range")]
    InvalidComponentId(ComponentId),
    /// A requested bond would connect two different disconnected components.
    #[error("bond endpoints must stay within one component")]
    CrossComponentBond,
    /// One requested leaf-only edit targeted a non-leaf atom.
    #[error("atom id {0} is not a leaf")]
    NotLeafAtom(AtomId),
    /// One requested subgraph or fragment selection was empty.
    #[error("subgraph selection must contain at least one atom")]
    EmptySubgraph,
    /// One requested fragment root does not exist in the source fragment.
    #[error("fragment root atom id {0} is out of range")]
    InvalidFragmentRoot(AtomId),
    /// One requested subtree replacement edge is not present.
    #[error("atom ids {0} and {1} are not directly bonded")]
    NotAdjacentAtoms(AtomId, AtomId),
    /// One requested ring-closing edit would create an illegal self-loop.
    #[error("ring edits cannot create a self-loop at atom id {0}")]
    SelfLoopBond(AtomId),
    /// One requested ring-closing edit would duplicate an existing edge.
    #[error("atoms {0} and {1} are already directly bonded")]
    BondAlreadyExists(AtomId, AtomId),
    /// One requested ring-opening edit targeted a non-cycle bond.
    #[error("bond id {0} is not currently part of a cycle")]
    NotCycleBond(BondId),
    /// One expression path does not resolve inside the current tree.
    #[error("expression path does not resolve inside the current tree")]
    InvalidExprPath,
    /// One expression edit expected to target a primitive node.
    #[error("expression path must target a primitive node")]
    ExpectedPrimitive,
    /// One expression edit would remove the whole expression tree.
    #[error("cannot remove the root expression node")]
    CannotRemoveRootExpression,
    /// One expression edit would leave an empty boolean operator.
    #[error("cannot remove the last child from a boolean expression node")]
    CannotRemoveLastExpression,
    /// One expression removal path must end at a direct child of an n-ary boolean node.
    #[error("expression removal requires an n-ary parent path ending in Child(index)")]
    UnsupportedExprRemovalPath,
    /// One normalization step produced an illegal empty expression.
    #[error("boolean expression cannot be empty after normalization")]
    EmptyExpressionTree,
    /// The edited query failed structural validation before commit.
    #[error("edited query is structurally invalid: {0}")]
    Validation(#[from] QueryValidationError),
}

/// Editable wrapper around a parsed SMARTS query graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditableQueryMol {
    mol: QueryMol,
}

impl QueryMol {
    /// Starts an editable session for the current query graph.
    #[must_use]
    pub fn edit(&self) -> EditableQueryMol {
        EditableQueryMol { mol: self.clone() }
    }

    /// Returns the leaf-like atoms in the query graph.
    ///
    /// Isolated atoms are treated as leaves.
    #[must_use]
    pub fn leaf_atoms(&self) -> Vec<AtomId> {
        self.atoms()
            .iter()
            .filter_map(|atom| (self.degree(atom.id) <= 1).then_some(atom.id))
            .collect()
    }

    /// Returns the branch-point atoms in the query graph.
    #[must_use]
    pub fn branch_atoms(&self) -> Vec<AtomId> {
        self.atoms()
            .iter()
            .filter_map(|atom| (self.degree(atom.id) >= 3).then_some(atom.id))
            .collect()
    }

    /// Returns whether one bond currently lies on a cycle in the query graph.
    #[must_use]
    pub fn is_cycle_edge(&self, bond_id: BondId) -> bool {
        let Some(removed_bond) = self.bond(bond_id) else {
            return false;
        };
        if removed_bond.src == removed_bond.dst {
            return true;
        }

        let mut visited = vec![false; self.atom_count()];
        let mut stack = vec![removed_bond.src];
        visited[removed_bond.src] = true;

        while let Some(current) = stack.pop() {
            for &incident_bond_id in self.incident_bonds(current).iter().rev() {
                if incident_bond_id == bond_id {
                    continue;
                }
                let Some(bond) = self.bond(incident_bond_id) else {
                    continue;
                };
                let Some(next) = other_endpoint(bond, current) else {
                    continue;
                };
                if !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }

        visited[removed_bond.dst]
    }

    /// Returns the bonds that currently lie on one cycle in the query graph.
    #[must_use]
    pub fn ring_bonds(&self) -> Vec<BondId> {
        self.bonds()
            .iter()
            .filter_map(|bond| self.is_cycle_edge(bond.id).then_some(bond.id))
            .collect()
    }

    /// Returns the bonds that are currently outside all cycles in the query graph.
    #[must_use]
    pub fn chain_bonds(&self) -> Vec<BondId> {
        self.bonds()
            .iter()
            .filter_map(|bond| (!self.is_cycle_edge(bond.id)).then_some(bond.id))
            .collect()
    }

    /// Returns the atoms reachable from `root` without traversing `blocked`.
    ///
    /// This is a graph reachability helper, not a tree-only primitive.
    #[must_use]
    pub fn rooted_subtree(&self, root: AtomId, blocked: Option<AtomId>) -> Vec<AtomId> {
        if self.atom(root).is_none() {
            return Vec::new();
        }

        let mut visited = vec![false; self.atom_count()];
        let mut stack = vec![root];
        let mut atoms = Vec::new();
        visited[root] = true;

        while let Some(current) = stack.pop() {
            atoms.push(current);
            for &next in self.neighbors(current).iter().rev() {
                if Some(next) == blocked || visited[next] {
                    continue;
                }
                visited[next] = true;
                stack.push(next);
            }
        }

        atoms.sort_unstable();
        atoms
    }

    /// Clones one induced subgraph and reindexes it into a standalone query.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::EmptySubgraph`] if `atom_ids` is empty or
    /// [`EditError::InvalidAtomId`] if any requested atom does not exist.
    pub fn clone_subgraph(&self, atom_ids: &[AtomId]) -> Result<Self, EditError> {
        if atom_ids.is_empty() {
            return Err(EditError::EmptySubgraph);
        }

        let mut selected = vec![false; self.atom_count()];
        for &atom_id in atom_ids {
            if self.atom(atom_id).is_none() {
                return Err(EditError::InvalidAtomId(atom_id));
            }
            selected[atom_id] = true;
        }

        let (atoms, bonds, _component_count, component_groups) = self.cloned_parts();
        let kept_atoms = atoms
            .into_iter()
            .filter(|atom| selected[atom.id])
            .collect::<Vec<_>>();
        let kept_bonds = bonds
            .into_iter()
            .filter(|bond| selected[bond.src] && selected[bond.dst])
            .collect::<Vec<_>>();

        Ok(rebuild_query_mol(kept_atoms, kept_bonds, &component_groups))
    }
}

impl BracketExpr {
    /// Normalizes the bracket expression tree in place.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::EmptyExpressionTree`] if normalization would leave
    /// an empty boolean operator.
    pub fn normalize(&mut self) -> Result<(), EditError> {
        normalize_bracket_tree(&mut self.tree)
    }

    /// Returns a normalized copy of the bracket expression.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::EmptyExpressionTree`] if normalization would leave
    /// an empty boolean operator.
    pub fn normalized(mut self) -> Result<Self, EditError> {
        self.normalize()?;
        Ok(self)
    }
}

impl BracketExprTree {
    /// Resolves one path inside the bracket-expression tree.
    #[must_use]
    pub fn get(&self, path: &ExprPath) -> Option<&Self> {
        let mut current = self;
        for segment in &path.0 {
            current = match (current, segment) {
                (Self::Not(inner), ExprPathSegment::Not) => inner,
                (
                    Self::HighAnd(items) | Self::Or(items) | Self::LowAnd(items),
                    ExprPathSegment::Child(index),
                ) => items.get(*index)?,
                _ => return None,
            };
        }
        Some(current)
    }

    /// Resolves one mutable path inside the bracket-expression tree.
    #[must_use]
    pub fn get_mut(&mut self, path: &ExprPath) -> Option<&mut Self> {
        let mut current = self;
        for segment in &path.0 {
            current = match (current, segment) {
                (Self::Not(inner), ExprPathSegment::Not) => inner,
                (
                    Self::HighAnd(items) | Self::Or(items) | Self::LowAnd(items),
                    ExprPathSegment::Child(index),
                ) => items.get_mut(*index)?,
                _ => return None,
            };
        }
        Some(current)
    }

    /// Enumerates all node paths in the bracket-expression tree, including the root.
    #[must_use]
    pub fn enumerate_paths(&self) -> Vec<ExprPath> {
        let mut paths = Vec::new();
        collect_bracket_paths(self, &mut Vec::new(), &mut paths);
        paths
    }
}

impl BondExprTree {
    /// Resolves one path inside the bond-expression tree.
    #[must_use]
    pub fn get(&self, path: &ExprPath) -> Option<&Self> {
        let mut current = self;
        for segment in &path.0 {
            current = match (current, segment) {
                (Self::Not(inner), ExprPathSegment::Not) => inner,
                (
                    Self::HighAnd(items) | Self::Or(items) | Self::LowAnd(items),
                    ExprPathSegment::Child(index),
                ) => items.get(*index)?,
                _ => return None,
            };
        }
        Some(current)
    }

    /// Resolves one mutable path inside the bond-expression tree.
    #[must_use]
    pub fn get_mut(&mut self, path: &ExprPath) -> Option<&mut Self> {
        let mut current = self;
        for segment in &path.0 {
            current = match (current, segment) {
                (Self::Not(inner), ExprPathSegment::Not) => inner,
                (
                    Self::HighAnd(items) | Self::Or(items) | Self::LowAnd(items),
                    ExprPathSegment::Child(index),
                ) => items.get_mut(*index)?,
                _ => return None,
            };
        }
        Some(current)
    }

    /// Enumerates all node paths in the bond-expression tree, including the root.
    #[must_use]
    pub fn enumerate_paths(&self) -> Vec<ExprPath> {
        let mut paths = Vec::new();
        collect_bond_paths(self, &mut Vec::new(), &mut paths);
        paths
    }
}

/// Normalizes one bracket-expression tree in place.
///
/// # Errors
///
/// Returns [`EditError::EmptyExpressionTree`] if normalization would leave
/// an empty boolean operator.
pub fn normalize_bracket_tree(tree: &mut BracketExprTree) -> Result<(), EditError> {
    *tree = normalized_bracket_tree(tree.clone())?;
    Ok(())
}

/// Normalizes one bond-expression tree in place.
///
/// # Errors
///
/// Returns [`EditError::EmptyExpressionTree`] if normalization would leave
/// an empty boolean operator.
pub fn normalize_bond_tree(tree: &mut BondExprTree) -> Result<(), EditError> {
    *tree = normalized_bond_tree(tree.clone())?;
    Ok(())
}

/// Adds one atom primitive to the bracket expression as a high-precedence conjunction.
///
/// # Errors
///
/// Returns any normalization error produced after insertion.
pub fn add_atom_primitive(expr: &mut BracketExpr, prim: AtomPrimitive) -> Result<(), EditError> {
    match &mut expr.tree {
        BracketExprTree::HighAnd(items) => items.push(BracketExprTree::Primitive(prim)),
        _ => {
            expr.tree =
                BracketExprTree::HighAnd(vec![expr.tree.clone(), BracketExprTree::Primitive(prim)]);
        }
    }
    expr.normalize()
}

/// Removes one atom primitive selected by `path`.
///
/// The path must end at a direct child of an n-ary boolean node.
///
/// # Errors
///
/// Returns [`EditError::InvalidExprPath`] if the path does not resolve,
/// [`EditError::ExpectedPrimitive`] if the target is not a primitive, or
/// [`EditError::UnsupportedExprRemovalPath`] if the path does not end at an
/// n-ary parent child.
pub fn remove_atom_primitive(
    expr: &mut BracketExpr,
    path: &ExprPath,
) -> Result<AtomPrimitive, EditError> {
    let removed = remove_bracket_primitive_at_path(&mut expr.tree, path)?;
    expr.normalize()?;
    Ok(removed)
}

/// Replaces one atom primitive selected by `path`.
///
/// # Errors
///
/// Returns [`EditError::InvalidExprPath`] if the path does not resolve or
/// [`EditError::ExpectedPrimitive`] if the target is not a primitive.
pub fn replace_atom_primitive(
    expr: &mut BracketExpr,
    path: &ExprPath,
    prim: AtomPrimitive,
) -> Result<AtomPrimitive, EditError> {
    let node = expr.tree.get_mut(path).ok_or(EditError::InvalidExprPath)?;
    if let BracketExprTree::Primitive(existing) = node {
        Ok(mem::replace(existing, prim))
    } else {
        Err(EditError::ExpectedPrimitive)
    }
}

/// Adds one bond primitive to the bond-expression tree as a high-precedence conjunction.
///
/// # Errors
///
/// Returns any normalization error produced after insertion.
pub fn add_bond_primitive(tree: &mut BondExprTree, prim: BondPrimitive) -> Result<(), EditError> {
    match tree {
        BondExprTree::HighAnd(items) => items.push(BondExprTree::Primitive(prim)),
        _ => {
            *tree = BondExprTree::HighAnd(vec![tree.clone(), BondExprTree::Primitive(prim)]);
        }
    }
    normalize_bond_tree(tree)
}

/// Removes one bond primitive selected by `path`.
///
/// The path must end at a direct child of an n-ary boolean node.
///
/// # Errors
///
/// Returns [`EditError::InvalidExprPath`] if the path does not resolve,
/// [`EditError::ExpectedPrimitive`] if the target is not a primitive, or
/// [`EditError::UnsupportedExprRemovalPath`] if the path does not end at an
/// n-ary parent child.
pub fn remove_bond_primitive(
    tree: &mut BondExprTree,
    path: &ExprPath,
) -> Result<BondPrimitive, EditError> {
    let removed = remove_bond_primitive_at_path(tree, path)?;
    normalize_bond_tree(tree)?;
    Ok(removed)
}

/// Replaces one bond primitive selected by `path`.
///
/// # Errors
///
/// Returns [`EditError::InvalidExprPath`] if the path does not resolve or
/// [`EditError::ExpectedPrimitive`] if the target is not a primitive.
pub fn replace_bond_primitive(
    tree: &mut BondExprTree,
    path: &ExprPath,
    prim: BondPrimitive,
) -> Result<BondPrimitive, EditError> {
    let node = tree.get_mut(path).ok_or(EditError::InvalidExprPath)?;
    if let BondExprTree::Primitive(existing) = node {
        Ok(mem::replace(existing, prim))
    } else {
        Err(EditError::ExpectedPrimitive)
    }
}

impl EditableQueryMol {
    /// Returns the current read-only query view.
    #[must_use]
    pub const fn as_query_mol(&self) -> &QueryMol {
        &self.mol
    }

    /// Returns one atom by dense identifier.
    #[must_use]
    pub fn atom(&self, atom_id: AtomId) -> Option<&QueryAtom> {
        self.mol.atom(atom_id)
    }

    /// Returns one bond by dense identifier.
    #[must_use]
    pub fn bond(&self, bond_id: BondId) -> Option<&QueryBond> {
        self.mol.bond(bond_id)
    }

    /// Adds one atom to an existing or new component.
    ///
    /// Passing `component == component_count` appends a new disconnected component.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidComponentId`] if `component` is larger than the
    /// next appendable component id.
    pub fn add_atom(
        &mut self,
        component: ComponentId,
        expr: AtomExpr,
    ) -> Result<AtomId, EditError> {
        let (mut atoms, bonds, mut component_count, mut component_groups) = self.mol.cloned_parts();
        if component > component_count {
            return Err(EditError::InvalidComponentId(component));
        }
        if component == component_count {
            component_count += 1;
            component_groups.push(None);
        }

        let atom_id = atoms.len();
        atoms.push(QueryAtom {
            id: atom_id,
            component,
            expr,
        });

        self.mol = QueryMol::from_parts(atoms, bonds, component_count, component_groups);
        Ok(atom_id)
    }

    /// Adds one bond between two existing atoms in the same component.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidAtomId`] if either endpoint does not exist,
    /// or [`EditError::CrossComponentBond`] if the bond would connect two
    /// disconnected components.
    pub fn add_bond(
        &mut self,
        src: AtomId,
        dst: AtomId,
        expr: BondExpr,
    ) -> Result<BondId, EditError> {
        let Some(src_atom) = self.mol.atom(src) else {
            return Err(EditError::InvalidAtomId(src));
        };
        let Some(dst_atom) = self.mol.atom(dst) else {
            return Err(EditError::InvalidAtomId(dst));
        };
        if src_atom.component != dst_atom.component {
            return Err(EditError::CrossComponentBond);
        }

        let (atoms, mut bonds, component_count, component_groups) = self.mol.cloned_parts();
        let bond_id = bonds.len();
        bonds.push(QueryBond {
            id: bond_id,
            src,
            dst,
            expr,
        });
        self.mol = QueryMol::from_parts(atoms, bonds, component_count, component_groups);
        Ok(bond_id)
    }

    /// Closes one ring by adding a new cycle edge between two existing atoms.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::SelfLoopBond`] for `a == b`,
    /// [`EditError::BondAlreadyExists`] if the atoms are already directly bonded,
    /// or any error produced by [`Self::add_bond`].
    pub fn close_ring(
        &mut self,
        a: AtomId,
        b: AtomId,
        expr: BondExpr,
    ) -> Result<BondId, EditError> {
        if a == b {
            return Err(EditError::SelfLoopBond(a));
        }
        if self.mol.bond_between(a, b).is_some() {
            return Err(EditError::BondAlreadyExists(a, b));
        }
        self.add_bond(a, b, expr)
    }

    /// Opens one ring by removing a bond that currently lies on a cycle.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidBondId`] if `bond_id` does not exist or
    /// [`EditError::NotCycleBond`] if the selected bond is not on a cycle.
    pub fn open_ring(&mut self, bond_id: BondId) -> Result<(), EditError> {
        if self.mol.bond(bond_id).is_none() {
            return Err(EditError::InvalidBondId(bond_id));
        }
        if !self.mol.is_cycle_edge(bond_id) {
            return Err(EditError::NotCycleBond(bond_id));
        }
        self.remove_bond(bond_id)
    }

    /// Replaces one atom expression in place.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidAtomId`] if `atom_id` does not exist.
    pub fn replace_atom_expr(&mut self, atom_id: AtomId, expr: AtomExpr) -> Result<(), EditError> {
        let (mut atoms, bonds, component_count, component_groups) = self.mol.cloned_parts();
        let Some(atom) = atoms.get_mut(atom_id) else {
            return Err(EditError::InvalidAtomId(atom_id));
        };
        atom.expr = expr;
        self.mol = QueryMol::from_parts(atoms, bonds, component_count, component_groups);
        Ok(())
    }

    /// Replaces one bond expression in place.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidBondId`] if `bond_id` does not exist.
    pub fn replace_bond_expr(&mut self, bond_id: BondId, expr: BondExpr) -> Result<(), EditError> {
        let (atoms, mut bonds, component_count, component_groups) = self.mol.cloned_parts();
        let Some(bond) = bonds.get_mut(bond_id) else {
            return Err(EditError::InvalidBondId(bond_id));
        };
        bond.expr = expr;
        self.mol = QueryMol::from_parts(atoms, bonds, component_count, component_groups);
        Ok(())
    }

    /// Removes one bond and reindexes the query graph.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidBondId`] if `bond_id` does not exist.
    pub fn remove_bond(&mut self, bond_id: BondId) -> Result<(), EditError> {
        let (atoms, bonds, _component_count, component_groups) = self.mol.cloned_parts();
        if bond_id >= bonds.len() {
            return Err(EditError::InvalidBondId(bond_id));
        }

        let kept_bonds = bonds
            .into_iter()
            .enumerate()
            .filter_map(|(index, bond)| (index != bond_id).then_some(bond))
            .collect::<Vec<_>>();
        self.mol = rebuild_query_mol(atoms, kept_bonds, &component_groups);
        Ok(())
    }

    /// Removes one atom and all of its incident bonds, then reindexes the graph.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidAtomId`] if `atom_id` does not exist.
    pub fn remove_atom(&mut self, atom_id: AtomId) -> Result<(), EditError> {
        let (atoms, bonds, _component_count, component_groups) = self.mol.cloned_parts();
        if atom_id >= atoms.len() {
            return Err(EditError::InvalidAtomId(atom_id));
        }

        let kept_atoms = atoms
            .into_iter()
            .enumerate()
            .filter_map(|(index, atom)| (index != atom_id).then_some(atom))
            .collect::<Vec<_>>();
        let kept_bonds = bonds
            .into_iter()
            .filter(|bond| bond.src != atom_id && bond.dst != atom_id)
            .collect::<Vec<_>>();
        self.mol = rebuild_query_mol(kept_atoms, kept_bonds, &component_groups);
        Ok(())
    }

    /// Attaches one leaf atom to an existing parent atom.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidAtomId`] if `parent` does not exist, or any
    /// error produced by the underlying atom and bond insertion steps.
    pub fn attach_leaf(
        &mut self,
        parent: AtomId,
        bond_expr: BondExpr,
        atom_expr: AtomExpr,
    ) -> Result<(BondId, AtomId), EditError> {
        let parent_component = self
            .mol
            .atom(parent)
            .ok_or(EditError::InvalidAtomId(parent))?
            .component;
        let atom_id = self.add_atom(parent_component, atom_expr)?;
        let bond_id = self.add_bond(parent, atom_id, bond_expr)?;
        Ok((bond_id, atom_id))
    }

    /// Inserts one atom in the middle of an existing bond.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidBondId`] if `bond_id` does not exist, or
    /// [`EditError::InvalidAtomId`] if one stored endpoint is somehow invalid.
    pub fn insert_atom_on_bond(
        &mut self,
        bond_id: BondId,
        left_expr: BondExpr,
        atom_expr: AtomExpr,
        right_expr: BondExpr,
    ) -> Result<(BondId, AtomId, BondId), EditError> {
        let bond = self
            .mol
            .bond(bond_id)
            .ok_or(EditError::InvalidBondId(bond_id))?
            .clone();
        let component = self
            .mol
            .atom(bond.src)
            .ok_or(EditError::InvalidAtomId(bond.src))?
            .component;
        let (mut atoms, bonds, component_count, component_groups) = self.mol.cloned_parts();
        let atom_id = atoms.len();
        atoms.push(QueryAtom {
            id: atom_id,
            component,
            expr: atom_expr,
        });

        let mut rebuilt_bonds = Vec::with_capacity(bonds.len() + 1);
        for (index, existing) in bonds.into_iter().enumerate() {
            if index == bond_id {
                continue;
            }
            rebuilt_bonds.push(QueryBond {
                id: rebuilt_bonds.len(),
                src: existing.src,
                dst: existing.dst,
                expr: existing.expr,
            });
        }
        let left_bond = rebuilt_bonds.len();
        rebuilt_bonds.push(QueryBond {
            id: left_bond,
            src: bond.src,
            dst: atom_id,
            expr: left_expr,
        });
        let right_bond = rebuilt_bonds.len();
        rebuilt_bonds.push(QueryBond {
            id: right_bond,
            src: atom_id,
            dst: bond.dst,
            expr: right_expr,
        });

        self.mol = QueryMol::from_parts(atoms, rebuilt_bonds, component_count, component_groups);
        Ok((left_bond, atom_id, right_bond))
    }

    /// Removes one leaf atom.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidAtomId`] if `atom_id` does not exist or
    /// [`EditError::NotLeafAtom`] if the target atom is not a leaf.
    pub fn remove_leaf_atom(&mut self, atom_id: AtomId) -> Result<(), EditError> {
        if self.mol.degree(atom_id) > 1 {
            return Err(EditError::NotLeafAtom(atom_id));
        }
        self.remove_atom(atom_id)
    }

    /// Grafts the connected component containing `fragment_root` onto `parent`.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidAtomId`] if `parent` does not exist,
    /// [`EditError::InvalidFragmentRoot`] if `fragment_root` does not exist in
    /// `fragment`, or any underlying insertion error while grafting the fragment.
    pub fn graft_subgraph(
        &mut self,
        parent: AtomId,
        bond_expr: BondExpr,
        fragment: &QueryMol,
        fragment_root: AtomId,
    ) -> Result<Vec<AtomId>, EditError> {
        let parent_component = self
            .mol
            .atom(parent)
            .ok_or(EditError::InvalidAtomId(parent))?
            .component;
        let fragment_root_atom = fragment
            .atom(fragment_root)
            .ok_or(EditError::InvalidFragmentRoot(fragment_root))?;
        let fragment_atom_ids = fragment.component_atoms(fragment_root_atom.component);

        let mut mapping = BTreeMap::new();
        let mut new_atom_ids = Vec::with_capacity(fragment_atom_ids.len());

        let new_root = self.add_atom(parent_component, fragment_root_atom.expr.clone())?;
        mapping.insert(fragment_root, new_root);
        new_atom_ids.push(new_root);

        for &fragment_atom_id in fragment_atom_ids {
            if fragment_atom_id == fragment_root {
                continue;
            }

            let fragment_atom = fragment
                .atom(fragment_atom_id)
                .ok_or(EditError::InvalidFragmentRoot(fragment_atom_id))?;
            let new_atom_id = self.add_atom(parent_component, fragment_atom.expr.clone())?;
            mapping.insert(fragment_atom_id, new_atom_id);
            new_atom_ids.push(new_atom_id);
        }

        self.add_bond(parent, new_root, bond_expr)?;
        for &fragment_bond_id in fragment.component_bonds(fragment_root_atom.component) {
            let fragment_bond = fragment
                .bond(fragment_bond_id)
                .ok_or(EditError::InvalidBondId(fragment_bond_id))?;
            let src = mapping[&fragment_bond.src];
            let dst = mapping[&fragment_bond.dst];
            self.add_bond(src, dst, fragment_bond.expr.clone())?;
        }

        new_atom_ids.sort_unstable();
        Ok(new_atom_ids)
    }

    /// Replaces the reachable subgraph rooted at `old_child` with `fragment`.
    ///
    /// The replacement preserves the original anchor-to-child bond expression.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidAtomId`] if `anchor` or `old_child` does not
    /// exist, [`EditError::NotAdjacentAtoms`] if the two atoms are not directly
    /// bonded, or any underlying rebuild or grafting error.
    pub fn replace_subtree(
        &mut self,
        anchor: AtomId,
        old_child: AtomId,
        fragment: &QueryMol,
        fragment_root: AtomId,
    ) -> Result<(), EditError> {
        self.mol
            .atom(anchor)
            .ok_or(EditError::InvalidAtomId(anchor))?;
        self.mol
            .atom(old_child)
            .ok_or(EditError::InvalidAtomId(old_child))?;
        let connecting_bond_id = self
            .mol
            .bond_between(anchor, old_child)
            .ok_or(EditError::NotAdjacentAtoms(anchor, old_child))?;
        let connecting_bond_expr = self
            .mol
            .bond(connecting_bond_id)
            .ok_or(EditError::InvalidBondId(connecting_bond_id))?
            .expr
            .clone();

        let removed_atoms = self.mol.rooted_subtree(old_child, Some(anchor));
        let (atoms, bonds, _component_count, component_groups) = self.mol.cloned_parts();
        let mut removed = vec![false; atoms.len()];
        for atom_id in removed_atoms {
            removed[atom_id] = true;
        }

        let kept_atoms = atoms
            .into_iter()
            .filter(|atom| !removed[atom.id])
            .collect::<Vec<_>>();
        let kept_bonds = bonds
            .into_iter()
            .filter(|bond| !removed[bond.src] && !removed[bond.dst])
            .collect::<Vec<_>>();
        let (rebuilt, atom_old_to_new) =
            rebuild_query_mol_with_atom_remap(kept_atoms, kept_bonds, &component_groups);
        let new_anchor = atom_old_to_new
            .get(anchor)
            .copied()
            .flatten()
            .ok_or(EditError::InvalidAtomId(anchor))?;

        self.mol = rebuilt;
        self.graft_subgraph(new_anchor, connecting_bond_expr, fragment, fragment_root)?;
        Ok(())
    }

    /// Validates the current editable graph.
    ///
    /// # Errors
    ///
    /// Returns [`QueryValidationError`] if the current edit state is structurally invalid.
    pub fn validate(&self) -> Result<(), QueryValidationError> {
        self.mol.validate()
    }

    /// Finishes the edit session and returns a validated query graph.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::Validation`] if the edited query is structurally invalid.
    pub fn into_query_mol(self) -> Result<QueryMol, EditError> {
        self.mol.validate()?;
        Ok(self.mol)
    }
}

fn collect_bracket_paths(
    tree: &BracketExprTree,
    current: &mut Vec<ExprPathSegment>,
    paths: &mut Vec<ExprPath>,
) {
    paths.push(ExprPath(current.clone()));
    match tree {
        BracketExprTree::Primitive(_) => {}
        BracketExprTree::Not(inner) => {
            current.push(ExprPathSegment::Not);
            collect_bracket_paths(inner, current, paths);
            current.pop();
        }
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            for (index, item) in items.iter().enumerate() {
                current.push(ExprPathSegment::Child(index));
                collect_bracket_paths(item, current, paths);
                current.pop();
            }
        }
    }
}

fn collect_bond_paths(
    tree: &BondExprTree,
    current: &mut Vec<ExprPathSegment>,
    paths: &mut Vec<ExprPath>,
) {
    paths.push(ExprPath(current.clone()));
    match tree {
        BondExprTree::Primitive(_) => {}
        BondExprTree::Not(inner) => {
            current.push(ExprPathSegment::Not);
            collect_bond_paths(inner, current, paths);
            current.pop();
        }
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            for (index, item) in items.iter().enumerate() {
                current.push(ExprPathSegment::Child(index));
                collect_bond_paths(item, current, paths);
                current.pop();
            }
        }
    }
}

fn normalized_bracket_tree(tree: BracketExprTree) -> Result<BracketExprTree, EditError> {
    match tree {
        BracketExprTree::Primitive(primitive) => Ok(BracketExprTree::Primitive(primitive)),
        BracketExprTree::Not(inner) => {
            let inner = normalized_bracket_tree(*inner)?;
            if let BracketExprTree::Not(grandchild) = inner {
                Ok(*grandchild)
            } else {
                Ok(BracketExprTree::Not(Box::new(inner)))
            }
        }
        BracketExprTree::HighAnd(items) => normalize_bracket_nary(items, BracketNaryKind::HighAnd),
        BracketExprTree::Or(items) => normalize_bracket_nary(items, BracketNaryKind::Or),
        BracketExprTree::LowAnd(items) => normalize_bracket_nary(items, BracketNaryKind::LowAnd),
    }
}

fn normalized_bond_tree(tree: BondExprTree) -> Result<BondExprTree, EditError> {
    match tree {
        BondExprTree::Primitive(primitive) => Ok(BondExprTree::Primitive(primitive)),
        BondExprTree::Not(inner) => {
            let inner = normalized_bond_tree(*inner)?;
            if let BondExprTree::Not(grandchild) = inner {
                Ok(*grandchild)
            } else {
                Ok(BondExprTree::Not(Box::new(inner)))
            }
        }
        BondExprTree::HighAnd(items) => normalize_bond_nary(items, BondNaryKind::HighAnd),
        BondExprTree::Or(items) => normalize_bond_nary(items, BondNaryKind::Or),
        BondExprTree::LowAnd(items) => normalize_bond_nary(items, BondNaryKind::LowAnd),
    }
}

#[derive(Clone, Copy)]
enum BracketNaryKind {
    HighAnd,
    Or,
    LowAnd,
}

#[derive(Clone, Copy)]
enum BondNaryKind {
    HighAnd,
    Or,
    LowAnd,
}

fn normalize_bracket_nary(
    items: Vec<BracketExprTree>,
    kind: BracketNaryKind,
) -> Result<BracketExprTree, EditError> {
    let mut flattened = Vec::new();
    for item in items {
        let item = normalized_bracket_tree(item)?;
        match (kind, item) {
            (BracketNaryKind::HighAnd, BracketExprTree::HighAnd(children))
            | (BracketNaryKind::Or, BracketExprTree::Or(children))
            | (BracketNaryKind::LowAnd, BracketExprTree::LowAnd(children)) => {
                flattened.extend(children);
            }
            (_, child) => flattened.push(child),
        }
    }

    match flattened.len() {
        0 => Err(EditError::EmptyExpressionTree),
        1 => Ok(flattened.pop().expect("one child after len check")),
        _ => Ok(match kind {
            BracketNaryKind::HighAnd => BracketExprTree::HighAnd(flattened),
            BracketNaryKind::Or => BracketExprTree::Or(flattened),
            BracketNaryKind::LowAnd => BracketExprTree::LowAnd(flattened),
        }),
    }
}

fn normalize_bond_nary(
    items: Vec<BondExprTree>,
    kind: BondNaryKind,
) -> Result<BondExprTree, EditError> {
    let mut flattened = Vec::new();
    for item in items {
        let item = normalized_bond_tree(item)?;
        match (kind, item) {
            (BondNaryKind::HighAnd, BondExprTree::HighAnd(children))
            | (BondNaryKind::Or, BondExprTree::Or(children))
            | (BondNaryKind::LowAnd, BondExprTree::LowAnd(children)) => {
                flattened.extend(children);
            }
            (_, child) => flattened.push(child),
        }
    }

    match flattened.len() {
        0 => Err(EditError::EmptyExpressionTree),
        1 => Ok(flattened.pop().expect("one child after len check")),
        _ => Ok(match kind {
            BondNaryKind::HighAnd => BondExprTree::HighAnd(flattened),
            BondNaryKind::Or => BondExprTree::Or(flattened),
            BondNaryKind::LowAnd => BondExprTree::LowAnd(flattened),
        }),
    }
}

fn remove_bracket_primitive_at_path(
    tree: &mut BracketExprTree,
    path: &ExprPath,
) -> Result<AtomPrimitive, EditError> {
    let (last, parent_path) = split_parent_path(path)?;
    let parent = tree
        .get_mut(&parent_path)
        .ok_or(EditError::InvalidExprPath)?;
    match (parent, last) {
        (
            BracketExprTree::HighAnd(items)
            | BracketExprTree::Or(items)
            | BracketExprTree::LowAnd(items),
            ExprPathSegment::Child(index),
        ) => remove_bracket_child(items, index),
        _ => Err(EditError::UnsupportedExprRemovalPath),
    }
}

fn remove_bond_primitive_at_path(
    tree: &mut BondExprTree,
    path: &ExprPath,
) -> Result<BondPrimitive, EditError> {
    let (last, parent_path) = split_parent_path(path)?;
    let parent = tree
        .get_mut(&parent_path)
        .ok_or(EditError::InvalidExprPath)?;
    match (parent, last) {
        (
            BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items),
            ExprPathSegment::Child(index),
        ) => remove_bond_child(items, index),
        _ => Err(EditError::UnsupportedExprRemovalPath),
    }
}

fn split_parent_path(path: &ExprPath) -> Result<(ExprPathSegment, ExprPath), EditError> {
    let Some((last, parent)) = path.0.split_last() else {
        return Err(EditError::CannotRemoveRootExpression);
    };
    Ok((*last, ExprPath(parent.to_vec())))
}

fn remove_bracket_child(
    items: &mut Vec<BracketExprTree>,
    index: usize,
) -> Result<AtomPrimitive, EditError> {
    let Some(item) = items.get(index) else {
        return Err(EditError::InvalidExprPath);
    };
    if items.len() == 1 {
        return Err(EditError::CannotRemoveLastExpression);
    }
    if !matches!(item, BracketExprTree::Primitive(_)) {
        return Err(EditError::ExpectedPrimitive);
    }
    let removed = items.remove(index);
    if let BracketExprTree::Primitive(primitive) = removed {
        Ok(primitive)
    } else {
        Err(EditError::ExpectedPrimitive)
    }
}

fn remove_bond_child(
    items: &mut Vec<BondExprTree>,
    index: usize,
) -> Result<BondPrimitive, EditError> {
    let Some(item) = items.get(index) else {
        return Err(EditError::InvalidExprPath);
    };
    if items.len() == 1 {
        return Err(EditError::CannotRemoveLastExpression);
    }
    if !matches!(item, BondExprTree::Primitive(_)) {
        return Err(EditError::ExpectedPrimitive);
    }
    let removed = items.remove(index);
    if let BondExprTree::Primitive(primitive) = removed {
        Ok(primitive)
    } else {
        Err(EditError::ExpectedPrimitive)
    }
}

const fn other_endpoint(bond: &QueryBond, atom_id: AtomId) -> Option<AtomId> {
    if bond.src == atom_id {
        Some(bond.dst)
    } else if bond.dst == atom_id {
        Some(bond.src)
    } else {
        None
    }
}

fn rebuild_query_mol(
    old_atoms: Vec<QueryAtom>,
    old_bonds: Vec<QueryBond>,
    old_component_groups: &[Option<ComponentGroupId>],
) -> QueryMol {
    rebuild_query_mol_with_atom_remap(old_atoms, old_bonds, old_component_groups).0
}

fn rebuild_query_mol_with_atom_remap(
    old_atoms: Vec<QueryAtom>,
    old_bonds: Vec<QueryBond>,
    old_component_groups: &[Option<ComponentGroupId>],
) -> (QueryMol, Vec<Option<AtomId>>) {
    let atom_map_len = old_atoms
        .iter()
        .map(|atom| atom.id)
        .max()
        .map_or(0, |max_atom_id| max_atom_id + 1);
    let mut atom_old_to_new = vec![None; atom_map_len];
    let mut old_component_by_new_atom = Vec::with_capacity(old_atoms.len());
    let mut atoms = Vec::with_capacity(old_atoms.len());
    for (new_id, atom) in old_atoms.into_iter().enumerate() {
        atom_old_to_new[atom.id] = Some(new_id);
        old_component_by_new_atom.push(atom.component);
        atoms.push(QueryAtom {
            id: new_id,
            component: 0,
            expr: atom.expr,
        });
    }

    let mut bonds = Vec::with_capacity(old_bonds.len());
    for bond in old_bonds {
        let Some(src) = atom_old_to_new.get(bond.src).copied().flatten() else {
            continue;
        };
        let Some(dst) = atom_old_to_new.get(bond.dst).copied().flatten() else {
            continue;
        };
        bonds.push(QueryBond {
            id: bonds.len(),
            src,
            dst,
            expr: bond.expr,
        });
    }

    let component_count = assign_components(&mut atoms, &bonds);
    let component_groups =
        rebuild_component_groups(&atoms, &old_component_by_new_atom, old_component_groups);

    (
        QueryMol::from_parts(atoms, bonds, component_count, component_groups),
        atom_old_to_new,
    )
}

fn assign_components(atoms: &mut [QueryAtom], bonds: &[QueryBond]) -> usize {
    if atoms.is_empty() {
        return 0;
    }

    let mut neighbors_by_atom = vec![Vec::new(); atoms.len()];
    for bond in bonds {
        if bond.src < neighbors_by_atom.len() {
            neighbors_by_atom[bond.src].push(bond.dst);
        }
        if bond.dst < neighbors_by_atom.len() && bond.dst != bond.src {
            neighbors_by_atom[bond.dst].push(bond.src);
        }
    }

    let mut visited = vec![false; atoms.len()];
    let mut component = 0usize;
    for atom_id in 0..atoms.len() {
        if visited[atom_id] {
            continue;
        }

        let mut stack = vec![atom_id];
        visited[atom_id] = true;
        while let Some(current) = stack.pop() {
            atoms[current].component = component;
            for &next in &neighbors_by_atom[current] {
                if next < visited.len() && !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }
        component += 1;
    }

    component
}

fn rebuild_component_groups(
    atoms: &[QueryAtom],
    old_component_by_new_atom: &[ComponentId],
    old_component_groups: &[Option<ComponentGroupId>],
) -> Vec<Option<ComponentGroupId>> {
    let component_count = atoms
        .iter()
        .map(|atom| atom.component)
        .max()
        .map_or(0, |max_component| max_component + 1);
    let mut groups = vec![None; component_count];

    for atom in atoms {
        let old_component = old_component_by_new_atom[atom.id];
        groups[atom.component] = old_component_groups.get(old_component).copied().flatten();
    }

    let mut remap = BTreeMap::new();
    let mut next_group = 0usize;
    for group in &mut groups {
        if let Some(old_group) = *group {
            let new_group = *remap.entry(old_group).or_insert_with(|| {
                let assigned = next_group;
                next_group += 1;
                assigned
            });
            *group = Some(new_group);
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, format, string::ToString, vec, vec::Vec};
    use elements_rs::Element;
    use smiles_parser::bond::Bond;

    use crate::{
        AtomExpr, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive, BracketExpr,
        BracketExprTree, HydrogenKind, NumericQuery, QueryMol,
    };

    use super::{
        add_atom_primitive, add_bond_primitive, normalize_bond_tree, normalize_bracket_tree,
        remove_atom_primitive, remove_bond_primitive, replace_atom_primitive,
        replace_bond_primitive, EditError, ExprPath, ExprPathSegment,
    };

    #[derive(Clone, Copy)]
    struct TinyRng(u64);

    impl TinyRng {
        const fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn pick(&mut self, upper: usize) -> usize {
            if upper == 0 {
                0
            } else {
                let upper_u64 = u64::try_from(upper).expect("usize must fit into u64");
                let value = self.next_u64() % upper_u64;
                usize::try_from(value).expect("modulo result must fit into usize")
            }
        }
    }

    fn sample_atom_primitive(rng: &mut TinyRng) -> AtomPrimitive {
        match rng.pick(6) {
            0 => AtomPrimitive::AtomicNumber(6),
            1 => AtomPrimitive::AtomicNumber(7),
            2 => AtomPrimitive::AtomicNumber(8),
            3 => AtomPrimitive::Degree(Some(NumericQuery::Exact(2))),
            4 => AtomPrimitive::Hydrogen(HydrogenKind::Total, Some(NumericQuery::Exact(1))),
            _ => AtomPrimitive::Charge(1),
        }
    }

    fn sample_atom_expr(rng: &mut TinyRng) -> AtomExpr {
        match rng.pick(6) {
            0 => AtomExpr::Wildcard,
            1 => AtomExpr::Bare {
                element: Element::C,
                aromatic: false,
            },
            2 => AtomExpr::Bare {
                element: Element::N,
                aromatic: false,
            },
            3 => AtomExpr::Bare {
                element: Element::O,
                aromatic: false,
            },
            4 => AtomExpr::Bracket(BracketExpr {
                tree: BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(6)),
                atom_map: None,
            }),
            _ => AtomExpr::Bracket(BracketExpr {
                tree: BracketExprTree::HighAnd(vec![
                    BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(7)),
                    BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                        HydrogenKind::Total,
                        Some(NumericQuery::Exact(1)),
                    )),
                ]),
                atom_map: None,
            }),
        }
    }

    fn sample_bond_primitive(rng: &mut TinyRng) -> BondPrimitive {
        match rng.pick(5) {
            0 => BondPrimitive::Bond(Bond::Single),
            1 => BondPrimitive::Bond(Bond::Double),
            2 => BondPrimitive::Bond(Bond::Triple),
            3 => BondPrimitive::Any,
            _ => BondPrimitive::Ring,
        }
    }

    fn sample_bond_expr(rng: &mut TinyRng) -> BondExpr {
        match rng.pick(5) {
            0 => BondExpr::Elided,
            _ => BondExpr::Query(BondExprTree::Primitive(sample_bond_primitive(rng))),
        }
    }

    fn sample_fragment(rng: &mut TinyRng) -> QueryMol {
        let choices = ["N(O)C", "S(=O)O", "C(C)N", "[#6]-[#8]"];
        choices[rng.pick(choices.len())]
            .parse::<QueryMol>()
            .unwrap()
    }

    fn sample_recursive_atom_expr(rng: &mut TinyRng) -> AtomExpr {
        AtomExpr::Bracket(BracketExpr {
            tree: BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(Box::new(
                sample_fragment(rng),
            ))),
            atom_map: None,
        })
    }

    fn assert_query_roundtrip_valid(query: &QueryMol) {
        query.validate().unwrap();
        let printed = query.to_string();
        let reparsed = printed.parse::<QueryMol>().unwrap();
        reparsed.validate().unwrap();
        assert_eq!(reparsed.to_string(), printed);
    }

    fn collect_recursive_bracket_paths(
        tree: &BracketExprTree,
        current: &mut Vec<ExprPathSegment>,
        paths: &mut Vec<ExprPath>,
    ) {
        if matches!(
            tree,
            BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(_))
        ) {
            paths.push(ExprPath(current.clone()));
        }

        match tree {
            BracketExprTree::Primitive(_) => {}
            BracketExprTree::Not(inner) => {
                current.push(ExprPathSegment::Not);
                collect_recursive_bracket_paths(inner, current, paths);
                current.pop();
            }
            BracketExprTree::HighAnd(items)
            | BracketExprTree::Or(items)
            | BracketExprTree::LowAnd(items) => {
                for (index, item) in items.iter().enumerate() {
                    current.push(ExprPathSegment::Child(index));
                    collect_recursive_bracket_paths(item, current, paths);
                    current.pop();
                }
            }
        }
    }

    fn recursive_atom_sites(query: &QueryMol) -> Vec<(usize, ExprPath)> {
        let mut sites = Vec::new();
        for atom in query.atoms() {
            let AtomExpr::Bracket(bracket) = &atom.expr else {
                continue;
            };
            let mut paths = Vec::new();
            collect_recursive_bracket_paths(&bracket.tree, &mut Vec::new(), &mut paths);
            sites.extend(paths.into_iter().map(|path| (atom.id, path)));
        }
        sites
    }

    fn removable_bracket_paths(tree: &BracketExprTree) -> Vec<ExprPath> {
        tree.enumerate_paths()
            .into_iter()
            .filter(|path| {
                let Some((last, parent_segments)) = path.0.split_last() else {
                    return false;
                };
                matches!(last, ExprPathSegment::Child(_))
                    && matches!(
                        tree.get(&ExprPath(parent_segments.to_vec())),
                        Some(
                            BracketExprTree::HighAnd(_)
                                | BracketExprTree::Or(_)
                                | BracketExprTree::LowAnd(_)
                        )
                    )
                    && matches!(tree.get(path), Some(BracketExprTree::Primitive(_)))
            })
            .collect()
    }

    fn removable_bond_paths(tree: &BondExprTree) -> Vec<ExprPath> {
        tree.enumerate_paths()
            .into_iter()
            .filter(|path| {
                let Some((last, parent_segments)) = path.0.split_last() else {
                    return false;
                };
                matches!(last, ExprPathSegment::Child(_))
                    && matches!(
                        tree.get(&ExprPath(parent_segments.to_vec())),
                        Some(
                            BondExprTree::HighAnd(_)
                                | BondExprTree::Or(_)
                                | BondExprTree::LowAnd(_)
                        )
                    )
                    && matches!(tree.get(path), Some(BondExprTree::Primitive(_)))
            })
            .collect()
    }

    #[test]
    fn editable_query_supports_topology_preserving_mutations() {
        let query = "CC".parse::<QueryMol>().unwrap();
        let mut editable = query.edit();

        let (_, leaf_id) = editable
            .attach_leaf(
                1,
                BondExpr::Elided,
                AtomExpr::Bare {
                    element: Element::O,
                    aromatic: false,
                },
            )
            .unwrap();
        assert_eq!(editable.as_query_mol().neighbors(1), &[0, 2]);
        editable
            .replace_bond_expr(
                1,
                BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double))),
            )
            .unwrap();
        editable.remove_leaf_atom(leaf_id).unwrap();

        let query = editable.into_query_mol().unwrap();
        assert_eq!(query.atom_count(), 2);
        assert_eq!(query.bond_count(), 1);
        assert_eq!(query.to_string(), "CC");
    }

    #[test]
    fn editable_query_can_insert_atoms_on_existing_bonds() {
        let mut editable = "CC".parse::<QueryMol>().unwrap().edit();
        editable
            .insert_atom_on_bond(
                0,
                BondExpr::Elided,
                AtomExpr::Bare {
                    element: Element::N,
                    aromatic: false,
                },
                BondExpr::Elided,
            )
            .unwrap();

        let query = editable.into_query_mol().unwrap();
        assert_eq!(query.atom_count(), 3);
        assert_eq!(query.bond_count(), 2);
        assert_eq!(query.to_string(), "CNC");
    }

    #[test]
    fn editable_query_reindexes_components_after_topology_changes() {
        let mut editable = "C.C".parse::<QueryMol>().unwrap().edit();
        assert_eq!(
            editable.add_bond(0, 1, BondExpr::Elided),
            Err(EditError::CrossComponentBond)
        );

        editable.remove_atom(0).unwrap();
        let query = editable.into_query_mol().unwrap();
        assert_eq!(query.atom_count(), 1);
        assert_eq!(query.component_count(), 1);
        assert_eq!(query.component_atoms(0), &[0]);
    }

    #[test]
    fn editable_query_rejects_invalid_ids_and_non_leaf_removals() {
        let mut editable = "CCC".parse::<QueryMol>().unwrap().edit();
        assert_eq!(
            editable.replace_atom_expr(9, AtomExpr::Wildcard),
            Err(EditError::InvalidAtomId(9))
        );
        assert_eq!(editable.remove_leaf_atom(1), Err(EditError::NotLeafAtom(1)));
        assert_eq!(editable.remove_bond(9), Err(EditError::InvalidBondId(9)));
    }

    #[test]
    fn query_subgraph_helpers_cover_tree_and_fragment_extraction() {
        let query = "CC(O)N.C".parse::<QueryMol>().unwrap();
        assert_eq!(query.leaf_atoms(), vec![0, 2, 3, 4]);
        assert_eq!(query.branch_atoms(), vec![1]);
        assert_eq!(query.ring_bonds(), Vec::<usize>::new());
        assert_eq!(query.rooted_subtree(2, Some(1)), vec![2]);
        assert_eq!(query.rooted_subtree(1, Some(0)), vec![1, 2, 3]);

        let fragment = query.clone_subgraph(&[1, 2, 3]).unwrap();
        assert_eq!(fragment.to_string(), "C(O)N");
        assert_eq!(fragment.atom_count(), 3);
        assert_eq!(fragment.component_count(), 1);
    }

    #[test]
    fn ring_helpers_detect_and_edit_cycle_edges() {
        let query = "C1CC1C".parse::<QueryMol>().unwrap();
        assert_eq!(query.ring_bonds(), vec![0, 1, 2]);
        assert_eq!(query.chain_bonds(), vec![3]);
        assert!(query.is_cycle_edge(1));
        assert!(!query.is_cycle_edge(3));

        let mut opened = "C1CC1".parse::<QueryMol>().unwrap().edit();
        opened.open_ring(0).unwrap();
        let opened_query = opened.into_query_mol().unwrap();
        assert_eq!(opened_query.bond_count(), 2);
        assert_eq!(opened_query.ring_bonds(), Vec::<usize>::new());
        assert_query_roundtrip_valid(&opened_query);

        let mut closed = "CCC".parse::<QueryMol>().unwrap().edit();
        let closed_bond = closed.close_ring(0, 2, BondExpr::Elided).unwrap();
        let closed_query = closed.into_query_mol().unwrap();
        assert!(closed_query.is_cycle_edge(closed_bond));
        assert_eq!(closed_query.atom_count(), 3);
        assert_eq!(closed_query.bond_count(), 3);
        assert_eq!(closed_query.ring_bonds(), vec![0, 1, 2]);
        assert_query_roundtrip_valid(&closed_query);
    }

    #[test]
    fn ring_helpers_reject_invalid_open_and_close_requests() {
        let mut chain = "CCCC".parse::<QueryMol>().unwrap().edit();
        assert_eq!(chain.open_ring(1), Err(EditError::NotCycleBond(1)));

        let mut query = "CC".parse::<QueryMol>().unwrap().edit();
        assert_eq!(
            query.close_ring(0, 1, BondExpr::Elided),
            Err(EditError::BondAlreadyExists(0, 1))
        );
        assert_eq!(
            query.close_ring(0, 0, BondExpr::Elided),
            Err(EditError::SelfLoopBond(0))
        );
    }

    #[test]
    fn editable_query_can_graft_fragment_components() {
        let mut editable = "CC".parse::<QueryMol>().unwrap().edit();
        let fragment = "N(O)C".parse::<QueryMol>().unwrap();

        let new_atom_ids = editable
            .graft_subgraph(1, BondExpr::Elided, &fragment, 0)
            .unwrap();
        let query = editable.into_query_mol().unwrap();

        assert_eq!(new_atom_ids.len(), 3);
        assert_eq!(query.to_string(), "CCN(O)C");
    }

    #[test]
    fn editable_query_can_replace_subtrees_with_fragments() {
        let mut editable = "CC(O)N".parse::<QueryMol>().unwrap().edit();
        let fragment = "S(=O)O".parse::<QueryMol>().unwrap();

        editable.replace_subtree(1, 2, &fragment, 0).unwrap();

        let query = editable.into_query_mol().unwrap();
        assert_eq!(query.to_string(), "CC(N)S(=O)O");
    }

    #[test]
    fn expression_paths_and_normalization_cover_bracket_and_bond_trees() {
        let mut bracket_tree = BracketExprTree::HighAnd(vec![
            BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(6)),
            BracketExprTree::HighAnd(vec![BracketExprTree::Primitive(AtomPrimitive::Degree(
                Some(NumericQuery::Exact(2)),
            ))]),
            BracketExprTree::Not(Box::new(BracketExprTree::Not(Box::new(
                BracketExprTree::Primitive(AtomPrimitive::Charge(1)),
            )))),
        ]);
        let paths = bracket_tree.enumerate_paths();
        assert!(paths.contains(&ExprPath(vec![])));
        assert!(paths.contains(&ExprPath(vec![ExprPathSegment::Child(0)])));
        assert!(paths.contains(&ExprPath(vec![
            ExprPathSegment::Child(2),
            ExprPathSegment::Not,
            ExprPathSegment::Not,
        ])));
        normalize_bracket_tree(&mut bracket_tree).unwrap();
        assert_eq!(
            bracket_tree,
            BracketExprTree::HighAnd(vec![
                BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(6)),
                BracketExprTree::Primitive(AtomPrimitive::Degree(Some(NumericQuery::Exact(2),))),
                BracketExprTree::Primitive(AtomPrimitive::Charge(1)),
            ])
        );

        let mut bond_tree = BondExprTree::Or(vec![
            BondExprTree::Primitive(BondPrimitive::Any),
            BondExprTree::Or(vec![BondExprTree::Primitive(BondPrimitive::Ring)]),
        ]);
        assert!(bond_tree
            .get(&ExprPath(vec![
                ExprPathSegment::Child(1),
                ExprPathSegment::Child(0)
            ]))
            .is_some());
        normalize_bond_tree(&mut bond_tree).unwrap();
        assert_eq!(
            bond_tree,
            BondExprTree::Or(vec![
                BondExprTree::Primitive(BondPrimitive::Any),
                BondExprTree::Primitive(BondPrimitive::Ring),
            ])
        );
    }

    #[test]
    fn primitive_helpers_support_add_replace_and_remove() {
        let mut bracket = BracketExpr {
            tree: BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(6)),
            atom_map: None,
        };
        add_atom_primitive(
            &mut bracket,
            AtomPrimitive::Degree(Some(NumericQuery::Exact(3))),
        )
        .unwrap();
        let replaced = replace_atom_primitive(
            &mut bracket,
            &ExprPath(vec![ExprPathSegment::Child(0)]),
            AtomPrimitive::AtomicNumber(7),
        )
        .unwrap();
        assert_eq!(replaced, AtomPrimitive::AtomicNumber(6));
        let removed =
            remove_atom_primitive(&mut bracket, &ExprPath(vec![ExprPathSegment::Child(1)]))
                .unwrap();
        assert_eq!(removed, AtomPrimitive::Degree(Some(NumericQuery::Exact(3))));
        assert_eq!(
            bracket.tree,
            BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(7))
        );

        let mut bond_tree = BondExprTree::Primitive(BondPrimitive::Any);
        add_bond_primitive(&mut bond_tree, BondPrimitive::Ring).unwrap();
        let replaced = replace_bond_primitive(
            &mut bond_tree,
            &ExprPath(vec![ExprPathSegment::Child(0)]),
            BondPrimitive::Bond(Bond::Single),
        )
        .unwrap();
        assert_eq!(replaced, BondPrimitive::Any);
        let removed =
            remove_bond_primitive(&mut bond_tree, &ExprPath(vec![ExprPathSegment::Child(1)]))
                .unwrap();
        assert_eq!(removed, BondPrimitive::Ring);
        assert_eq!(
            bond_tree,
            BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single))
        );
    }

    #[test]
    fn primitive_helpers_reject_invalid_paths_and_unsupported_removals() {
        let mut bracket = BracketExpr {
            tree: BracketExprTree::Not(Box::new(BracketExprTree::Primitive(
                AtomPrimitive::AtomicNumber(6),
            ))),
            atom_map: None,
        };
        assert_eq!(
            replace_atom_primitive(&mut bracket, &ExprPath(vec![]), AtomPrimitive::Wildcard),
            Err(EditError::ExpectedPrimitive)
        );
        assert_eq!(
            remove_atom_primitive(&mut bracket, &ExprPath(vec![ExprPathSegment::Not])),
            Err(EditError::UnsupportedExprRemovalPath)
        );
        assert_eq!(
            remove_atom_primitive(
                &mut BracketExpr {
                    tree: BracketExprTree::Primitive(AtomPrimitive::Wildcard),
                    atom_map: None,
                },
                &ExprPath(vec![]),
            ),
            Err(EditError::CannotRemoveRootExpression)
        );
    }

    #[test]
    fn deterministic_edit_sequences_roundtrip_and_validate() {
        let seeds = ["C", "CC", "C(O)N", "[#6]-[#8]", "C1CC1", "C.C"];

        for (case_index, smarts) in seeds.iter().enumerate() {
            let mut editable = smarts.parse::<QueryMol>().unwrap().edit();
            let mut rng = TinyRng::new((case_index as u64) + 1);

            for _ in 0..48 {
                let snapshot = editable.as_query_mol().clone();
                let atom_count = snapshot.atom_count();
                let bond_count = snapshot.bond_count();

                match rng.pick(7) {
                    0 => {
                        let atom_id = rng.pick(atom_count);
                        editable
                            .replace_atom_expr(atom_id, sample_atom_expr(&mut rng))
                            .unwrap();
                    }
                    1 => {
                        if bond_count > 0 {
                            let bond_id = rng.pick(bond_count);
                            editable
                                .replace_bond_expr(bond_id, sample_bond_expr(&mut rng))
                                .unwrap();
                        }
                    }
                    2 => {
                        let parent = rng.pick(atom_count);
                        editable
                            .attach_leaf(
                                parent,
                                sample_bond_expr(&mut rng),
                                sample_atom_expr(&mut rng),
                            )
                            .unwrap();
                    }
                    3 => {
                        if bond_count > 0 {
                            let bond_id = rng.pick(bond_count);
                            editable
                                .insert_atom_on_bond(
                                    bond_id,
                                    sample_bond_expr(&mut rng),
                                    sample_atom_expr(&mut rng),
                                    sample_bond_expr(&mut rng),
                                )
                                .unwrap();
                        }
                    }
                    4 => {
                        if atom_count > 1 {
                            let leaves = snapshot.leaf_atoms();
                            if !leaves.is_empty() {
                                let atom_id = leaves[rng.pick(leaves.len())];
                                editable.remove_leaf_atom(atom_id).unwrap();
                            }
                        }
                    }
                    5 => {
                        let parent = rng.pick(atom_count);
                        let fragment = sample_fragment(&mut rng);
                        editable
                            .graft_subgraph(parent, sample_bond_expr(&mut rng), &fragment, 0)
                            .unwrap();
                    }
                    _ => {
                        let branch_atoms = snapshot.branch_atoms();
                        if !branch_atoms.is_empty() {
                            let anchor = branch_atoms[rng.pick(branch_atoms.len())];
                            let neighbors = snapshot.neighbors(anchor);
                            if !neighbors.is_empty() {
                                let old_child = neighbors[rng.pick(neighbors.len())];
                                let removed = snapshot.rooted_subtree(old_child, Some(anchor));
                                if !removed.is_empty() && removed.len() < atom_count {
                                    let fragment = sample_fragment(&mut rng);
                                    editable
                                        .replace_subtree(anchor, old_child, &fragment, 0)
                                        .unwrap();
                                }
                            }
                        }
                    }
                }

                assert_query_roundtrip_valid(editable.as_query_mol());
            }

            let final_query = editable.into_query_mol().unwrap();
            assert_query_roundtrip_valid(&final_query);
        }
    }

    #[test]
    fn deterministic_expression_edit_sequences_roundtrip() {
        let mut rng = TinyRng::new(17);
        let mut bracket = BracketExpr {
            tree: BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(6)),
            atom_map: None,
        };
        let mut bond_tree = BondExprTree::Primitive(BondPrimitive::Any);

        for _ in 0..64 {
            match rng.pick(3) {
                0 => add_atom_primitive(&mut bracket, sample_atom_primitive(&mut rng)).unwrap(),
                1 => {
                    let primitive_paths = bracket
                        .tree
                        .enumerate_paths()
                        .into_iter()
                        .filter(|path| {
                            matches!(bracket.tree.get(path), Some(BracketExprTree::Primitive(_)))
                        })
                        .collect::<Vec<_>>();
                    let path = primitive_paths[rng.pick(primitive_paths.len())].clone();
                    replace_atom_primitive(&mut bracket, &path, sample_atom_primitive(&mut rng))
                        .unwrap();
                }
                _ => {
                    let removable = removable_bracket_paths(&bracket.tree);
                    if removable.is_empty() {
                        add_atom_primitive(&mut bracket, sample_atom_primitive(&mut rng)).unwrap();
                    } else {
                        let path = removable[rng.pick(removable.len())].clone();
                        remove_atom_primitive(&mut bracket, &path).unwrap();
                    }
                }
            }
            bracket.normalize().unwrap();
            let text = format!("[{bracket}]");
            let reparsed = text.parse::<QueryMol>().unwrap();
            reparsed.validate().unwrap();
            assert_eq!(reparsed.to_string(), text);

            match rng.pick(3) {
                0 => add_bond_primitive(&mut bond_tree, sample_bond_primitive(&mut rng)).unwrap(),
                1 => {
                    let primitive_paths = bond_tree
                        .enumerate_paths()
                        .into_iter()
                        .filter(|path| {
                            matches!(bond_tree.get(path), Some(BondExprTree::Primitive(_)))
                        })
                        .collect::<Vec<_>>();
                    let path = primitive_paths[rng.pick(primitive_paths.len())].clone();
                    replace_bond_primitive(&mut bond_tree, &path, sample_bond_primitive(&mut rng))
                        .unwrap();
                }
                _ => {
                    let removable = removable_bond_paths(&bond_tree);
                    if removable.is_empty() {
                        add_bond_primitive(&mut bond_tree, sample_bond_primitive(&mut rng))
                            .unwrap();
                    } else {
                        let path = removable[rng.pick(removable.len())].clone();
                        remove_bond_primitive(&mut bond_tree, &path).unwrap();
                    }
                }
            }
            normalize_bond_tree(&mut bond_tree).unwrap();
            let text = format!("C{}C", BondExpr::Query(bond_tree.clone()));
            let reparsed = text.parse::<QueryMol>().unwrap();
            reparsed.validate().unwrap();
            assert_eq!(reparsed.to_string(), text);
        }
    }

    #[test]
    fn deterministic_recursive_query_edit_sequences_roundtrip_and_validate() {
        let seeds = ["[$(CC)]", "C[$(O)]N", "[$(C1CC1)]", "N[$(C(C)O)]"];

        for (case_index, smarts) in seeds.iter().enumerate() {
            let mut editable = smarts.parse::<QueryMol>().unwrap().edit();
            let mut rng = TinyRng::new(100 + (case_index as u64));

            for _ in 0..40 {
                let snapshot = editable.as_query_mol().clone();
                let recursive_sites = recursive_atom_sites(&snapshot);

                if recursive_sites.is_empty() {
                    let atom_id = rng.pick(snapshot.atom_count());
                    editable
                        .replace_atom_expr(atom_id, sample_recursive_atom_expr(&mut rng))
                        .unwrap();
                    assert_query_roundtrip_valid(editable.as_query_mol());
                    continue;
                }

                let (atom_id, path) = recursive_sites[rng.pick(recursive_sites.len())].clone();
                let mut expr = editable.atom(atom_id).unwrap().expr.clone();
                let AtomExpr::Bracket(bracket) = &mut expr else {
                    panic!("recursive site must live inside a bracket expression");
                };
                let Some(BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(inner))) =
                    bracket.tree.get_mut(&path)
                else {
                    panic!("recursive site path must resolve to a recursive primitive");
                };

                let mut nested = inner.edit();
                let nested_snapshot = nested.as_query_mol().clone();
                let nested_atom_count = nested_snapshot.atom_count();
                let nested_bond_count = nested_snapshot.bond_count();

                match rng.pick(5) {
                    0 => {
                        let nested_atom = rng.pick(nested_atom_count);
                        nested
                            .replace_atom_expr(nested_atom, sample_atom_expr(&mut rng))
                            .unwrap();
                    }
                    1 => {
                        let parent = rng.pick(nested_atom_count);
                        nested
                            .attach_leaf(
                                parent,
                                sample_bond_expr(&mut rng),
                                sample_atom_expr(&mut rng),
                            )
                            .unwrap();
                    }
                    2 => {
                        if nested_atom_count > 1 {
                            let leaves = nested_snapshot.leaf_atoms();
                            if !leaves.is_empty() {
                                nested
                                    .remove_leaf_atom(leaves[rng.pick(leaves.len())])
                                    .unwrap();
                            }
                        }
                    }
                    3 => {
                        if nested_bond_count > 0 {
                            let bond_id = rng.pick(nested_bond_count);
                            nested
                                .insert_atom_on_bond(
                                    bond_id,
                                    sample_bond_expr(&mut rng),
                                    sample_atom_expr(&mut rng),
                                    sample_bond_expr(&mut rng),
                                )
                                .unwrap();
                        }
                    }
                    _ => {
                        let fragment = sample_fragment(&mut rng);
                        let parent = rng.pick(nested_atom_count);
                        nested
                            .graft_subgraph(parent, sample_bond_expr(&mut rng), &fragment, 0)
                            .unwrap();
                    }
                }

                **inner = nested.into_query_mol().unwrap();
                bracket.normalize().unwrap();
                editable.replace_atom_expr(atom_id, expr).unwrap();
                assert_query_roundtrip_valid(editable.as_query_mol());
            }

            let final_query = editable.into_query_mol().unwrap();
            assert_query_roundtrip_valid(&final_query);
        }
    }

    #[test]
    fn deterministic_ring_rich_edit_sequences_roundtrip_and_validate() {
        let seeds = [
            "C1CC1",
            "C1CCC(CC1)O",
            "C1=CC=CC=C1",
            "C1CC2CC1C2",
            "C1CC1.C1CC1",
        ];

        for (case_index, smarts) in seeds.iter().enumerate() {
            let mut editable = smarts.parse::<QueryMol>().unwrap().edit();
            let mut rng = TinyRng::new(200 + (case_index as u64));

            for _ in 0..56 {
                let snapshot = editable.as_query_mol().clone();
                let atom_count = snapshot.atom_count();
                let bond_count = snapshot.bond_count();

                match rng.pick(6) {
                    0 => {
                        let atom_id = rng.pick(atom_count);
                        editable
                            .replace_atom_expr(atom_id, sample_atom_expr(&mut rng))
                            .unwrap();
                    }
                    1 => {
                        if bond_count > 0 {
                            let bond_id = rng.pick(bond_count);
                            editable
                                .replace_bond_expr(bond_id, sample_bond_expr(&mut rng))
                                .unwrap();
                        }
                    }
                    2 => {
                        let parent = rng.pick(atom_count);
                        editable
                            .attach_leaf(
                                parent,
                                sample_bond_expr(&mut rng),
                                sample_atom_expr(&mut rng),
                            )
                            .unwrap();
                    }
                    3 => {
                        if bond_count > 0 {
                            let bond_id = rng.pick(bond_count);
                            editable
                                .insert_atom_on_bond(
                                    bond_id,
                                    sample_bond_expr(&mut rng),
                                    sample_atom_expr(&mut rng),
                                    sample_bond_expr(&mut rng),
                                )
                                .unwrap();
                        }
                    }
                    4 => {
                        let fragment = sample_fragment(&mut rng);
                        let parent = rng.pick(atom_count);
                        editable
                            .graft_subgraph(parent, sample_bond_expr(&mut rng), &fragment, 0)
                            .unwrap();
                    }
                    _ => {
                        let branch_atoms = snapshot.branch_atoms();
                        if !branch_atoms.is_empty() {
                            let anchor = branch_atoms[rng.pick(branch_atoms.len())];
                            let neighbors = snapshot.neighbors(anchor);
                            if !neighbors.is_empty() {
                                let old_child = neighbors[rng.pick(neighbors.len())];
                                let removed = snapshot.rooted_subtree(old_child, Some(anchor));
                                if !removed.is_empty() && removed.len() < atom_count {
                                    let fragment = sample_fragment(&mut rng);
                                    editable
                                        .replace_subtree(anchor, old_child, &fragment, 0)
                                        .unwrap();
                                }
                            }
                        }
                    }
                }

                assert_query_roundtrip_valid(editable.as_query_mol());
            }

            let final_query = editable.into_query_mol().unwrap();
            assert_query_roundtrip_valid(&final_query);
        }
    }
}
