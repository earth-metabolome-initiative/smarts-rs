use alloc::{
    boxed::Box,
    collections::BTreeMap,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use elements_rs::{AtomicNumber, ElementVariant, Isotope, MassNumber};
use geometric_traits::{
    impls::SortedVec,
    naive_structs::{UndiEdgesBuilder, UndiGraph},
    prelude::CanonicalLabeling,
    traits::EdgesBuilder,
};
use smiles_parser::{atom::bracketed::chirality::Chirality, bond::Bond};

use crate::{
    edit::normalize_bond_tree,
    query::{
        AtomExpr, AtomId, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive, BracketExpr,
        BracketExprTree, ComponentId, HydrogenKind, NumericQuery, NumericRange, QueryAtom,
        QueryBond, QueryMol,
    },
};

mod boolean_ops;

use self::boolean_ops::{
    collapse_disjunction_consensus_terms, relax_absorbed_disjunction_conjunction_alternatives,
    relax_complemented_disjunction_terms, relax_covered_negated_disjunction_terms,
    relax_negated_conjunction_terms, remove_absorbed_disjunction_terms,
    remove_redundant_disjunction_consensus_terms, AbsorbedDisjunctionConjunctionOps,
    CoveredNegatedDisjunctionOps,
};

/// Canonical labeling result for a [`QueryMol`] graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryCanonicalLabeling {
    order: Vec<AtomId>,
    new_index_of_old_atom: Vec<AtomId>,
}

impl QueryCanonicalLabeling {
    fn new(order: Vec<AtomId>) -> Self {
        let mut new_index_of_old_atom = vec![usize::MAX; order.len()];
        for (new_index, old_atom) in order.iter().copied().enumerate() {
            new_index_of_old_atom[old_atom] = new_index;
        }
        Self {
            order,
            new_index_of_old_atom,
        }
    }

    /// Returns original atom ids in canonical order.
    #[inline]
    #[must_use]
    pub fn order(&self) -> &[AtomId] {
        &self.order
    }

    /// Returns the canonical index for each original atom id.
    #[inline]
    #[must_use]
    pub fn new_index_of_old_atom(&self) -> &[AtomId] {
        &self.new_index_of_old_atom
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalQueryStateKey {
    atom_labels: Vec<String>,
    bond_edges: Vec<(usize, usize, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalWholeQueryStateKey {
    component_groups: Vec<Option<ComponentId>>,
    atom_labels: Vec<String>,
    bond_edges: Vec<(usize, usize, String)>,
}

struct CanonicalQueryStateParts {
    atom_labels: Vec<String>,
    bond_edges: Vec<(usize, usize, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalTopLevelEntryStateKey {
    grouped: bool,
    component_states: Vec<CanonicalQueryStateKey>,
}

#[derive(Debug, Clone)]
struct CanonicalizedComponent {
    original_atom_order: Vec<AtomId>,
    canonicalized: QueryMol,
    state_key: CanonicalQueryStateKey,
}

#[derive(Debug, Clone)]
struct CanonicalizedEntry {
    grouped: bool,
    components: Vec<CanonicalizedComponent>,
    state_key: CanonicalTopLevelEntryStateKey,
}

#[derive(Debug, Clone)]
struct TopLevelComponentEntry {
    grouped: bool,
    component_ids: Vec<ComponentId>,
}

#[derive(Default)]
struct RecursiveCanonicalizationContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecursiveCanonicalizationMode {
    Canonicalize,
    Preserve,
}

impl RecursiveCanonicalizationContext {
    fn canonicalize_recursive_query(&mut self, query: &QueryMol) -> QueryMol {
        query.canonicalize_with_labeling_with_context(self).0
    }
}

impl QueryMol {
    /// Returns the canonical atom labeling of the query graph.
    #[must_use]
    pub fn canonical_labeling(&self) -> QueryCanonicalLabeling {
        self.canonicalize_with_labeling().1
    }

    /// Returns the query rewritten into canonical graph and expression order.
    #[must_use]
    pub fn canonicalize(&self) -> Self {
        self.canonicalize_with_labeling().0
    }

    /// Returns whether the query is already in canonical form.
    #[must_use]
    pub fn is_canonical(&self) -> bool {
        *self == self.canonicalize()
    }

    /// Returns one canonical SMARTS string for the current query.
    #[must_use]
    pub fn to_canonical_smarts(&self) -> String {
        self.canonicalize().to_string()
    }

    fn canonicalize_with_labeling(&self) -> (Self, QueryCanonicalLabeling) {
        let mut recursive_context = RecursiveCanonicalizationContext;
        self.canonicalize_with_labeling_with_context(&mut recursive_context)
    }

    fn canonicalize_with_labeling_with_context(
        &self,
        recursive_context: &mut RecursiveCanonicalizationContext,
    ) -> (Self, QueryCanonicalLabeling) {
        let (first_query, first_labeling) = self.canonicalize_once_with_labeling(
            recursive_context,
            RecursiveCanonicalizationMode::Canonicalize,
        );
        if first_query == *self {
            return (first_query, first_labeling);
        }
        let (second_query, second_step_labeling) = first_query.canonicalize_once_with_labeling(
            recursive_context,
            RecursiveCanonicalizationMode::Preserve,
        );
        if second_query == first_query {
            return (first_query, first_labeling);
        }
        let first_key = canonical_whole_query_state_key(&first_query);
        let second_key = canonical_whole_query_state_key(&second_query);
        let second_labeling = compose_labelings(&first_labeling, &second_step_labeling);

        let mut seen = BTreeMap::new();
        seen.insert(first_key.clone(), 0usize);
        seen.insert(second_key.clone(), 1usize);

        let mut states = vec![
            (first_query, first_labeling, first_key),
            (second_query, second_labeling, second_key),
        ];
        let cycle_start = loop {
            let current_index = states.len() - 1;
            let (next_query, next_step_labeling) =
                states[current_index].0.canonicalize_once_with_labeling(
                    recursive_context,
                    RecursiveCanonicalizationMode::Preserve,
                );
            let next_labeling = compose_labelings(&states[current_index].1, &next_step_labeling);
            let next_key = canonical_whole_query_state_key(&next_query);
            if let Some(&repeat_index) = seen.get(&next_key) {
                break repeat_index;
            }
            let next_index = states.len();
            seen.insert(next_key.clone(), next_index);
            states.push((next_query, next_labeling, next_key));
        };

        let (best_query, best_labeling, _) = states
            .into_iter()
            .skip(cycle_start)
            .min_by(|left, right| left.2.cmp(&right.2))
            .unwrap_or_else(|| unreachable!("canonical orbit is never empty"));
        (best_query, best_labeling)
    }

    fn canonicalize_once_with_labeling(
        &self,
        recursive_context: &mut RecursiveCanonicalizationContext,
        recursive_mode: RecursiveCanonicalizationMode,
    ) -> (Self, QueryCanonicalLabeling) {
        let normalized = self.canonicalization_normal_form(recursive_context, recursive_mode);
        if normalized.atom_count() <= 1 && normalized.component_count() <= 1 {
            let labeling = QueryCanonicalLabeling::new((0..normalized.atom_count()).collect());
            return (normalized, labeling);
        }
        if normalized.component_count() <= 1 {
            let labeling = normalized.exact_canonical_labeling_whole_graph();
            let canonicalized = normalized.exact_canonicalize_with_labeling(&labeling);
            return (canonicalized, labeling);
        }

        normalized.componentwise_canonicalize_with_labeling()
    }

    fn canonicalization_normal_form(
        &self,
        recursive_context: &mut RecursiveCanonicalizationContext,
        recursive_mode: RecursiveCanonicalizationMode,
    ) -> Self {
        let atoms = self
            .atoms()
            .iter()
            .map(|atom| QueryAtom {
                id: atom.id,
                component: atom.component,
                expr: canonical_atom_expr(&atom.expr, recursive_context, recursive_mode),
            })
            .collect::<Vec<_>>();
        let bonds = self
            .bonds()
            .iter()
            .map(|bond| QueryBond {
                id: bond.id,
                src: bond.src,
                dst: bond.dst,
                expr: canonical_bond_expr(&bond.expr),
            })
            .collect::<Vec<_>>();

        Self::from_parts(
            atoms,
            bonds,
            self.component_count(),
            canonical_component_groups(self),
        )
    }

    fn exact_canonical_labeling_whole_graph(&self) -> QueryCanonicalLabeling {
        if self.atom_count() == 0 {
            return QueryCanonicalLabeling::new(Vec::new());
        }
        if self.atom_count() == 2 {
            let first_label = self.atoms()[0].expr.to_string();
            let second_label = self.atoms()[1].expr.to_string();
            let order = if second_label < first_label {
                vec![1, 0]
            } else {
                vec![0, 1]
            };
            return QueryCanonicalLabeling::new(order);
        }

        let atom_labels = self
            .atoms()
            .iter()
            .map(|atom| atom.expr.to_string())
            .collect::<Vec<_>>();

        let bond_multiset_labels = build_bond_multiset_labels(self);
        let nodes = SortedVec::try_from((0..self.atom_count()).collect::<Vec<_>>())
            .unwrap_or_else(|_| unreachable!("dense node ids are always sorted"));
        let edges = bond_multiset_labels
            .iter()
            .map(|(edge, _)| *edge)
            .collect::<Vec<_>>();
        let edge_count = edges.len();
        let edges = UndiEdgesBuilder::default()
            .expected_number_of_edges(edge_count)
            .expected_shape(self.atom_count())
            .edges(edges.into_iter())
            .build()
            .unwrap_or_else(|_| unreachable!("query graph uses dense atom ids"));
        let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));

        let result = CanonicalLabeling::canonical_labeling(
            &graph,
            |node_id| atom_labels[node_id].as_str(),
            |node_a, node_b| {
                bond_multiset_label_for_edge(&bond_multiset_labels, node_a, node_b).map_or_else(
                    || unreachable!("canonizer only queries edges that exist"),
                    |(_, label)| label.as_str(),
                )
            },
        );
        QueryCanonicalLabeling::new(result.order)
    }

    fn exact_canonicalize_with_labeling(&self, labeling: &QueryCanonicalLabeling) -> Self {
        if let Some(canonicalized) = self.exact_canonicalize_single_bond_with_labeling(labeling) {
            return canonicalized;
        }

        let order = labeling.order();
        let new_index_of_old_atom = labeling.new_index_of_old_atom();
        let has_chirality = self
            .atoms()
            .iter()
            .any(|atom| atom_expr_contains_chirality(&atom.expr));
        let has_directional_bonds = self
            .bonds()
            .iter()
            .any(|bond| bond_expr_contains_direction(&bond.expr));

        let provisional_atoms = order
            .iter()
            .copied()
            .map(|old_atom| QueryAtom {
                id: new_index_of_old_atom[old_atom],
                component: 0,
                expr: self.atoms()[old_atom].expr.clone(),
            })
            .collect::<Vec<_>>();

        let mut bonds = self
            .bonds()
            .iter()
            .map(|bond| {
                let src = new_index_of_old_atom[bond.src];
                let dst = new_index_of_old_atom[bond.dst];
                let expr = if src <= dst {
                    bond.expr.clone()
                } else {
                    flip_directional_bond_expr(&bond.expr)
                };
                let (src, dst) = if src <= dst { (src, dst) } else { (dst, src) };
                (src, dst, expr)
            })
            .collect::<Vec<_>>();
        bonds.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.1.cmp(&right.1))
                .then(left.2.to_string().cmp(&right.2.to_string()))
        });
        let bonds = bonds
            .into_iter()
            .enumerate()
            .map(|(bond_id, (src, dst, expr))| QueryBond {
                id: bond_id,
                src,
                dst,
                expr,
            })
            .collect::<Vec<_>>();
        let provisional =
            Self::from_parts(provisional_atoms, bonds, 1, vec![self.component_group(0)]);
        let provisional = if has_directional_bonds {
            normalize_directional_double_bond_pairs(&provisional)
        } else {
            provisional
        };
        if !has_chirality {
            return provisional;
        }
        let atoms =
            chirality_adjusted_canonical_atoms(self, &provisional, order, new_index_of_old_atom);

        Self::from_parts(
            atoms,
            provisional.bonds().to_vec(),
            1,
            vec![self.component_group(0)],
        )
    }

    fn exact_canonicalize_single_bond_with_labeling(
        &self,
        labeling: &QueryCanonicalLabeling,
    ) -> Option<Self> {
        if self.atom_count() != 2 || self.bond_count() != 1 {
            return None;
        }
        if self
            .atoms()
            .iter()
            .any(|atom| atom_expr_contains_chirality(&atom.expr))
        {
            return None;
        }

        let bond = &self.bonds()[0];
        if bond_expr_contains_direction(&bond.expr) {
            return None;
        }

        let order = labeling.order();
        let new_index_of_old_atom = labeling.new_index_of_old_atom();
        let atoms = order
            .iter()
            .copied()
            .map(|old_atom| QueryAtom {
                id: new_index_of_old_atom[old_atom],
                component: 0,
                expr: self.atoms()[old_atom].expr.clone(),
            })
            .collect::<Vec<_>>();
        let src = new_index_of_old_atom[bond.src];
        let dst = new_index_of_old_atom[bond.dst];
        let (src, dst) = if src <= dst { (src, dst) } else { (dst, src) };
        let bonds = vec![QueryBond {
            id: 0,
            src,
            dst,
            expr: bond.expr.clone(),
        }];

        Some(Self::from_parts(
            atoms,
            bonds,
            1,
            vec![self.component_group(0)],
        ))
    }

    fn componentwise_canonicalize_with_labeling(&self) -> (Self, QueryCanonicalLabeling) {
        let mut keyed_entries = build_top_level_component_entries(self)
            .into_iter()
            .map(|entry| {
                let mut components = entry
                    .component_ids
                    .iter()
                    .copied()
                    .map(|component_id| {
                        let old_atoms = self.component_atoms(component_id).to_vec();
                        let component = self
                            .clone_subgraph(&old_atoms)
                            .unwrap_or_else(|_| unreachable!("component extraction is valid"));
                        let component_labeling = component.exact_canonical_labeling_whole_graph();
                        let canonicalized =
                            component.exact_canonicalize_with_labeling(&component_labeling);
                        CanonicalizedComponent {
                            original_atom_order: component_labeling
                                .order()
                                .iter()
                                .copied()
                                .map(|local_old_atom| old_atoms[local_old_atom])
                                .collect(),
                            state_key: canonical_query_state_key(&canonicalized),
                            canonicalized,
                        }
                    })
                    .collect::<Vec<_>>();
                components.sort_unstable_by(|left, right| left.state_key.cmp(&right.state_key));
                let state_key = CanonicalTopLevelEntryStateKey {
                    grouped: entry.grouped,
                    component_states: components
                        .iter()
                        .map(|component| component.state_key.clone())
                        .collect(),
                };
                CanonicalizedEntry {
                    grouped: entry.grouped,
                    components,
                    state_key,
                }
            })
            .collect::<Vec<_>>();
        keyed_entries.sort_unstable_by(|left, right| left.state_key.cmp(&right.state_key));

        let labeling = QueryCanonicalLabeling::new(flatten_original_atom_order(&keyed_entries));
        let canonicalized = rebuild_query_from_entries(&keyed_entries);
        (canonicalized, labeling)
    }
}

fn atom_expr_contains_chirality(expr: &AtomExpr) -> bool {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => false,
        AtomExpr::Bracket(bracket) => bracket_tree_contains_chirality(&bracket.tree),
    }
}

fn chirality_adjusted_canonical_atoms(
    original: &QueryMol,
    provisional: &QueryMol,
    order: &[AtomId],
    new_index_of_old_atom: &[AtomId],
) -> Vec<QueryAtom> {
    let old_parent_bond_by_atom = spanning_forest_parent_bonds(original);
    let new_parent_bond_by_atom = spanning_forest_parent_bonds(provisional);
    let old_ring_neighbors = ring_neighbors_by_atom(original, &old_parent_bond_by_atom);
    let new_ring_neighbors = ring_neighbors_by_atom(provisional, &new_parent_bond_by_atom);
    let old_emitted_neighbors = emitted_stereo_neighbors(original);
    let new_emitted_neighbors = emitted_stereo_neighbors(provisional);

    order
        .iter()
        .copied()
        .map(|old_atom| {
            let new_atom = new_index_of_old_atom[old_atom];
            let mapped_old_neighbors = old_emitted_neighbors[old_atom]
                .iter()
                .copied()
                .map(|neighbor| new_index_of_old_atom[neighbor])
                .collect::<Vec<_>>();
            let expr = if atom_expr_supports_emitted_parity(
                &original.atoms()[old_atom].expr,
                &mapped_old_neighbors,
                &new_emitted_neighbors[new_atom],
                old_ring_neighbors[old_atom].is_empty() && new_ring_neighbors[new_atom].is_empty(),
            ) && emitted_neighbor_permutation_is_odd(
                &mapped_old_neighbors,
                &new_emitted_neighbors[new_atom],
            ) {
                invert_atom_expr_chirality(original.atoms()[old_atom].expr.clone())
            } else {
                original.atoms()[old_atom].expr.clone()
            };
            QueryAtom {
                id: new_atom,
                component: 0,
                expr,
            }
        })
        .collect()
}

fn atom_expr_supports_emitted_parity(
    expr: &AtomExpr,
    from_neighbors: &[AtomId],
    to_neighbors: &[AtomId],
    has_no_ring_neighbors: bool,
) -> bool {
    if !has_no_ring_neighbors {
        return false;
    }
    if !atom_expr_has_single_atomic_identity(expr) {
        return false;
    }
    let emitted_degree = from_neighbors.len().max(to_neighbors.len());
    match atom_expr_chirality(expr) {
        Some(Chirality::At | Chirality::AtAt | Chirality::TH(1 | 2)) => {
            emitted_degree == 4 || (emitted_degree == 3 && atom_expr_has_single_hydrogen(expr))
        }
        Some(Chirality::AL(1 | 2)) => emitted_degree == 2,
        Some(_) | None => false,
    }
}

fn atom_expr_chirality(expr: &AtomExpr) -> Option<Chirality> {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => None,
        AtomExpr::Bracket(bracket) => bracket_tree_chirality(&bracket.tree),
    }
}

fn bracket_tree_contains_chirality(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Chirality(_)) => true,
        BracketExprTree::Primitive(_) => false,
        BracketExprTree::Not(inner) => bracket_tree_contains_chirality(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => items.iter().any(bracket_tree_contains_chirality),
    }
}

fn bracket_tree_chirality(tree: &BracketExprTree) -> Option<Chirality> {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Chirality(chirality)) => Some(*chirality),
        BracketExprTree::Primitive(_) => None,
        BracketExprTree::Not(inner) => bracket_tree_chirality(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => items.iter().find_map(bracket_tree_chirality),
    }
}

fn atom_expr_has_single_hydrogen(expr: &AtomExpr) -> bool {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => false,
        AtomExpr::Bracket(bracket) => bracket_tree_has_single_hydrogen(&bracket.tree),
    }
}

fn atom_expr_has_single_atomic_identity(expr: &AtomExpr) -> bool {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => true,
        AtomExpr::Bracket(bracket) => bracket_tree_atomic_identity_count(&bracket.tree) == 1,
    }
}

fn bracket_tree_has_single_hydrogen(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
            _,
            None | Some(NumericQuery::Exact(1)),
        )) => true,
        BracketExprTree::Primitive(_) => false,
        BracketExprTree::Not(inner) => bracket_tree_has_single_hydrogen(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => items.iter().any(bracket_tree_has_single_hydrogen),
    }
}

fn bracket_tree_atomic_identity_count(tree: &BracketExprTree) -> usize {
    match tree {
        BracketExprTree::Primitive(primitive) => {
            usize::from(atom_primitive_is_atomic_identity(primitive))
        }
        BracketExprTree::Not(inner) => bracket_tree_atomic_identity_count(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            items.iter().map(bracket_tree_atomic_identity_count).sum()
        }
    }
}

const fn atom_primitive_is_atomic_identity(primitive: &AtomPrimitive) -> bool {
    matches!(
        primitive,
        AtomPrimitive::Wildcard
            | AtomPrimitive::AliphaticAny
            | AtomPrimitive::AromaticAny
            | AtomPrimitive::Symbol { .. }
            | AtomPrimitive::Isotope { .. }
            | AtomPrimitive::IsotopeWildcard(_)
            | AtomPrimitive::AtomicNumber(_)
    )
}

fn flatten_original_atom_order(entries: &[CanonicalizedEntry]) -> Vec<AtomId> {
    entries
        .iter()
        .flat_map(|entry| entry.components.iter())
        .flat_map(|component| component.original_atom_order.iter().copied())
        .collect()
}

fn rebuild_query_from_entries(entries: &[CanonicalizedEntry]) -> QueryMol {
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut component_groups = Vec::new();
    let mut component_count = 0usize;
    let mut next_group_id = 0usize;

    for entry in entries {
        let group = entry.grouped.then(|| {
            let group_id = next_group_id;
            next_group_id += 1;
            group_id
        });

        for component in &entry.components {
            let atom_offset = atoms.len();
            let component_id = component_count;
            component_count += 1;
            component_groups.push(group);

            for atom in component.canonicalized.atoms() {
                atoms.push(QueryAtom {
                    id: atoms.len(),
                    component: component_id,
                    expr: atom.expr.clone(),
                });
            }
            for bond in component.canonicalized.bonds() {
                bonds.push(QueryBond {
                    id: bonds.len(),
                    src: atom_offset + bond.src,
                    dst: atom_offset + bond.dst,
                    expr: bond.expr.clone(),
                });
            }
        }
    }

    QueryMol::from_parts(atoms, bonds, component_count, component_groups)
}

fn build_top_level_component_entries(query: &QueryMol) -> Vec<TopLevelComponentEntry> {
    let mut entries = Vec::new();
    let mut seen_group_ids = Vec::new();
    for component_id in 0..query.component_count() {
        if let Some(group_id) = query.component_group(component_id) {
            if seen_group_ids.contains(&group_id) {
                continue;
            }
            seen_group_ids.push(group_id);
            let component_ids = (0..query.component_count())
                .filter(|&candidate_id| query.component_group(candidate_id) == Some(group_id))
                .collect();
            entries.push(TopLevelComponentEntry {
                grouped: true,
                component_ids,
            });
        } else {
            entries.push(TopLevelComponentEntry {
                grouped: false,
                component_ids: vec![component_id],
            });
        }
    }
    entries
}

fn canonical_component_groups(query: &QueryMol) -> Vec<Option<ComponentId>> {
    let mut component_groups = vec![None; query.component_count()];
    let mut next_group_id = 0usize;
    for entry in build_top_level_component_entries(query) {
        if entry.grouped && entry.component_ids.len() > 1 {
            let group_id = next_group_id;
            next_group_id += 1;
            for component_id in entry.component_ids {
                component_groups[component_id] = Some(group_id);
            }
        }
    }
    component_groups
}

fn canonical_query_state_key(query: &QueryMol) -> CanonicalQueryStateKey {
    let parts = canonical_query_state_parts(query);
    CanonicalQueryStateKey {
        atom_labels: parts.atom_labels,
        bond_edges: parts.bond_edges,
    }
}

fn canonical_whole_query_state_key(query: &QueryMol) -> CanonicalWholeQueryStateKey {
    let parts = canonical_query_state_parts(query);
    CanonicalWholeQueryStateKey {
        component_groups: query.component_groups().to_vec(),
        atom_labels: parts.atom_labels,
        bond_edges: parts.bond_edges,
    }
}

fn canonical_query_state_parts(query: &QueryMol) -> CanonicalQueryStateParts {
    let atom_labels = query
        .atoms()
        .iter()
        .map(|atom| atom.expr.to_string())
        .collect();
    let mut bond_edges = query
        .bonds()
        .iter()
        .map(|bond| {
            (
                bond.src.min(bond.dst),
                bond.src.max(bond.dst),
                undirected_bond_expr_key(&bond.expr),
            )
        })
        .collect::<Vec<_>>();
    bond_edges.sort_unstable();
    CanonicalQueryStateParts {
        atom_labels,
        bond_edges,
    }
}

fn compose_labelings(
    left: &QueryCanonicalLabeling,
    right: &QueryCanonicalLabeling,
) -> QueryCanonicalLabeling {
    let order = right
        .order()
        .iter()
        .copied()
        .map(|mid_atom| left.order()[mid_atom])
        .collect();
    QueryCanonicalLabeling::new(order)
}

fn canonical_atom_expr(
    expr: &AtomExpr,
    recursive_context: &mut RecursiveCanonicalizationContext,
    recursive_mode: RecursiveCanonicalizationMode,
) -> AtomExpr {
    match expr {
        AtomExpr::Wildcard => AtomExpr::Wildcard,
        AtomExpr::Bare { element, aromatic } => AtomExpr::Bare {
            element: *element,
            aromatic: *aromatic,
        },
        AtomExpr::Bracket(expr) => {
            let canonical = canonical_bracket_expr(expr, recursive_context, recursive_mode);
            if canonical.atom_map.is_none() {
                match canonical.tree {
                    BracketExprTree::Primitive(AtomPrimitive::Wildcard) => AtomExpr::Wildcard,
                    BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                        HydrogenKind::Total,
                        None,
                    )) => AtomExpr::Bracket(BracketExpr {
                        tree: BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(1)),
                        atom_map: None,
                    }),
                    _ => AtomExpr::Bracket(canonical),
                }
            } else {
                AtomExpr::Bracket(canonical)
            }
        }
    }
}

fn canonical_bracket_expr(
    expr: &BracketExpr,
    recursive_context: &mut RecursiveCanonicalizationContext,
    recursive_mode: RecursiveCanonicalizationMode,
) -> BracketExpr {
    let mut normalized = expr.clone();
    normalized
        .normalize()
        .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
    let mut tree = canonical_bracket_tree(normalized.tree, recursive_context, recursive_mode);
    tree = simplify_bracket_tree(tree);
    let mut bracket = BracketExpr {
        tree,
        atom_map: normalized.atom_map,
    };
    bracket
        .normalize()
        .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
    let mut tree = canonical_bracket_tree(
        bracket.tree,
        recursive_context,
        RecursiveCanonicalizationMode::Preserve,
    );
    tree = simplify_bracket_tree(tree);
    tree = simplify_top_level_negated_numeric_primitive(tree);
    let mut bracket = BracketExpr {
        tree,
        atom_map: bracket.atom_map,
    };
    bracket
        .normalize()
        .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
    BracketExpr {
        tree: bracket.tree,
        atom_map: bracket.atom_map,
    }
}

fn canonical_bracket_tree(
    tree: BracketExprTree,
    recursive_context: &mut RecursiveCanonicalizationContext,
    recursive_mode: RecursiveCanonicalizationMode,
) -> BracketExprTree {
    match tree {
        BracketExprTree::Primitive(primitive) => BracketExprTree::Primitive(
            canonical_atom_primitive(primitive, recursive_context, recursive_mode),
        ),
        BracketExprTree::Not(inner) => BracketExprTree::Not(Box::new(canonical_bracket_tree(
            *inner,
            recursive_context,
            recursive_mode,
        ))),
        BracketExprTree::HighAnd(items) => {
            let mut items = items
                .into_iter()
                .map(|item| canonical_bracket_tree(item, recursive_context, recursive_mode))
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BracketExprTree::to_string);
            items.dedup();
            BracketExprTree::HighAnd(items)
        }
        BracketExprTree::Or(items) => {
            let mut items = items
                .into_iter()
                .map(|item| canonical_bracket_tree(item, recursive_context, recursive_mode))
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BracketExprTree::to_string);
            items.dedup();
            BracketExprTree::Or(items)
        }
        BracketExprTree::LowAnd(items) => {
            let mut items = items
                .into_iter()
                .map(|item| canonical_bracket_tree(item, recursive_context, recursive_mode))
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BracketExprTree::to_string);
            items.dedup();
            BracketExprTree::LowAnd(items)
        }
    }
}

fn canonical_atom_primitive(
    primitive: AtomPrimitive,
    recursive_context: &mut RecursiveCanonicalizationContext,
    recursive_mode: RecursiveCanonicalizationMode,
) -> AtomPrimitive {
    match primitive {
        AtomPrimitive::Symbol {
            element: elements_rs::Element::H,
            aromatic: false,
        } => AtomPrimitive::AtomicNumber(1),
        AtomPrimitive::Degree(query) => canonical_count_primitive(query, AtomPrimitive::Degree),
        AtomPrimitive::Connectivity(query) => {
            canonical_count_primitive(query, AtomPrimitive::Connectivity)
        }
        AtomPrimitive::Valence(query) => canonical_count_primitive(query, AtomPrimitive::Valence),
        AtomPrimitive::Hydrogen(kind, query) => {
            canonical_count_primitive(query, |query| AtomPrimitive::Hydrogen(kind, query))
        }
        AtomPrimitive::RingMembership(query) => {
            canonical_ring_primitive(query, AtomPrimitive::RingMembership)
        }
        AtomPrimitive::RingSize(query) => canonical_ring_primitive(query, AtomPrimitive::RingSize),
        AtomPrimitive::RingConnectivity(query) => {
            canonical_ring_primitive(query, AtomPrimitive::RingConnectivity)
        }
        AtomPrimitive::Hybridization(query) => {
            let query = canonical_numeric_query(query);
            if numeric_query_is_unbounded_from_zero(query) {
                AtomPrimitive::Wildcard
            } else {
                AtomPrimitive::Hybridization(query)
            }
        }
        AtomPrimitive::HeteroNeighbor(query) => {
            canonical_count_primitive(query, AtomPrimitive::HeteroNeighbor)
        }
        AtomPrimitive::AliphaticHeteroNeighbor(query) => {
            canonical_count_primitive(query, AtomPrimitive::AliphaticHeteroNeighbor)
        }
        AtomPrimitive::RecursiveQuery(query) => match recursive_mode {
            RecursiveCanonicalizationMode::Canonicalize => AtomPrimitive::RecursiveQuery(Box::new(
                recursive_context.canonicalize_recursive_query(&query),
            )),
            RecursiveCanonicalizationMode::Preserve => AtomPrimitive::RecursiveQuery(query),
        },
        other => other,
    }
}

fn canonical_count_primitive(
    query: Option<NumericQuery>,
    build: impl FnOnce(Option<NumericQuery>) -> AtomPrimitive,
) -> AtomPrimitive {
    let query = canonical_count_query(query);
    if optional_numeric_query_is_unbounded_from_zero(query) {
        AtomPrimitive::Wildcard
    } else {
        build(query)
    }
}

fn canonical_ring_primitive(
    query: Option<NumericQuery>,
    build: impl FnOnce(Option<NumericQuery>) -> AtomPrimitive,
) -> AtomPrimitive {
    let query = query.map(canonical_numeric_query);
    if optional_numeric_query_is_unbounded_from_zero(query) {
        AtomPrimitive::Wildcard
    } else if optional_numeric_query_is_unbounded_from_one(query) {
        build(None)
    } else {
        build(query)
    }
}

const fn optional_numeric_query_is_unbounded_from_zero(query: Option<NumericQuery>) -> bool {
    matches!(
        query,
        Some(NumericQuery::Range(NumericRange {
            min: None,
            max: None
        }))
    )
}

const fn numeric_query_is_unbounded_from_zero(query: NumericQuery) -> bool {
    matches!(
        query,
        NumericQuery::Range(NumericRange {
            min: None,
            max: None
        })
    )
}

const fn optional_numeric_query_is_unbounded_from_one(query: Option<NumericQuery>) -> bool {
    matches!(
        query,
        Some(NumericQuery::Range(NumericRange {
            min: Some(1),
            max: None
        }))
    )
}

fn canonical_count_query(query: Option<NumericQuery>) -> Option<NumericQuery> {
    match query.map(canonical_numeric_query) {
        Some(NumericQuery::Exact(1)) => None,
        other => other,
    }
}

fn canonical_numeric_query(query: NumericQuery) -> NumericQuery {
    match query {
        NumericQuery::Range(range) => {
            let range = canonical_numeric_range(range);
            let lower = range.min.unwrap_or(0);
            if range.max == Some(lower) {
                NumericQuery::Exact(lower)
            } else {
                NumericQuery::Range(range)
            }
        }
        NumericQuery::Exact(value) => NumericQuery::Exact(value),
    }
}

fn canonical_numeric_range(mut range: NumericRange) -> NumericRange {
    if range.min == Some(0) {
        range.min = None;
    }
    range
}

fn simplify_bracket_tree(tree: BracketExprTree) -> BracketExprTree {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(query)) => {
            simplify_single_atom_recursive_query(*query)
        }
        BracketExprTree::Primitive(primitive) => BracketExprTree::Primitive(primitive),
        BracketExprTree::Not(inner) => simplify_bracket_not(simplify_bracket_tree(*inner)),
        BracketExprTree::HighAnd(items) => simplify_bracket_and(items, BracketAndKind::High),
        BracketExprTree::LowAnd(items) => simplify_bracket_and(items, BracketAndKind::Low),
        BracketExprTree::Or(items) => simplify_bracket_or(items),
    }
}

fn simplify_single_atom_recursive_query(query: QueryMol) -> BracketExprTree {
    if query.atom_count() != 1 || query.bond_count() != 0 {
        return BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(Box::new(query)));
    }

    single_atom_recursive_query_equivalent(&query).map_or_else(
        || BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(Box::new(query))),
        simplify_bracket_tree,
    )
}

fn single_atom_recursive_query_equivalent(query: &QueryMol) -> Option<BracketExprTree> {
    match &query.atoms()[0].expr {
        AtomExpr::Wildcard => Some(BracketExprTree::Primitive(AtomPrimitive::Wildcard)),
        AtomExpr::Bare { element, aromatic } => {
            Some(BracketExprTree::Primitive(AtomPrimitive::Symbol {
                element: *element,
                aromatic: *aromatic,
            }))
        }
        AtomExpr::Bracket(expr) if expr.atom_map.is_none() => Some(expr.tree.clone()),
        AtomExpr::Bracket(_) => None,
    }
}

fn simplify_bracket_not(inner: BracketExprTree) -> BracketExprTree {
    if is_false_bracket_tree(&inner) {
        BracketExprTree::Primitive(AtomPrimitive::Wildcard)
    } else if let BracketExprTree::Not(grandchild) = inner {
        *grandchild
    } else if let BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) = inner {
        simplify_bracket_or(items.into_iter().map(simplify_bracket_not).collect())
    } else if let BracketExprTree::Or(items) = inner {
        simplify_bracket_and(
            items.into_iter().map(simplify_bracket_not).collect(),
            BracketAndKind::High,
        )
    } else if matches!(
        inner,
        BracketExprTree::Primitive(AtomPrimitive::AliphaticAny)
    ) {
        BracketExprTree::Primitive(AtomPrimitive::AromaticAny)
    } else if matches!(
        inner,
        BracketExprTree::Primitive(AtomPrimitive::AromaticAny)
    ) {
        BracketExprTree::Primitive(AtomPrimitive::AliphaticAny)
    } else {
        BracketExprTree::Not(Box::new(inner))
    }
}

fn simplify_top_level_negated_numeric_primitive(tree: BracketExprTree) -> BracketExprTree {
    let BracketExprTree::Not(inner) = tree else {
        return tree;
    };
    let BracketExprTree::Primitive(primitive) = inner.as_ref() else {
        return BracketExprTree::Not(inner);
    };
    simplify_negated_numeric_primitive(primitive).unwrap_or(BracketExprTree::Not(inner))
}

fn simplify_negated_numeric_primitive(primitive: &AtomPrimitive) -> Option<BracketExprTree> {
    match primitive {
        AtomPrimitive::Degree(query) => complement_count_query(*query, AtomPrimitive::Degree),
        AtomPrimitive::Connectivity(query) => {
            complement_count_query(*query, AtomPrimitive::Connectivity)
        }
        AtomPrimitive::Valence(query) => complement_count_query(*query, AtomPrimitive::Valence),
        AtomPrimitive::Hydrogen(kind, query) => {
            complement_count_query(*query, |query| AtomPrimitive::Hydrogen(*kind, query))
        }
        AtomPrimitive::RingMembership(query) => {
            complement_ring_query(*query, AtomPrimitive::RingMembership)
        }
        AtomPrimitive::RingSize(query) => complement_ring_query(*query, AtomPrimitive::RingSize),
        AtomPrimitive::RingConnectivity(query) => {
            complement_ring_query(*query, AtomPrimitive::RingConnectivity)
        }
        AtomPrimitive::Hybridization(query) => {
            complement_zero_based_numeric_range(numeric_query_range(*query), |range| {
                BracketExprTree::Primitive(AtomPrimitive::Hybridization(numeric_query_from_range(
                    range,
                )))
            })
        }
        AtomPrimitive::HeteroNeighbor(query) => {
            complement_count_query(*query, AtomPrimitive::HeteroNeighbor)
        }
        AtomPrimitive::AliphaticHeteroNeighbor(query) => {
            complement_count_query(*query, AtomPrimitive::AliphaticHeteroNeighbor)
        }
        _ => None,
    }
}

fn complement_count_query(
    query: Option<NumericQuery>,
    build: impl Fn(Option<NumericQuery>) -> AtomPrimitive,
) -> Option<BracketExprTree> {
    complement_zero_based_numeric_range(numeric_count_range(query), |range| {
        BracketExprTree::Primitive(build(count_query_from_range(range).into_option()))
    })
}

fn complement_ring_query(
    query: Option<NumericQuery>,
    build: impl Fn(Option<NumericQuery>) -> AtomPrimitive,
) -> Option<BracketExprTree> {
    complement_zero_based_numeric_range(numeric_ring_range(query), |range| {
        BracketExprTree::Primitive(build(ring_query_from_range(range).into_option()))
    })
}

fn complement_zero_based_numeric_range(
    range: NumericRange,
    build: impl Fn(NumericRange) -> BracketExprTree,
) -> Option<BracketExprTree> {
    let range = canonical_numeric_range(range);
    if range.min.unwrap_or(0) != 0 {
        return None;
    }

    if range.max != Some(0) {
        return None;
    }

    Some(build(NumericRange {
        min: Some(1),
        max: None,
    }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BracketAndKind {
    High,
    Low,
}

fn simplify_bracket_and(items: Vec<BracketExprTree>, kind: BracketAndKind) -> BracketExprTree {
    let mut items = items
        .into_iter()
        .map(simplify_bracket_tree)
        .collect::<Vec<_>>();
    flatten_bracket_and_items(&mut items, kind);
    while replace_negated_numeric_pair_with_residual(&mut items) {}
    if items.iter().any(is_false_bracket_tree)
        || bracket_and_contains_complement_pair(&items)
        || bracket_and_contains_mutually_exclusive_pair(&items)
    {
        return false_bracket_tree();
    }
    if items.len() > 1 {
        items.retain(|item| !matches!(item, BracketExprTree::Primitive(AtomPrimitive::Wildcard)));
    }
    merge_numeric_bracket_conjunctions(&mut items);
    synthesize_symbol_conjunctions(&mut items);
    synthesize_isotope_conjunctions(&mut items);
    synthesize_negated_symbol_conjunctions(&mut items);
    while let Some(changed) = prune_bracket_disjunctions_against_conjunction_terms(&mut items) {
        if !changed {
            break;
        }
    }
    while subtract_negated_isotope_conjunction_terms(&mut items) {}
    while subtract_negated_bracket_conjunction_terms(&mut items) {}
    while replace_negated_numeric_pair_with_residual(&mut items) {}
    if items.iter().any(is_false_bracket_tree)
        || bracket_and_contains_complement_pair(&items)
        || bracket_and_contains_mutually_exclusive_pair(&items)
    {
        return false_bracket_tree();
    }
    if kind == BracketAndKind::Low {
        loop {
            let mut changed = flatten_bracket_and_items(&mut items, kind);
            while remove_redundant_bracket_conjunction_consensus_terms(&mut items) {
                changed = true;
            }
            while factor_common_bracket_conjunction_disjunction_terms(&mut items) {
                changed = true;
            }
            while replace_negated_bracket_disjunction_alternatives_with_constrained_residual(
                &mut items,
            ) {
                changed = true;
            }
            while distribute_bracket_conjunction_term_into_disjunction(&mut items) {
                changed = true;
            }
            while let Some(pruned) =
                prune_bracket_disjunctions_against_conjunction_terms(&mut items)
            {
                if !pruned {
                    break;
                }
                changed = true;
            }
            if !changed {
                break;
            }
            if items.iter().any(is_false_bracket_tree) {
                return false_bracket_tree();
            }
        }
    }
    remove_implied_bracket_conjunction_terms(&mut items);
    items.sort_by_cached_key(BracketExprTree::to_string);
    items.dedup();

    match items.as_slice() {
        [] => BracketExprTree::Primitive(AtomPrimitive::Wildcard),
        [single] => single.clone(),
        _ if !can_render_bracket_and_as_high_and(&items) => BracketExprTree::LowAnd(items),
        _ => BracketExprTree::HighAnd(items),
    }
}

fn flatten_bracket_and_items(items: &mut Vec<BracketExprTree>, kind: BracketAndKind) -> bool {
    let mut flattened = Vec::with_capacity(items.len());
    let mut changed = false;
    for item in items.drain(..) {
        match item {
            BracketExprTree::HighAnd(nested) => {
                changed = true;
                flattened.extend(nested);
            }
            BracketExprTree::LowAnd(nested) if kind == BracketAndKind::Low => {
                changed = true;
                flattened.extend(nested);
            }
            other => flattened.push(other),
        }
    }
    *items = flattened;
    changed
}

fn false_bracket_tree() -> BracketExprTree {
    BracketExprTree::Not(Box::new(BracketExprTree::Primitive(
        AtomPrimitive::Wildcard,
    )))
}

fn is_false_bracket_tree(tree: &BracketExprTree) -> bool {
    matches!(
        tree,
        BracketExprTree::Not(inner)
            if matches!(inner.as_ref(), BracketExprTree::Primitive(AtomPrimitive::Wildcard))
    )
}

fn prune_bracket_disjunctions_against_conjunction_terms(
    items: &mut [BracketExprTree],
) -> Option<bool> {
    for disjunction_index in 0..items.len() {
        let BracketExprTree::Or(alternatives) = &items[disjunction_index] else {
            continue;
        };
        let conjunction_contains_negation = items.iter().enumerate().any(|(other_index, other)| {
            other_index != disjunction_index && bracket_tree_contains_negation(other)
        });
        let retained = alternatives
            .iter()
            .filter(|alternative| {
                !items.iter().enumerate().any(|(other_index, other)| {
                    other_index != disjunction_index
                        && bracket_trees_are_mutually_exclusive(alternative, other)
                }) && !bracket_alternative_is_inconsistent_with_conjunction(
                    alternative,
                    items,
                    disjunction_index,
                    conjunction_contains_negation,
                )
            })
            .cloned()
            .collect::<Vec<_>>();

        if retained.is_empty() {
            items[disjunction_index] = false_bracket_tree();
            return None;
        }
        if retained.len() != alternatives.len() {
            items[disjunction_index] = simplify_bracket_or(retained);
            return Some(true);
        }
    }

    Some(false)
}

fn bracket_alternative_is_inconsistent_with_conjunction(
    alternative: &BracketExprTree,
    items: &[BracketExprTree],
    disjunction_index: usize,
    conjunction_contains_negation: bool,
) -> bool {
    if !conjunction_contains_negation && !bracket_tree_contains_negation(alternative) {
        return false;
    }

    let mut conjunction = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (index != disjunction_index).then_some(item.clone()))
        .collect::<Vec<_>>();
    conjunction.push(alternative.clone());
    flatten_bracket_and_items(&mut conjunction, BracketAndKind::Low);
    if conjunction
        .iter()
        .any(|item| matches!(item, BracketExprTree::Or(_)))
    {
        return false;
    }

    is_false_bracket_tree(&simplify_bracket_and(conjunction, BracketAndKind::High))
}

fn bracket_tree_contains_negation(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Not(_) => true,
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => items.iter().any(bracket_tree_contains_negation),
        BracketExprTree::Primitive(_) => false,
    }
}

fn factor_common_bracket_conjunction_disjunction_terms(items: &mut Vec<BracketExprTree>) -> bool {
    for left_index in 0..items.len() {
        let left_alternatives = bracket_disjunction_factor_items(&items[left_index]);
        if left_alternatives.len() < 2 {
            continue;
        }

        for right_index in (left_index + 1)..items.len() {
            let right_alternatives = bracket_disjunction_factor_items(&items[right_index]);
            if right_alternatives.len() < 2 {
                continue;
            }

            let common = left_alternatives
                .iter()
                .filter(|item| right_alternatives.contains(item))
                .cloned()
                .collect::<Vec<_>>();
            if common.is_empty() {
                continue;
            }

            let Some(replacement) =
                factor_bracket_disjunction_pair(&left_alternatives, &right_alternatives, common)
            else {
                continue;
            };
            items.remove(right_index);
            items.remove(left_index);
            items.push(replacement);
            return true;
        }
    }

    false
}

fn factor_bracket_disjunction_pair(
    left_alternatives: &[BracketExprTree],
    right_alternatives: &[BracketExprTree],
    mut common: Vec<BracketExprTree>,
) -> Option<BracketExprTree> {
    let left_remainder = disjunction_remainder(left_alternatives, &common);
    let right_remainder = disjunction_remainder(right_alternatives, &common);
    if left_remainder.is_empty() || right_remainder.is_empty() {
        return Some(simplify_bracket_or(common));
    }

    let remainder = simplify_bracket_and(
        vec![
            simplify_bracket_or(left_remainder),
            simplify_bracket_or(right_remainder),
        ],
        BracketAndKind::Low,
    );
    if matches!(remainder, BracketExprTree::LowAnd(_)) {
        return None;
    }
    if !is_false_bracket_tree(&remainder) {
        common.push(remainder);
    }
    Some(simplify_bracket_or(common))
}

fn remove_redundant_bracket_conjunction_consensus_terms(items: &mut Vec<BracketExprTree>) -> bool {
    for candidate_index in 0..items.len() {
        for left_index in 0..items.len() {
            if left_index == candidate_index {
                continue;
            }

            for right_index in (left_index + 1)..items.len() {
                if right_index == candidate_index
                    || !bracket_conjunction_consensus_term_is_redundant(
                        &items[left_index],
                        &items[right_index],
                        &items[candidate_index],
                    )
                {
                    continue;
                }

                items.remove(candidate_index);
                return true;
            }
        }
    }

    false
}

fn replace_negated_bracket_disjunction_alternatives_with_constrained_residual(
    items: &mut [BracketExprTree],
) -> bool {
    for disjunction_index in 0..items.len() {
        let BracketExprTree::Or(alternatives) = &items[disjunction_index] else {
            continue;
        };
        let mut base_terms = items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| (index != disjunction_index).then_some(item.clone()))
            .collect::<Vec<_>>();
        flatten_bracket_and_items(&mut base_terms, BracketAndKind::Low);
        if base_terms
            .iter()
            .any(|item| matches!(item, BracketExprTree::Or(_)))
        {
            continue;
        }

        for alternative_index in 0..alternatives.len() {
            let BracketExprTree::Not(excluded) = &alternatives[alternative_index] else {
                continue;
            };
            let Some(residual) = subtract_from_bracket_conjunction_terms(&base_terms, excluded)
            else {
                continue;
            };
            if bracket_trees_are_equivalent(&residual, &alternatives[alternative_index]) {
                continue;
            }

            let mut replacement = alternatives.clone();
            replacement[alternative_index] = residual;
            items[disjunction_index] = simplify_bracket_or(replacement);
            return true;
        }
    }

    false
}

fn subtract_from_bracket_conjunction_terms(
    base_terms: &[BracketExprTree],
    excluded: &BracketExprTree,
) -> Option<BracketExprTree> {
    let base = simplify_bracket_and(base_terms.to_vec(), BracketAndKind::High);
    if let Some(residual) = subtract_bracket_tree(&base, excluded) {
        return Some(residual);
    }

    base_terms
        .iter()
        .find_map(|term| subtract_bracket_tree(term, excluded))
}

fn distribute_bracket_conjunction_term_into_disjunction(items: &mut Vec<BracketExprTree>) -> bool {
    for disjunction_index in 0..items.len() {
        let BracketExprTree::Or(alternatives) = &items[disjunction_index] else {
            continue;
        };
        let alternatives = alternatives.clone();

        for term_index in 0..items.len() {
            if term_index == disjunction_index {
                continue;
            }
            let term = items[term_index].clone();
            if matches!(term, BracketExprTree::Or(_)) {
                continue;
            }
            let implied_by_alternative = alternatives
                .iter()
                .map(|alternative| bracket_tree_implies(alternative, &term))
                .collect::<Vec<_>>();

            if !implied_by_alternative.iter().any(|implied| *implied)
                || implied_by_alternative.iter().all(|implied| *implied)
            {
                continue;
            }
            let replacement = alternatives
                .into_iter()
                .zip(implied_by_alternative)
                .map(|(alternative, implied)| {
                    if implied {
                        alternative
                    } else {
                        simplify_bracket_conjunction_pair(term.clone(), alternative)
                    }
                })
                .collect::<Vec<_>>();
            remove_indices_descending(items, &[disjunction_index, term_index]);
            push_flattened_low_bracket_and_item(items, simplify_bracket_or(replacement));
            return true;
        }
    }

    false
}

fn push_flattened_low_bracket_and_item(items: &mut Vec<BracketExprTree>, item: BracketExprTree) {
    match item {
        BracketExprTree::HighAnd(nested) | BracketExprTree::LowAnd(nested) => items.extend(nested),
        other => items.push(other),
    }
}

fn simplify_bracket_conjunction_pair(
    left: BracketExprTree,
    right: BracketExprTree,
) -> BracketExprTree {
    let items = vec![left, right];
    let kind = if can_render_bracket_and_as_high_and(&items) {
        BracketAndKind::High
    } else {
        BracketAndKind::Low
    };
    simplify_bracket_and(items, kind)
}

fn bracket_conjunction_consensus_term_is_redundant(
    left: &BracketExprTree,
    right: &BracketExprTree,
    candidate: &BracketExprTree,
) -> bool {
    let left_alternatives = bracket_disjunction_factor_items(left);
    let right_alternatives = bracket_disjunction_factor_items(right);
    if left_alternatives.len() < 2 || right_alternatives.len() < 2 {
        return false;
    }

    for left_complement_index in 0..left_alternatives.len() {
        for right_complement_index in 0..right_alternatives.len() {
            if !bracket_trees_are_complements(
                &left_alternatives[left_complement_index],
                &right_alternatives[right_complement_index],
            ) {
                continue;
            }

            let mut consensus =
                bracket_items_without_index(&left_alternatives, left_complement_index);
            consensus.extend(bracket_items_without_index(
                &right_alternatives,
                right_complement_index,
            ));
            sort_and_dedup_bracket_items(&mut consensus);
            let consensus = simplify_bracket_or(consensus);
            if bracket_tree_implies(&consensus, candidate) {
                return true;
            }
        }
    }

    false
}

fn bracket_disjunction_factor_items(tree: &BracketExprTree) -> Vec<BracketExprTree> {
    let mut alternatives = match tree {
        BracketExprTree::Or(items) => items.clone(),
        other => vec![other.clone()],
    };
    sort_and_dedup_bracket_items(&mut alternatives);
    alternatives
}

fn disjunction_remainder<T: Clone + PartialEq>(items: &[T], common: &[T]) -> Vec<T> {
    items
        .iter()
        .filter(|item| !common.contains(item))
        .cloned()
        .collect()
}

fn subtract_negated_isotope_conjunction_terms(items: &mut Vec<BracketExprTree>) -> bool {
    for negated_index in 0..items.len() {
        let BracketExprTree::Not(excluded) = &items[negated_index] else {
            continue;
        };
        let BracketExprTree::Primitive(AtomPrimitive::Isotope { isotope, aromatic }) =
            excluded.as_ref()
        else {
            continue;
        };
        let isotope = *isotope;
        let aromatic = *aromatic;

        let Some(atomic_number_index) = find_atomic_number_item(items, isotope) else {
            continue;
        };
        let Some(isotope_wildcard_index) = find_isotope_wildcard_item(items, isotope) else {
            continue;
        };
        if atomic_number_index == negated_index || isotope_wildcard_index == negated_index {
            continue;
        }

        remove_indices_descending(
            items,
            &[negated_index, atomic_number_index, isotope_wildcard_index],
        );
        items.push(isotope_complement_tree(isotope, aromatic));
        return true;
    }

    false
}

fn subtract_negated_bracket_conjunction_terms(items: &mut Vec<BracketExprTree>) -> bool {
    for negated_index in 0..items.len() {
        let BracketExprTree::Not(excluded) = &items[negated_index] else {
            continue;
        };
        for base_index in 0..items.len() {
            if base_index == negated_index {
                continue;
            }
            let Some(replacement) = subtract_bracket_tree(&items[base_index], excluded) else {
                continue;
            };

            if base_index > negated_index {
                items.remove(base_index);
                items.remove(negated_index);
            } else {
                items.remove(negated_index);
                items.remove(base_index);
            }
            items.push(replacement);
            return true;
        }
    }

    false
}

fn subtract_bracket_tree(
    base: &BracketExprTree,
    excluded: &BracketExprTree,
) -> Option<BracketExprTree> {
    match (base, excluded) {
        (
            BracketExprTree::Primitive(base_primitive),
            BracketExprTree::Primitive(excluded_primitive),
        ) => subtract_atom_primitive(base_primitive, excluded_primitive),
        (
            BracketExprTree::HighAnd(base_items) | BracketExprTree::LowAnd(base_items),
            BracketExprTree::Primitive(AtomPrimitive::Isotope { isotope, aromatic }),
        ) => subtract_isotope_from_conjunction(base_items, *isotope, *aromatic),
        _ => None,
    }
}

fn subtract_isotope_from_conjunction(
    base_items: &[BracketExprTree],
    excluded_isotope: Isotope,
    excluded_aromatic: bool,
) -> Option<BracketExprTree> {
    let atomic_number_index = find_atomic_number_item(base_items, excluded_isotope)?;
    let isotope_wildcard_index = find_isotope_wildcard_item(base_items, excluded_isotope)?;

    let mut retained = base_items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            (index != atomic_number_index && index != isotope_wildcard_index)
                .then_some(item.clone())
        })
        .collect::<Vec<_>>();
    retained.push(isotope_complement_tree(excluded_isotope, excluded_aromatic));
    Some(simplify_bracket_and(retained, BracketAndKind::High))
}

fn find_atomic_number_item(items: &[BracketExprTree], isotope: Isotope) -> Option<usize> {
    items.iter().position(|item| {
        matches!(
            item,
            BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(atomic_number))
                if *atomic_number == u16::from(isotope.element().atomic_number())
        )
    })
}

fn find_isotope_wildcard_item(items: &[BracketExprTree], isotope: Isotope) -> Option<usize> {
    items.iter().position(|item| {
        matches!(
            item,
            BracketExprTree::Primitive(AtomPrimitive::IsotopeWildcard(mass_number))
                if *mass_number == isotope.mass_number()
        )
    })
}

fn isotope_complement_tree(isotope: Isotope, excluded_aromatic: bool) -> BracketExprTree {
    if excluded_aromatic {
        BracketExprTree::Primitive(AtomPrimitive::Isotope {
            isotope,
            aromatic: false,
        })
    } else if element_supports_aromatic_bracket_symbol(isotope.element()) {
        BracketExprTree::Primitive(AtomPrimitive::Isotope {
            isotope,
            aromatic: true,
        })
    } else {
        false_bracket_tree()
    }
}

fn remove_indices_descending<T>(items: &mut Vec<T>, indices: &[usize]) {
    let mut indices = indices.to_vec();
    indices.sort_unstable();
    indices.dedup();
    for index in indices.into_iter().rev() {
        items.remove(index);
    }
}

fn replace_negated_numeric_pair_with_residual(items: &mut Vec<BracketExprTree>) -> bool {
    for left_index in 0..items.len() {
        let Some(left) = negated_numeric_primitive(&items[left_index]) else {
            continue;
        };
        for right_index in (left_index + 1)..items.len() {
            let Some(right) = negated_numeric_primitive(&items[right_index]) else {
                continue;
            };
            let Some(replacement) = residual_for_negated_numeric_pair(left, right) else {
                continue;
            };
            remove_indices_descending(items, &[left_index, right_index]);
            items.push(replacement);
            return true;
        }
    }

    false
}

fn negated_numeric_primitive(tree: &BracketExprTree) -> Option<&AtomPrimitive> {
    let BracketExprTree::Not(inner) = tree else {
        return None;
    };
    let BracketExprTree::Primitive(primitive) = inner.as_ref() else {
        return None;
    };
    Some(primitive)
}

fn residual_for_negated_numeric_pair(
    left: &AtomPrimitive,
    right: &AtomPrimitive,
) -> Option<BracketExprTree> {
    match (left, right) {
        (AtomPrimitive::Degree(left), AtomPrimitive::Degree(right)) => {
            residual_for_negated_count_pair(*left, *right, AtomPrimitive::Degree)
        }
        (AtomPrimitive::Connectivity(left), AtomPrimitive::Connectivity(right)) => {
            residual_for_negated_count_pair(*left, *right, AtomPrimitive::Connectivity)
        }
        (AtomPrimitive::Valence(left), AtomPrimitive::Valence(right)) => {
            residual_for_negated_count_pair(*left, *right, AtomPrimitive::Valence)
        }
        (AtomPrimitive::HeteroNeighbor(left), AtomPrimitive::HeteroNeighbor(right)) => {
            residual_for_negated_count_pair(*left, *right, AtomPrimitive::HeteroNeighbor)
        }
        (
            AtomPrimitive::AliphaticHeteroNeighbor(left),
            AtomPrimitive::AliphaticHeteroNeighbor(right),
        ) => residual_for_negated_count_pair(*left, *right, AtomPrimitive::AliphaticHeteroNeighbor),
        (AtomPrimitive::Hydrogen(left_kind, left), AtomPrimitive::Hydrogen(right_kind, right))
            if left_kind == right_kind =>
        {
            residual_for_negated_count_pair(*left, *right, |query| {
                AtomPrimitive::Hydrogen(*left_kind, query)
            })
        }
        (AtomPrimitive::RingMembership(left), AtomPrimitive::RingMembership(right)) => {
            residual_for_negated_ring_pair(*left, *right, AtomPrimitive::RingMembership)
        }
        (AtomPrimitive::RingSize(left), AtomPrimitive::RingSize(right)) => {
            residual_for_negated_ring_pair(*left, *right, AtomPrimitive::RingSize)
        }
        (AtomPrimitive::RingConnectivity(left), AtomPrimitive::RingConnectivity(right)) => {
            residual_for_negated_ring_pair(*left, *right, AtomPrimitive::RingConnectivity)
        }
        _ => None,
    }
}

fn residual_for_negated_count_pair(
    left: Option<NumericQuery>,
    right: Option<NumericQuery>,
    build: impl Fn(Option<NumericQuery>) -> AtomPrimitive,
) -> Option<BracketExprTree> {
    residual_for_negated_numeric_ranges(
        numeric_count_range(left),
        numeric_count_range(right),
        |range| BracketExprTree::Primitive(build(count_query_from_range(range).into_option())),
    )
}

fn residual_for_negated_ring_pair(
    left: Option<NumericQuery>,
    right: Option<NumericQuery>,
    build: impl Fn(Option<NumericQuery>) -> AtomPrimitive,
) -> Option<BracketExprTree> {
    residual_for_negated_numeric_ranges(
        numeric_ring_range(left),
        numeric_ring_range(right),
        |range| BracketExprTree::Primitive(build(ring_query_from_range(range).into_option())),
    )
}

fn residual_for_negated_numeric_ranges(
    left: NumericRange,
    right: NumericRange,
    build: impl Fn(NumericRange) -> BracketExprTree,
) -> Option<BracketExprTree> {
    let retained = nonnegative_range_union_complement(left, right);

    match retained.as_slice() {
        [] => Some(false_bracket_tree()),
        [single] => Some(build(canonical_numeric_range(*single))),
        _ => None,
    }
}

fn nonnegative_range_union_complement(
    left: NumericRange,
    right: NumericRange,
) -> Vec<NumericRange> {
    let left_min = left.min.unwrap_or(0);
    let right_min = right.min.unwrap_or(0);
    let (first, second, second_min) = if left_min <= right_min {
        (left, right, right_min)
    } else {
        (right, left, left_min)
    };

    let covered = if first
        .max
        .is_some_and(|first_max| first_max.saturating_add(1) < second_min)
    {
        vec![first, second]
    } else {
        let max = match (first.max, second.max) {
            (None, _) | (_, None) => None,
            (Some(first_max), Some(second_max)) => Some(first_max.max(second_max)),
        };
        vec![NumericRange {
            min: Some(first.min.unwrap_or(0)),
            max,
        }]
    };
    nonnegative_ranges_complement(&covered)
}

fn nonnegative_ranges_complement(covered: &[NumericRange]) -> Vec<NumericRange> {
    let mut next_retained_min = Some(0u16);
    let mut retained = Vec::new();

    for range in covered {
        let Some(start) = next_retained_min else {
            break;
        };
        let range_min = range.min.unwrap_or(0);
        if start < range_min {
            retained.push(NumericRange {
                min: Some(start),
                max: Some(range_min - 1),
            });
        }
        next_retained_min = range.max.and_then(|max| max.checked_add(1));
    }

    if let Some(start) = next_retained_min {
        retained.push(NumericRange {
            min: Some(start),
            max: None,
        });
    }

    retained
}

fn subtract_atom_primitive(
    base: &AtomPrimitive,
    excluded: &AtomPrimitive,
) -> Option<BracketExprTree> {
    subtract_atomic_number_primitive(base, excluded)
        .or_else(|| subtract_numeric_atom_primitive(base, excluded))
}

fn subtract_atomic_number_primitive(
    base: &AtomPrimitive,
    excluded: &AtomPrimitive,
) -> Option<BracketExprTree> {
    let AtomPrimitive::AtomicNumber(atomic_number) = base else {
        return None;
    };
    let (element, aromatic) = match excluded {
        AtomPrimitive::Symbol { element, aromatic } => (*element, !*aromatic),
        AtomPrimitive::AliphaticAny => {
            (element_for_symbol_conjunction(*atomic_number, true)?, true)
        }
        AtomPrimitive::AromaticAny => (
            element_for_symbol_conjunction(*atomic_number, false)?,
            false,
        ),
        _ => return None,
    };
    if u16::from(element.atomic_number()) != *atomic_number {
        return None;
    }
    if aromatic && !element_supports_aromatic_bracket_symbol(element) {
        return None;
    }
    Some(BracketExprTree::Primitive(AtomPrimitive::Symbol {
        element,
        aromatic,
    }))
}

fn subtract_numeric_atom_primitive(
    base: &AtomPrimitive,
    excluded: &AtomPrimitive,
) -> Option<BracketExprTree> {
    match (base, excluded) {
        (AtomPrimitive::Degree(base), AtomPrimitive::Degree(excluded)) => {
            subtract_count_query(*base, *excluded, AtomPrimitive::Degree)
        }
        (AtomPrimitive::Connectivity(base), AtomPrimitive::Connectivity(excluded)) => {
            subtract_count_query(*base, *excluded, AtomPrimitive::Connectivity)
        }
        (AtomPrimitive::Valence(base), AtomPrimitive::Valence(excluded)) => {
            subtract_count_query(*base, *excluded, AtomPrimitive::Valence)
        }
        (AtomPrimitive::HeteroNeighbor(base), AtomPrimitive::HeteroNeighbor(excluded)) => {
            subtract_count_query(*base, *excluded, AtomPrimitive::HeteroNeighbor)
        }
        (
            AtomPrimitive::AliphaticHeteroNeighbor(base),
            AtomPrimitive::AliphaticHeteroNeighbor(excluded),
        ) => subtract_count_query(*base, *excluded, AtomPrimitive::AliphaticHeteroNeighbor),
        (
            AtomPrimitive::Hydrogen(base_kind, base),
            AtomPrimitive::Hydrogen(excluded_kind, excluded),
        ) if base_kind == excluded_kind => subtract_numeric_query_range(
            numeric_count_range(*base),
            numeric_count_range(*excluded),
            |range| {
                BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                    *base_kind,
                    count_query_from_range(range).into_option(),
                ))
            },
        ),
        (AtomPrimitive::RingMembership(base), AtomPrimitive::RingMembership(excluded)) => {
            subtract_ring_query(*base, *excluded, AtomPrimitive::RingMembership)
        }
        (AtomPrimitive::RingSize(base), AtomPrimitive::RingSize(excluded)) => {
            subtract_ring_query(*base, *excluded, AtomPrimitive::RingSize)
        }
        (AtomPrimitive::RingConnectivity(base), AtomPrimitive::RingConnectivity(excluded)) => {
            subtract_ring_query(*base, *excluded, AtomPrimitive::RingConnectivity)
        }
        (AtomPrimitive::Hybridization(base), AtomPrimitive::Hybridization(excluded)) => {
            subtract_numeric_query_range(
                numeric_query_range(*base),
                numeric_query_range(*excluded),
                |range| {
                    BracketExprTree::Primitive(AtomPrimitive::Hybridization(
                        numeric_query_from_range(range),
                    ))
                },
            )
        }
        _ => None,
    }
}

fn subtract_count_query(
    base: Option<NumericQuery>,
    excluded: Option<NumericQuery>,
    build: impl Fn(Option<NumericQuery>) -> AtomPrimitive,
) -> Option<BracketExprTree> {
    subtract_numeric_query_range(
        numeric_count_range(base),
        numeric_count_range(excluded),
        |range| BracketExprTree::Primitive(build(count_query_from_range(range).into_option())),
    )
}

fn subtract_ring_query(
    base: Option<NumericQuery>,
    excluded: Option<NumericQuery>,
    build: impl Fn(Option<NumericQuery>) -> AtomPrimitive,
) -> Option<BracketExprTree> {
    subtract_numeric_query_range(
        numeric_ring_range(base),
        numeric_ring_range(excluded),
        |range| BracketExprTree::Primitive(build(ring_query_from_range(range).into_option())),
    )
}

fn subtract_numeric_query_range(
    base: NumericRange,
    excluded: NumericRange,
    build: impl Fn(NumericRange) -> BracketExprTree,
) -> Option<BracketExprTree> {
    let overlap = intersect_numeric_ranges(base, excluded)?;
    if overlap == base {
        return Some(false_bracket_tree());
    }

    let base_min = base.min.unwrap_or(0);
    let overlap_min = overlap.min.unwrap_or(0);
    let mut retained = Vec::new();
    if base_min < overlap_min {
        retained.push(NumericRange {
            min: Some(base_min),
            max: Some(overlap_min - 1),
        });
    }
    if let Some(overlap_max) = overlap.max {
        let next_min = overlap_max.saturating_add(1);
        if base.max.is_none_or(|base_max| next_min <= base_max) {
            retained.push(NumericRange {
                min: Some(next_min),
                max: base.max,
            });
        }
    }

    match retained.as_slice() {
        [] => Some(false_bracket_tree()),
        [single] => Some(build(canonical_numeric_range(*single))),
        _ => Some(simplify_bracket_or(
            retained
                .into_iter()
                .map(|range| build(canonical_numeric_range(range)))
                .collect(),
        )),
    }
}

fn synthesize_symbol_conjunctions(items: &mut Vec<BracketExprTree>) {
    while let Some((atomic_number_index, any_index, element, aromatic)) =
        find_symbol_conjunction_pair(items)
    {
        if atomic_number_index > any_index {
            items.remove(atomic_number_index);
            items.remove(any_index);
        } else {
            items.remove(any_index);
            items.remove(atomic_number_index);
        }
        items.push(BracketExprTree::Primitive(AtomPrimitive::Symbol {
            element,
            aromatic,
        }));
    }
}

fn find_symbol_conjunction_pair(
    items: &[BracketExprTree],
) -> Option<(usize, usize, elements_rs::Element, bool)> {
    for (atomic_number_index, item) in items.iter().enumerate() {
        let BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(atomic_number)) = item else {
            continue;
        };
        for (any_index, any_item) in items.iter().enumerate() {
            if any_index == atomic_number_index {
                continue;
            }
            let aromatic = match any_item {
                BracketExprTree::Primitive(AtomPrimitive::AliphaticAny) => false,
                BracketExprTree::Primitive(AtomPrimitive::AromaticAny) => true,
                _ => continue,
            };
            let Some(element) = element_for_symbol_conjunction(*atomic_number, aromatic) else {
                continue;
            };
            return Some((atomic_number_index, any_index, element, aromatic));
        }
    }
    None
}

fn synthesize_isotope_conjunctions(items: &mut Vec<BracketExprTree>) {
    while let Some((isotope_wildcard_index, symbol_index, isotope, aromatic)) =
        find_isotope_conjunction_pair(items)
    {
        if isotope_wildcard_index > symbol_index {
            items.remove(isotope_wildcard_index);
            items.remove(symbol_index);
        } else {
            items.remove(symbol_index);
            items.remove(isotope_wildcard_index);
        }
        items.push(BracketExprTree::Primitive(AtomPrimitive::Isotope {
            isotope,
            aromatic,
        }));
    }
}

fn find_isotope_conjunction_pair(
    items: &[BracketExprTree],
) -> Option<(usize, usize, Isotope, bool)> {
    for (isotope_wildcard_index, item) in items.iter().enumerate() {
        let BracketExprTree::Primitive(AtomPrimitive::IsotopeWildcard(mass_number)) = item else {
            continue;
        };
        if *mass_number == 0 {
            continue;
        }
        for (symbol_index, symbol_item) in items.iter().enumerate() {
            if symbol_index == isotope_wildcard_index {
                continue;
            }
            let BracketExprTree::Primitive(AtomPrimitive::Symbol { element, aromatic }) =
                symbol_item
            else {
                continue;
            };
            let Ok(isotope) = Isotope::try_from((*element, *mass_number)) else {
                continue;
            };
            return Some((isotope_wildcard_index, symbol_index, isotope, *aromatic));
        }
    }
    None
}

fn synthesize_negated_symbol_conjunctions(items: &mut Vec<BracketExprTree>) {
    while let Some((left_index, right_index, replacement)) =
        find_negated_symbol_conjunction_pair(items)
    {
        remove_indices_descending(items, &[left_index, right_index]);
        items.push(replacement);
    }
}

fn find_negated_symbol_conjunction_pair(
    items: &[BracketExprTree],
) -> Option<(usize, usize, BracketExprTree)> {
    for (left_index, left) in items.iter().enumerate() {
        for (right_index, right) in items.iter().enumerate().skip(left_index + 1) {
            let Some(replacement) = negated_symbol_conjunction_replacement(left, right) else {
                continue;
            };
            return Some((left_index, right_index, replacement));
        }
    }
    None
}

fn negated_symbol_conjunction_replacement(
    left: &BracketExprTree,
    right: &BracketExprTree,
) -> Option<BracketExprTree> {
    let left = negated_bracket_primitive(left)?;
    let right = negated_bracket_primitive(right)?;
    match (left, right) {
        (AtomPrimitive::AliphaticAny, AtomPrimitive::AromaticAny)
        | (AtomPrimitive::AromaticAny, AtomPrimitive::AliphaticAny) => Some(false_bracket_tree()),
        (
            AtomPrimitive::Symbol {
                element: left_element,
                aromatic: false,
            },
            AtomPrimitive::Symbol {
                element: right_element,
                aromatic: true,
            },
        )
        | (
            AtomPrimitive::Symbol {
                element: right_element,
                aromatic: true,
            },
            AtomPrimitive::Symbol {
                element: left_element,
                aromatic: false,
            },
        ) if left_element == right_element => {
            Some(simplify_bracket_not(BracketExprTree::Primitive(
                AtomPrimitive::AtomicNumber(u16::from(left_element.atomic_number())),
            )))
        }
        _ => None,
    }
}

fn negated_bracket_primitive(tree: &BracketExprTree) -> Option<&AtomPrimitive> {
    let BracketExprTree::Not(inner) = tree else {
        return None;
    };
    let BracketExprTree::Primitive(primitive) = inner.as_ref() else {
        return None;
    };
    Some(primitive)
}

fn element_for_symbol_conjunction(
    atomic_number: u16,
    aromatic: bool,
) -> Option<elements_rs::Element> {
    let element = elements_rs::Element::try_from(u8::try_from(atomic_number).ok()?).ok()?;
    if aromatic && !element_supports_aromatic_bracket_symbol(element) {
        return None;
    }
    Some(element)
}

const fn element_supports_aromatic_bracket_symbol(element: elements_rs::Element) -> bool {
    matches!(
        element,
        elements_rs::Element::B
            | elements_rs::Element::C
            | elements_rs::Element::N
            | elements_rs::Element::O
            | elements_rs::Element::P
            | elements_rs::Element::S
            | elements_rs::Element::As
            | elements_rs::Element::Se
    )
}

fn remove_implied_bracket_conjunction_terms(items: &mut Vec<BracketExprTree>) {
    let mut index = 0usize;
    while index < items.len() {
        let is_implied = items.iter().enumerate().any(|(other_index, other)| {
            other_index != index && bracket_tree_implies(other, &items[index])
        });
        if is_implied {
            items.remove(index);
        } else {
            index += 1;
        }
    }
}

fn bracket_tree_implies(specific: &BracketExprTree, general: &BracketExprTree) -> bool {
    match (specific, general) {
        (specific, general) if specific == general => true,
        (BracketExprTree::Not(specific), BracketExprTree::Not(general)) => {
            bracket_tree_implies(general, specific)
        }
        (BracketExprTree::Or(items), _) => {
            items.iter().all(|item| bracket_tree_implies(item, general))
        }
        (_, BracketExprTree::Or(items)) => items
            .iter()
            .any(|item| bracket_tree_implies(specific, item)),
        (_, BracketExprTree::Not(excluded)) => {
            bracket_trees_are_mutually_exclusive(specific, excluded)
        }
        (
            BracketExprTree::HighAnd(_) | BracketExprTree::LowAnd(_),
            BracketExprTree::HighAnd(general_items) | BracketExprTree::LowAnd(general_items),
        ) => general_items
            .iter()
            .all(|general_item| bracket_tree_implies(specific, general_item)),
        (BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items), _) => {
            items.iter().any(|item| bracket_tree_implies(item, general))
        }
        (_, BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items)) => items
            .iter()
            .all(|item| bracket_tree_implies(specific, item)),
        (
            BracketExprTree::Primitive(AtomPrimitive::Symbol { element, .. }),
            BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(atomic_number)),
        ) => u16::from(element.atomic_number()) == *atomic_number,
        (
            BracketExprTree::Primitive(AtomPrimitive::Symbol {
                aromatic: false, ..
            }),
            BracketExprTree::Primitive(AtomPrimitive::AliphaticAny),
        )
        | (
            BracketExprTree::Primitive(AtomPrimitive::Symbol { aromatic: true, .. }),
            BracketExprTree::Primitive(AtomPrimitive::AromaticAny),
        ) => true,
        (BracketExprTree::Primitive(specific), BracketExprTree::Primitive(general)) => {
            atom_primitive_implies(specific, general)
        }
        _ => false,
    }
}

fn bracket_and_contains_complement_pair(items: &[BracketExprTree]) -> bool {
    items.iter().any(|item| {
        items.iter().any(|other| match other {
            BracketExprTree::Not(inner) => inner.as_ref() == item,
            _ => false,
        })
    })
}

fn bracket_and_contains_mutually_exclusive_pair(items: &[BracketExprTree]) -> bool {
    items.iter().enumerate().any(|(left_index, left)| {
        items.iter().enumerate().any(|(right_index, right)| {
            left_index != right_index && bracket_trees_are_mutually_exclusive(left, right)
        })
    })
}

fn merge_numeric_bracket_conjunctions(items: &mut Vec<BracketExprTree>) {
    while merge_numeric_bracket_items(items, NumericMerge::Intersection) {}
}

fn merge_numeric_bracket_disjunctions(items: &mut Vec<BracketExprTree>) {
    while merge_numeric_bracket_items(items, NumericMerge::Union) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericMerge {
    Intersection,
    Union,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergedOptionalNumericQuery {
    Omitted,
    Query(NumericQuery),
}

impl MergedOptionalNumericQuery {
    const fn into_option(self) -> Option<NumericQuery> {
        match self {
            Self::Omitted => None,
            Self::Query(query) => Some(query),
        }
    }
}

fn merge_numeric_bracket_items(items: &mut Vec<BracketExprTree>, merge: NumericMerge) -> bool {
    for left_index in 0..items.len() {
        let BracketExprTree::Primitive(left) = &items[left_index] else {
            continue;
        };
        for right_index in (left_index + 1)..items.len() {
            let BracketExprTree::Primitive(right) = &items[right_index] else {
                continue;
            };
            let Some(merged) = merge_numeric_atom_primitives(left, right, merge) else {
                continue;
            };
            items.remove(right_index);
            items.remove(left_index);
            items.push(BracketExprTree::Primitive(merged));
            return true;
        }
    }
    false
}

fn merge_numeric_atom_primitives(
    left: &AtomPrimitive,
    right: &AtomPrimitive,
    merge: NumericMerge,
) -> Option<AtomPrimitive> {
    match (left, right) {
        (AtomPrimitive::Degree(left), AtomPrimitive::Degree(right)) => Some(AtomPrimitive::Degree(
            merge_count_queries(*left, *right, merge)?.into_option(),
        )),
        (AtomPrimitive::Connectivity(left), AtomPrimitive::Connectivity(right)) => Some(
            AtomPrimitive::Connectivity(merge_count_queries(*left, *right, merge)?.into_option()),
        ),
        (AtomPrimitive::Valence(left), AtomPrimitive::Valence(right)) => Some(
            AtomPrimitive::Valence(merge_count_queries(*left, *right, merge)?.into_option()),
        ),
        (AtomPrimitive::HeteroNeighbor(left), AtomPrimitive::HeteroNeighbor(right)) => Some(
            AtomPrimitive::HeteroNeighbor(merge_count_queries(*left, *right, merge)?.into_option()),
        ),
        (
            AtomPrimitive::AliphaticHeteroNeighbor(left),
            AtomPrimitive::AliphaticHeteroNeighbor(right),
        ) => Some(AtomPrimitive::AliphaticHeteroNeighbor(
            merge_count_queries(*left, *right, merge)?.into_option(),
        )),
        (AtomPrimitive::Hydrogen(left_kind, left), AtomPrimitive::Hydrogen(right_kind, right))
            if left_kind == right_kind =>
        {
            Some(AtomPrimitive::Hydrogen(
                *left_kind,
                merge_count_queries(*left, *right, merge)?.into_option(),
            ))
        }
        (AtomPrimitive::RingMembership(left), AtomPrimitive::RingMembership(right)) => Some(
            AtomPrimitive::RingMembership(merge_ring_queries(*left, *right, merge)?.into_option()),
        ),
        (AtomPrimitive::RingSize(left), AtomPrimitive::RingSize(right)) => Some(
            AtomPrimitive::RingSize(merge_ring_queries(*left, *right, merge)?.into_option()),
        ),
        (AtomPrimitive::RingConnectivity(left), AtomPrimitive::RingConnectivity(right)) => {
            Some(AtomPrimitive::RingConnectivity(
                merge_ring_queries(*left, *right, merge)?.into_option(),
            ))
        }
        (AtomPrimitive::Hybridization(left), AtomPrimitive::Hybridization(right)) => Some(
            AtomPrimitive::Hybridization(merge_numeric_queries(*left, *right, merge)?),
        ),
        _ => None,
    }
}

fn merge_count_queries(
    left: Option<NumericQuery>,
    right: Option<NumericQuery>,
    merge: NumericMerge,
) -> Option<MergedOptionalNumericQuery> {
    merge_numeric_ranges(numeric_count_range(left), numeric_count_range(right), merge)
        .map(count_query_from_range)
}

fn merge_ring_queries(
    left: Option<NumericQuery>,
    right: Option<NumericQuery>,
    merge: NumericMerge,
) -> Option<MergedOptionalNumericQuery> {
    merge_numeric_ranges(numeric_ring_range(left), numeric_ring_range(right), merge)
        .map(ring_query_from_range)
}

fn merge_numeric_queries(
    left: NumericQuery,
    right: NumericQuery,
    merge: NumericMerge,
) -> Option<NumericQuery> {
    merge_numeric_ranges(numeric_query_range(left), numeric_query_range(right), merge)
        .map(numeric_query_from_range)
}

fn merge_numeric_ranges(
    left: NumericRange,
    right: NumericRange,
    merge: NumericMerge,
) -> Option<NumericRange> {
    match merge {
        NumericMerge::Intersection => intersect_numeric_ranges(left, right),
        NumericMerge::Union => union_numeric_ranges(left, right),
    }
}

fn intersect_numeric_ranges(left: NumericRange, right: NumericRange) -> Option<NumericRange> {
    let min = left.min.unwrap_or(0).max(right.min.unwrap_or(0));
    let max = match (left.max, right.max) {
        (Some(left_max), Some(right_max)) => Some(left_max.min(right_max)),
        (Some(max), None) | (None, Some(max)) => Some(max),
        (None, None) => None,
    };
    if max.is_some_and(|max| min > max) {
        None
    } else {
        Some(canonical_numeric_range(NumericRange {
            min: Some(min),
            max,
        }))
    }
}

fn union_numeric_ranges(left: NumericRange, right: NumericRange) -> Option<NumericRange> {
    let left_min = left.min.unwrap_or(0);
    let right_min = right.min.unwrap_or(0);
    let (first, second, first_min, second_min) = if left_min <= right_min {
        (left, right, left_min, right_min)
    } else {
        (right, left, right_min, left_min)
    };

    if let Some(first_max) = first.max {
        if first_max.saturating_add(1) < second_min {
            return None;
        }
    }

    let max = match (first.max, second.max) {
        (None, _) | (_, None) => None,
        (Some(first_max), Some(second_max)) => Some(first_max.max(second_max)),
    };
    Some(canonical_numeric_range(NumericRange {
        min: Some(first_min),
        max,
    }))
}

fn count_query_from_range(range: NumericRange) -> MergedOptionalNumericQuery {
    let query = numeric_query_from_range(range);
    if query == NumericQuery::Exact(1) {
        MergedOptionalNumericQuery::Omitted
    } else {
        MergedOptionalNumericQuery::Query(query)
    }
}

fn ring_query_from_range(range: NumericRange) -> MergedOptionalNumericQuery {
    let range = canonical_numeric_range(range);
    if range.min == Some(1) && range.max.is_none() {
        MergedOptionalNumericQuery::Omitted
    } else {
        MergedOptionalNumericQuery::Query(numeric_query_from_range(range))
    }
}

fn numeric_query_from_range(range: NumericRange) -> NumericQuery {
    canonical_numeric_query(NumericQuery::Range(range))
}

fn atom_primitive_implies(specific: &AtomPrimitive, general: &AtomPrimitive) -> bool {
    match (specific, general) {
        (specific, general) if specific == general => true,
        (AtomPrimitive::Isotope { isotope, .. }, AtomPrimitive::AtomicNumber(atomic_number)) => {
            u16::from(isotope.element().atomic_number()) == *atomic_number
        }
        (
            AtomPrimitive::Isotope { isotope, aromatic },
            AtomPrimitive::Symbol {
                element,
                aromatic: symbol_aromatic,
            },
        ) => isotope.element() == *element && aromatic == symbol_aromatic,
        (
            AtomPrimitive::Isotope {
                aromatic: false, ..
            },
            AtomPrimitive::AliphaticAny,
        )
        | (AtomPrimitive::Isotope { aromatic: true, .. }, AtomPrimitive::AromaticAny) => true,
        (AtomPrimitive::Isotope { isotope, .. }, AtomPrimitive::IsotopeWildcard(mass_number)) => {
            isotope.mass_number() == *mass_number
        }
        (AtomPrimitive::RecursiveQuery(specific), AtomPrimitive::RecursiveQuery(general)) => {
            recursive_query_implies(specific, general)
        }
        (AtomPrimitive::Degree(specific), AtomPrimitive::Degree(general))
        | (AtomPrimitive::Connectivity(specific), AtomPrimitive::Connectivity(general))
        | (AtomPrimitive::Valence(specific), AtomPrimitive::Valence(general))
        | (AtomPrimitive::HeteroNeighbor(specific), AtomPrimitive::HeteroNeighbor(general))
        | (
            AtomPrimitive::AliphaticHeteroNeighbor(specific),
            AtomPrimitive::AliphaticHeteroNeighbor(general),
        ) => count_query_implies(*specific, *general),
        (
            AtomPrimitive::Hydrogen(specific_kind, specific),
            AtomPrimitive::Hydrogen(general_kind, general),
        ) if specific_kind == general_kind => count_query_implies(*specific, *general),
        (AtomPrimitive::RingMembership(specific), AtomPrimitive::RingMembership(general))
        | (AtomPrimitive::RingSize(specific), AtomPrimitive::RingSize(general))
        | (AtomPrimitive::RingConnectivity(specific), AtomPrimitive::RingConnectivity(general)) => {
            ring_query_implies(*specific, *general)
        }
        (AtomPrimitive::Hybridization(specific), AtomPrimitive::Hybridization(general)) => {
            numeric_query_implies(*specific, *general)
        }
        _ => false,
    }
}

fn recursive_query_implies(specific: &QueryMol, general: &QueryMol) -> bool {
    if specific.atom_count() != 1
        || specific.bond_count() != 0
        || general.atom_count() != 1
        || general.bond_count() != 0
    {
        return false;
    }

    atom_expr_implies(&specific.atoms()[0].expr, &general.atoms()[0].expr)
}

fn atom_expr_implies(specific: &AtomExpr, general: &AtomExpr) -> bool {
    match (specific, general) {
        (specific, general) if specific == general => true,
        (_, AtomExpr::Wildcard) => true,
        (AtomExpr::Wildcard, _) => false,
        (
            AtomExpr::Bare {
                element: specific_element,
                aromatic: specific_aromatic,
            },
            AtomExpr::Bare {
                element: general_element,
                aromatic: general_aromatic,
            },
        ) => specific_element == general_element && specific_aromatic == general_aromatic,
        (AtomExpr::Bare { element, aromatic }, AtomExpr::Bracket(general)) => {
            bracket_tree_implies(&symbol_bracket_tree(*element, *aromatic), &general.tree)
        }
        (AtomExpr::Bracket(specific), AtomExpr::Bare { element, aromatic }) => {
            bracket_tree_implies(&specific.tree, &symbol_bracket_tree(*element, *aromatic))
        }
        (AtomExpr::Bracket(specific), AtomExpr::Bracket(general)) => {
            bracket_tree_implies(&specific.tree, &general.tree)
        }
    }
}

const fn symbol_bracket_tree(element: elements_rs::Element, aromatic: bool) -> BracketExprTree {
    BracketExprTree::Primitive(AtomPrimitive::Symbol { element, aromatic })
}

fn count_query_implies(specific: Option<NumericQuery>, general: Option<NumericQuery>) -> bool {
    numeric_range_implies(numeric_count_range(specific), numeric_count_range(general))
}

fn ring_query_implies(specific: Option<NumericQuery>, general: Option<NumericQuery>) -> bool {
    numeric_range_implies(numeric_ring_range(specific), numeric_ring_range(general))
}

fn numeric_query_implies(specific: NumericQuery, general: NumericQuery) -> bool {
    numeric_range_implies(numeric_query_range(specific), numeric_query_range(general))
}

fn numeric_count_range(query: Option<NumericQuery>) -> NumericRange {
    query.map_or(
        NumericRange {
            min: Some(1),
            max: Some(1),
        },
        numeric_query_range,
    )
}

fn numeric_ring_range(query: Option<NumericQuery>) -> NumericRange {
    query.map_or(
        NumericRange {
            min: Some(1),
            max: None,
        },
        numeric_query_range,
    )
}

const fn numeric_query_range(query: NumericQuery) -> NumericRange {
    match query {
        NumericQuery::Exact(value) => NumericRange {
            min: Some(value),
            max: Some(value),
        },
        NumericQuery::Range(range) => range,
    }
}

fn numeric_range_implies(specific: NumericRange, general: NumericRange) -> bool {
    let lower_bound_is_subset = specific.min.unwrap_or(0) >= general.min.unwrap_or(0);
    let upper_bound_is_subset = match (specific.max, general.max) {
        (_, None) => true,
        (Some(specific_max), Some(general_max)) => specific_max <= general_max,
        (None, Some(_)) => false,
    };
    lower_bound_is_subset && upper_bound_is_subset
}

fn simplify_bracket_or(items: Vec<BracketExprTree>) -> BracketExprTree {
    let mut items = items
        .into_iter()
        .map(simplify_bracket_tree)
        .collect::<Vec<_>>();
    flatten_bracket_or_items(&mut items);
    if items.len() > 1 {
        items.retain(|item| !is_false_bracket_tree(item));
    }
    synthesize_atomic_number_disjunctions(&mut items);
    synthesize_isotope_disjunctions(&mut items);
    merge_numeric_bracket_disjunctions(&mut items);
    while relax_aromaticity_any_negated_atomic_number_disjunction_terms(&mut items) {}
    if items
        .iter()
        .any(|item| matches!(item, BracketExprTree::Primitive(AtomPrimitive::Wildcard)))
        || bracket_or_contains_complement_pair(&items)
        || bracket_or_contains_implied_complement_pair(&items)
        || bracket_or_contains_negated_mutually_exclusive_pair(&items)
        || bracket_or_contains_aliphatic_and_aromatic_any(&items)
    {
        return BracketExprTree::Primitive(AtomPrimitive::Wildcard);
    }

    while relax_complemented_bracket_disjunction_terms(&mut items) {}
    while relax_negated_bracket_conjunction_terms(&mut items) {}
    while relax_negated_broader_bracket_disjunction_terms(&mut items) {}
    while relax_covered_negated_bracket_disjunction_terms(&mut items) {}
    while relax_conjunctive_complement_bracket_disjunction_terms(&mut items) {}
    while collapse_covered_complement_partition_bracket_disjunction_terms(&mut items) {}
    while remove_redundant_bracket_disjunction_consensus_terms(&mut items) {}
    while collapse_bracket_disjunction_consensus_terms(&mut items) {}
    while relax_absorbed_bracket_disjunction_conjunction_alternatives(&mut items) {}
    if let Some(factored) = factor_common_bracket_disjunction_terms(&items) {
        return simplify_bracket_tree(factored);
    }
    remove_absorbed_bracket_disjunction_terms(&mut items);
    if let Some(distributed) = distribute_low_precedence_bracket_disjunction(&items) {
        return distributed;
    }
    items.sort_by_cached_key(BracketExprTree::to_string);
    items.dedup();
    match items.as_slice() {
        [] => false_bracket_tree(),
        [single] => single.clone(),
        _ => BracketExprTree::Or(items),
    }
}

fn flatten_bracket_or_items(items: &mut Vec<BracketExprTree>) -> bool {
    let mut flattened = Vec::with_capacity(items.len());
    let mut changed = false;
    for item in items.drain(..) {
        if let BracketExprTree::Or(children) = item {
            changed = true;
            flattened.extend(children);
        } else {
            flattened.push(item);
        }
    }
    *items = flattened;
    changed
}

fn distribute_low_precedence_bracket_disjunction(
    items: &[BracketExprTree],
) -> Option<BracketExprTree> {
    let (distributed_index, terms) =
        items
            .iter()
            .enumerate()
            .find_map(|(index, item)| match item {
                BracketExprTree::LowAnd(terms) => Some((index, terms)),
                _ => None,
            })?;

    let rest = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (index != distributed_index).then_some(item.clone()))
        .collect::<Vec<_>>();
    let factors = terms
        .iter()
        .map(|term| {
            let mut alternatives = rest.clone();
            alternatives.push(term.clone());
            flatten_bracket_or_items(&mut alternatives);
            simplify_bracket_or(alternatives)
        })
        .collect::<Vec<_>>();

    Some(simplify_bracket_and(factors, BracketAndKind::Low))
}

fn synthesize_atomic_number_disjunctions(items: &mut Vec<BracketExprTree>) {
    while let Some((aliphatic_index, aromatic_index, atomic_number)) =
        find_atomic_number_disjunction_pair(items)
    {
        if aliphatic_index > aromatic_index {
            items.remove(aliphatic_index);
            items.remove(aromatic_index);
        } else {
            items.remove(aromatic_index);
            items.remove(aliphatic_index);
        }
        items.push(BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(
            atomic_number,
        )));
    }
}

fn find_atomic_number_disjunction_pair(items: &[BracketExprTree]) -> Option<(usize, usize, u16)> {
    for (aliphatic_index, item) in items.iter().enumerate() {
        let BracketExprTree::Primitive(AtomPrimitive::Symbol {
            element,
            aromatic: false,
        }) = item
        else {
            continue;
        };
        for (aromatic_index, aromatic_item) in items.iter().enumerate() {
            if aromatic_index == aliphatic_index {
                continue;
            }
            if matches!(
                aromatic_item,
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: aromatic_element,
                    aromatic: true,
                }) if aromatic_element == element
            ) {
                return Some((
                    aliphatic_index,
                    aromatic_index,
                    u16::from(element.atomic_number()),
                ));
            }
        }
    }
    None
}

fn synthesize_isotope_disjunctions(items: &mut Vec<BracketExprTree>) {
    while let Some((aliphatic_index, aromatic_index, isotope)) =
        find_isotope_disjunction_pair(items)
    {
        if aliphatic_index > aromatic_index {
            items.remove(aliphatic_index);
            items.remove(aromatic_index);
        } else {
            items.remove(aromatic_index);
            items.remove(aliphatic_index);
        }
        items.push(BracketExprTree::HighAnd(vec![
            BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(u16::from(
                isotope.element().atomic_number(),
            ))),
            BracketExprTree::Primitive(AtomPrimitive::IsotopeWildcard(isotope.mass_number())),
        ]));
    }
}

fn find_isotope_disjunction_pair(items: &[BracketExprTree]) -> Option<(usize, usize, Isotope)> {
    for (aliphatic_index, item) in items.iter().enumerate() {
        let BracketExprTree::Primitive(AtomPrimitive::Isotope {
            isotope,
            aromatic: false,
        }) = item
        else {
            continue;
        };
        for (aromatic_index, aromatic_item) in items.iter().enumerate() {
            if aromatic_index == aliphatic_index {
                continue;
            }
            if matches!(
                aromatic_item,
                BracketExprTree::Primitive(AtomPrimitive::Isotope {
                    isotope: aromatic_isotope,
                    aromatic: true,
                }) if aromatic_isotope == isotope
            ) {
                return Some((aliphatic_index, aromatic_index, *isotope));
            }
        }
    }
    None
}

fn bracket_or_contains_complement_pair(items: &[BracketExprTree]) -> bool {
    items.iter().any(|item| {
        items.iter().any(|other| match other {
            BracketExprTree::Not(inner) => inner.as_ref() == item,
            _ => false,
        })
    })
}

fn bracket_or_contains_implied_complement_pair(items: &[BracketExprTree]) -> bool {
    items.iter().any(|item| {
        items.iter().any(|other| match other {
            BracketExprTree::Not(inner) => bracket_tree_implies(inner, item),
            _ => bracket_tree_is_conjunctive_complement(item, other),
        })
    })
}

fn bracket_tree_is_conjunctive_complement(
    candidate: &BracketExprTree,
    complement: &BracketExprTree,
) -> bool {
    let Some(excluded) = negated_bracket_conjunction_items(complement) else {
        return false;
    };
    let (excluded, residual) = split_negated_bracket_conjunction_items(&excluded);
    if excluded.len() < 2 || !residual.is_empty() {
        return false;
    }
    let excluded = simplify_bracket_or(excluded);
    bracket_tree_implies(candidate, &excluded) && bracket_tree_implies(&excluded, candidate)
}

fn negated_bracket_conjunction_items(tree: &BracketExprTree) -> Option<Vec<BracketExprTree>> {
    let (BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items)) = tree else {
        return None;
    };
    Some(items.clone())
}

fn negated_bracket_tree_inner(tree: &BracketExprTree) -> Option<&BracketExprTree> {
    let BracketExprTree::Not(inner) = tree else {
        return None;
    };
    Some(inner.as_ref())
}

fn relax_conjunctive_complement_bracket_disjunction_terms(items: &mut [BracketExprTree]) -> bool {
    for base_index in 0..items.len() {
        for candidate_index in 0..items.len() {
            if base_index == candidate_index {
                continue;
            }
            let Some(candidate_items) = negated_bracket_conjunction_items(&items[candidate_index])
            else {
                continue;
            };
            let (excluded, residual) = split_negated_bracket_conjunction_items(&candidate_items);
            if excluded.len() < 2 || residual.is_empty() {
                continue;
            }
            let excluded = simplify_bracket_or(excluded);
            if !bracket_trees_are_equivalent(&items[base_index], &excluded) {
                continue;
            }
            items[candidate_index] = simplify_bracket_residual_conjunction(residual);
            return true;
        }
    }

    false
}

fn split_negated_bracket_conjunction_items(
    items: &[BracketExprTree],
) -> (Vec<BracketExprTree>, Vec<BracketExprTree>) {
    let mut excluded = Vec::new();
    let mut residual = Vec::new();
    for item in items {
        match item {
            BracketExprTree::Not(inner) => excluded.push(inner.as_ref().clone()),
            _ => residual.push(item.clone()),
        }
    }
    (excluded, residual)
}

fn bracket_trees_are_equivalent(left: &BracketExprTree, right: &BracketExprTree) -> bool {
    bracket_tree_implies(left, right) && bracket_tree_implies(right, left)
}

fn simplify_bracket_residual_conjunction(items: Vec<BracketExprTree>) -> BracketExprTree {
    let kind = if can_render_bracket_and_as_high_and(&items) {
        BracketAndKind::High
    } else {
        BracketAndKind::Low
    };
    simplify_bracket_and(items, kind)
}

fn bracket_or_contains_negated_mutually_exclusive_pair(items: &[BracketExprTree]) -> bool {
    items.iter().any(|item| {
        let BracketExprTree::Not(left) = item else {
            return false;
        };
        items.iter().any(|other| match other {
            BracketExprTree::Not(right) => bracket_trees_are_mutually_exclusive(left, right),
            _ => false,
        })
    })
}

fn relax_aromaticity_any_negated_atomic_number_disjunction_terms(
    items: &mut Vec<BracketExprTree>,
) -> bool {
    for any_index in 0..items.len() {
        let replacement_aromaticity = match &items[any_index] {
            BracketExprTree::Primitive(AtomPrimitive::AliphaticAny) => Some(true),
            BracketExprTree::Primitive(AtomPrimitive::AromaticAny) => Some(false),
            _ => None,
        };
        let Some(replacement_aromaticity) = replacement_aromaticity else {
            continue;
        };

        for negated_index in 0..items.len() {
            if negated_index == any_index {
                continue;
            }
            let Some(atomic_number) = negated_atomic_number(&items[negated_index]) else {
                continue;
            };

            let Some(element) = element_for_symbol_conjunction(atomic_number, false) else {
                continue;
            };
            if replacement_aromaticity && !element_supports_aromatic_bracket_symbol(element) {
                remove_indices_descending(items, &[any_index, negated_index]);
                items.push(BracketExprTree::Primitive(AtomPrimitive::Wildcard));
                return true;
            }

            let replacement =
                simplify_bracket_not(BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element,
                    aromatic: replacement_aromaticity,
                }));
            remove_indices_descending(items, &[any_index, negated_index]);
            items.push(replacement);
            return true;
        }
    }

    false
}

fn negated_atomic_number(tree: &BracketExprTree) -> Option<u16> {
    let BracketExprTree::Not(inner) = tree else {
        return None;
    };
    let BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(atomic_number)) = inner.as_ref()
    else {
        return None;
    };
    Some(*atomic_number)
}

fn relax_complemented_bracket_disjunction_terms(items: &mut [BracketExprTree]) -> bool {
    relax_complemented_disjunction_terms(items, remove_negated_bracket_term_from_conjunction)
}

fn relax_negated_bracket_conjunction_terms(items: &mut [BracketExprTree]) -> bool {
    relax_negated_conjunction_terms(
        items,
        negated_bracket_tree_inner,
        remove_bracket_term_from_conjunction,
    )
}

fn remove_bracket_term_from_conjunction(
    candidate: &BracketExprTree,
    removed: &BracketExprTree,
) -> Option<BracketExprTree> {
    let (items, kind) = match candidate {
        BracketExprTree::HighAnd(items) => (items, BracketAndKind::High),
        BracketExprTree::LowAnd(items) => (items, BracketAndKind::Low),
        _ => return None,
    };
    let removed_index = items.iter().position(|item| item == removed)?;
    Some(simplify_bracket_and(
        bracket_items_without_index(items, removed_index),
        kind,
    ))
}

fn relax_negated_broader_bracket_disjunction_terms(items: &mut Vec<BracketExprTree>) -> bool {
    for specific_index in 0..items.len() {
        for negated_index in 0..items.len() {
            if specific_index == negated_index {
                continue;
            }
            let BracketExprTree::Not(general) = &items[negated_index] else {
                continue;
            };
            if !bracket_tree_implies(&items[specific_index], general) {
                continue;
            }
            let Some(residual) = subtract_bracket_tree(general, &items[specific_index]) else {
                continue;
            };
            let Some(relaxed) = complement_subtracted_bracket_tree(residual) else {
                continue;
            };

            remove_indices_descending(items, &[specific_index, negated_index]);
            items.push(relaxed);
            return true;
        }
    }

    false
}

fn relax_covered_negated_bracket_disjunction_terms(items: &mut Vec<BracketExprTree>) -> bool {
    relax_covered_negated_disjunction_terms(
        items,
        &CoveredNegatedDisjunctionOps {
            consensus_items: bracket_consensus_items,
            sort_and_dedup: sort_and_dedup_bracket_items,
            trees_are_complements: bracket_trees_are_complements,
            items_without_index: bracket_items_without_index,
            can_render_and_as_high: can_render_bracket_and_as_high_and,
            simplify_and: simplify_bracket_and,
            high_kind: BracketAndKind::High,
            low_kind: BracketAndKind::Low,
        },
    )
}

fn collapse_covered_complement_partition_bracket_disjunction_terms(
    items: &mut Vec<BracketExprTree>,
) -> bool {
    for specific_index in 0..items.len() {
        let mut coverages: Vec<(Vec<BracketExprTree>, Vec<BracketExprTree>, Vec<usize>)> =
            Vec::new();

        for (alternative_index, alternative) in items.iter().enumerate() {
            if alternative_index == specific_index {
                continue;
            }

            for (mut common, excluded) in constrained_complement_alternatives(alternative) {
                sort_and_dedup_bracket_items(&mut common);
                if let Some((_, excluded_terms, alternative_indices)) = coverages
                    .iter_mut()
                    .find(|(known_common, _, _)| known_common.as_slice() == common.as_slice())
                {
                    if !excluded_terms.contains(&excluded) {
                        excluded_terms.push(excluded);
                    }
                    if !alternative_indices.contains(&alternative_index) {
                        alternative_indices.push(alternative_index);
                    }
                } else {
                    coverages.push((common, vec![excluded], vec![alternative_index]));
                }
            }
        }

        for (common, mut excluded_terms, alternative_indices) in coverages {
            if common
                .iter()
                .any(|term| !bracket_tree_implies(&items[specific_index], term))
            {
                continue;
            }

            sort_and_dedup_bracket_items(&mut excluded_terms);
            let mut covered_partition = common.clone();
            covered_partition.extend(excluded_terms);
            let partition_kind = if can_render_bracket_and_as_high_and(&covered_partition) {
                BracketAndKind::High
            } else {
                BracketAndKind::Low
            };
            let covered_partition = simplify_bracket_and(covered_partition, partition_kind);
            if !bracket_trees_are_equivalent(&covered_partition, &items[specific_index]) {
                continue;
            }

            let replacement_kind = if can_render_bracket_and_as_high_and(&common) {
                BracketAndKind::High
            } else {
                BracketAndKind::Low
            };
            let replacement = simplify_bracket_and(common, replacement_kind);
            let mut removed = alternative_indices;
            removed.push(specific_index);
            remove_indices_descending(items, &removed);
            items.push(replacement);
            return true;
        }
    }

    false
}

fn constrained_complement_alternatives(
    tree: &BracketExprTree,
) -> Vec<(Vec<BracketExprTree>, BracketExprTree)> {
    let items = bracket_direct_conjunction_items(tree);
    let mut alternatives = Vec::new();
    let complemented = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            complement_excluded_tree(item).map(|excluded| (index, excluded))
        })
        .collect::<Vec<_>>();
    if complemented.len() >= 2 {
        let excluded_indices = complemented
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>();
        let mut common = items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                (!excluded_indices.contains(&index)).then_some(item.clone())
            })
            .collect::<Vec<_>>();
        sort_and_dedup_bracket_items(&mut common);
        alternatives.push((
            common,
            simplify_bracket_or(
                complemented
                    .iter()
                    .map(|(_, excluded)| excluded.clone())
                    .collect(),
            ),
        ));
    }

    for (term_index, term) in items.iter().enumerate() {
        match term {
            term if complement_excluded_tree(term).is_some() => {
                alternatives.push((
                    bracket_items_without_index(&items, term_index),
                    complement_excluded_tree(term).expect("complement was checked by match guard"),
                ));
            }
            BracketExprTree::Or(disjunction) => {
                let mut excluded_terms = Vec::with_capacity(disjunction.len());
                for alternative in disjunction {
                    let Some(excluded) = complement_excluded_tree(alternative) else {
                        excluded_terms.clear();
                        break;
                    };
                    excluded_terms.push(excluded);
                }
                if excluded_terms.is_empty() {
                    continue;
                }

                let common = bracket_items_without_index(&items, term_index);
                alternatives.extend(
                    excluded_terms
                        .into_iter()
                        .map(|excluded| (common.clone(), excluded)),
                );
            }
            _ => {}
        }
    }

    alternatives
}

fn complement_excluded_tree(tree: &BracketExprTree) -> Option<BracketExprTree> {
    match tree {
        BracketExprTree::Not(excluded) => Some(excluded.as_ref().clone()),
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => {
            let excluded = items
                .iter()
                .map(|item| {
                    let BracketExprTree::Not(excluded) = item else {
                        return None;
                    };
                    Some(excluded.as_ref().clone())
                })
                .collect::<Option<Vec<_>>>()?;
            (excluded.len() >= 2).then(|| simplify_bracket_or(excluded))
        }
        other => numeric_zero_complement_excluded_tree(other),
    }
}

fn bracket_direct_conjunction_items(tree: &BracketExprTree) -> Vec<BracketExprTree> {
    match tree {
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => items.clone(),
        other => vec![other.clone()],
    }
}

fn complement_subtracted_bracket_tree(residual: BracketExprTree) -> Option<BracketExprTree> {
    let residual = simplify_bracket_tree(residual);
    if is_false_bracket_tree(&residual) {
        return Some(BracketExprTree::Primitive(AtomPrimitive::Wildcard));
    }

    match residual {
        BracketExprTree::Primitive(_) => Some(simplify_bracket_not(residual)),
        BracketExprTree::Or(items)
            if items
                .iter()
                .all(|item| matches!(item, BracketExprTree::Primitive(_))) =>
        {
            Some(simplify_bracket_and(
                items.into_iter().map(simplify_bracket_not).collect(),
                BracketAndKind::High,
            ))
        }
        _ => None,
    }
}

fn remove_negated_bracket_term_from_conjunction(
    candidate: &BracketExprTree,
    base: &BracketExprTree,
) -> Option<BracketExprTree> {
    let (items, kind) = match candidate {
        BracketExprTree::HighAnd(items) => (items, BracketAndKind::High),
        BracketExprTree::LowAnd(items) => (items, BracketAndKind::Low),
        _ => return None,
    };
    let complement_index = items.iter().position(|item| match item {
        BracketExprTree::Not(inner) => bracket_tree_implies(inner, base),
        _ => false,
    })?;
    Some(simplify_bracket_and(
        bracket_items_without_index(items, complement_index),
        kind,
    ))
}

fn collapse_bracket_disjunction_consensus_terms(items: &mut Vec<BracketExprTree>) -> bool {
    collapse_disjunction_consensus_terms(items, bracket_disjunction_consensus)
}

fn remove_redundant_bracket_disjunction_consensus_terms(items: &mut Vec<BracketExprTree>) -> bool {
    remove_redundant_disjunction_consensus_terms(
        items,
        bracket_disjunction_consensus_term_is_redundant,
    )
}

fn bracket_disjunction_consensus_term_is_redundant(
    left: &BracketExprTree,
    right: &BracketExprTree,
    candidate: &BracketExprTree,
) -> bool {
    let (left_items, left_kind) = bracket_consensus_items(left);
    let (right_items, right_kind) = bracket_consensus_items(right);
    let kind = if left_kind == BracketAndKind::Low || right_kind == BracketAndKind::Low {
        BracketAndKind::Low
    } else {
        BracketAndKind::High
    };

    for left_complement_index in 0..left_items.len() {
        for right_complement_index in 0..right_items.len() {
            if !bracket_trees_are_complements(
                &left_items[left_complement_index],
                &right_items[right_complement_index],
            ) {
                continue;
            }

            let mut consensus = bracket_items_without_index(&left_items, left_complement_index);
            consensus.extend(bracket_items_without_index(
                &right_items,
                right_complement_index,
            ));
            sort_and_dedup_bracket_items(&mut consensus);
            let consensus = simplify_bracket_and(consensus, kind);
            if bracket_tree_implies(candidate, &consensus) {
                return true;
            }
        }
    }

    false
}

fn factor_common_bracket_disjunction_terms(items: &[BracketExprTree]) -> Option<BracketExprTree> {
    if items.len() < 2 {
        return None;
    }

    let alternatives = items
        .iter()
        .map(|item| {
            let (mut items, _) = bracket_consensus_items(item);
            sort_and_dedup_bracket_items(&mut items);
            items
        })
        .collect::<Vec<_>>();

    let common = alternatives
        .first()?
        .iter()
        .try_fold(Vec::new(), |mut common, candidate| {
            if alternatives
                .iter()
                .all(|alternative| alternative.contains(candidate))
            {
                common.push(candidate.clone());
            }
            Some(common)
        })?;
    if common.is_empty() {
        return None;
    }

    let mut reduced = Vec::new();
    for alternative in alternatives {
        let remainder = alternative
            .into_iter()
            .filter(|item| !common.contains(item))
            .collect::<Vec<_>>();
        if remainder.is_empty() {
            return Some(simplify_bracket_and(common, BracketAndKind::High));
        }
        reduced.push(simplify_bracket_and(remainder, BracketAndKind::High));
    }

    let disjunction = simplify_bracket_or(reduced);
    if matches!(
        disjunction,
        BracketExprTree::Primitive(AtomPrimitive::Wildcard)
    ) {
        return Some(simplify_bracket_and(common, BracketAndKind::High));
    }

    let kind = if matches!(
        disjunction,
        BracketExprTree::Or(_) | BracketExprTree::LowAnd(_)
    ) {
        BracketAndKind::Low
    } else {
        BracketAndKind::High
    };
    let mut factored = common;
    factored.push(disjunction);
    Some(simplify_bracket_and(factored, kind))
}

fn bracket_disjunction_consensus(
    left: &BracketExprTree,
    right: &BracketExprTree,
) -> Option<BracketExprTree> {
    bracket_disjunction_consensus_with_items(left, right, bracket_consensus_items).or_else(|| {
        bracket_disjunction_consensus_with_items(left, right, bracket_complement_consensus_items)
    })
}

fn bracket_disjunction_consensus_with_items<F>(
    left: &BracketExprTree,
    right: &BracketExprTree,
    items_for: F,
) -> Option<BracketExprTree>
where
    F: Fn(&BracketExprTree) -> (Vec<BracketExprTree>, BracketAndKind),
{
    let (left_items, left_kind) = items_for(left);
    let (right_items, right_kind) = items_for(right);
    let kind = if left_kind == BracketAndKind::Low || right_kind == BracketAndKind::Low {
        BracketAndKind::Low
    } else {
        BracketAndKind::High
    };

    for left_complement_index in 0..left_items.len() {
        for right_complement_index in 0..right_items.len() {
            if !bracket_trees_are_complements(
                &left_items[left_complement_index],
                &right_items[right_complement_index],
            ) {
                continue;
            }

            let mut left_remaining =
                bracket_items_without_index(&left_items, left_complement_index);
            let mut right_remaining =
                bracket_items_without_index(&right_items, right_complement_index);
            sort_and_dedup_bracket_items(&mut left_remaining);
            sort_and_dedup_bracket_items(&mut right_remaining);
            if left_remaining == right_remaining {
                return Some(simplify_bracket_and(left_remaining, kind));
            }
        }
    }

    None
}

fn bracket_complement_consensus_items(
    tree: &BracketExprTree,
) -> (Vec<BracketExprTree>, BracketAndKind) {
    match tree {
        BracketExprTree::HighAnd(items) => (
            expanded_bracket_complement_consensus_items(items),
            BracketAndKind::High,
        ),
        BracketExprTree::LowAnd(items) => (
            expanded_bracket_complement_consensus_items(items),
            BracketAndKind::Low,
        ),
        other => (
            expanded_bracket_complement_consensus_item(other.clone()),
            BracketAndKind::High,
        ),
    }
}

fn expanded_bracket_complement_consensus_items(items: &[BracketExprTree]) -> Vec<BracketExprTree> {
    items
        .iter()
        .cloned()
        .flat_map(expanded_bracket_complement_consensus_item)
        .collect()
}

fn expanded_bracket_complement_consensus_item(item: BracketExprTree) -> Vec<BracketExprTree> {
    match item {
        BracketExprTree::Primitive(AtomPrimitive::Symbol { element, aromatic }) => {
            vec![
                BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(u16::from(
                    element.atomic_number(),
                ))),
                aromaticity_class_tree(aromatic),
            ]
        }
        BracketExprTree::Primitive(AtomPrimitive::Isotope { isotope, aromatic }) => {
            vec![
                BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(u16::from(
                    isotope.element().atomic_number(),
                ))),
                BracketExprTree::Primitive(AtomPrimitive::IsotopeWildcard(isotope.mass_number())),
                aromaticity_class_tree(aromatic),
            ]
        }
        other => expanded_bracket_consensus_item(other),
    }
}

const fn aromaticity_class_tree(aromatic: bool) -> BracketExprTree {
    BracketExprTree::Primitive(if aromatic {
        AtomPrimitive::AromaticAny
    } else {
        AtomPrimitive::AliphaticAny
    })
}

fn bracket_consensus_items(tree: &BracketExprTree) -> (Vec<BracketExprTree>, BracketAndKind) {
    match tree {
        BracketExprTree::HighAnd(items) => (
            expanded_bracket_consensus_items(items),
            BracketAndKind::High,
        ),
        BracketExprTree::LowAnd(items) => {
            (expanded_bracket_consensus_items(items), BracketAndKind::Low)
        }
        other => (
            expanded_bracket_consensus_item(other.clone()),
            BracketAndKind::High,
        ),
    }
}

fn expanded_bracket_consensus_items(items: &[BracketExprTree]) -> Vec<BracketExprTree> {
    items
        .iter()
        .cloned()
        .flat_map(expanded_bracket_consensus_item)
        .collect()
}

fn expanded_bracket_consensus_item(item: BracketExprTree) -> Vec<BracketExprTree> {
    let BracketExprTree::Primitive(AtomPrimitive::Isotope { isotope, aromatic }) = item else {
        return vec![item];
    };

    vec![
        isotope_symbol_tree(isotope, aromatic),
        BracketExprTree::Primitive(AtomPrimitive::IsotopeWildcard(isotope.mass_number())),
    ]
}

fn isotope_symbol_tree(isotope: Isotope, aromatic: bool) -> BracketExprTree {
    if isotope.element() == elements_rs::Element::H && !aromatic {
        BracketExprTree::Primitive(AtomPrimitive::AtomicNumber(1))
    } else {
        BracketExprTree::Primitive(AtomPrimitive::Symbol {
            element: isotope.element(),
            aromatic,
        })
    }
}

fn bracket_items_without_index(
    items: &[BracketExprTree],
    excluded_index: usize,
) -> Vec<BracketExprTree> {
    items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (index != excluded_index).then_some(item.clone()))
        .collect()
}

fn sort_and_dedup_bracket_items(items: &mut Vec<BracketExprTree>) {
    items.sort_by_cached_key(BracketExprTree::to_string);
    items.dedup();
}

fn bracket_trees_are_complements(left: &BracketExprTree, right: &BracketExprTree) -> bool {
    match (left, right) {
        (BracketExprTree::Not(left), right) => left.as_ref() == right,
        (left, BracketExprTree::Not(right)) => right.as_ref() == left,
        _ => {
            numeric_zero_complement_excluded_tree(left).is_some_and(|excluded| &excluded == right)
                || numeric_zero_complement_excluded_tree(right)
                    .is_some_and(|excluded| &excluded == left)
        }
    }
}

fn numeric_zero_complement_excluded_tree(tree: &BracketExprTree) -> Option<BracketExprTree> {
    let BracketExprTree::Primitive(primitive) = tree else {
        return None;
    };
    numeric_zero_complement_excluded_primitive(primitive).map(BracketExprTree::Primitive)
}

fn numeric_zero_complement_excluded_primitive(primitive: &AtomPrimitive) -> Option<AtomPrimitive> {
    let complement_range = NumericRange {
        min: Some(1),
        max: None,
    };
    match primitive {
        AtomPrimitive::Degree(query) if numeric_count_range(*query) == complement_range => {
            Some(AtomPrimitive::Degree(Some(NumericQuery::Exact(0))))
        }
        AtomPrimitive::Connectivity(query) if numeric_count_range(*query) == complement_range => {
            Some(AtomPrimitive::Connectivity(Some(NumericQuery::Exact(0))))
        }
        AtomPrimitive::Valence(query) if numeric_count_range(*query) == complement_range => {
            Some(AtomPrimitive::Valence(Some(NumericQuery::Exact(0))))
        }
        AtomPrimitive::Hydrogen(kind, query) if numeric_count_range(*query) == complement_range => {
            Some(AtomPrimitive::Hydrogen(*kind, Some(NumericQuery::Exact(0))))
        }
        AtomPrimitive::RingMembership(query) if numeric_ring_range(*query) == complement_range => {
            Some(AtomPrimitive::RingMembership(Some(NumericQuery::Exact(0))))
        }
        AtomPrimitive::RingSize(query) if numeric_ring_range(*query) == complement_range => {
            Some(AtomPrimitive::RingSize(Some(NumericQuery::Exact(0))))
        }
        AtomPrimitive::RingConnectivity(query)
            if numeric_ring_range(*query) == complement_range =>
        {
            Some(AtomPrimitive::RingConnectivity(Some(NumericQuery::Exact(
                0,
            ))))
        }
        AtomPrimitive::Hybridization(query) if numeric_query_range(*query) == complement_range => {
            Some(AtomPrimitive::Hybridization(NumericQuery::Exact(0)))
        }
        AtomPrimitive::HeteroNeighbor(query) if numeric_count_range(*query) == complement_range => {
            Some(AtomPrimitive::HeteroNeighbor(Some(NumericQuery::Exact(0))))
        }
        AtomPrimitive::AliphaticHeteroNeighbor(query)
            if numeric_count_range(*query) == complement_range =>
        {
            Some(AtomPrimitive::AliphaticHeteroNeighbor(Some(
                NumericQuery::Exact(0),
            )))
        }
        _ => None,
    }
}

fn bracket_trees_are_mutually_exclusive(left: &BracketExprTree, right: &BracketExprTree) -> bool {
    if bracket_tree_is_conjunctive_complement(left, right)
        || bracket_tree_is_conjunctive_complement(right, left)
    {
        return true;
    }

    match (left, right) {
        (BracketExprTree::Or(items), other) | (other, BracketExprTree::Or(items)) => {
            return items
                .iter()
                .all(|item| bracket_trees_are_mutually_exclusive(item, other));
        }
        _ => {}
    }

    let left_items = bracket_conjunction_items(left);
    let right_items = bracket_conjunction_items(right);
    left_items.iter().any(|left_item| {
        right_items
            .iter()
            .any(|right_item| bracket_tree_pair_is_mutually_exclusive(left_item, right_item))
    })
}

fn bracket_conjunction_items(tree: &BracketExprTree) -> &[BracketExprTree] {
    match tree {
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => items,
        other => core::slice::from_ref(other),
    }
}

fn bracket_tree_pair_is_mutually_exclusive(
    left: &BracketExprTree,
    right: &BracketExprTree,
) -> bool {
    match (left, right) {
        (BracketExprTree::Not(_), BracketExprTree::Not(_)) => false,
        (BracketExprTree::Not(excluded), candidate)
        | (candidate, BracketExprTree::Not(excluded)) => bracket_tree_implies(candidate, excluded),
        (BracketExprTree::Or(items), other) | (other, BracketExprTree::Or(items)) => items
            .iter()
            .all(|item| bracket_trees_are_mutually_exclusive(item, other)),
        (
            BracketExprTree::Primitive(left_primitive),
            BracketExprTree::Primitive(right_primitive),
        ) => atom_primitives_are_mutually_exclusive(left_primitive, right_primitive),
        _ => false,
    }
}

fn atom_primitives_are_mutually_exclusive(left: &AtomPrimitive, right: &AtomPrimitive) -> bool {
    if left == right {
        return false;
    }
    atom_aromaticity_classes_are_mutually_exclusive(left, right)
        || atom_element_identities_are_mutually_exclusive(left, right)
        || atom_isotope_identities_are_mutually_exclusive(left, right)
        || atom_numeric_queries_are_mutually_exclusive(left, right)
        || matches!((left, right), (AtomPrimitive::Charge(left), AtomPrimitive::Charge(right)) if left != right)
}

const fn atom_aromaticity_classes_are_mutually_exclusive(
    left: &AtomPrimitive,
    right: &AtomPrimitive,
) -> bool {
    match (
        atom_primitive_aromaticity_class(left),
        atom_primitive_aromaticity_class(right),
    ) {
        (Some(left), Some(right)) => left != right,
        _ => false,
    }
}

const fn atom_primitive_aromaticity_class(primitive: &AtomPrimitive) -> Option<bool> {
    match primitive {
        AtomPrimitive::AliphaticAny => Some(false),
        AtomPrimitive::AromaticAny => Some(true),
        AtomPrimitive::Symbol { aromatic, .. } | AtomPrimitive::Isotope { aromatic, .. } => {
            Some(*aromatic)
        }
        _ => None,
    }
}

fn atom_element_identities_are_mutually_exclusive(
    left: &AtomPrimitive,
    right: &AtomPrimitive,
) -> bool {
    match (left, right) {
        (
            AtomPrimitive::Symbol {
                element: left_element,
                ..
            },
            AtomPrimitive::Symbol {
                element: right_element,
                ..
            },
        ) => left_element != right_element,
        (AtomPrimitive::Symbol { element, .. }, AtomPrimitive::Isotope { isotope, .. })
        | (AtomPrimitive::Isotope { isotope, .. }, AtomPrimitive::Symbol { element, .. }) => {
            element != &isotope.element()
        }
        (
            AtomPrimitive::Symbol {
                element,
                aromatic: _,
            },
            AtomPrimitive::AtomicNumber(atomic_number),
        )
        | (
            AtomPrimitive::AtomicNumber(atomic_number),
            AtomPrimitive::Symbol {
                element,
                aromatic: _,
            },
        ) => u16::from(element.atomic_number()) != *atomic_number,
        (
            AtomPrimitive::Isotope {
                isotope,
                aromatic: _,
            },
            AtomPrimitive::AtomicNumber(atomic_number),
        )
        | (
            AtomPrimitive::AtomicNumber(atomic_number),
            AtomPrimitive::Isotope {
                isotope,
                aromatic: _,
            },
        ) => u16::from(isotope.element().atomic_number()) != *atomic_number,
        (AtomPrimitive::AtomicNumber(left), AtomPrimitive::AtomicNumber(right)) => left != right,
        _ => false,
    }
}

fn atom_isotope_identities_are_mutually_exclusive(
    left: &AtomPrimitive,
    right: &AtomPrimitive,
) -> bool {
    match (left, right) {
        (
            AtomPrimitive::Isotope {
                isotope: left_isotope,
                ..
            },
            AtomPrimitive::Isotope {
                isotope: right_isotope,
                ..
            },
        ) => left_isotope != right_isotope,
        (AtomPrimitive::Isotope { isotope, .. }, AtomPrimitive::IsotopeWildcard(mass_number))
        | (AtomPrimitive::IsotopeWildcard(mass_number), AtomPrimitive::Isotope { isotope, .. }) => {
            isotope.mass_number() != *mass_number
        }
        (AtomPrimitive::IsotopeWildcard(left), AtomPrimitive::IsotopeWildcard(right)) => {
            left != right
        }
        _ => false,
    }
}

fn atom_numeric_queries_are_mutually_exclusive(
    left: &AtomPrimitive,
    right: &AtomPrimitive,
) -> bool {
    match (left, right) {
        (AtomPrimitive::Degree(left), AtomPrimitive::Degree(right))
        | (AtomPrimitive::Connectivity(left), AtomPrimitive::Connectivity(right))
        | (AtomPrimitive::Valence(left), AtomPrimitive::Valence(right))
        | (AtomPrimitive::HeteroNeighbor(left), AtomPrimitive::HeteroNeighbor(right))
        | (
            AtomPrimitive::AliphaticHeteroNeighbor(left),
            AtomPrimitive::AliphaticHeteroNeighbor(right),
        ) => intersect_numeric_ranges(numeric_count_range(*left), numeric_count_range(*right))
            .is_none(),
        (AtomPrimitive::Hydrogen(left_kind, left), AtomPrimitive::Hydrogen(right_kind, right))
            if left_kind == right_kind =>
        {
            intersect_numeric_ranges(numeric_count_range(*left), numeric_count_range(*right))
                .is_none()
        }
        (AtomPrimitive::RingMembership(left), AtomPrimitive::RingMembership(right))
        | (AtomPrimitive::RingSize(left), AtomPrimitive::RingSize(right))
        | (AtomPrimitive::RingConnectivity(left), AtomPrimitive::RingConnectivity(right)) => {
            intersect_numeric_ranges(numeric_ring_range(*left), numeric_ring_range(*right))
                .is_none()
        }
        (AtomPrimitive::Hybridization(left), AtomPrimitive::Hybridization(right)) => {
            intersect_numeric_ranges(numeric_query_range(*left), numeric_query_range(*right))
                .is_none()
        }
        _ => false,
    }
}

fn bracket_or_contains_aliphatic_and_aromatic_any(items: &[BracketExprTree]) -> bool {
    items.iter().any(is_aliphatic_any) && items.iter().any(is_aromatic_any)
}

fn remove_absorbed_bracket_disjunction_terms(items: &mut Vec<BracketExprTree>) {
    remove_absorbed_disjunction_terms(items, bracket_or_item_absorbs);
}

fn relax_absorbed_bracket_disjunction_conjunction_alternatives(
    items: &mut Vec<BracketExprTree>,
) -> bool {
    relax_absorbed_disjunction_conjunction_alternatives(
        items,
        &AbsorbedDisjunctionConjunctionOps {
            conjunction_items: bracket_conjunction_items_with_kind,
            disjunction_items: bracket_disjunction_items,
            tree_implies: bracket_tree_implies,
            simplify_or: simplify_bracket_or,
            simplify_and: simplify_bracket_and,
        },
    )
}

fn bracket_or_item_absorbs(base: &BracketExprTree, candidate: &BracketExprTree) -> bool {
    if bracket_tree_implies(candidate, base) {
        return true;
    }
    match candidate {
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => items
            .iter()
            .any(|item| item == base || bracket_tree_implies(item, base)),
        _ => false,
    }
}

fn bracket_conjunction_items_with_kind(
    tree: &BracketExprTree,
) -> Option<(&[BracketExprTree], BracketAndKind)> {
    match tree {
        BracketExprTree::HighAnd(items) => Some((items, BracketAndKind::High)),
        BracketExprTree::LowAnd(items) => Some((items, BracketAndKind::Low)),
        _ => None,
    }
}

fn bracket_disjunction_items(tree: &BracketExprTree) -> Option<&[BracketExprTree]> {
    let BracketExprTree::Or(items) = tree else {
        return None;
    };
    Some(items)
}

fn can_render_bracket_and_as_high_and(items: &[BracketExprTree]) -> bool {
    items
        .iter()
        .all(|item| !matches!(item, BracketExprTree::Or(_) | BracketExprTree::LowAnd(_)))
}

const fn is_aliphatic_any(item: &BracketExprTree) -> bool {
    matches!(
        item,
        BracketExprTree::Primitive(AtomPrimitive::AliphaticAny)
    )
}

const fn is_aromatic_any(item: &BracketExprTree) -> bool {
    matches!(item, BracketExprTree::Primitive(AtomPrimitive::AromaticAny))
}

fn canonical_bond_expr(expr: &BondExpr) -> BondExpr {
    match expr {
        BondExpr::Elided => BondExpr::Elided,
        BondExpr::Query(tree) => {
            let canonical = canonical_bond_tree_candidate(tree.clone());
            if !bond_tree_contains_direction(tree, Bond::Up)
                && !bond_tree_contains_direction(tree, Bond::Down)
                && !bond_tree_contains_direction(&canonical, Bond::Up)
                && !bond_tree_contains_direction(&canonical, Bond::Down)
            {
                return BondExpr::Query(canonical);
            }

            let flipped = canonical_bond_tree_candidate(flip_directional_bond_tree(tree.clone()));
            let canonical_string = canonical.to_string();
            let flipped_string = flipped.to_string();
            let chosen = if flipped_string < canonical_string {
                flipped
            } else {
                canonical
            };
            BondExpr::Query(chosen)
        }
    }
}

fn canonical_bond_tree_candidate(tree: BondExprTree) -> BondExprTree {
    let mut normalized = tree;
    normalize_bond_tree(&mut normalized)
        .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
    let mut canonical = canonical_bond_tree(normalized);
    canonical = simplify_bond_tree(canonical);
    normalize_bond_tree(&mut canonical)
        .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
    if bond_tree_contains_direction(&canonical, Bond::Up)
        && bond_tree_contains_direction(&canonical, Bond::Down)
    {
        canonical = collapse_mixed_directional_polarity(canonical);
        canonical = simplify_bond_tree(canonical);
        normalize_bond_tree(&mut canonical)
            .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
    }
    canonical_bond_tree(canonical)
}

fn bond_tree_contains_direction(tree: &BondExprTree, direction: Bond) -> bool {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(bond)) => *bond == direction,
        BondExprTree::Primitive(_) => false,
        BondExprTree::Not(inner) => bond_tree_contains_direction(inner, direction),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            items
                .iter()
                .any(|item| bond_tree_contains_direction(item, direction))
        }
    }
}

fn bond_tree_contains_negated_direction(tree: &BondExprTree) -> bool {
    match tree {
        BondExprTree::Primitive(_) => false,
        BondExprTree::Not(inner) => {
            bond_tree_contains_direction(inner, Bond::Up)
                || bond_tree_contains_direction(inner, Bond::Down)
        }
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            items.iter().any(bond_tree_contains_negated_direction)
        }
    }
}

fn collapse_mixed_directional_polarity(tree: BondExprTree) -> BondExprTree {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(Bond::Down)) => {
            BondExprTree::Primitive(BondPrimitive::Bond(Bond::Up))
        }
        BondExprTree::Primitive(primitive) => BondExprTree::Primitive(primitive),
        BondExprTree::Not(inner) => {
            BondExprTree::Not(Box::new(collapse_mixed_directional_polarity(*inner)))
        }
        BondExprTree::HighAnd(items) => BondExprTree::HighAnd(
            items
                .into_iter()
                .map(collapse_mixed_directional_polarity)
                .collect(),
        ),
        BondExprTree::Or(items) => BondExprTree::Or(
            items
                .into_iter()
                .map(collapse_mixed_directional_polarity)
                .collect(),
        ),
        BondExprTree::LowAnd(items) => BondExprTree::LowAnd(
            items
                .into_iter()
                .map(collapse_mixed_directional_polarity)
                .collect(),
        ),
    }
}

fn canonical_bond_tree(tree: BondExprTree) -> BondExprTree {
    match tree {
        BondExprTree::Primitive(primitive) => BondExprTree::Primitive(primitive),
        BondExprTree::Not(inner) => BondExprTree::Not(Box::new(canonical_bond_tree(*inner))),
        BondExprTree::HighAnd(items) => {
            let mut items = items
                .into_iter()
                .map(canonical_bond_tree)
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BondExprTree::to_string);
            items.dedup();
            BondExprTree::HighAnd(items)
        }
        BondExprTree::Or(items) => {
            let mut items = items
                .into_iter()
                .map(canonical_bond_tree)
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BondExprTree::to_string);
            items.dedup();
            BondExprTree::Or(items)
        }
        BondExprTree::LowAnd(items) => {
            let mut items = items
                .into_iter()
                .map(canonical_bond_tree)
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BondExprTree::to_string);
            items.dedup();
            BondExprTree::LowAnd(items)
        }
    }
}

fn simplify_bond_tree(tree: BondExprTree) -> BondExprTree {
    match tree {
        BondExprTree::Primitive(primitive) => BondExprTree::Primitive(primitive),
        BondExprTree::Not(inner) => simplify_bond_not(simplify_bond_tree(*inner)),
        BondExprTree::HighAnd(items) => simplify_bond_and(items, BondAndKind::High),
        BondExprTree::LowAnd(items) => simplify_bond_and(items, BondAndKind::Low),
        BondExprTree::Or(items) => simplify_bond_or(items),
    }
}

fn simplify_bond_not(inner: BondExprTree) -> BondExprTree {
    if is_false_bond_tree(&inner) {
        BondExprTree::Primitive(BondPrimitive::Any)
    } else if let BondExprTree::Not(grandchild) = inner {
        *grandchild
    } else {
        BondExprTree::Not(Box::new(inner))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BondAndKind {
    High,
    Low,
}

fn simplify_bond_and(items: Vec<BondExprTree>, kind: BondAndKind) -> BondExprTree {
    let mut items = items
        .into_iter()
        .map(simplify_bond_tree)
        .collect::<Vec<_>>();
    flatten_bond_and_items(&mut items);
    if items.iter().any(is_false_bond_tree) || bond_and_contains_mutually_exclusive_pair(&items) {
        return false_bond_tree();
    }
    if items.len() > 1 {
        items.retain(|item| !matches!(item, BondExprTree::Primitive(BondPrimitive::Any)));
    }
    while let Some(changed) = prune_bond_disjunctions_against_conjunction_terms(&mut items) {
        if !changed {
            break;
        }
    }
    if items.iter().any(is_false_bond_tree) || bond_and_contains_mutually_exclusive_pair(&items) {
        return false_bond_tree();
    }
    if kind == BondAndKind::Low {
        loop {
            let mut changed = false;
            while remove_redundant_bond_conjunction_consensus_terms(&mut items) {
                changed = true;
            }
            while factor_common_bond_conjunction_disjunction_terms(&mut items) {
                changed = true;
            }
            while resolve_complementary_bond_conjunction_disjunction_terms(&mut items) {
                changed = true;
            }
            while factor_crossed_bond_conjunction_disjunction_terms(&mut items) {
                changed = true;
            }
            while factor_covered_bond_conjunction_disjunction_terms(&mut items) {
                changed = true;
            }
            while distribute_pruned_bond_conjunction_disjunction_pair(&mut items) {
                changed = true;
            }
            while relax_covered_bond_conjunction_disjunction_alternatives(&mut items) {
                changed = true;
            }
            while remove_context_absorbed_bond_disjunction_alternatives(&mut items) {
                changed = true;
            }
            while normalize_rotated_bond_disjunction_cover_terms(&mut items) {
                changed = true;
            }
            while distribute_bond_conjunction_term_into_disjunction(&mut items) {
                changed = true;
            }
            while let Some(pruned) = prune_bond_disjunctions_against_conjunction_terms(&mut items) {
                if !pruned {
                    break;
                }
                changed = true;
            }
            if !changed {
                break;
            }
            if items.iter().any(is_false_bond_tree) {
                return false_bond_tree();
            }
        }
    }
    remove_implied_bond_conjunction_terms(&mut items);
    sort_and_dedup_bond_items(&mut items);
    match items.as_slice() {
        [] => BondExprTree::Primitive(BondPrimitive::Any),
        [single] => single.clone(),
        _ if !can_render_bond_and_as_high_and(&items) => BondExprTree::LowAnd(items),
        _ => BondExprTree::HighAnd(items),
    }
}

fn flatten_bond_and_items(items: &mut Vec<BondExprTree>) -> bool {
    let mut flattened = Vec::with_capacity(items.len());
    let mut changed = false;
    for item in items.drain(..) {
        match item {
            BondExprTree::HighAnd(children) | BondExprTree::LowAnd(children) => {
                changed = true;
                flattened.extend(children);
            }
            other => flattened.push(other),
        }
    }
    *items = flattened;
    changed
}

fn simplify_bond_or(items: Vec<BondExprTree>) -> BondExprTree {
    let mut items = items
        .into_iter()
        .map(simplify_bond_tree)
        .collect::<Vec<_>>();
    flatten_bond_or_items(&mut items);
    if items.len() > 1 {
        items.retain(|item| !is_false_bond_tree(item));
    }
    if items
        .iter()
        .any(|item| matches!(item, BondExprTree::Primitive(BondPrimitive::Any)))
        || bond_or_contains_complement_pair(&items)
        || bond_or_contains_implied_complement_pair(&items)
    {
        return BondExprTree::Primitive(BondPrimitive::Any);
    }
    while relax_complemented_bond_disjunction_terms(&mut items) {}
    while collapse_bond_disjunction_consensus_terms(&mut items) {}
    while relax_negated_bond_conjunction_terms(&mut items) {}
    while relax_covered_negated_bond_disjunction_terms(&mut items) {}
    while relax_conjunctive_complement_bond_disjunction_terms(&mut items) {}
    while relax_partitioned_bond_disjunction_terms(&mut items) {}
    while remove_redundant_bond_disjunction_consensus_terms(&mut items) {}
    while relax_absorbed_bond_disjunction_conjunction_alternatives(&mut items) {}
    if let Some(factored) = factor_common_bond_disjunction_terms(&items) {
        return factored;
    }
    remove_absorbed_bond_disjunction_terms(&mut items);
    if let Some(distributed) = distribute_low_precedence_bond_disjunction(&items) {
        return distributed;
    }
    sort_and_dedup_bond_items(&mut items);
    match items.as_slice() {
        [] => false_bond_tree(),
        [single] => single.clone(),
        _ => BondExprTree::Or(items),
    }
}

fn flatten_bond_or_items(items: &mut Vec<BondExprTree>) -> bool {
    let mut flattened = Vec::with_capacity(items.len());
    let mut changed = false;
    for item in items.drain(..) {
        if let BondExprTree::Or(children) = item {
            changed = true;
            flattened.extend(children);
        } else {
            flattened.push(item);
        }
    }
    *items = flattened;
    changed
}

fn distribute_low_precedence_bond_disjunction(items: &[BondExprTree]) -> Option<BondExprTree> {
    let (distributed_index, terms) =
        items
            .iter()
            .enumerate()
            .find_map(|(index, item)| match item {
                BondExprTree::LowAnd(terms) => Some((index, terms)),
                _ => None,
            })?;

    let rest = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (index != distributed_index).then_some(item.clone()))
        .collect::<Vec<_>>();
    let factors = terms
        .iter()
        .map(|term| {
            let mut alternatives = rest.clone();
            alternatives.push(term.clone());
            flatten_bond_or_items(&mut alternatives);
            simplify_bond_or(alternatives)
        })
        .collect::<Vec<_>>();

    Some(simplify_bond_and(factors, BondAndKind::Low))
}

fn false_bond_tree() -> BondExprTree {
    BondExprTree::Not(Box::new(BondExprTree::Primitive(BondPrimitive::Any)))
}

fn is_false_bond_tree(tree: &BondExprTree) -> bool {
    matches!(
        tree,
        BondExprTree::Not(inner)
            if matches!(inner.as_ref(), BondExprTree::Primitive(BondPrimitive::Any))
    )
}

fn prune_bond_disjunctions_against_conjunction_terms(items: &mut [BondExprTree]) -> Option<bool> {
    for disjunction_index in 0..items.len() {
        let BondExprTree::Or(alternatives) = &items[disjunction_index] else {
            continue;
        };
        let retained = alternatives
            .iter()
            .filter(|alternative| {
                !items.iter().enumerate().any(|(other_index, other)| {
                    other_index != disjunction_index
                        && bond_trees_are_mutually_exclusive(alternative, other)
                })
            })
            .cloned()
            .collect::<Vec<_>>();

        if retained.is_empty() {
            items[disjunction_index] = false_bond_tree();
            return None;
        }
        if retained.len() != alternatives.len() {
            items[disjunction_index] = simplify_bond_or(retained);
            return Some(true);
        }
    }

    Some(false)
}

fn factor_common_bond_conjunction_disjunction_terms(items: &mut Vec<BondExprTree>) -> bool {
    for left_index in 0..items.len() {
        let left_alternatives = bond_disjunction_factor_items(&items[left_index]);
        if left_alternatives.len() < 2 {
            continue;
        }

        for right_index in (left_index + 1)..items.len() {
            let right_alternatives = bond_disjunction_factor_items(&items[right_index]);
            if right_alternatives.len() < 2 {
                continue;
            }

            let common = left_alternatives
                .iter()
                .filter(|item| right_alternatives.contains(item))
                .cloned()
                .collect::<Vec<_>>();
            if common.is_empty() {
                continue;
            }

            let Some(replacement) =
                factor_bond_disjunction_pair(&left_alternatives, &right_alternatives, common)
            else {
                continue;
            };
            items.remove(right_index);
            items.remove(left_index);
            items.push(replacement);
            return true;
        }
    }

    false
}

fn factor_bond_disjunction_pair(
    left_alternatives: &[BondExprTree],
    right_alternatives: &[BondExprTree],
    mut common: Vec<BondExprTree>,
) -> Option<BondExprTree> {
    let left_remainder = disjunction_remainder(left_alternatives, &common);
    let right_remainder = disjunction_remainder(right_alternatives, &common);
    if left_remainder.is_empty() || right_remainder.is_empty() {
        return Some(simplify_bond_or(common));
    }

    let remainder = simplify_bond_and(
        vec![
            simplify_bond_or(left_remainder),
            simplify_bond_or(right_remainder),
        ],
        BondAndKind::Low,
    );
    if matches!(remainder, BondExprTree::LowAnd(_)) {
        return None;
    }
    if !is_false_bond_tree(&remainder) {
        common.push(remainder);
    }
    Some(simplify_bond_or(common))
}

fn resolve_complementary_bond_conjunction_disjunction_terms(items: &mut Vec<BondExprTree>) -> bool {
    for left_index in 0..items.len() {
        let left_alternatives = bond_disjunction_factor_items(&items[left_index]);
        if left_alternatives.len() < 2 {
            continue;
        }

        for right_index in (left_index + 1)..items.len() {
            let right_alternatives = bond_disjunction_factor_items(&items[right_index]);
            if right_alternatives.len() < 2 {
                continue;
            }

            for left_complement_index in 0..left_alternatives.len() {
                for right_complement_index in 0..right_alternatives.len() {
                    if !bond_trees_are_complements(
                        &left_alternatives[left_complement_index],
                        &right_alternatives[right_complement_index],
                    ) {
                        continue;
                    }

                    let left_condition = left_alternatives[left_complement_index].clone();
                    let right_condition = right_alternatives[right_complement_index].clone();
                    let left_remainder =
                        bond_items_without_index(&left_alternatives, left_complement_index);
                    let right_remainder =
                        bond_items_without_index(&right_alternatives, right_complement_index);
                    let left_remainder = simplify_bond_or(left_remainder);
                    let right_remainder = simplify_bond_or(right_remainder);
                    if matches!(
                        left_remainder,
                        BondExprTree::Or(_) | BondExprTree::LowAnd(_)
                    ) || matches!(
                        right_remainder,
                        BondExprTree::Or(_) | BondExprTree::LowAnd(_)
                    ) {
                        continue;
                    }

                    let left_branch =
                        simplify_bond_conjunction_pair(left_condition, right_remainder);
                    let right_branch =
                        simplify_bond_conjunction_pair(right_condition, left_remainder);
                    if matches!(left_branch, BondExprTree::Or(_) | BondExprTree::LowAnd(_))
                        || matches!(right_branch, BondExprTree::Or(_) | BondExprTree::LowAnd(_))
                    {
                        continue;
                    }

                    let replacement = simplify_bond_or(vec![left_branch, right_branch]);

                    remove_indices_descending(items, &[left_index, right_index]);
                    items.push(replacement);
                    return true;
                }
            }
        }
    }

    false
}

fn factor_crossed_bond_conjunction_disjunction_terms(items: &mut Vec<BondExprTree>) -> bool {
    for left_index in 0..items.len() {
        let left_alternatives = bond_disjunction_factor_items(&items[left_index]);
        if left_alternatives.len() != 2 {
            continue;
        }

        for right_index in (left_index + 1)..items.len() {
            let right_alternatives = bond_disjunction_factor_items(&items[right_index]);
            if right_alternatives.len() != 2 {
                continue;
            }

            let Some(replacement) =
                factor_crossed_bond_disjunction_pair(&left_alternatives, &right_alternatives)
            else {
                continue;
            };
            remove_indices_descending(items, &[left_index, right_index]);
            items.push(replacement);
            return true;
        }
    }

    false
}

fn factor_crossed_bond_disjunction_pair(
    left_alternatives: &[BondExprTree],
    right_alternatives: &[BondExprTree],
) -> Option<BondExprTree> {
    for left_conjunction_index in 0..2 {
        let left_bare = &left_alternatives[1 - left_conjunction_index];
        let (left_items, _) = bond_consensus_items(&left_alternatives[left_conjunction_index]);

        for right_conjunction_index in 0..2 {
            let right_bare = &right_alternatives[1 - right_conjunction_index];
            let Some(left_common) = bond_items_without_item(&left_items, right_bare) else {
                continue;
            };
            let (right_items, _) =
                bond_consensus_items(&right_alternatives[right_conjunction_index]);
            let Some(right_common) = bond_items_without_item(&right_items, left_bare) else {
                continue;
            };

            if !bond_trees_are_mutually_exclusive(left_bare, right_bare) {
                continue;
            }
            if left_common != right_common {
                continue;
            }

            let mut factored = left_common;
            factored.push(simplify_bond_or(vec![
                left_bare.clone(),
                right_bare.clone(),
            ]));
            return Some(simplify_bond_and(factored, BondAndKind::Low));
        }
    }

    None
}

fn bond_items_without_item(
    items: &[BondExprTree],
    removed: &BondExprTree,
) -> Option<Vec<BondExprTree>> {
    let index = items.iter().position(|item| item == removed)?;
    let mut remainder = bond_items_without_index(items, index);
    sort_and_dedup_bond_items(&mut remainder);
    Some(remainder)
}

fn factor_covered_bond_conjunction_disjunction_terms(items: &mut Vec<BondExprTree>) -> bool {
    for left_index in 0..items.len() {
        let left_alternatives = bond_disjunction_factor_items(&items[left_index]);
        if left_alternatives.len() != 2 {
            continue;
        }

        for right_index in (left_index + 1)..items.len() {
            let right_alternatives = bond_disjunction_factor_items(&items[right_index]);
            if right_alternatives.len() != 2 {
                continue;
            }

            let replacement =
                factor_covered_bond_disjunction_pair(&left_alternatives, &right_alternatives)
                    .or_else(|| {
                        factor_covered_bond_disjunction_pair(
                            &right_alternatives,
                            &left_alternatives,
                        )
                    });
            let Some(replacement) = replacement else {
                continue;
            };
            remove_indices_descending(items, &[left_index, right_index]);
            items.push(replacement);
            return true;
        }
    }

    false
}

fn factor_covered_bond_disjunction_pair(
    expanded_alternatives: &[BondExprTree],
    covering_alternatives: &[BondExprTree],
) -> Option<BondExprTree> {
    for expanded_index in 0..2 {
        let exclusive = &expanded_alternatives[1 - expanded_index];
        let (expanded_items, _) = bond_consensus_items(&expanded_alternatives[expanded_index]);
        if expanded_items.len() < 2 {
            continue;
        }

        for term_index in 0..expanded_items.len() {
            let term = &expanded_items[term_index];
            if !bond_trees_are_mutually_exclusive(term, exclusive) {
                continue;
            }

            let mut common = bond_items_without_index(&expanded_items, term_index);
            sort_and_dedup_bond_items(&mut common);
            if common.is_empty() {
                continue;
            }
            let common_tree = simplify_bond_and(common.clone(), BondAndKind::High);
            let covers_term = covering_alternatives.iter().any(|item| item == term);
            let covers_common = covering_alternatives
                .iter()
                .any(|item| bond_trees_are_equivalent(item, &common_tree));
            if !covers_term || !covers_common {
                continue;
            }

            common.push(simplify_bond_or(vec![term.clone(), exclusive.clone()]));
            return Some(simplify_bond_and(common, BondAndKind::Low));
        }
    }

    None
}

fn distribute_pruned_bond_conjunction_disjunction_pair(items: &mut Vec<BondExprTree>) -> bool {
    const MAX_DISTRIBUTED_BRANCHES: usize = 16;

    for left_index in 0..items.len() {
        let left_alternatives = bond_disjunction_factor_items(&items[left_index]);
        if left_alternatives.len() < 2 {
            continue;
        }

        for right_index in (left_index + 1)..items.len() {
            let right_alternatives = bond_disjunction_factor_items(&items[right_index]);
            if right_alternatives.len() < 2 {
                continue;
            }
            if left_alternatives.len() * right_alternatives.len() > MAX_DISTRIBUTED_BRANCHES {
                continue;
            }

            let Some(replacement) =
                pruned_bond_disjunction_product(&left_alternatives, &right_alternatives)
            else {
                continue;
            };
            remove_indices_descending(items, &[left_index, right_index]);
            items.push(replacement);
            return true;
        }
    }

    false
}

fn pruned_bond_disjunction_product(
    left_alternatives: &[BondExprTree],
    right_alternatives: &[BondExprTree],
) -> Option<BondExprTree> {
    let mut pruned = false;
    let mut branches = Vec::new();

    for left in left_alternatives {
        for right in right_alternatives {
            if bond_trees_are_mutually_exclusive(left, right) {
                pruned = true;
                continue;
            }

            let branch = simplify_bond_conjunction_pair(left.clone(), right.clone());
            if matches!(branch, BondExprTree::Or(_) | BondExprTree::LowAnd(_)) {
                return None;
            }
            if is_false_bond_tree(&branch) {
                pruned = true;
                continue;
            }
            branches.push(branch);
        }
    }

    sort_and_dedup_bond_items(&mut branches);
    if !pruned || branches.is_empty() {
        return None;
    }

    let replacement = simplify_bond_or(branches);
    if matches!(replacement, BondExprTree::LowAnd(_)) {
        return None;
    }
    Some(replacement)
}

fn relax_covered_bond_conjunction_disjunction_alternatives(items: &mut [BondExprTree]) -> bool {
    for base_index in 0..items.len() {
        let base_alternatives = bond_disjunction_factor_items(&items[base_index]);
        if base_alternatives.len() < 2 {
            continue;
        }

        let mut candidate_index = 0usize;
        while candidate_index < items.len() {
            if base_index == candidate_index {
                candidate_index += 1;
                continue;
            }
            let candidate_alternatives = bond_disjunction_factor_items(&items[candidate_index]);
            if candidate_alternatives.len() < 2 {
                candidate_index += 1;
                continue;
            }

            for base_alternative_index in 0..base_alternatives.len() {
                if !base_alternative_remainder_is_covered(
                    &base_alternatives,
                    base_alternative_index,
                    &candidate_alternatives,
                ) {
                    continue;
                }

                for candidate_alternative_index in 0..candidate_alternatives.len() {
                    let Some(residual) = remove_bond_term_from_conjunction(
                        &candidate_alternatives[candidate_alternative_index],
                        &base_alternatives[base_alternative_index],
                    ) else {
                        continue;
                    };

                    let mut replacement = candidate_alternatives.clone();
                    replacement[candidate_alternative_index] = residual;
                    items[candidate_index] = simplify_bond_or(replacement);
                    return true;
                }
            }
            candidate_index += 1;
        }
    }

    false
}

fn base_alternative_remainder_is_covered(
    base_alternatives: &[BondExprTree],
    base_alternative_index: usize,
    candidate_alternatives: &[BondExprTree],
) -> bool {
    base_alternatives
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != base_alternative_index)
        .all(|(_, base_alternative)| {
            candidate_alternatives.iter().any(|candidate_alternative| {
                bond_tree_implies(base_alternative, candidate_alternative)
            })
        })
}

fn remove_context_absorbed_bond_disjunction_alternatives(items: &mut [BondExprTree]) -> bool {
    for disjunction_index in 0..items.len() {
        let BondExprTree::Or(alternatives) = &items[disjunction_index] else {
            continue;
        };

        for candidate_index in 0..alternatives.len() {
            for absorber_index in 0..alternatives.len() {
                if candidate_index == absorber_index
                    || !bond_alternative_is_absorbed_by_conjunction_context(
                        &alternatives[candidate_index],
                        &alternatives[absorber_index],
                        items,
                        disjunction_index,
                    )
                {
                    continue;
                }

                let replacement = bond_items_without_index(alternatives, candidate_index);
                items[disjunction_index] = simplify_bond_or(replacement);
                return true;
            }
        }
    }

    false
}

fn normalize_rotated_bond_disjunction_cover_terms(items: &mut Vec<BondExprTree>) -> bool {
    for left_index in 0..items.len() {
        for right_index in (left_index + 1)..items.len() {
            let candidates =
                rotated_bond_disjunction_cover_candidates(&items[left_index], &items[right_index]);
            if candidates.is_empty() {
                continue;
            }

            let mut best_key = bond_low_and_display_key(items);
            let mut best_items = None;
            for replacement_pair in candidates {
                let mut candidate = items
                    .iter()
                    .enumerate()
                    .filter_map(|(index, item)| {
                        (index != left_index && index != right_index).then_some(item.clone())
                    })
                    .collect::<Vec<_>>();
                candidate.extend(replacement_pair);
                sort_and_dedup_bond_items(&mut candidate);

                let candidate_key = bond_low_and_display_key(&candidate);
                if candidate_key < best_key {
                    best_key = candidate_key;
                    best_items = Some(candidate);
                }
            }

            if let Some(replacement) = best_items {
                *items = replacement;
                return true;
            }
        }
    }

    false
}

fn rotated_bond_disjunction_cover_candidates(
    left: &BondExprTree,
    right: &BondExprTree,
) -> Vec<Vec<BondExprTree>> {
    let mut candidates = Vec::new();
    if let Some((bare, first, second)) = bond_disjunction_cover_terms(left, right) {
        push_bond_disjunction_cover_rotations(&mut candidates, &bare, &first, &second);
    }
    if let Some((bare, first, second)) = bond_disjunction_cover_terms(right, left) {
        push_bond_disjunction_cover_rotations(&mut candidates, &bare, &first, &second);
    }
    candidates
}

fn bond_disjunction_cover_terms(
    expanded: &BondExprTree,
    cover: &BondExprTree,
) -> Option<(BondExprTree, BondExprTree, BondExprTree)> {
    let expanded_alternatives = bond_disjunction_factor_items(expanded);
    let cover_alternatives = bond_disjunction_factor_items(cover);
    if expanded_alternatives.len() != 2 || cover_alternatives.len() != 2 {
        return None;
    }

    let mut cover_terms = cover_alternatives;
    sort_and_dedup_bond_items(&mut cover_terms);
    if cover_terms
        .iter()
        .any(|term| matches!(term, BondExprTree::Or(_) | BondExprTree::LowAnd(_)))
    {
        return None;
    }

    let mut bare = None;
    let mut covered_conjunction = None;
    for alternative in expanded_alternatives {
        let (mut conjunction_terms, kind) = bond_consensus_items(&alternative);
        sort_and_dedup_bond_items(&mut conjunction_terms);
        if kind == BondAndKind::High
            && conjunction_terms.len() == 2
            && conjunction_terms == cover_terms
        {
            covered_conjunction = Some(alternative);
        } else if !matches!(alternative, BondExprTree::Or(_) | BondExprTree::LowAnd(_)) {
            bare = Some(alternative);
        } else {
            return None;
        }
    }

    covered_conjunction?;
    let bare = bare?;
    Some((bare, cover_terms[0].clone(), cover_terms[1].clone()))
}

fn push_bond_disjunction_cover_rotations(
    candidates: &mut Vec<Vec<BondExprTree>>,
    bare: &BondExprTree,
    first: &BondExprTree,
    second: &BondExprTree,
) {
    for (rotated_bare, rotated_first, rotated_second) in [
        (bare, first, second),
        (first, second, bare),
        (second, bare, first),
    ] {
        if let Some(candidate) =
            bond_disjunction_cover_rotation(rotated_bare, rotated_first, rotated_second)
        {
            candidates.push(candidate);
        }
    }
}

fn bond_disjunction_cover_rotation(
    bare: &BondExprTree,
    first: &BondExprTree,
    second: &BondExprTree,
) -> Option<Vec<BondExprTree>> {
    let conjunction = simplify_bond_and(vec![first.clone(), second.clone()], BondAndKind::High);
    if !matches!(conjunction, BondExprTree::HighAnd(_)) {
        return None;
    }

    let mut replacement = vec![
        simplify_bond_or(vec![bare.clone(), conjunction]),
        simplify_bond_or(vec![first.clone(), second.clone()]),
    ];
    sort_and_dedup_bond_items(&mut replacement);
    Some(replacement)
}

fn bond_low_and_display_key(items: &[BondExprTree]) -> String {
    let mut sorted = items.to_vec();
    sort_and_dedup_bond_items(&mut sorted);
    BondExprTree::LowAnd(sorted).to_string()
}

fn bond_alternative_is_absorbed_by_conjunction_context(
    candidate: &BondExprTree,
    absorber: &BondExprTree,
    items: &[BondExprTree],
    disjunction_index: usize,
) -> bool {
    if bond_tree_implies(candidate, absorber) {
        return true;
    }

    items.iter().enumerate().any(|(context_index, context)| {
        context_index != disjunction_index
            && bond_context_makes_alternative_absorbed(candidate, absorber, context)
    })
}

fn bond_context_makes_alternative_absorbed(
    candidate: &BondExprTree,
    absorber: &BondExprTree,
    context: &BondExprTree,
) -> bool {
    match context {
        BondExprTree::Or(alternatives) => alternatives.iter().all(|alternative| {
            bond_trees_are_mutually_exclusive(candidate, alternative)
                || bond_tree_implies(alternative, absorber)
        }),
        BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => items.iter().any(|item| {
            bond_trees_are_mutually_exclusive(candidate, item) || bond_tree_implies(item, absorber)
        }),
        other => {
            bond_trees_are_mutually_exclusive(candidate, other)
                || bond_tree_implies(other, absorber)
        }
    }
}

fn remove_redundant_bond_conjunction_consensus_terms(items: &mut Vec<BondExprTree>) -> bool {
    for candidate_index in 0..items.len() {
        for left_index in 0..items.len() {
            if left_index == candidate_index {
                continue;
            }

            for right_index in (left_index + 1)..items.len() {
                if right_index == candidate_index
                    || !bond_conjunction_consensus_term_is_redundant(
                        &items[left_index],
                        &items[right_index],
                        &items[candidate_index],
                    )
                {
                    continue;
                }

                items.remove(candidate_index);
                return true;
            }
        }
    }

    false
}

fn distribute_bond_conjunction_term_into_disjunction(items: &mut Vec<BondExprTree>) -> bool {
    const MAX_DISTRIBUTED_ALTERNATIVES: usize = 4;

    for disjunction_index in 0..items.len() {
        let BondExprTree::Or(alternatives) = &items[disjunction_index] else {
            continue;
        };
        let alternatives = alternatives.clone();

        for term_index in 0..items.len() {
            if term_index == disjunction_index {
                continue;
            }
            let term = items[term_index].clone();
            if matches!(term, BondExprTree::Or(_)) {
                continue;
            }
            let implied_by_alternative = alternatives
                .iter()
                .map(|alternative| bond_tree_implies(alternative, &term))
                .collect::<Vec<_>>();
            let partially_implied = implied_by_alternative.iter().any(|implied| *implied)
                && !implied_by_alternative.iter().all(|implied| *implied);
            let small_negated_directional_product = alternatives.len()
                <= MAX_DISTRIBUTED_ALTERNATIVES
                && (bond_tree_contains_negated_direction(&term)
                    || alternatives
                        .iter()
                        .any(bond_tree_contains_negated_direction));

            if !partially_implied && !small_negated_directional_product {
                continue;
            }
            if alternatives.iter().any(|alternative| {
                matches!(alternative, BondExprTree::Or(_) | BondExprTree::LowAnd(_))
            }) {
                continue;
            }

            let replacement = alternatives
                .iter()
                .cloned()
                .zip(implied_by_alternative)
                .map(|(alternative, implied)| {
                    if implied {
                        alternative
                    } else {
                        simplify_bond_conjunction_pair(term.clone(), alternative)
                    }
                })
                .collect::<Vec<_>>();
            if replacement.iter().any(|alternative| {
                matches!(alternative, BondExprTree::Or(_) | BondExprTree::LowAnd(_))
            }) {
                continue;
            }

            remove_indices_descending(items, &[disjunction_index, term_index]);
            items.push(simplify_bond_or(replacement));
            return true;
        }
    }

    false
}

fn simplify_bond_conjunction_pair(left: BondExprTree, right: BondExprTree) -> BondExprTree {
    let items = vec![left, right];
    let kind = if can_render_bond_and_as_high_and(&items) {
        BondAndKind::High
    } else {
        BondAndKind::Low
    };
    simplify_bond_and(items, kind)
}

fn bond_conjunction_consensus_term_is_redundant(
    left: &BondExprTree,
    right: &BondExprTree,
    candidate: &BondExprTree,
) -> bool {
    let left_alternatives = bond_disjunction_factor_items(left);
    let right_alternatives = bond_disjunction_factor_items(right);
    if left_alternatives.len() < 2 || right_alternatives.len() < 2 {
        return false;
    }

    for left_complement_index in 0..left_alternatives.len() {
        for right_complement_index in 0..right_alternatives.len() {
            if !bond_trees_are_complements(
                &left_alternatives[left_complement_index],
                &right_alternatives[right_complement_index],
            ) {
                continue;
            }

            let mut consensus = bond_items_without_index(&left_alternatives, left_complement_index);
            consensus.extend(bond_items_without_index(
                &right_alternatives,
                right_complement_index,
            ));
            sort_and_dedup_bond_items(&mut consensus);
            let consensus = simplify_bond_or(consensus);
            if bond_tree_implies(&consensus, candidate) {
                return true;
            }
        }
    }

    false
}

fn bond_disjunction_factor_items(tree: &BondExprTree) -> Vec<BondExprTree> {
    let mut alternatives = match tree {
        BondExprTree::Or(items) => items.clone(),
        other => vec![other.clone()],
    };
    sort_and_dedup_bond_items(&mut alternatives);
    alternatives
}

fn remove_implied_bond_conjunction_terms(items: &mut Vec<BondExprTree>) {
    let mut index = 0usize;
    while index < items.len() {
        let is_implied = items.iter().enumerate().any(|(other_index, other)| {
            other_index != index && bond_tree_implies(other, &items[index])
        });
        if is_implied {
            items.remove(index);
        } else {
            index += 1;
        }
    }
}

fn bond_and_contains_mutually_exclusive_pair(items: &[BondExprTree]) -> bool {
    items.iter().enumerate().any(|(left_index, left)| {
        items.iter().enumerate().any(|(right_index, right)| {
            left_index != right_index && bond_trees_are_mutually_exclusive(left, right)
        })
    })
}

fn bond_or_contains_complement_pair(items: &[BondExprTree]) -> bool {
    items.iter().any(|item| {
        items.iter().any(|other| match other {
            BondExprTree::Not(inner) => inner.as_ref() == item,
            _ => false,
        })
    })
}

fn bond_or_contains_implied_complement_pair(items: &[BondExprTree]) -> bool {
    items.iter().any(|item| {
        items.iter().any(|other| match other {
            BondExprTree::Not(inner) => bond_tree_implies(inner, item),
            _ => bond_tree_is_conjunctive_complement(item, other),
        })
    })
}

fn bond_tree_is_conjunctive_complement(
    candidate: &BondExprTree,
    complement: &BondExprTree,
) -> bool {
    let Some(excluded) = negated_bond_conjunction_items(complement) else {
        return false;
    };
    let (excluded, residual) = split_negated_bond_conjunction_items(&excluded);
    if excluded.len() < 2 || !residual.is_empty() {
        return false;
    }
    let excluded = simplify_bond_or(excluded);
    bond_tree_implies(candidate, &excluded) && bond_tree_implies(&excluded, candidate)
}

fn negated_bond_conjunction_items(tree: &BondExprTree) -> Option<Vec<BondExprTree>> {
    let (BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items)) = tree else {
        return None;
    };
    Some(items.clone())
}

fn negated_bond_tree_inner(tree: &BondExprTree) -> Option<&BondExprTree> {
    let BondExprTree::Not(inner) = tree else {
        return None;
    };
    Some(inner.as_ref())
}

fn relax_conjunctive_complement_bond_disjunction_terms(items: &mut [BondExprTree]) -> bool {
    for base_index in 0..items.len() {
        for candidate_index in 0..items.len() {
            if base_index == candidate_index {
                continue;
            }
            let Some(candidate_items) = negated_bond_conjunction_items(&items[candidate_index])
            else {
                continue;
            };
            let (excluded, residual) = split_negated_bond_conjunction_items(&candidate_items);
            if excluded.len() < 2 || residual.is_empty() {
                continue;
            }
            let excluded = simplify_bond_or(excluded);
            if !bond_trees_are_equivalent(&items[base_index], &excluded) {
                continue;
            }
            items[candidate_index] = simplify_bond_residual_conjunction(residual);
            return true;
        }
    }

    false
}

fn relax_partitioned_bond_disjunction_terms(items: &mut [BondExprTree]) -> bool {
    for candidate_index in 0..items.len() {
        let (candidate_items, candidate_kind) = bond_consensus_items(&items[candidate_index]);
        for partition_index in 0..candidate_items.len() {
            let partition = candidate_items[partition_index].clone();
            let residual = bond_conjunction_from_items(
                bond_items_without_index(&candidate_items, partition_index),
                candidate_kind,
            );
            if residual == items[candidate_index] {
                continue;
            }
            let complement_branch = bond_conjunction_from_items(
                vec![complement_bond_partition_term(partition), residual.clone()],
                candidate_kind,
            );

            for cover_index in 0..items.len() {
                let (cover_items, _) = bond_consensus_items(&items[cover_index]);
                if cover_index == candidate_index
                    || cover_items.len() < 2
                    || !bond_tree_implies(&complement_branch, &items[cover_index])
                {
                    continue;
                }

                items[candidate_index] = residual;
                return true;
            }
        }
    }

    false
}

fn bond_conjunction_from_items(mut items: Vec<BondExprTree>, kind: BondAndKind) -> BondExprTree {
    match items.len() {
        0 => BondExprTree::Primitive(BondPrimitive::Any),
        1 => items.pop().expect("one item after len check"),
        _ if kind == BondAndKind::Low => BondExprTree::LowAnd(items),
        _ => BondExprTree::HighAnd(items),
    }
}

fn complement_bond_partition_term(term: BondExprTree) -> BondExprTree {
    match term {
        BondExprTree::Not(inner) => *inner,
        other => BondExprTree::Not(Box::new(other)),
    }
}

fn split_negated_bond_conjunction_items(
    items: &[BondExprTree],
) -> (Vec<BondExprTree>, Vec<BondExprTree>) {
    let mut excluded = Vec::new();
    let mut residual = Vec::new();
    for item in items {
        match item {
            BondExprTree::Not(inner) => excluded.push(inner.as_ref().clone()),
            _ => residual.push(item.clone()),
        }
    }
    (excluded, residual)
}

fn bond_trees_are_equivalent(left: &BondExprTree, right: &BondExprTree) -> bool {
    bond_tree_implies(left, right) && bond_tree_implies(right, left)
}

fn simplify_bond_residual_conjunction(items: Vec<BondExprTree>) -> BondExprTree {
    let kind = if can_render_bond_and_as_high_and(&items) {
        BondAndKind::High
    } else {
        BondAndKind::Low
    };
    simplify_bond_and(items, kind)
}

fn remove_absorbed_bond_disjunction_terms(items: &mut Vec<BondExprTree>) {
    remove_absorbed_disjunction_terms(items, bond_or_item_absorbs);
}

fn relax_absorbed_bond_disjunction_conjunction_alternatives(items: &mut Vec<BondExprTree>) -> bool {
    relax_absorbed_disjunction_conjunction_alternatives(
        items,
        &AbsorbedDisjunctionConjunctionOps {
            conjunction_items: bond_conjunction_items_with_kind,
            disjunction_items: bond_disjunction_items,
            tree_implies: bond_tree_implies,
            simplify_or: simplify_bond_or,
            simplify_and: simplify_bond_and,
        },
    )
}

fn collapse_bond_disjunction_consensus_terms(items: &mut Vec<BondExprTree>) -> bool {
    collapse_disjunction_consensus_terms(items, bond_disjunction_consensus)
}

fn relax_negated_bond_conjunction_terms(items: &mut [BondExprTree]) -> bool {
    relax_negated_conjunction_terms(
        items,
        negated_bond_tree_inner,
        remove_bond_term_from_conjunction,
    )
}

fn relax_complemented_bond_disjunction_terms(items: &mut [BondExprTree]) -> bool {
    relax_complemented_disjunction_terms(items, remove_negated_bond_term_from_conjunction)
}

fn remove_negated_bond_term_from_conjunction(
    candidate: &BondExprTree,
    base: &BondExprTree,
) -> Option<BondExprTree> {
    let (items, kind) = match candidate {
        BondExprTree::HighAnd(items) => (items, BondAndKind::High),
        BondExprTree::LowAnd(items) => (items, BondAndKind::Low),
        _ => return None,
    };
    let complement_index = items.iter().position(|item| match item {
        BondExprTree::Not(inner) => bond_tree_implies(inner, base),
        _ => false,
    })?;
    Some(simplify_bond_and(
        bond_items_without_index(items, complement_index),
        kind,
    ))
}

fn relax_covered_negated_bond_disjunction_terms(items: &mut Vec<BondExprTree>) -> bool {
    relax_covered_negated_disjunction_terms(
        items,
        &CoveredNegatedDisjunctionOps {
            consensus_items: bond_consensus_items,
            sort_and_dedup: sort_and_dedup_bond_items,
            trees_are_complements: bond_trees_are_complements,
            items_without_index: bond_items_without_index,
            can_render_and_as_high: can_render_bond_and_as_high_and,
            simplify_and: simplify_bond_and,
            high_kind: BondAndKind::High,
            low_kind: BondAndKind::Low,
        },
    )
}

fn remove_bond_term_from_conjunction(
    candidate: &BondExprTree,
    removed: &BondExprTree,
) -> Option<BondExprTree> {
    let (items, kind) = match candidate {
        BondExprTree::HighAnd(items) => (items, BondAndKind::High),
        BondExprTree::LowAnd(items) => (items, BondAndKind::Low),
        _ => return None,
    };
    let removed_index = items.iter().position(|item| item == removed)?;
    Some(simplify_bond_and(
        bond_items_without_index(items, removed_index),
        kind,
    ))
}

fn remove_redundant_bond_disjunction_consensus_terms(items: &mut Vec<BondExprTree>) -> bool {
    remove_redundant_disjunction_consensus_terms(
        items,
        bond_disjunction_consensus_term_is_redundant,
    )
}

fn bond_disjunction_consensus_term_is_redundant(
    left: &BondExprTree,
    right: &BondExprTree,
    candidate: &BondExprTree,
) -> bool {
    let (left_items, left_kind) = bond_consensus_items(left);
    let (right_items, right_kind) = bond_consensus_items(right);
    let kind = if left_kind == BondAndKind::Low || right_kind == BondAndKind::Low {
        BondAndKind::Low
    } else {
        BondAndKind::High
    };

    for left_complement_index in 0..left_items.len() {
        for right_complement_index in 0..right_items.len() {
            if !bond_trees_are_complements(
                &left_items[left_complement_index],
                &right_items[right_complement_index],
            ) {
                continue;
            }

            let mut consensus = bond_items_without_index(&left_items, left_complement_index);
            consensus.extend(bond_items_without_index(
                &right_items,
                right_complement_index,
            ));
            sort_and_dedup_bond_items(&mut consensus);
            let consensus = simplify_bond_and(consensus, kind);
            if bond_tree_implies(candidate, &consensus) {
                return true;
            }
        }
    }

    false
}

fn bond_disjunction_consensus(left: &BondExprTree, right: &BondExprTree) -> Option<BondExprTree> {
    let (left_items, left_kind) = bond_consensus_items(left);
    let (right_items, right_kind) = bond_consensus_items(right);
    let kind = if left_kind == BondAndKind::Low || right_kind == BondAndKind::Low {
        BondAndKind::Low
    } else {
        BondAndKind::High
    };

    for left_complement_index in 0..left_items.len() {
        for right_complement_index in 0..right_items.len() {
            if !bond_trees_are_complements(
                &left_items[left_complement_index],
                &right_items[right_complement_index],
            ) {
                continue;
            }

            let mut left_remaining = bond_items_without_index(&left_items, left_complement_index);
            let mut right_remaining =
                bond_items_without_index(&right_items, right_complement_index);
            sort_and_dedup_bond_items(&mut left_remaining);
            sort_and_dedup_bond_items(&mut right_remaining);
            if left_remaining == right_remaining {
                return Some(simplify_bond_and(left_remaining, kind));
            }
        }
    }

    None
}

fn factor_common_bond_disjunction_terms(items: &[BondExprTree]) -> Option<BondExprTree> {
    if items.len() < 2 {
        return None;
    }
    if items.iter().any(bond_tree_contains_negated_direction) {
        return None;
    }

    let alternatives = items
        .iter()
        .map(|item| {
            let (mut items, kind) = bond_consensus_items(item);
            sort_and_dedup_bond_items(&mut items);
            (items, kind)
        })
        .collect::<Vec<_>>();
    let common = alternatives
        .first()?
        .0
        .iter()
        .try_fold(Vec::new(), |mut common, candidate| {
            if alternatives
                .iter()
                .all(|(alternative, _)| alternative.contains(candidate))
            {
                common.push(candidate.clone());
            }
            Some(common)
        })?;
    if common.is_empty() {
        return None;
    }

    let mut reduced = Vec::new();
    for (alternative, kind) in alternatives {
        let remainder = alternative
            .into_iter()
            .filter(|item| !common.contains(item))
            .collect::<Vec<_>>();
        if remainder.is_empty() {
            return Some(simplify_bond_and(common, kind));
        }
        reduced.push(simplify_bond_and(remainder, kind));
    }

    let disjunction = simplify_bond_or(reduced);
    if matches!(disjunction, BondExprTree::Primitive(BondPrimitive::Any)) {
        return Some(simplify_bond_and(common, BondAndKind::High));
    }

    let kind = if matches!(disjunction, BondExprTree::Or(_) | BondExprTree::LowAnd(_)) {
        BondAndKind::Low
    } else {
        BondAndKind::High
    };
    let mut factored = common;
    factored.push(disjunction);
    Some(simplify_bond_and(factored, kind))
}

fn bond_consensus_items(tree: &BondExprTree) -> (Vec<BondExprTree>, BondAndKind) {
    match tree {
        BondExprTree::HighAnd(items) => (items.clone(), BondAndKind::High),
        BondExprTree::LowAnd(items) => (items.clone(), BondAndKind::Low),
        other => (vec![other.clone()], BondAndKind::High),
    }
}

fn bond_items_without_index(items: &[BondExprTree], excluded_index: usize) -> Vec<BondExprTree> {
    items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (index != excluded_index).then_some(item.clone()))
        .collect()
}

fn sort_and_dedup_bond_items(items: &mut Vec<BondExprTree>) {
    items.sort();
    items.dedup();
}

fn bond_trees_are_complements(left: &BondExprTree, right: &BondExprTree) -> bool {
    match (left, right) {
        (BondExprTree::Not(left), right) => left.as_ref() == right,
        (left, BondExprTree::Not(right)) => right.as_ref() == left,
        _ => false,
    }
}

fn bond_or_item_absorbs(base: &BondExprTree, candidate: &BondExprTree) -> bool {
    if bond_tree_implies(candidate, base) {
        return true;
    }
    match candidate {
        BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => items
            .iter()
            .any(|item| item == base || bond_tree_implies(item, base)),
        _ => false,
    }
}

fn bond_conjunction_items_with_kind(tree: &BondExprTree) -> Option<(&[BondExprTree], BondAndKind)> {
    match tree {
        BondExprTree::HighAnd(items) => Some((items, BondAndKind::High)),
        BondExprTree::LowAnd(items) => Some((items, BondAndKind::Low)),
        _ => None,
    }
}

fn bond_disjunction_items(tree: &BondExprTree) -> Option<&[BondExprTree]> {
    let BondExprTree::Or(items) = tree else {
        return None;
    };
    Some(items)
}

fn bond_tree_implies(specific: &BondExprTree, general: &BondExprTree) -> bool {
    match (specific, general) {
        (specific, general) if specific == general => true,
        (BondExprTree::Not(specific), BondExprTree::Not(general)) => {
            bond_tree_implies(general, specific)
        }
        (BondExprTree::Or(items), _) => items.iter().all(|item| bond_tree_implies(item, general)),
        (_, BondExprTree::Or(items)) => items.iter().any(|item| bond_tree_implies(specific, item)),
        (_, BondExprTree::Not(excluded)) => bond_trees_are_mutually_exclusive(specific, excluded),
        (
            BondExprTree::HighAnd(_) | BondExprTree::LowAnd(_),
            BondExprTree::HighAnd(general_items) | BondExprTree::LowAnd(general_items),
        ) => general_items
            .iter()
            .all(|general_item| bond_tree_implies(specific, general_item)),
        (BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items), _) => {
            items.iter().any(|item| bond_tree_implies(item, general))
        }
        (_, BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items)) => {
            items.iter().all(|item| bond_tree_implies(specific, item))
        }
        (BondExprTree::Primitive(specific), BondExprTree::Primitive(general)) => {
            bond_primitive_implies(*specific, *general)
        }
        _ => false,
    }
}

fn bond_primitive_implies(specific: BondPrimitive, general: BondPrimitive) -> bool {
    specific == general || matches!(general, BondPrimitive::Any)
}

fn bond_trees_are_mutually_exclusive(left: &BondExprTree, right: &BondExprTree) -> bool {
    if bond_tree_is_conjunctive_complement(left, right)
        || bond_tree_is_conjunctive_complement(right, left)
    {
        return true;
    }

    match (left, right) {
        (BondExprTree::Or(items), other) | (other, BondExprTree::Or(items)) => {
            return items
                .iter()
                .all(|item| bond_trees_are_mutually_exclusive(item, other));
        }
        _ => {}
    }

    let left_items = bond_conjunction_items(left);
    let right_items = bond_conjunction_items(right);
    left_items.iter().any(|left_item| {
        right_items
            .iter()
            .any(|right_item| bond_tree_pair_is_mutually_exclusive(left_item, right_item))
    })
}

fn bond_conjunction_items(tree: &BondExprTree) -> &[BondExprTree] {
    match tree {
        BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => items,
        other => core::slice::from_ref(other),
    }
}

fn bond_tree_pair_is_mutually_exclusive(left: &BondExprTree, right: &BondExprTree) -> bool {
    match (left, right) {
        (BondExprTree::Not(_), BondExprTree::Not(_)) => false,
        (BondExprTree::Not(excluded), candidate) | (candidate, BondExprTree::Not(excluded)) => {
            bond_tree_implies(candidate, excluded)
        }
        (BondExprTree::Or(items), other) | (other, BondExprTree::Or(items)) => items
            .iter()
            .all(|item| bond_trees_are_mutually_exclusive(item, other)),
        (BondExprTree::Primitive(left), BondExprTree::Primitive(right)) => {
            bond_primitives_are_mutually_exclusive(*left, *right)
        }
        _ => false,
    }
}

const fn bond_primitives_are_mutually_exclusive(left: BondPrimitive, right: BondPrimitive) -> bool {
    match (left, right) {
        (BondPrimitive::Any | BondPrimitive::Ring, _)
        | (_, BondPrimitive::Any | BondPrimitive::Ring) => false,
        (BondPrimitive::Bond(left), BondPrimitive::Bond(right)) => {
            bond_kind_class(left) != bond_kind_class(right)
        }
    }
}

const fn bond_kind_class(bond: Bond) -> u8 {
    match bond {
        Bond::Single | Bond::Up | Bond::Down => 0,
        Bond::Double => 1,
        Bond::Triple => 2,
        Bond::Aromatic => 3,
        Bond::Quadruple => 4,
    }
}

fn can_render_bond_and_as_high_and(items: &[BondExprTree]) -> bool {
    items
        .iter()
        .all(|item| !matches!(item, BondExprTree::Or(_) | BondExprTree::LowAnd(_)))
}

fn flip_directional_bond_expr(expr: &BondExpr) -> BondExpr {
    match expr {
        BondExpr::Elided => BondExpr::Elided,
        BondExpr::Query(tree) => BondExpr::Query(flip_directional_bond_tree(tree.clone())),
    }
}

fn flip_directional_bond_tree(tree: BondExprTree) -> BondExprTree {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(Bond::Up)) => {
            BondExprTree::Primitive(BondPrimitive::Bond(Bond::Down))
        }
        BondExprTree::Primitive(BondPrimitive::Bond(Bond::Down)) => {
            BondExprTree::Primitive(BondPrimitive::Bond(Bond::Up))
        }
        BondExprTree::Primitive(primitive) => BondExprTree::Primitive(primitive),
        BondExprTree::Not(inner) => BondExprTree::Not(Box::new(flip_directional_bond_tree(*inner))),
        BondExprTree::HighAnd(items) => {
            BondExprTree::HighAnd(items.into_iter().map(flip_directional_bond_tree).collect())
        }
        BondExprTree::Or(items) => {
            BondExprTree::Or(items.into_iter().map(flip_directional_bond_tree).collect())
        }
        BondExprTree::LowAnd(items) => {
            BondExprTree::LowAnd(items.into_iter().map(flip_directional_bond_tree).collect())
        }
    }
}

fn invert_atom_expr_chirality(expr: AtomExpr) -> AtomExpr {
    match expr {
        AtomExpr::Bracket(bracket) => AtomExpr::Bracket(BracketExpr {
            tree: invert_bracket_tree_chirality(bracket.tree),
            atom_map: bracket.atom_map,
        }),
        other => other,
    }
}

fn invert_bracket_tree_chirality(tree: BracketExprTree) -> BracketExprTree {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Chirality(chirality)) => {
            BracketExprTree::Primitive(AtomPrimitive::Chirality(invert_chirality(chirality)))
        }
        BracketExprTree::Primitive(primitive) => BracketExprTree::Primitive(primitive),
        BracketExprTree::Not(inner) => {
            BracketExprTree::Not(Box::new(invert_bracket_tree_chirality(*inner)))
        }
        BracketExprTree::HighAnd(items) => BracketExprTree::HighAnd(
            items
                .into_iter()
                .map(invert_bracket_tree_chirality)
                .collect(),
        ),
        BracketExprTree::Or(items) => BracketExprTree::Or(
            items
                .into_iter()
                .map(invert_bracket_tree_chirality)
                .collect(),
        ),
        BracketExprTree::LowAnd(items) => BracketExprTree::LowAnd(
            items
                .into_iter()
                .map(invert_bracket_tree_chirality)
                .collect(),
        ),
    }
}

const fn invert_chirality(chirality: Chirality) -> Chirality {
    match chirality {
        Chirality::At => Chirality::AtAt,
        Chirality::AtAt => Chirality::At,
        Chirality::TH(1) => Chirality::TH(2),
        Chirality::TH(2) => Chirality::TH(1),
        Chirality::AL(1) => Chirality::AL(2),
        Chirality::AL(2) => Chirality::AL(1),
        other => other,
    }
}

fn emitted_neighbor_permutation_is_odd(from_neighbors: &[AtomId], to_neighbors: &[AtomId]) -> bool {
    if from_neighbors.len() != to_neighbors.len() {
        return false;
    }
    if has_duplicate_neighbors(from_neighbors) || has_duplicate_neighbors(to_neighbors) {
        return false;
    }

    let permutation = permutation_from_neighbors(from_neighbors, to_neighbors);
    permutation.is_some_and(|permutation| permutation_is_odd(&permutation))
}

fn has_duplicate_neighbors(neighbors: &[AtomId]) -> bool {
    for (index, &neighbor) in neighbors.iter().enumerate() {
        if neighbors[index + 1..].contains(&neighbor) {
            return true;
        }
    }
    false
}

fn permutation_from_neighbors(
    from_neighbors: &[AtomId],
    to_neighbors: &[AtomId],
) -> Option<Vec<usize>> {
    let mut used = vec![false; from_neighbors.len()];
    let mut permutation = Vec::with_capacity(to_neighbors.len());

    for &to_neighbor in to_neighbors {
        let index = from_neighbors
            .iter()
            .enumerate()
            .find_map(|(index, &from_neighbor)| {
                (!used[index] && from_neighbor == to_neighbor).then_some(index)
            })?;
        used[index] = true;
        permutation.push(index);
    }

    Some(permutation)
}

fn permutation_is_odd(permutation: &[usize]) -> bool {
    let mut visited = vec![false; permutation.len()];
    let mut transpositions = 0usize;

    for start in 0..permutation.len() {
        if visited[start] {
            continue;
        }
        let mut cursor = start;
        let mut cycle_len = 0usize;
        while !visited[cursor] {
            visited[cursor] = true;
            cursor = permutation[cursor];
            cycle_len += 1;
        }
        if cycle_len > 0 {
            transpositions += cycle_len - 1;
        }
    }

    transpositions % 2 == 1
}

fn emitted_stereo_neighbors(query: &QueryMol) -> Vec<Vec<AtomId>> {
    let parent_bond_by_atom = spanning_forest_parent_bonds(query);
    let mut children_by_atom = vec![Vec::new(); query.atom_count()];
    for (atom_id, parent_bond) in parent_bond_by_atom.iter().copied().enumerate() {
        let Some(parent_bond) = parent_bond else {
            continue;
        };
        let bond = &query.bonds()[parent_bond];
        let parent_atom = if bond.src == atom_id {
            bond.dst
        } else {
            bond.src
        };
        children_by_atom[parent_atom].push(atom_id);
    }
    for children in &mut children_by_atom {
        children.sort_unstable();
    }

    let ring_neighbors_by_atom = ring_neighbors_by_atom(query, &parent_bond_by_atom);
    (0..query.atom_count())
        .map(|atom_id| {
            let mut neighbors = Vec::new();
            if let Some(parent_bond) = parent_bond_by_atom[atom_id] {
                let bond = &query.bonds()[parent_bond];
                neighbors.push(if bond.src == atom_id {
                    bond.dst
                } else {
                    bond.src
                });
            }
            neighbors.extend_from_slice(&ring_neighbors_by_atom[atom_id]);
            neighbors.extend_from_slice(&children_by_atom[atom_id]);
            neighbors
        })
        .collect()
}

fn normalize_directional_double_bond_pairs(query: &QueryMol) -> QueryMol {
    let mut bonds = query.bonds().to_vec();
    let mut incident_bonds_by_atom = vec![Vec::new(); query.atom_count()];
    let parent_bond_by_atom = spanning_forest_parent_bonds(query);
    let preorder_indices = query_preorder_indices(query, &parent_bond_by_atom);
    let mut paired_directional_bonds = vec![false; bonds.len()];
    for bond in &bonds {
        incident_bonds_by_atom[bond.src].push(bond.id);
        if bond.dst != bond.src {
            incident_bonds_by_atom[bond.dst].push(bond.id);
        }
    }

    for bond_id in 0..bonds.len() {
        let bond = &bonds[bond_id];
        if !matches!(
            bond.expr,
            BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)))
        ) {
            continue;
        }

        let left = neighboring_directional_bonds(
            &bonds,
            &incident_bonds_by_atom[bond.src],
            bond.id,
            bond.src,
        );
        let right = neighboring_directional_bonds(
            &bonds,
            &incident_bonds_by_atom[bond.dst],
            bond.id,
            bond.dst,
        );
        let (left_bond_id, left_dir) = match left.as_slice() {
            [single] => *single,
            _ => continue,
        };
        let (right_bond_id, right_dir) = match right.as_slice() {
            [single] => *single,
            _ => continue,
        };

        let same_parity = left_dir == right_dir;
        let left_key = canonical_directional_edge_key(&bonds[left_bond_id], &preorder_indices);
        let right_key = canonical_directional_edge_key(&bonds[right_bond_id], &preorder_indices);
        paired_directional_bonds[left_bond_id] = true;
        paired_directional_bonds[right_bond_id] = true;
        if left_key <= right_key {
            let right_canonical_dir = if same_parity { Bond::Up } else { Bond::Down };
            set_simple_bond_direction(&mut bonds[left_bond_id].expr, Bond::Up);
            set_simple_bond_direction(&mut bonds[right_bond_id].expr, right_canonical_dir);
        } else {
            let left_canonical_dir = if same_parity { Bond::Up } else { Bond::Down };
            set_simple_bond_direction(&mut bonds[right_bond_id].expr, Bond::Up);
            set_simple_bond_direction(&mut bonds[left_bond_id].expr, left_canonical_dir);
        }
    }

    for (bond_id, bond) in bonds.iter_mut().enumerate() {
        if paired_directional_bonds[bond_id] {
            continue;
        }
        if matches!(
            simple_bond_direction(&bond.expr),
            Some(Bond::Up | Bond::Down)
        ) {
            set_simple_bond_direction(&mut bond.expr, Bond::Up);
        }
    }

    for bond in &mut bonds {
        bond.expr = canonical_bond_expr(&bond.expr);
    }

    bonds.sort_unstable_by(|left, right| {
        left.src
            .cmp(&right.src)
            .then(left.dst.cmp(&right.dst))
            .then(left.expr.to_string().cmp(&right.expr.to_string()))
    });
    for (bond_id, bond) in bonds.iter_mut().enumerate() {
        bond.id = bond_id;
    }

    QueryMol::from_parts(
        query.atoms().to_vec(),
        bonds,
        query.component_count(),
        query.component_groups().to_vec(),
    )
}

fn canonical_directional_edge_key(bond: &QueryBond, preorder_indices: &[usize]) -> (usize, usize) {
    let left = preorder_indices[bond.src];
    let right = preorder_indices[bond.dst];
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn neighboring_directional_bonds(
    bonds: &[QueryBond],
    incident_bonds: &[usize],
    excluded_bond_id: usize,
    center_atom: AtomId,
) -> Vec<(usize, Bond)> {
    incident_bonds
        .iter()
        .copied()
        .filter(|&bond_id| bond_id != excluded_bond_id)
        .filter_map(|bond_id| {
            let bond = &bonds[bond_id];
            ((bond.src == center_atom) || (bond.dst == center_atom))
                .then(|| {
                    simple_bond_direction_relative_to_center(&bond.expr, bond.src == center_atom)
                        .map(|direction| (bond_id, direction))
                })
                .flatten()
        })
        .collect()
}

fn simple_bond_direction(expr: &BondExpr) -> Option<Bond> {
    match expr {
        BondExpr::Query(tree) => simple_bond_direction_in_tree(tree),
        BondExpr::Elided => None,
    }
}

fn simple_bond_direction_in_tree(tree: &BondExprTree) -> Option<Bond> {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(bond @ (Bond::Up | Bond::Down))) => Some(*bond),
        BondExprTree::Primitive(_) | BondExprTree::Not(_) | BondExprTree::Or(_) => None,
        BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => {
            let mut direction = None;
            for item in items {
                let item_direction = simple_bond_direction_in_tree(item);
                match (direction, item_direction) {
                    (None, Some(found)) => direction = Some(found),
                    (Some(current), Some(found)) if current == found => {}
                    (Some(_), Some(_)) => return None,
                    (None | Some(_), None) => {}
                }
            }
            direction
        }
    }
}

fn simple_bond_direction_relative_to_center(expr: &BondExpr, center_is_src: bool) -> Option<Bond> {
    let direction = simple_bond_direction(expr)?;
    Some(if center_is_src {
        invert_simple_direction(direction)
    } else {
        direction
    })
}

const fn invert_simple_direction(direction: Bond) -> Bond {
    match direction {
        Bond::Up => Bond::Down,
        Bond::Down => Bond::Up,
        other => other,
    }
}

fn set_simple_bond_direction(expr: &mut BondExpr, direction: Bond) {
    match expr {
        BondExpr::Elided => {
            *expr = BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(direction)));
        }
        BondExpr::Query(tree) => {
            if !set_simple_bond_direction_in_tree(tree, direction) {
                *expr = BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(direction)));
            }
        }
    }
}

fn set_simple_bond_direction_in_tree(tree: &mut BondExprTree, direction: Bond) -> bool {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(bond @ (Bond::Up | Bond::Down))) => {
            *bond = direction;
            true
        }
        BondExprTree::Primitive(_) | BondExprTree::Not(_) | BondExprTree::Or(_) => false,
        BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => {
            let mut changed = false;
            for item in items {
                changed |= set_simple_bond_direction_in_tree(item, direction);
            }
            changed
        }
    }
}

fn query_preorder_indices(query: &QueryMol, parent_bond_by_atom: &[Option<usize>]) -> Vec<usize> {
    let mut children_by_atom = vec![Vec::new(); query.atom_count()];
    for (atom_id, parent_bond) in parent_bond_by_atom.iter().copied().enumerate() {
        let Some(parent_bond) = parent_bond else {
            continue;
        };
        let bond = &query.bonds()[parent_bond];
        let parent_atom = if bond.src == atom_id {
            bond.dst
        } else {
            bond.src
        };
        children_by_atom[parent_atom].push(atom_id);
    }
    for children in &mut children_by_atom {
        children.sort_unstable();
    }

    let mut preorder = vec![usize::MAX; query.atom_count()];
    let mut next = 0usize;
    for atom_id in 0..query.atom_count() {
        if parent_bond_by_atom[atom_id].is_some() || preorder[atom_id] != usize::MAX {
            continue;
        }
        assign_query_preorder(atom_id, &children_by_atom, &mut preorder, &mut next);
    }
    preorder
}

fn assign_query_preorder(
    atom_id: AtomId,
    children_by_atom: &[Vec<AtomId>],
    preorder: &mut [usize],
    next: &mut usize,
) {
    if preorder[atom_id] != usize::MAX {
        return;
    }
    preorder[atom_id] = *next;
    *next += 1;
    for &child in &children_by_atom[atom_id] {
        assign_query_preorder(child, children_by_atom, preorder, next);
    }
}

fn spanning_forest_parent_bonds(query: &QueryMol) -> Vec<Option<usize>> {
    let mut parent_bond_by_atom = vec![None; query.atom_count()];
    let mut bond_ids_by_atom = vec![Vec::new(); query.atom_count()];
    for bond in query.bonds() {
        if bond.src < bond_ids_by_atom.len() {
            bond_ids_by_atom[bond.src].push(bond.id);
        }
        if bond.dst < bond_ids_by_atom.len() && bond.dst != bond.src {
            bond_ids_by_atom[bond.dst].push(bond.id);
        }
    }
    for (atom_id, bond_ids) in bond_ids_by_atom.iter_mut().enumerate() {
        bond_ids.sort_by_key(|&bond_id| {
            let bond = &query.bonds()[bond_id];
            (
                if bond.src == atom_id {
                    bond.dst
                } else {
                    bond.src
                },
                canonicalization_bond_expr_rank(&bond.expr),
                bond.id,
            )
        });
    }

    let mut visited = vec![false; query.atom_count()];
    for atom in query.atoms() {
        if visited[atom.id] {
            continue;
        }
        visited[atom.id] = true;
        let mut stack = vec![atom.id];
        while let Some(current_atom) = stack.pop() {
            for &bond_id in bond_ids_by_atom[current_atom].iter().rev() {
                let bond = &query.bonds()[bond_id];
                let next_atom = if bond.src == current_atom {
                    bond.dst
                } else {
                    bond.src
                };
                if next_atom >= visited.len() || visited[next_atom] {
                    continue;
                }
                visited[next_atom] = true;
                parent_bond_by_atom[next_atom] = Some(bond_id);
                stack.push(next_atom);
            }
        }
    }

    parent_bond_by_atom
}

fn ring_neighbors_by_atom(
    query: &QueryMol,
    parent_bond_by_atom: &[Option<usize>],
) -> Vec<Vec<AtomId>> {
    let mut is_parent_bond = vec![false; query.bond_count()];
    for parent_bond in parent_bond_by_atom.iter().flatten().copied() {
        is_parent_bond[parent_bond] = true;
    }

    let mut ring_neighbors = vec![Vec::new(); query.atom_count()];
    let mut ring_bond_ids = query
        .bonds()
        .iter()
        .filter(|bond| !is_parent_bond[bond.id])
        .map(|bond| bond.id)
        .collect::<Vec<_>>();
    ring_bond_ids.sort_by_key(|&bond_id| {
        let bond = &query.bonds()[bond_id];
        (
            bond.src.min(bond.dst),
            bond.src.max(bond.dst),
            canonicalization_bond_expr_rank(&bond.expr),
            bond.id,
        )
    });

    for bond_id in ring_bond_ids {
        let bond = &query.bonds()[bond_id];
        ring_neighbors[bond.src].push(bond.dst);
        ring_neighbors[bond.dst].push(bond.src);
    }

    ring_neighbors
}

fn build_bond_multiset_labels(query: &QueryMol) -> Vec<((AtomId, AtomId), String)> {
    let mut labels = query
        .bonds()
        .iter()
        .map(|bond| {
            (
                (bond.src.min(bond.dst), bond.src.max(bond.dst)),
                undirected_bond_expr_key(&bond.expr),
            )
        })
        .collect::<Vec<_>>();
    labels.sort_unstable();

    let mut grouped = Vec::<((AtomId, AtomId), String)>::with_capacity(labels.len());
    for (edge, label) in labels {
        if let Some((last_edge, last_label)) = grouped.last_mut() {
            if *last_edge == edge {
                last_label.push('\u{1f}');
                last_label.push_str(&label);
                continue;
            }
        }
        grouped.push((edge, label));
    }
    grouped
}

fn bond_multiset_label_for_edge(
    labels: &[((AtomId, AtomId), String)],
    node_a: AtomId,
    node_b: AtomId,
) -> Option<&((AtomId, AtomId), String)> {
    let edge = (node_a.min(node_b), node_a.max(node_b));
    labels
        .binary_search_by_key(&edge, |(candidate, _)| *candidate)
        .ok()
        .map(|index| &labels[index])
}

fn canonicalization_bond_expr_rank(expr: &BondExpr) -> (u8, String) {
    let rank = match expr {
        BondExpr::Elided => 0,
        BondExpr::Query(BondExprTree::Primitive(_)) => 1,
        BondExpr::Query(BondExprTree::Not(_)) => 2,
        BondExpr::Query(BondExprTree::HighAnd(_)) => 3,
        BondExpr::Query(BondExprTree::Or(_)) => 4,
        BondExpr::Query(BondExprTree::LowAnd(_)) => 5,
    };
    (rank, undirected_bond_expr_key(expr))
}

fn undirected_bond_expr_key(expr: &BondExpr) -> String {
    if !bond_expr_contains_direction(expr) {
        return expr.to_string();
    }
    let direct = expr.to_string();
    let flipped = flip_directional_bond_expr(expr).to_string();
    if direct <= flipped {
        direct
    } else {
        flipped
    }
}

fn bond_expr_contains_direction(expr: &BondExpr) -> bool {
    match expr {
        BondExpr::Elided => false,
        BondExpr::Query(tree) => {
            bond_tree_contains_direction(tree, Bond::Up)
                || bond_tree_contains_direction(tree, Bond::Down)
        }
    }
}

#[cfg(test)]
mod tests;
