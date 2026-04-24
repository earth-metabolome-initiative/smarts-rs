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
        let (first_query, first_labeling) = self.canonicalize_once_with_labeling();
        let (second_query, second_step_labeling) = first_query.canonicalize_once_with_labeling();
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
                states[current_index].0.canonicalize_once_with_labeling();
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

    fn canonicalize_once_with_labeling(&self) -> (Self, QueryCanonicalLabeling) {
        let normalized = self.canonicalization_normal_form();
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

    fn canonicalization_normal_form(&self) -> Self {
        let atoms = self
            .atoms()
            .iter()
            .map(|atom| QueryAtom {
                id: atom.id,
                component: atom.component,
                expr: canonical_atom_expr(&atom.expr),
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
            |node_id| atom_labels[node_id].clone(),
            |node_a, node_b| {
                bond_multiset_label_for_edge(&bond_multiset_labels, node_a, node_b).map_or_else(
                    || unreachable!("canonizer only queries edges that exist"),
                    |(_, label)| label.clone(),
                )
            },
        );
        QueryCanonicalLabeling::new(result.order)
    }

    fn exact_canonicalize_with_labeling(&self, labeling: &QueryCanonicalLabeling) -> Self {
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
    let mut component_id = 0usize;
    while component_id < query.component_count() {
        if let Some(group_id) = query.component_group(component_id) {
            let mut component_ids = vec![component_id];
            component_id += 1;
            while component_id < query.component_count()
                && query.component_group(component_id) == Some(group_id)
            {
                component_ids.push(component_id);
                component_id += 1;
            }
            entries.push(TopLevelComponentEntry {
                grouped: true,
                component_ids,
            });
        } else {
            entries.push(TopLevelComponentEntry {
                grouped: false,
                component_ids: vec![component_id],
            });
            component_id += 1;
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
    CanonicalQueryStateKey {
        atom_labels,
        bond_edges,
    }
}

fn canonical_whole_query_state_key(query: &QueryMol) -> CanonicalWholeQueryStateKey {
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
    CanonicalWholeQueryStateKey {
        component_groups: query.component_groups().to_vec(),
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

fn canonical_atom_expr(expr: &AtomExpr) -> AtomExpr {
    match expr {
        AtomExpr::Wildcard => AtomExpr::Wildcard,
        AtomExpr::Bare { element, aromatic } => AtomExpr::Bare {
            element: *element,
            aromatic: *aromatic,
        },
        AtomExpr::Bracket(expr) => {
            let canonical = canonical_bracket_expr(expr);
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

fn canonical_bracket_expr(expr: &BracketExpr) -> BracketExpr {
    let mut normalized = expr.clone();
    normalized
        .normalize()
        .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
    let mut tree = canonical_bracket_tree(normalized.tree);
    tree = simplify_bracket_tree(tree);
    let mut bracket = BracketExpr {
        tree,
        atom_map: normalized.atom_map,
    };
    bracket
        .normalize()
        .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
    let mut tree = canonical_bracket_tree(bracket.tree);
    tree = simplify_bracket_tree(tree);
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

fn canonical_bracket_tree(tree: BracketExprTree) -> BracketExprTree {
    match tree {
        BracketExprTree::Primitive(primitive) => {
            BracketExprTree::Primitive(canonical_atom_primitive(primitive))
        }
        BracketExprTree::Not(inner) => {
            BracketExprTree::Not(Box::new(canonical_bracket_tree(*inner)))
        }
        BracketExprTree::HighAnd(items) => {
            let mut items = items
                .into_iter()
                .map(canonical_bracket_tree)
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BracketExprTree::to_string);
            items.dedup();
            BracketExprTree::HighAnd(items)
        }
        BracketExprTree::Or(items) => {
            let mut items = items
                .into_iter()
                .map(canonical_bracket_tree)
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BracketExprTree::to_string);
            items.dedup();
            BracketExprTree::Or(items)
        }
        BracketExprTree::LowAnd(items) => {
            let mut items = items
                .into_iter()
                .map(canonical_bracket_tree)
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BracketExprTree::to_string);
            items.dedup();
            BracketExprTree::LowAnd(items)
        }
    }
}

fn canonical_atom_primitive(primitive: AtomPrimitive) -> AtomPrimitive {
    match primitive {
        AtomPrimitive::Symbol {
            element: elements_rs::Element::H,
            aromatic: false,
        } => AtomPrimitive::AtomicNumber(1),
        AtomPrimitive::Degree(query) => AtomPrimitive::Degree(canonical_count_query(query)),
        AtomPrimitive::Connectivity(query) => {
            AtomPrimitive::Connectivity(canonical_count_query(query))
        }
        AtomPrimitive::Valence(query) => AtomPrimitive::Valence(canonical_count_query(query)),
        AtomPrimitive::Hydrogen(kind, query) => {
            AtomPrimitive::Hydrogen(kind, canonical_count_query(query))
        }
        AtomPrimitive::RingMembership(query) => {
            AtomPrimitive::RingMembership(query.map(canonical_numeric_query))
        }
        AtomPrimitive::RingSize(query) => {
            AtomPrimitive::RingSize(query.map(canonical_numeric_query))
        }
        AtomPrimitive::RingConnectivity(query) => {
            AtomPrimitive::RingConnectivity(query.map(canonical_numeric_query))
        }
        AtomPrimitive::Hybridization(query) => {
            AtomPrimitive::Hybridization(canonical_numeric_query(query))
        }
        AtomPrimitive::HeteroNeighbor(query) => {
            AtomPrimitive::HeteroNeighbor(canonical_count_query(query))
        }
        AtomPrimitive::AliphaticHeteroNeighbor(query) => {
            AtomPrimitive::AliphaticHeteroNeighbor(canonical_count_query(query))
        }
        AtomPrimitive::RecursiveQuery(query) => {
            AtomPrimitive::RecursiveQuery(Box::new(query.canonicalize()))
        }
        other => other,
    }
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
        BracketExprTree::Primitive(primitive) => BracketExprTree::Primitive(primitive),
        BracketExprTree::Not(inner) => simplify_bracket_not(simplify_bracket_tree(*inner)),
        BracketExprTree::HighAnd(items) => simplify_bracket_and(items, BracketAndKind::High),
        BracketExprTree::LowAnd(items) => simplify_bracket_and(items, BracketAndKind::Low),
        BracketExprTree::Or(items) => simplify_bracket_or(items),
    }
}

fn simplify_bracket_not(inner: BracketExprTree) -> BracketExprTree {
    if is_false_bracket_tree(&inner) {
        BracketExprTree::Primitive(AtomPrimitive::Wildcard)
    } else {
        BracketExprTree::Not(Box::new(inner))
    }
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
    remove_implied_bracket_conjunction_terms(&mut items);
    items.sort_by_cached_key(BracketExprTree::to_string);
    items.dedup();

    match items.as_slice() {
        [] => BracketExprTree::Primitive(AtomPrimitive::Wildcard),
        [single] => single.clone(),
        _ if kind == BracketAndKind::Low && !can_render_bracket_and_as_high_and(&items) => {
            BracketExprTree::LowAnd(items)
        }
        _ => BracketExprTree::HighAnd(items),
    }
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
        (_, BracketExprTree::Not(excluded)) => {
            bracket_trees_are_mutually_exclusive(specific, excluded)
        }
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
    if items.len() > 1 {
        items.retain(|item| !is_false_bracket_tree(item));
    }
    synthesize_atomic_number_disjunctions(&mut items);
    synthesize_isotope_disjunctions(&mut items);
    merge_numeric_bracket_disjunctions(&mut items);
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
    remove_absorbed_bracket_disjunction_terms(&mut items);
    items.sort_by_cached_key(BracketExprTree::to_string);
    items.dedup();
    match items.as_slice() {
        [] => false_bracket_tree(),
        [single] => single.clone(),
        _ => BracketExprTree::Or(items),
    }
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
            _ => false,
        })
    })
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

fn relax_complemented_bracket_disjunction_terms(items: &mut [BracketExprTree]) -> bool {
    for base_index in 0..items.len() {
        for candidate_index in 0..items.len() {
            if base_index == candidate_index {
                continue;
            }
            let Some(relaxed) = remove_negated_bracket_term_from_conjunction(
                &items[candidate_index],
                &items[base_index],
            ) else {
                continue;
            };
            items[candidate_index] = relaxed;
            return true;
        }
    }
    false
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
        BracketExprTree::Not(inner) => inner.as_ref() == base,
        _ => false,
    })?;
    let remaining = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (index != complement_index).then_some(item.clone()))
        .collect::<Vec<_>>();
    Some(simplify_bracket_and(remaining, kind))
}

fn bracket_trees_are_mutually_exclusive(left: &BracketExprTree, right: &BracketExprTree) -> bool {
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
    let mut index = 0usize;
    while index < items.len() {
        let is_absorbed = items.iter().enumerate().any(|(other_index, other)| {
            other_index != index && bracket_or_item_absorbs(other, &items[index])
        });
        if is_absorbed {
            items.remove(index);
        } else {
            index += 1;
        }
    }
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
            let mut normalized = tree.clone();
            normalize_bond_tree(&mut normalized)
                .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
            let mut canonical = canonical_bond_tree(normalized);
            canonical = simplify_bond_tree(canonical);
            normalize_bond_tree(&mut canonical)
                .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
            let has_up = bond_tree_contains_direction(&canonical, Bond::Up);
            let has_down = bond_tree_contains_direction(&canonical, Bond::Down);
            if !has_up && !has_down {
                return BondExpr::Query(canonical);
            }
            if has_up && has_down {
                canonical = collapse_mixed_directional_polarity(canonical);
                canonical = simplify_bond_tree(canonical);
                normalize_bond_tree(&mut canonical)
                    .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
            }
            let flipped = flip_directional_bond_tree(canonical.clone());
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
    if items.iter().any(is_false_bond_tree) || bond_and_contains_mutually_exclusive_pair(&items) {
        return false_bond_tree();
    }
    if items.len() > 1 {
        items.retain(|item| !matches!(item, BondExprTree::Primitive(BondPrimitive::Any)));
    }
    items.sort_by_cached_key(BondExprTree::to_string);
    items.dedup();
    match items.as_slice() {
        [] => BondExprTree::Primitive(BondPrimitive::Any),
        [single] => single.clone(),
        _ if kind == BondAndKind::Low && !can_render_bond_and_as_high_and(&items) => {
            BondExprTree::LowAnd(items)
        }
        _ => BondExprTree::HighAnd(items),
    }
}

fn simplify_bond_or(items: Vec<BondExprTree>) -> BondExprTree {
    let mut items = items
        .into_iter()
        .map(simplify_bond_tree)
        .collect::<Vec<_>>();
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
    remove_absorbed_bond_disjunction_terms(&mut items);
    items.sort_by_cached_key(BondExprTree::to_string);
    items.dedup();
    match items.as_slice() {
        [] => false_bond_tree(),
        [single] => single.clone(),
        _ => BondExprTree::Or(items),
    }
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
            _ => false,
        })
    })
}

fn remove_absorbed_bond_disjunction_terms(items: &mut Vec<BondExprTree>) {
    let mut index = 0usize;
    while index < items.len() {
        let is_absorbed = items.iter().enumerate().any(|(other_index, other)| {
            other_index != index && bond_or_item_absorbs(other, &items[index])
        });
        if is_absorbed {
            items.remove(index);
        } else {
            index += 1;
        }
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

fn bond_tree_implies(specific: &BondExprTree, general: &BondExprTree) -> bool {
    match (specific, general) {
        (specific, general) if specific == general => true,
        (BondExprTree::Not(specific), BondExprTree::Not(general)) => {
            bond_tree_implies(general, specific)
        }
        (_, BondExprTree::Not(excluded)) => bond_trees_are_mutually_exclusive(specific, excluded),
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
        BondExprTree::Primitive(_) | BondExprTree::Or(_) => None,
        BondExprTree::Not(inner) => simple_bond_direction_in_tree(inner),
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
        BondExprTree::Primitive(_) | BondExprTree::Or(_) => false,
        BondExprTree::Not(inner) => set_simple_bond_direction_in_tree(inner, direction),
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
    let mut labels_by_edge = BTreeMap::<(AtomId, AtomId), Vec<String>>::new();
    for bond in query.bonds() {
        labels_by_edge
            .entry((bond.src.min(bond.dst), bond.src.max(bond.dst)))
            .or_default()
            .push(undirected_bond_expr_key(&bond.expr));
    }

    labels_by_edge
        .into_iter()
        .map(|(edge, mut labels)| {
            labels.sort_unstable();
            (edge, labels.join("\u{1f}"))
        })
        .collect()
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
mod tests {
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };
    use core::str::FromStr;

    use smiles_parser::bond::Bond;

    use super::QueryCanonicalLabeling;
    use crate::{BondExpr, BondExprTree, BondPrimitive, QueryAtom, QueryBond, QueryMol};

    fn canonical_string(source: &str) -> String {
        QueryMol::from_str(source)
            .unwrap()
            .canonicalize()
            .to_string()
    }

    fn all_permutations(items: &[usize]) -> Vec<Vec<usize>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }

        let mut permutations = Vec::new();
        for (index, &item) in items.iter().enumerate() {
            let mut remaining = items.to_vec();
            remaining.remove(index);
            for mut tail in all_permutations(&remaining) {
                let mut permutation = Vec::with_capacity(items.len());
                permutation.push(item);
                permutation.append(&mut tail);
                permutations.push(permutation);
            }
        }
        permutations
    }

    fn permute_atoms(query: &QueryMol, atom_order: &[usize]) -> QueryMol {
        assert_eq!(atom_order.len(), query.atom_count());

        let mut new_index_of_old = vec![usize::MAX; query.atom_count()];
        for (new_id, old_id) in atom_order.iter().copied().enumerate() {
            new_index_of_old[old_id] = new_id;
        }

        let atoms = atom_order
            .iter()
            .copied()
            .enumerate()
            .map(|(new_id, old_id)| {
                let old_atom = &query.atoms()[old_id];
                QueryAtom {
                    id: new_id,
                    component: old_atom.component,
                    expr: old_atom.expr.clone(),
                }
            })
            .collect::<Vec<_>>();

        let mut bonds = query
            .bonds()
            .iter()
            .rev()
            .map(|bond| QueryBond {
                id: 0,
                src: new_index_of_old[bond.src],
                dst: new_index_of_old[bond.dst],
                expr: bond.expr.clone(),
            })
            .collect::<Vec<_>>();
        for (bond_id, bond) in bonds.iter_mut().enumerate() {
            bond.id = bond_id;
        }

        QueryMol::from_parts(
            atoms,
            bonds,
            query.component_count(),
            query.component_groups().to_vec(),
        )
    }

    fn permute_top_level_entries(query: &QueryMol, entry_order: &[usize]) -> QueryMol {
        let entries = super::build_top_level_component_entries(query);
        assert_eq!(entry_order.len(), entries.len());

        let reordered_entries = entry_order
            .iter()
            .copied()
            .map(|index| entries[index].clone())
            .collect::<Vec<_>>();
        let mut new_component_of_old = vec![usize::MAX; query.component_count()];
        let mut next_component_id = 0usize;
        let mut component_groups = Vec::with_capacity(query.component_count());
        let mut next_group_id = 0usize;

        for entry in reordered_entries {
            let group = (entry.grouped && entry.component_ids.len() > 1).then(|| {
                let group_id = next_group_id;
                next_group_id += 1;
                group_id
            });
            for old_component_id in entry.component_ids {
                new_component_of_old[old_component_id] = next_component_id;
                component_groups.push(group);
                next_component_id += 1;
            }
        }

        let atoms = query
            .atoms()
            .iter()
            .map(|atom| QueryAtom {
                id: atom.id,
                component: new_component_of_old[atom.component],
                expr: atom.expr.clone(),
            })
            .collect::<Vec<_>>();

        QueryMol::from_parts(
            atoms,
            query.bonds().to_vec(),
            query.component_count(),
            component_groups,
        )
    }

    fn assert_same_canonical_group(group: &[&str]) {
        let expected = canonical_string(group[0]);
        for source in &group[1..] {
            assert_eq!(
                expected,
                canonical_string(source),
                "group did not converge: {group:?}"
            );
        }
    }

    fn assert_all_atom_permutations_converge(source: &str) {
        let query = QueryMol::from_str(source).unwrap();
        let expected = query.canonicalize();
        let atom_ids = (0..query.atom_count()).collect::<Vec<_>>();
        for atom_order in all_permutations(&atom_ids) {
            let permuted = permute_atoms(&query, &atom_order);
            let canonicalized = permuted.canonicalize();
            assert_eq!(
                expected, canonicalized,
                "atom permutation changed canonical form for {source}: {atom_order:?}"
            );
            assert!(canonicalized.is_canonical());
        }
    }

    fn assert_all_top_level_entry_permutations_converge(source: &str) {
        let query = QueryMol::from_str(source).unwrap();
        let expected = query.canonicalize();
        let entry_ids =
            (0..super::build_top_level_component_entries(&query).len()).collect::<Vec<_>>();
        for entry_order in all_permutations(&entry_ids) {
            let permuted = permute_top_level_entries(&query, &entry_order);
            let canonicalized = permuted.canonicalize();
            assert_eq!(
                expected, canonicalized,
                "top-level entry permutation changed canonical form for {source}: {entry_order:?}"
            );
            assert!(canonicalized.is_canonical());
        }
    }

    #[test]
    fn canonical_labeling_of_empty_query_is_empty() {
        let query = QueryMol::from_str("").err();
        assert!(query.is_some());

        let empty = QueryMol::from_parts(Vec::new(), Vec::new(), 0, Vec::new());
        let labeling = empty.canonical_labeling();
        assert_eq!(labeling, QueryCanonicalLabeling::new(Vec::new()));
        assert!(empty.is_canonical());
    }

    #[test]
    fn canonicalize_parse_smarts_empty_query_roundtrips() {
        assert!(crate::parse_smarts("").is_err());
        let empty = QueryMol::from_parts(Vec::new(), Vec::new(), 0, Vec::new());
        let canonical = empty.canonicalize();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), 0);
        assert_eq!(canonical.bond_count(), 0);
        assert_eq!(canonical.component_count(), 0);
        assert_eq!(canonical.to_string(), "");
        assert_eq!(canonical.to_canonical_smarts(), "");
        assert!(crate::parse_smarts(&canonical.to_canonical_smarts()).is_err());
        assert_eq!(
            canonical,
            QueryMol::from_parts(Vec::new(), Vec::new(), 0, Vec::new())
        );
    }

    #[test]
    fn canonical_labeling_inverse_matches_order() {
        let query = QueryMol::from_str("OC").unwrap();
        let labeling = query.canonical_labeling();

        assert_eq!(labeling.order().len(), 2);
        assert_eq!(labeling.new_index_of_old_atom()[labeling.order()[0]], 0);
        assert_eq!(labeling.new_index_of_old_atom()[labeling.order()[1]], 1);
    }

    #[test]
    fn canonicalize_is_idempotent() {
        let query = QueryMol::from_str("N(C)C").unwrap();
        let once = query.canonicalize();
        let twice = once.canonicalize();

        assert_eq!(once, twice);
        assert!(once.is_canonical());
    }

    #[test]
    fn canonicalize_converges_component_permutations() {
        assert_eq!(canonical_string("C.N"), canonical_string("N.C"));
        assert_eq!(canonical_string("(N.C).O"), canonical_string("O.(C.N)"));
        assert_all_top_level_entry_permutations_converge("C.N.O");
        assert_all_top_level_entry_permutations_converge("(C.N).O");
        assert_all_top_level_entry_permutations_converge("(C.N).(O.S)");
    }

    #[test]
    fn canonicalize_converges_linear_spellings() {
        assert_eq!(canonical_string("CO"), canonical_string("OC"));
        assert_eq!(canonical_string("OCN"), canonical_string("NCO"));
        assert_eq!(canonical_string("C(N)O"), canonical_string("C(O)N"));
        assert_all_atom_permutations_converge("CO");
        assert_all_atom_permutations_converge("CCO");
        assert_all_atom_permutations_converge("C(N)O");
    }

    #[test]
    fn canonicalize_sorts_bracket_boolean_children() {
        assert_eq!(canonical_string("[H1;C]"), canonical_string("[C;H1]"));
        assert_eq!(canonical_string("[O,C,N]"), canonical_string("[N,O,C]"));
        assert_eq!(canonical_string("[C&X4&H2]"), canonical_string("[H2&C&X4]"));
        assert_same_canonical_group(&["[C;H1;X4]", "[X4;C;H1]", "[H1;X4;C]", "[C;X4;H1]"]);
        assert_same_canonical_group(&["[C,N,O]", "[O,N,C]", "[N,O,C]", "[C,O,N]"]);
    }

    #[test]
    fn canonicalize_deduplicates_equivalent_bracket_boolean_children() {
        assert_eq!(canonical_string("[#6&#6]"), "[#6]");
        assert_eq!(canonical_string("[#6;#6]"), "[#6]");
        assert_eq!(canonical_string("[#6,#6]"), "[#6]");
        assert_eq!(canonical_string("[!#6&!#6]"), "[!#6]");
        assert_eq!(canonical_string("[#6;#6;#6]"), "[#6]");

        for term in [
            "#7", "C", "c", "N", "n", "R", "r6", "+0", "H0", "D3", "X4", "v4", "x2", "A", "a",
            "$([#6])",
        ] {
            let expected = format!("[{term}]");
            assert_eq!(canonical_string(&format!("[{term}&{term}]")), expected);
            assert_eq!(canonical_string(&format!("[{term};{term}]")), expected);
            assert_eq!(canonical_string(&format!("[{term},{term}]")), expected);
            assert_eq!(
                canonical_string(&format!("[!{term}&!{term}]")),
                format!("[!{term}]")
            );
        }
    }

    #[test]
    fn canonicalize_simplifies_equivalent_atom_boolean_forms() {
        for (source, expected) in [
            ("[#6;R]", "[#6&R]"),
            ("[#6;#7]", "[!*]"),
            ("[#6,#6&R]", "[#6]"),
            ("[C,C&R]", "[C]"),
            ("[#6,!#6]", "*"),
            ("[A,a]", "*"),
            ("[C&#6]", "[C]"),
            ("[c&#6]", "[c]"),
            ("[N&#7]", "[N]"),
            ("[n&#7]", "[n]"),
            ("[O&#8]", "[O]"),
            ("[o&#8]", "[o]"),
            ("[C&A]", "[C]"),
            ("[c&a]", "[c]"),
            ("[#1&A]", "[#1]"),
            ("[#5&A]", "[B]"),
            ("[#5&a]", "[b]"),
            ("[#6&A]", "[C]"),
            ("[#6&a]", "[c]"),
            ("[#7&A]", "[N]"),
            ("[#7&a]", "[n]"),
            ("[#8&A]", "[O]"),
            ("[#8&a]", "[o]"),
            ("[#15&A]", "[P]"),
            ("[#15&a]", "[p]"),
            ("[#16&A]", "[S]"),
            ("[#16&a]", "[s]"),
            ("[#33&a]", "[as]"),
            ("[#34&a]", "[se]"),
            ("[#9&A]", "[F]"),
            ("[#17&A]", "[Cl]"),
            ("[#35&A]", "[Br]"),
            ("[#53&A]", "[I]"),
            ("[#6,#6&H3,#6&R]", "[#6]"),
            ("[!#6,!#6&R]", "[!#6]"),
            ("[C,#6]", "[#6]"),
            ("[c,#6]", "[#6]"),
            ("[C,A]", "[A]"),
            ("[c,a]", "[a]"),
            ("[#6,C&H3]", "[#6]"),
            ("[#6,12C&H3]", "[#6]"),
            ("[C,12C&H3]", "[C]"),
            ("[A,12C&H3]", "[A]"),
            ("[C,c]", "[#6]"),
            ("[B,b]", "[#5]"),
            ("[N,n]", "[#7]"),
            ("[O,o]", "[#8]"),
            ("[P,p]", "[#15]"),
            ("[S,s]", "[#16]"),
            ("[As,as]", "[#33]"),
            ("[Se,se]", "[#34]"),
            ("[#6&A,#6&a]", "[#6]"),
            ("[C,c,!#6]", "*"),
            ("[R,!R]", "*"),
            ("[r6,!r6]", "*"),
            ("[H0,!H0]", "*"),
            ("[#6,!C]", "*"),
            ("[#6,!c]", "*"),
            ("[12*,!12C]", "*"),
            ("[D{2-4},!D3]", "*"),
            ("[#6&A,!C]", "*"),
            ("[12*&C,!12C]", "*"),
            ("[D{2-4}&D3,!D3]", "*"),
            ("[#6;R;H1]", "[#6&H&R]"),
        ] {
            assert_eq!(canonical_string(source), expected, "{source}");
        }
    }

    #[test]
    fn canonicalize_simplifies_negated_mutually_exclusive_atom_forms() {
        for (source, expected) in [
            ("[!C,!c]", "*"),
            ("[!#6,!#7]", "*"),
            ("[!A,!a]", "*"),
            ("[!D1,!D2]", "*"),
            ("[!R1,!R2]", "*"),
            ("[!+0,!+]", "*"),
            ("[#6,!C,!c]", "*"),
            ("[!12C,!12c]", "*"),
            ("[#6,!#6&!#7]", "[!#7]"),
            ("[!#6&!C]", "[!#6]"),
            ("[!D{2-4}&!D3]", "[!D{2-4}]"),
            ("[!#6,!C]", "[!C]"),
            ("[!D{2-4},!D3]", "[!D3]"),
        ] {
            assert_eq!(canonical_string(source), expected, "{source}");
        }
    }

    #[test]
    fn canonicalize_simplifies_false_atom_boolean_forms() {
        for (source, expected) in [
            ("[!*]", "[!*]"),
            ("[#6&!#6]", "[!*]"),
            ("[C&!C]", "[!*]"),
            ("[#6&#7]", "[!*]"),
            ("[C&c]", "[!*]"),
            ("[A&a]", "[!*]"),
            ("[12C&13C]", "[!*]"),
            ("[12C&13*]", "[!*]"),
            ("[+0&+]", "[!*]"),
            ("[D1&D2]", "[!*]"),
            ("[D{1-2}&D{3-4}]", "[!*]"),
            ("[R0&R]", "[!*]"),
            ("[r0&r]", "[!*]"),
            ("[C&!#6]", "[!*]"),
            ("[c&!#6]", "[!*]"),
            ("[12C&!12*]", "[!*]"),
            ("[D&!D1]", "[!*]"),
            ("[D3&!D{2-4}]", "[!*]"),
            ("[R&!R{1-}]", "[!*]"),
            ("[#6&A&!C]", "[!*]"),
            ("[#6&a&!c]", "[!*]"),
            ("[12*&C&!12C]", "[!*]"),
            ("[D{2-4}&D3&!D3]", "[!*]"),
            ("[!*,#6]", "[#6]"),
            ("[!*&#6]", "[!*]"),
            ("[!*,!*]", "[!*]"),
            ("[#6&!#6,#7]", "[#7]"),
            ("[#6&!#6,#7&!#7]", "[!*]"),
            ("[#6,!*]", "[#6]"),
            ("[#6,!#6&!*]", "[#6]"),
        ] {
            assert_eq!(canonical_string(source), expected, "{source}");
        }
    }

    #[test]
    fn canonicalize_simplifies_isotope_and_numeric_atom_forms() {
        for (source, expected) in [
            ("[12C&#6]", "[12C]"),
            ("[12C&C]", "[12C]"),
            ("[12C&A]", "[12C]"),
            ("[12c&#6]", "[12c]"),
            ("[12c&c]", "[12c]"),
            ("[12c&a]", "[12c]"),
            ("[2H&#1]", "[2H]"),
            ("[12C,#6]", "[#6]"),
            ("[12C,C]", "[C]"),
            ("[12C,A]", "[A]"),
            ("[12c,#6]", "[#6]"),
            ("[12c,c]", "[c]"),
            ("[12c,a]", "[a]"),
            ("[12C&12*]", "[12C]"),
            ("[12C,12*]", "[12*]"),
            ("[12*&C]", "[12C]"),
            ("[12*&c]", "[12c]"),
            ("[12*&#6&A]", "[12C]"),
            ("[12C,12c]", "[#6&12*]"),
            ("[12*&C,12*&c]", "[#6&12*]"),
            ("[12*&C,12*]", "[12*]"),
            ("[12C,12c,12*]", "[12*]"),
            ("[12C,12c,#6]", "[#6]"),
            ("[12C,12c,!#6]", "[!#6,#6&12*]"),
            ("[D1]", "[D]"),
            ("[D{1-1}]", "[D]"),
            ("[D{-0}]", "[D0]"),
            ("[D{0-}]", "[D{-}]"),
            ("[X1]", "[X]"),
            ("[v1]", "[v]"),
            ("[H1]", "[#1]"),
            ("[#6&H1]", "[#6&H]"),
            ("[h1]", "[h]"),
            ("[z1]", "[z]"),
            ("[Z1]", "[Z]"),
            ("[R{1-1}]", "[R1]"),
            ("[r{5-5}]", "[r5]"),
            ("[x{1-1}]", "[x1]"),
            ("[D&D1]", "[D]"),
            ("[D,D1]", "[D]"),
            ("[R1&R]", "[R1]"),
            ("[R1,R]", "[R]"),
            ("[r5&r]", "[r5]"),
            ("[r5,r]", "[r]"),
            ("[D3&D{2-4}]", "[D3]"),
            ("[D3,D{2-4}]", "[D{2-4}]"),
            ("[D{2-3}&D{1-4}]", "[D{2-3}]"),
            ("[D{2-3},D{1-4}]", "[D{1-4}]"),
            ("[D{2-4}&D{3-5}]", "[D{3-4}]"),
            ("[D{2-4},D{3-5}]", "[D{2-5}]"),
            ("[R1&R{1-3}]", "[R1]"),
            ("[R1,R{2-3}]", "[R{1-3}]"),
            ("[r5&r{4-6}]", "[r5]"),
            ("[r4,r{5-6}]", "[r{4-6}]"),
            ("[D,D{2-}]", "[D{1-}]"),
            ("[D0,D{1-}]", "[D{-}]"),
            ("[D{0-2},D{3-}]", "[D{-}]"),
            ("[R,R0]", "[R{-}]"),
            ("[R0,R{1-}]", "[R{-}]"),
            ("[r,r0]", "[r{-}]"),
            ("[x,x0]", "[x{-}]"),
            ("[z,z0]", "[z{-1}]"),
            ("[Z,Z0]", "[Z{-1}]"),
        ] {
            assert_eq!(canonical_string(source), expected, "{source}");
        }
    }

    #[test]
    fn canonicalize_simplifies_equivalent_bond_boolean_forms() {
        for (source, expected) in [
            ("[#6]-&~[#7]", "[#6]-[#7]"),
            ("[#6]-,~[#7]", "[#6]~[#7]"),
            ("[#6]@&~[#7]", "[#6]@[#7]"),
            ("[#6]-;@[#7]", "[#6]-&@[#7]"),
            ("[#6]-&!~[#7]", "[#6]!~[#7]"),
            ("[#6]-&!-[#7]", "[#6]!~[#7]"),
            ("[#6]~&!-&!:[#7]", "[#6]!-&!:[#7]"),
            ("[#6]-,!-[#7]", "[#6]~[#7]"),
            ("[#6]-,!~[#7]", "[#6]-[#7]"),
            ("[#6]-,@&-[#7]", "[#6]-[#7]"),
            ("[#6]@,@&-[#7]", "[#6]@[#7]"),
            ("[#6]-&~&@[#7]", "[#6]-&@[#7]"),
            ("[#6]-;~;@[#7]", "[#6]-&@[#7]"),
            ("[#6]-&-&~[#7]", "[#6]-[#7]"),
            ("[#6]-,=,-[#7]", "[#6]-,=[#7]"),
            ("[#6]~,!~[#7]", "[#6]~[#7]"),
        ] {
            assert_eq!(canonical_string(source), expected, "{source}");
        }
    }

    #[test]
    fn canonicalize_sorts_bond_boolean_children() {
        assert_eq!(canonical_string("C-,=N"), canonical_string("C=,-N"));
        assert_eq!(canonical_string("C-;@N"), canonical_string("C@;-N"));
        assert_same_canonical_group(&["C-,=N", "C=,-N", "C=,-N"]);
        assert_same_canonical_group(&["C-;@N", "C@;-N"]);
    }

    #[test]
    fn canonicalize_recursively_canonicalizes_nested_queries() {
        assert_eq!(canonical_string("[$(OC)]"), canonical_string("[$(CO)]"));
        assert_eq!(
            canonical_string("[$([H1;C])]"),
            canonical_string("[$([C;H1])]")
        );
        assert_same_canonical_group(&["[$(OC)]", "[$(CO)]"]);
        assert_eq!(canonical_string("[$((CO))]"), "[$(CO)]");
        assert_all_atom_permutations_converge("[$(CO)]N");
        assert_all_atom_permutations_converge("C[$([O,N])]");
    }

    #[test]
    fn canonicalize_runs_expression_simplifications_through_recursion_and_bonds() {
        assert_eq!(canonical_string("[!!#6]"), "[#6]");
        assert_eq!(canonical_string("[!!-2]"), "[-2]");
        assert_eq!(canonical_string("[$([!!#6;*])]"), "[$([#6])]");

        let query = QueryMol::from_str("C-N").unwrap();
        let mut bonds = query.bonds().to_vec();
        bonds[0].expr = BondExpr::Query(BondExprTree::Not(Box::new(BondExprTree::Not(Box::new(
            BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
        )))));
        let query = QueryMol::from_parts(
            query.atoms().to_vec(),
            bonds,
            query.component_count(),
            query.component_groups().to_vec(),
        );

        assert_eq!(query.canonicalize().to_string(), "C-N");
    }

    #[test]
    fn canonicalize_preserves_group_structure_but_sorts_group_contents() {
        assert_eq!(canonical_string("(N.C)"), canonical_string("(C.N)"));
        assert_ne!(canonical_string("(C.N)"), canonical_string("C.N"));
    }

    #[test]
    fn canonicalize_converges_ring_and_symmetric_permutations() {
        assert_same_canonical_group(&["C1CC1O", "OC1CC1", "C1(O)CC1"]);
        assert_all_atom_permutations_converge("C1CC1");
        assert_all_atom_permutations_converge("C1CC1O");
        assert_all_atom_permutations_converge("C1NC1O");
        assert_all_atom_permutations_converge("C1=CC=C1");
    }

    #[test]
    fn canonicalize_handles_parallel_bonds_between_the_same_atoms() {
        let query = QueryMol::from_str("C1C1").unwrap();

        let canonical = query.canonicalize();
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), 2);
        assert_eq!(canonical.bond_count(), 2);
        assert_eq!(canonical.to_string(), canonical.canonicalize().to_string());
        assert_eq!(
            canonical,
            QueryMol::from_str(&canonical.to_string())
                .unwrap()
                .canonicalize()
        );
    }

    #[test]
    fn canonicalize_makes_atomic_hydrogen_unambiguous_in_recursive_queries() {
        let query = QueryMol::from_str("[!$([H-])]").unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(rendered, "[!$([#1&-])]");
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_disconnected_sulfur_rich_fuzz_artifact() {
        let query = QueryMol::from_str(
            "CCCC.OCCCSSSSSSSSSSCC.CCC.OCCCC.CCC.OCCCSSSSSSSSSSCCSOSSSSSSSSSCC.OCOC",
        )
        .unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_recursive_high_and_fuzz_artifact() {
        let query = QueryMol::from_str("[$(COcccccccc)]([RRRRRRRRRSmRR+])").unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_recursive_oom_fuzz_artifact() {
        let source = String::from_utf8(vec![
            67, 67, 67, 67, 91, 36, 40, 67, 115, 91, 36, 40, 42, 65, 65, 70, 80, 80, 80, 80, 80,
            80, 80, 80, 80, 80, 67, 80, 80, 80, 80, 80, 64, 67, 45, 67, 65, 99, 42, 79, 46, 98, 42,
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 91, 43, 72, 93, 42, 42, 42, 42, 42, 42,
            46, 115, 42, 42, 42, 42, 42, 65, 67, 65, 80, 80, 80, 79, 41, 93, 79, 110, 79, 79, 41,
            93, 79, 46, 79, 80, 64, 65, 65, 67, 67, 65, 65, 65, 65,
        ])
        .unwrap();
        let query = QueryMol::from_str(&source).unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_recursive_component_oom_fuzz_artifact() {
        let query = QueryMol::from_str("[$(Cs[$(*.sO)]O.OO)]O.O").unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_recursive_expression_oom_fuzz_artifact() {
        let source = String::from_utf8(vec![
            0x5b, 0x21, 0x24, 0x28, 0x5b, 0x43, 0x21, 0x24, 0x28, 0x5b, 0x42, 0x72, 0x4e, 0x40,
            0x48, 0x2d, 0x21, 0x24, 0x28, 0x5b, 0x43, 0x2c, 0x4e, 0x40, 0x48, 0x2d, 0x42, 0x72,
            0x5d, 0x29, 0x42, 0x72, 0x5d, 0x29, 0x4e, 0x41, 0x2d, 0x26, 0x42, 0x72, 0x5d, 0x29,
            0x5d, 0x0a,
        ])
        .unwrap();
        let query = QueryMol::from_str(source.trim_end()).unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_recursive_disconnected_triple_oom_fuzz_artifact() {
        let query = QueryMol::from_str("[$(C#[$(CsO)].OO)]O.O").unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_duplicate_wildcard_bracket_fuzz_artifact() {
        let source = String::from_utf8(vec![0x5b, 0x2a, 0x2a, 0x5d, 0x0c, 0x43]).unwrap();
        let query = QueryMol::from_str(&source).unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_unpaired_directional_single_bond_fuzz_artifact() {
        let query = QueryMol::from_str("C/C").unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_repeated_directional_single_bond_fuzz_artifact() {
        let query = QueryMol::from_str("CC//C").unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_directional_single_bond_conjunction_fuzz_artifact() {
        let query = QueryMol::from_str(r"C\;-O\O/n").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_lone_bracket_hydrogen_fuzz_artifact() {
        let query = QueryMol::from_str("C. [*H].O").unwrap();

        let canonical = query.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_repeated_ring_bond_conjunction_fuzz_artifact() {
        let query = QueryMol::from_str(r"CC\--@@@@NCC").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_negated_directional_single_bond_fuzz_artifact() {
        let query = QueryMol::from_str("ACPaOCCCCCC!/CC").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_recursive_hydrogen_bundle_fuzz_artifact() {
        let query = QueryMol::from_str("[!$([N;Tm*H-;H-])]").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_recursive_oom_bundle_fuzz_artifact() {
        let query = QueryMol::from_str("[$(Cs[$(CsO[$(Cs[$(CsO)]O.OO)]O.O)]O.OO)]O.O").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_repeated_ring_topology_bundle_fuzz_artifact() {
        let query = QueryMol::from_str("C-;@;@@CCC-;@CC--;@CP-;@CC@a").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_directional_bond_bundle_fuzz_artifact() {
        let query = QueryMol::from_str(r"C#[R0]\\;\\\\,\\\\\\\\\\C\\\\\\\\\\\\\\C\:CCC").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed.canonicalize());
    }

    #[test]
    fn canonicalize_handles_directional_bond_bundle_2_fuzz_artifact() {
        let query =
            QueryMol::from_str(r"CC:CC\\\\\\,\\\\\\\\#\\\\\\\\\\\\\\\\\\\\\\\\C:CCACC").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();
        let reparsed_canonical = reparsed.canonicalize();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed_canonical);
    }

    #[test]
    fn canonicalize_handles_chiral_directional_timeout_fuzz_artifact() {
        let query = QueryMol::from_str(r"C\[Cn@@@@@B@A@B@A@@N@]\\C").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();
        let reparsed_canonical = reparsed.canonicalize();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed_canonical);
    }

    #[test]
    fn canonicalize_handles_trigonal_bipyramidal_parallel_bond_fuzz_artifact() {
        let query = QueryMol::from_str(r"[@PasBP]17O[@TBP]17O").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();
        let reparsed_canonical = reparsed.canonicalize();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed_canonical);
    }

    #[test]
    fn canonicalize_handles_underconstrained_ring_chirality_fuzz_artifact() {
        let query = QueryMol::from_str("C[No]1[21Al43AlB]2[21AlBNo@@]1[21Al24AlB]2[21Al]").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();
        let reparsed_canonical = reparsed.canonicalize();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed_canonical);
    }

    #[test]
    fn canonicalize_handles_underconstrained_ring_chirality_with_h_fuzz_artifact() {
        let query = QueryMol::from_str("C[No]1[21Al43AlB]2[21AlBNo@H]1[21Al40AlB]2[21Al]").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();
        let reparsed_canonical = reparsed.canonicalize();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed_canonical);
    }

    #[test]
    fn canonicalize_handles_fused_ring_generic_chirality_fuzz_artifact() {
        let query = QueryMol::from_str("c[@]8[C@@]71cOC7=8s1").unwrap();

        let canonical = query.canonicalize();
        let recanonicalized = canonical.canonicalize();
        let rendered = canonical.to_string();
        let reparsed = QueryMol::from_str(&rendered).unwrap();
        let reparsed_canonical = reparsed.canonicalize();

        assert_eq!(canonical, recanonicalized);
        assert!(canonical.is_canonical());
        assert_eq!(canonical.atom_count(), query.atom_count());
        assert_eq!(canonical.bond_count(), query.bond_count());
        assert_eq!(canonical.component_count(), query.component_count());
        assert_eq!(rendered, canonical.to_canonical_smarts());
        assert_eq!(canonical, reparsed_canonical);
    }

    #[test]
    fn canonicalize_is_stable_under_manual_graph_reordering() {
        let query = QueryMol::from_str("[$(CO)]C1NC1.O").unwrap();
        let canonical = query.canonicalize();

        let atom_order = vec![4, 2, 0, 3, 1];
        let permuted_atoms = permute_atoms(&query, &atom_order);
        let permuted_both = permute_top_level_entries(&permuted_atoms, &[1, 0]);

        assert_eq!(canonical, permuted_atoms.canonicalize());
        assert_eq!(canonical, permuted_both.canonicalize());
    }

    #[test]
    fn canonicalize_preserves_distinctions_between_non_equivalent_queries() {
        assert_ne!(canonical_string("C"), canonical_string("[#6]"));
        assert_ne!(canonical_string("C-N"), canonical_string("C=N"));
        assert_ne!(canonical_string("C.N"), canonical_string("(C.N)"));
        assert_ne!(canonical_string("[$(CO)]"), canonical_string("[$(CN)]"));
        assert_ne!(canonical_string("[C;H1]"), canonical_string("[C;H2]"));
        assert_ne!(canonical_string("C1CC1"), canonical_string("CCC"));
    }

    #[test]
    fn canonicalize_recursive_charge_bundle_subcase_redundant_wildcard_charge_converges() {
        assert_eq!(
            canonical_string("[$([*+]~[*-])]"),
            canonical_string("[$([+]~[-])]")
        );
    }

    #[test]
    fn canonicalize_recursive_charge_bundle_subcase_x4_charge_rooting_converges() {
        assert_eq!(
            canonical_string("[$([OX1-,OH1][#7X4+]([*])([*])([*]))]"),
            canonical_string("[$(*[#7&+&X4](*)(*)[-&O&X1,H1&O])]")
        );
    }

    #[test]
    fn canonicalize_recursive_charge_bundle_subcase_x4v5_rooting_converges() {
        assert_eq!(
            canonical_string("[$([OX1]=[#7X4v5]([*])([*])([*]))]"),
            canonical_string("[$(*[#7&X4&v5](*)(*)=[O&X1])]")
        );
    }
}
