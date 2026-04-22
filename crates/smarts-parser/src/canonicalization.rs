use alloc::{
    boxed::Box,
    collections::BTreeMap,
    string::{String, ToString},
    vec,
    vec::Vec,
};

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
        BracketExprTree, ComponentId, HydrogenKind, QueryAtom, QueryBond, QueryMol,
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
        let old_emitted_neighbors = emitted_stereo_neighbors(self);
        let new_emitted_neighbors = emitted_stereo_neighbors(&provisional);
        let atoms = order
            .iter()
            .copied()
            .map(|old_atom| {
                let new_atom = new_index_of_old_atom[old_atom];
                let mapped_old_neighbors = old_emitted_neighbors[old_atom]
                    .iter()
                    .copied()
                    .map(|neighbor| new_index_of_old_atom[neighbor])
                    .collect::<Vec<_>>();
                let expr = if emitted_neighbor_permutation_is_odd(
                    &mapped_old_neighbors,
                    &new_emitted_neighbors[new_atom],
                ) {
                    invert_atom_expr_chirality(self.atoms()[old_atom].expr.clone())
                } else {
                    self.atoms()[old_atom].expr.clone()
                };
                QueryAtom {
                    id: new_atom,
                    component: 0,
                    expr,
                }
            })
            .collect::<Vec<_>>();

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
    tree = simplify_redundant_wildcards(tree);
    let mut bracket = BracketExpr {
        tree,
        atom_map: normalized.atom_map,
    };
    bracket
        .normalize()
        .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
    let mut tree = canonical_bracket_tree(bracket.tree);
    tree = simplify_redundant_wildcards(tree);
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
            BracketExprTree::HighAnd(items)
        }
        BracketExprTree::Or(items) => {
            let mut items = items
                .into_iter()
                .map(canonical_bracket_tree)
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BracketExprTree::to_string);
            BracketExprTree::Or(items)
        }
        BracketExprTree::LowAnd(items) => {
            let mut items = items
                .into_iter()
                .map(canonical_bracket_tree)
                .collect::<Vec<_>>();
            items.sort_by_cached_key(BracketExprTree::to_string);
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
        AtomPrimitive::RecursiveQuery(query) => {
            AtomPrimitive::RecursiveQuery(Box::new(query.canonicalize()))
        }
        other => other,
    }
}

fn simplify_redundant_wildcards(tree: BracketExprTree) -> BracketExprTree {
    match tree {
        BracketExprTree::Primitive(primitive) => BracketExprTree::Primitive(primitive),
        BracketExprTree::Not(inner) => {
            BracketExprTree::Not(Box::new(simplify_redundant_wildcards(*inner)))
        }
        BracketExprTree::HighAnd(items) => {
            let mut items = items
                .into_iter()
                .map(simplify_redundant_wildcards)
                .collect::<Vec<_>>();
            if items.len() > 1 {
                items.retain(|item| {
                    !matches!(item, BracketExprTree::Primitive(AtomPrimitive::Wildcard))
                });
            }
            match items.as_slice() {
                [] => BracketExprTree::Primitive(AtomPrimitive::Wildcard),
                [single] => single.clone(),
                _ => BracketExprTree::HighAnd(items),
            }
        }
        BracketExprTree::LowAnd(items) => {
            let mut items = items
                .into_iter()
                .map(simplify_redundant_wildcards)
                .collect::<Vec<_>>();
            if items.len() > 1 {
                items.retain(|item| {
                    !matches!(item, BracketExprTree::Primitive(AtomPrimitive::Wildcard))
                });
            }
            match items.as_slice() {
                [] => BracketExprTree::Primitive(AtomPrimitive::Wildcard),
                [single] => single.clone(),
                _ => BracketExprTree::LowAnd(items),
            }
        }
        BracketExprTree::Or(items) => {
            let items = items
                .into_iter()
                .map(simplify_redundant_wildcards)
                .collect::<Vec<_>>();
            if items
                .iter()
                .any(|item| matches!(item, BracketExprTree::Primitive(AtomPrimitive::Wildcard)))
            {
                BracketExprTree::Primitive(AtomPrimitive::Wildcard)
            } else {
                BracketExprTree::Or(items)
            }
        }
    }
}

fn canonical_bond_expr(expr: &BondExpr) -> BondExpr {
    match expr {
        BondExpr::Elided => BondExpr::Elided,
        BondExpr::Query(tree) => {
            let mut normalized = tree.clone();
            normalize_bond_tree(&mut normalized)
                .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
            let mut canonical = canonical_bond_tree(normalized);
            normalize_bond_tree(&mut canonical)
                .unwrap_or_else(|_| unreachable!("parsed SMARTS expressions are never empty"));
            let has_up = bond_tree_contains_direction(&canonical, Bond::Up);
            let has_down = bond_tree_contains_direction(&canonical, Bond::Down);
            if !has_up && !has_down {
                return BondExpr::Query(canonical);
            }
            if has_up && has_down {
                canonical = collapse_mixed_directional_polarity(canonical);
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
        string::{String, ToString},
        vec,
        vec::Vec,
    };
    use core::str::FromStr;

    use super::QueryCanonicalLabeling;
    use crate::{QueryAtom, QueryBond, QueryMol};

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
