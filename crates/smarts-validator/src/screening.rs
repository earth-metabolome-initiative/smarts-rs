//! Conservative screening for many-SMARTS × many-SMILES workloads.
//!
//! The screen never proves a match. It only rules out impossible pairs using
//! cheap lower bounds derived from the query and cheap summary counts derived
//! from the prepared target.

use alloc::collections::BTreeMap;

use elements_rs::{Element, ElementVariant};
use smarts_parser::{
    AtomExpr, AtomPrimitive, BracketExprTree, ComponentGroupId, HydrogenKind, NumericQuery,
    QueryMol,
};

use crate::prepared::PreparedTarget;

/// Conservative summary of one compiled SMARTS query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryScreen {
    /// Minimum number of target atoms required by the query.
    pub min_atom_count: usize,
    /// Minimum number of target connected components required by grouped
    /// disconnected SMARTS.
    pub min_target_component_count: usize,
    /// Lower bounds for exact element occurrences the target must contain.
    pub required_element_counts: BTreeMap<Element, usize>,
    /// Minimum number of aromatic atoms required by the query.
    pub min_aromatic_atom_count: usize,
    /// Minimum number of ring-member atoms required by the query.
    pub min_ring_atom_count: usize,
}

impl QueryScreen {
    /// Builds a conservative screen summary from one parsed SMARTS query.
    #[must_use]
    pub fn new(query: &QueryMol) -> Self {
        let mut required_element_counts = BTreeMap::new();
        let mut min_aromatic_atom_count = 0usize;
        let mut min_ring_atom_count = 0usize;

        for atom in query.atoms() {
            let requirement = forced_atom_requirement(&atom.expr);
            if let Some(element) = requirement.element {
                *required_element_counts.entry(element).or_insert(0) += 1;
            }
            if requirement.requires_aromatic {
                min_aromatic_atom_count += 1;
            }
            if requirement.requires_ring {
                min_ring_atom_count += 1;
            }
        }

        Self {
            min_atom_count: query.atom_count(),
            min_target_component_count: grouped_component_count(query.component_groups()),
            required_element_counts,
            min_aromatic_atom_count,
            min_ring_atom_count,
        }
    }

    /// Returns whether the target summary could still satisfy the query.
    #[must_use]
    pub fn may_match(&self, target: &TargetScreen) -> bool {
        if target.atom_count < self.min_atom_count {
            return false;
        }
        if target.connected_component_count < self.min_target_component_count {
            return false;
        }
        if target.aromatic_atom_count < self.min_aromatic_atom_count {
            return false;
        }
        if target.ring_atom_count < self.min_ring_atom_count {
            return false;
        }
        self.required_element_counts
            .iter()
            .all(|(element, required)| {
                target
                    .element_counts
                    .get(element)
                    .copied()
                    .unwrap_or_default()
                    >= *required
            })
    }
}

/// Conservative summary of one prepared target molecule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetScreen {
    /// Number of atoms in the target.
    pub atom_count: usize,
    /// Number of connected components in the target.
    pub connected_component_count: usize,
    /// Exact element occurrence counts in the target.
    pub element_counts: BTreeMap<Element, usize>,
    /// Number of aromatic atoms in the target.
    pub aromatic_atom_count: usize,
    /// Number of ring-member atoms in the target.
    pub ring_atom_count: usize,
}

impl TargetScreen {
    /// Builds a conservative target summary from one prepared target.
    #[must_use]
    pub fn new(target: &PreparedTarget) -> Self {
        let mut element_counts = BTreeMap::new();
        let mut aromatic_atom_count = 0usize;
        let mut ring_atom_count = 0usize;
        let mut max_component = None::<usize>;

        for atom_id in 0..target.atom_count() {
            if let Some(element) = target
                .atom(atom_id)
                .and_then(smiles_parser::atom::Atom::element)
            {
                *element_counts.entry(element).or_insert(0) += 1;
            }
            if target.is_aromatic(atom_id) {
                aromatic_atom_count += 1;
            }
            if target.is_ring_atom(atom_id) {
                ring_atom_count += 1;
            }
            if let Some(component_id) = target.connected_component(atom_id) {
                max_component =
                    Some(max_component.map_or(component_id, |current| current.max(component_id)));
            }
        }

        Self {
            atom_count: target.atom_count(),
            connected_component_count: max_component.map_or(0, |value| value + 1),
            element_counts,
            aromatic_atom_count,
            ring_atom_count,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AtomRequirement {
    element: Option<Element>,
    requires_aromatic: bool,
    requires_ring: bool,
}

fn grouped_component_count(component_groups: &[Option<ComponentGroupId>]) -> usize {
    let mut groups = alloc::vec::Vec::new();
    for group in component_groups.iter().flatten().copied() {
        if !groups.contains(&group) {
            groups.push(group);
        }
    }
    groups.len()
}

fn forced_atom_requirement(expr: &AtomExpr) -> AtomRequirement {
    match expr {
        AtomExpr::Wildcard => AtomRequirement::default(),
        AtomExpr::Bare { element, aromatic } => AtomRequirement {
            element: Some(*element),
            requires_aromatic: *aromatic,
            requires_ring: false,
        },
        AtomExpr::Bracket(expr) => forced_bracket_requirement(&expr.tree),
    }
}

fn forced_bracket_requirement(tree: &BracketExprTree) -> AtomRequirement {
    match tree {
        BracketExprTree::Primitive(primitive) => forced_primitive_requirement(primitive),
        BracketExprTree::Not(_) => AtomRequirement::default(),
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => items
            .iter()
            .fold(AtomRequirement::default(), merge_and_requirement),
        BracketExprTree::Or(items) => items
            .iter()
            .map(forced_bracket_requirement)
            .reduce(intersect_or_requirement)
            .unwrap_or_default(),
    }
}

fn forced_primitive_requirement(primitive: &AtomPrimitive) -> AtomRequirement {
    match primitive {
        AtomPrimitive::Symbol { element, aromatic } => AtomRequirement {
            element: Some(*element),
            requires_aromatic: *aromatic,
            requires_ring: false,
        },
        AtomPrimitive::Isotope { isotope, aromatic } => AtomRequirement {
            element: Some(isotope.element()),
            requires_aromatic: *aromatic,
            requires_ring: false,
        },
        AtomPrimitive::AtomicNumber(number) => {
            Element::try_from(u8::try_from(*number).unwrap_or(u8::MAX))
                .ok()
                .map_or_else(AtomRequirement::default, |element| AtomRequirement {
                    element: Some(element),
                    requires_aromatic: false,
                    requires_ring: false,
                })
        }
        AtomPrimitive::AromaticAny => AtomRequirement {
            element: None,
            requires_aromatic: true,
            requires_ring: false,
        },
        AtomPrimitive::RingMembership(expected)
        | AtomPrimitive::RingSize(expected)
        | AtomPrimitive::RingConnectivity(expected) => AtomRequirement {
            element: None,
            requires_aromatic: false,
            requires_ring: numeric_query_requires_positive(*expected),
        },
        AtomPrimitive::Hybridization(_)
        | AtomPrimitive::HeteroNeighbor(_)
        | AtomPrimitive::AliphaticHeteroNeighbor(_)
        | AtomPrimitive::IsotopeWildcard(_)
        | AtomPrimitive::Hydrogen(
            HydrogenKind::Total | HydrogenKind::Implicit,
            Some(NumericQuery::Exact(0)),
        )
        | AtomPrimitive::Wildcard
        | AtomPrimitive::AliphaticAny
        | AtomPrimitive::Degree(_)
        | AtomPrimitive::Connectivity(_)
        | AtomPrimitive::Valence(_)
        | AtomPrimitive::RecursiveQuery(_)
        | AtomPrimitive::Hydrogen(_, _)
        | AtomPrimitive::Chirality(_)
        | AtomPrimitive::Charge(_) => AtomRequirement::default(),
    }
}

fn numeric_query_requires_positive(query: Option<NumericQuery>) -> bool {
    match query {
        None => true,
        Some(NumericQuery::Exact(value)) => value > 0,
        Some(NumericQuery::Range(range)) => range.min.is_some_and(|value| value > 0),
    }
}

fn merge_and_requirement(left: AtomRequirement, right: &BracketExprTree) -> AtomRequirement {
    let right = forced_bracket_requirement(right);
    AtomRequirement {
        element: match (left.element, right.element) {
            (Some(left), Some(right)) if left == right => Some(left),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            _ => None,
        },
        requires_aromatic: left.requires_aromatic || right.requires_aromatic,
        requires_ring: left.requires_ring || right.requires_ring,
    }
}

fn intersect_or_requirement(left: AtomRequirement, right: AtomRequirement) -> AtomRequirement {
    AtomRequirement {
        element: match (left.element, right.element) {
            (Some(left), Some(right)) if left == right => Some(left),
            _ => None,
        },
        requires_aromatic: left.requires_aromatic && right.requires_aromatic,
        requires_ring: left.requires_ring && right.requires_ring,
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use core::str::FromStr;

    use serde::Deserialize;
    use smarts_parser::QueryMol;
    use smiles_parser::Smiles;

    use super::{QueryScreen, TargetScreen};
    use crate::prepared::PreparedTarget;

    #[derive(Debug, Deserialize)]
    struct ExpectedCase {
        smarts: String,
        smiles: String,
        expected_match: bool,
    }

    #[test]
    fn query_screen_extracts_conservative_lower_bounds() {
        let query = QueryMol::from_str("([c].[c]).[R]").unwrap();
        let screen = QueryScreen::new(&query);

        assert_eq!(screen.min_atom_count, 3);
        assert_eq!(screen.min_target_component_count, 1);
        assert_eq!(
            screen.required_element_counts.get(&elements_rs::Element::C),
            Some(&2)
        );
        assert_eq!(screen.min_aromatic_atom_count, 2);
        assert_eq!(screen.min_ring_atom_count, 1);
    }

    #[test]
    fn target_screen_summarizes_prepared_target() {
        let prepared = PreparedTarget::new(Smiles::from_str("c1ccccc1.O").unwrap());
        let screen = TargetScreen::new(&prepared);

        assert_eq!(screen.atom_count, 7);
        assert_eq!(screen.connected_component_count, 2);
        assert_eq!(
            screen.element_counts.get(&elements_rs::Element::C),
            Some(&6)
        );
        assert_eq!(
            screen.element_counts.get(&elements_rs::Element::O),
            Some(&1)
        );
        assert_eq!(screen.aromatic_atom_count, 6);
        assert_eq!(screen.ring_atom_count, 6);
    }

    #[test]
    fn screen_rejects_obvious_non_matches() {
        let query = QueryScreen::new(&QueryMol::from_str("(C).(C)").unwrap());
        let same_component =
            TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
        let two_components =
            TargetScreen::new(&PreparedTarget::new(Smiles::from_str("C.C").unwrap()));

        assert!(!query.may_match(&same_component));
        assert!(query.may_match(&two_components));
    }

    #[test]
    fn screen_rejects_missing_atoms_elements_aromaticity_and_ring_membership() {
        let atom_count_query = QueryScreen::new(&QueryMol::from_str("CCC").unwrap());
        let aromatic_query = QueryScreen::new(&QueryMol::from_str("c").unwrap());
        let ring_query = QueryScreen::new(&QueryMol::from_str("[R]").unwrap());
        let element_query = QueryScreen::new(&QueryMol::from_str("[Cl]").unwrap());

        let small_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
        let aliphatic_target =
            TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
        let acyclic_target =
            TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CCC").unwrap()));
        let oxygen_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("O").unwrap()));

        assert!(!atom_count_query.may_match(&small_target));
        assert!(!aromatic_query.may_match(&aliphatic_target));
        assert!(!ring_query.may_match(&acyclic_target));
        assert!(!element_query.may_match(&oxygen_target));
    }

    #[test]
    fn query_screen_extracts_atomic_number_isotope_and_nonpositive_ring_bounds_conservatively() {
        let atomic_number = QueryScreen::new(&QueryMol::from_str("[#8]").unwrap());
        let isotope = QueryScreen::new(&QueryMol::from_str("[18O]").unwrap());
        let ring_zero = QueryScreen::new(&QueryMol::from_str("[R0]").unwrap());
        let ring_range_zero = QueryScreen::new(&QueryMol::from_str("[r{0-2}]").unwrap());

        assert_eq!(
            atomic_number
                .required_element_counts
                .get(&elements_rs::Element::O),
            Some(&1)
        );
        assert_eq!(
            isotope
                .required_element_counts
                .get(&elements_rs::Element::O),
            Some(&1)
        );
        assert_eq!(ring_zero.min_ring_atom_count, 0);
        assert_eq!(ring_range_zero.min_ring_atom_count, 0);
    }

    #[test]
    fn screen_never_filters_true_matches_from_frozen_fixtures() {
        for fixture in [
            include_str!("../../../corpus/validator/single-atom-v0.rdkit.json"),
            include_str!("../../../corpus/validator/connected-v0.rdkit.json"),
            include_str!("../../../corpus/validator/ring-v0.rdkit.json"),
            include_str!("../../../corpus/validator/counts-v0.rdkit.json"),
            include_str!("../../../corpus/validator/disconnected-v0.rdkit.json"),
            include_str!("../../../corpus/validator/recursive-v0.rdkit.json"),
            include_str!("../../../corpus/validator/stereo-v0.rdkit.json"),
        ] {
            let cases: alloc::vec::Vec<ExpectedCase> =
                serde_json::from_str(fixture).expect("valid frozen fixture");
            for case in cases {
                if !case.expected_match {
                    continue;
                }
                let query = QueryMol::from_str(&case.smarts).expect("valid SMARTS");
                let target =
                    PreparedTarget::new(case.smiles.parse::<Smiles>().expect("valid SMILES"));
                let query_screen = QueryScreen::new(&query);
                let target_screen = TargetScreen::new(&target);
                assert!(
                    query_screen.may_match(&target_screen),
                    "screen rejected known true match: SMARTS {:?} vs SMILES {:?}",
                    case.smarts,
                    case.smiles
                );
            }
        }
    }
}
