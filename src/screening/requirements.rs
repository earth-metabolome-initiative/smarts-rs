use crate::{
    AtomExpr, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive, BracketExprTree, HydrogenKind,
    NumericQuery,
};
use elements_rs::{Element, ElementVariant};
use smiles_parser::bond::Bond;

use super::features::RequiredBondKind;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct AtomRequirement {
    pub(super) element: Option<Element>,
    pub(super) requires_aromatic: bool,
    pub(super) requires_ring: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ExactAtomRequirement {
    pub(super) element: Option<Element>,
    pub(super) aromatic: Option<bool>,
    pub(super) requires_ring: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct AtomCountRequirement {
    pub(super) degree: Option<u16>,
    pub(super) total_hydrogens: Option<u16>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct BondRequirement {
    pub(super) kind: Option<RequiredBondKind>,
    pub(super) requires_ring: bool,
}

pub(super) fn forced_atom_requirement(expr: &AtomExpr) -> AtomRequirement {
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

pub(super) fn exact_atom_requirement(expr: &AtomExpr) -> ExactAtomRequirement {
    match expr {
        AtomExpr::Wildcard => ExactAtomRequirement::default(),
        AtomExpr::Bare { element, aromatic } => ExactAtomRequirement {
            element: Some(*element),
            aromatic: Some(*aromatic),
            requires_ring: false,
        },
        AtomExpr::Bracket(expr) => exact_bracket_requirement(&expr.tree),
    }
}

pub(super) fn forced_atom_count_requirement(expr: &AtomExpr) -> AtomCountRequirement {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => AtomCountRequirement::default(),
        AtomExpr::Bracket(expr) => forced_bracket_count_requirement(&expr.tree),
    }
}

pub(super) fn forced_bond_requirement(expr: &BondExpr) -> BondRequirement {
    match expr {
        BondExpr::Elided => BondRequirement::default(),
        BondExpr::Query(tree) => forced_bond_tree_requirement(tree),
    }
}

fn exact_bracket_requirement(tree: &BracketExprTree) -> ExactAtomRequirement {
    match tree {
        BracketExprTree::Primitive(primitive) => exact_primitive_requirement(primitive),
        BracketExprTree::Not(_) | BracketExprTree::Or(_) => ExactAtomRequirement::default(),
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => items
            .iter()
            .fold(ExactAtomRequirement::default(), merge_and_exact_requirement),
    }
}

fn exact_primitive_requirement(primitive: &AtomPrimitive) -> ExactAtomRequirement {
    match primitive {
        AtomPrimitive::Symbol { element, aromatic } => ExactAtomRequirement {
            element: Some(*element),
            aromatic: Some(*aromatic),
            requires_ring: false,
        },
        AtomPrimitive::Isotope { isotope, aromatic } => ExactAtomRequirement {
            element: Some(isotope.element()),
            aromatic: Some(*aromatic),
            requires_ring: false,
        },
        AtomPrimitive::AtomicNumber(number) => {
            Element::try_from(u8::try_from(*number).unwrap_or(u8::MAX))
                .ok()
                .map_or_else(ExactAtomRequirement::default, |element| {
                    ExactAtomRequirement {
                        element: Some(element),
                        aromatic: None,
                        requires_ring: false,
                    }
                })
        }
        AtomPrimitive::AromaticAny => ExactAtomRequirement {
            element: None,
            aromatic: Some(true),
            requires_ring: false,
        },
        AtomPrimitive::AliphaticAny => ExactAtomRequirement {
            element: None,
            aromatic: Some(false),
            requires_ring: false,
        },
        AtomPrimitive::RingMembership(expected)
        | AtomPrimitive::RingSize(expected)
        | AtomPrimitive::RingConnectivity(expected) => ExactAtomRequirement {
            element: None,
            aromatic: None,
            requires_ring: numeric_query_requires_positive(*expected),
        },
        _ => ExactAtomRequirement::default(),
    }
}

fn forced_bracket_count_requirement(tree: &BracketExprTree) -> AtomCountRequirement {
    match tree {
        BracketExprTree::Primitive(primitive) => forced_primitive_count_requirement(primitive),
        BracketExprTree::Not(_) => AtomCountRequirement::default(),
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => items
            .iter()
            .fold(AtomCountRequirement::default(), merge_and_count_requirement),
        BracketExprTree::Or(items) => items
            .iter()
            .map(forced_bracket_count_requirement)
            .reduce(intersect_or_count_requirement)
            .unwrap_or_default(),
    }
}

fn forced_primitive_count_requirement(primitive: &AtomPrimitive) -> AtomCountRequirement {
    match primitive {
        AtomPrimitive::Degree(expected) => AtomCountRequirement {
            degree: exact_count_query_value(*expected, 1),
            total_hydrogens: None,
        },
        AtomPrimitive::Hydrogen(HydrogenKind::Total, expected) => AtomCountRequirement {
            degree: None,
            total_hydrogens: exact_count_query_value(*expected, 1),
        },
        _ => AtomCountRequirement::default(),
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

fn forced_bond_tree_requirement(tree: &BondExprTree) -> BondRequirement {
    match tree {
        BondExprTree::Primitive(primitive) => forced_bond_primitive_requirement(*primitive),
        BondExprTree::Not(_) => BondRequirement::default(),
        BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => items
            .iter()
            .fold(BondRequirement::default(), merge_and_bond_requirement),
        BondExprTree::Or(items) => items
            .iter()
            .map(forced_bond_tree_requirement)
            .reduce(intersect_or_bond_requirement)
            .unwrap_or_default(),
    }
}

fn forced_bond_primitive_requirement(primitive: BondPrimitive) -> BondRequirement {
    match primitive {
        BondPrimitive::Bond(Bond::Single | Bond::Up | Bond::Down) => BondRequirement {
            kind: Some(RequiredBondKind::Single),
            requires_ring: false,
        },
        BondPrimitive::Bond(Bond::Double) => BondRequirement {
            kind: Some(RequiredBondKind::Double),
            requires_ring: false,
        },
        BondPrimitive::Bond(Bond::Triple) => BondRequirement {
            kind: Some(RequiredBondKind::Triple),
            requires_ring: false,
        },
        BondPrimitive::Bond(Bond::Aromatic) => BondRequirement {
            kind: Some(RequiredBondKind::Aromatic),
            requires_ring: false,
        },
        BondPrimitive::Ring => BondRequirement {
            kind: None,
            requires_ring: true,
        },
        BondPrimitive::Bond(Bond::Quadruple) | BondPrimitive::Any => BondRequirement::default(),
    }
}

fn numeric_query_requires_positive(query: Option<NumericQuery>) -> bool {
    match query {
        None => true,
        Some(NumericQuery::Exact(value)) => value > 0,
        Some(NumericQuery::Range(range)) => range.min.is_some_and(|value| value > 0),
    }
}

const fn exact_count_query_value(query: Option<NumericQuery>, omitted_default: u16) -> Option<u16> {
    match query {
        None => Some(omitted_default),
        Some(NumericQuery::Exact(value)) => Some(value),
        Some(NumericQuery::Range(_)) => None,
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

fn merge_and_count_requirement(
    left: AtomCountRequirement,
    right: &BracketExprTree,
) -> AtomCountRequirement {
    let right = forced_bracket_count_requirement(right);
    AtomCountRequirement {
        degree: merge_exact_count_requirement(left.degree, right.degree),
        total_hydrogens: merge_exact_count_requirement(left.total_hydrogens, right.total_hydrogens),
    }
}

const fn merge_exact_count_requirement(left: Option<u16>, right: Option<u16>) -> Option<u16> {
    match (left, right) {
        (Some(left), Some(right)) if left == right => Some(left),
        (Some(value), None) | (None, Some(value)) => Some(value),
        _ => None,
    }
}

fn merge_and_exact_requirement(
    left: ExactAtomRequirement,
    right: &BracketExprTree,
) -> ExactAtomRequirement {
    let right = exact_bracket_requirement(right);
    ExactAtomRequirement {
        element: match (left.element, right.element) {
            (Some(left), Some(right)) if left == right => Some(left),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            _ => None,
        },
        aromatic: match (left.aromatic, right.aromatic) {
            (Some(left), Some(right)) if left == right => Some(left),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            _ => None,
        },
        requires_ring: left.requires_ring || right.requires_ring,
    }
}

const fn intersect_or_count_requirement(
    left: AtomCountRequirement,
    right: AtomCountRequirement,
) -> AtomCountRequirement {
    AtomCountRequirement {
        degree: intersect_exact_count_requirement(left.degree, right.degree),
        total_hydrogens: intersect_exact_count_requirement(
            left.total_hydrogens,
            right.total_hydrogens,
        ),
    }
}

const fn intersect_exact_count_requirement(left: Option<u16>, right: Option<u16>) -> Option<u16> {
    match (left, right) {
        (Some(left), Some(right)) if left == right => Some(left),
        _ => None,
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

fn merge_and_bond_requirement(left: BondRequirement, right: &BondExprTree) -> BondRequirement {
    let right = forced_bond_tree_requirement(right);
    BondRequirement {
        kind: match (left.kind, right.kind) {
            (Some(left), Some(right)) if left == right => Some(left),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            _ => None,
        },
        requires_ring: left.requires_ring || right.requires_ring,
    }
}

fn intersect_or_bond_requirement(left: BondRequirement, right: BondRequirement) -> BondRequirement {
    BondRequirement {
        kind: match (left.kind, right.kind) {
            (Some(left), Some(right)) if left == right => Some(left),
            _ => None,
        },
        requires_ring: left.requires_ring && right.requires_ring,
    }
}
