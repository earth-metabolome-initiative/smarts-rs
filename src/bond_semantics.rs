//! Meaning of a SMARTS bond expression, shared by the matcher and the canonicalizer.
//!
//! A bond expression matches a set of bond states, each a [`BondLabel`] on a ring
//! or chain bond. It also carries the stereo direction of its first non-negated
//! `/` or `\`. A negated direction makes the bond match nothing, and a bond that
//! matches no single bond carries no direction.
#![expect(
    clippy::redundant_pub_crate,
    reason = "items are shared across crate modules, and unreachable_pub rejects plain pub"
)]

use alloc::borrow::Cow;

use smiles_rs::bond::Bond;

use crate::{
    parse::parse_bond_expr_text, target::BondLabel, BondExpr, BondExprTree, BondPrimitive,
};

mod spellings;
#[cfg(test)]
mod tests;

const BOND_LABEL_STATE_COUNT: usize = 7;
const BOND_STATE_MASK_ALL: u16 = (1u16 << (BOND_LABEL_STATE_COUNT * 2)) - 1;
const ELIDED_STATES: u16 =
    single_like_bond_state_mask() | bond_label_state_mask(BondLabel::Aromatic);

/// Labels that a bond expression can tell apart. `Up` and `Down` target bonds
/// always share the states of `Single`, so the spelling tables index the five
/// distinguishable labels on chain bonds, then the same five on ring bonds.
const DISTINGUISHABLE_LABELS: [BondLabel; 5] = [
    BondLabel::Single,
    BondLabel::Double,
    BondLabel::Triple,
    BondLabel::Aromatic,
    BondLabel::Any,
];

const fn bond_label_state_index(label: BondLabel) -> usize {
    match label {
        BondLabel::Single => 0,
        BondLabel::Double => 1,
        BondLabel::Triple => 2,
        BondLabel::Aromatic => 3,
        BondLabel::Up => 4,
        BondLabel::Down => 5,
        BondLabel::Any => 6,
    }
}

pub(crate) const fn bond_state_bit(label: BondLabel, ring: bool) -> u16 {
    let offset = if ring { BOND_LABEL_STATE_COUNT } else { 0 };
    1u16 << (bond_label_state_index(label) + offset)
}

const fn bond_label_state_mask(label: BondLabel) -> u16 {
    bond_state_bit(label, false) | bond_state_bit(label, true)
}

const fn single_like_bond_state_mask() -> u16 {
    bond_label_state_mask(BondLabel::Single)
        | bond_label_state_mask(BondLabel::Up)
        | bond_label_state_mask(BondLabel::Down)
}

const fn ring_bond_state_mask() -> u16 {
    BOND_STATE_MASK_ALL & !((1u16 << BOND_LABEL_STATE_COUNT) - 1)
}

pub(crate) const fn bond_state_mask_is_ring_sensitive(mask: u16) -> bool {
    let chain = mask & ((1u16 << BOND_LABEL_STATE_COUNT) - 1);
    let ring = mask >> BOND_LABEL_STATE_COUNT;
    chain != ring
}

const fn primitive_states(primitive: BondPrimitive) -> u16 {
    match primitive {
        BondPrimitive::Bond(Bond::Single | Bond::Up | Bond::Down) => single_like_bond_state_mask(),
        BondPrimitive::Bond(Bond::Double) => bond_label_state_mask(BondLabel::Double),
        BondPrimitive::Bond(Bond::Triple) => bond_label_state_mask(BondLabel::Triple),
        BondPrimitive::Aromatic => bond_label_state_mask(BondLabel::Aromatic),
        BondPrimitive::Bond(Bond::Quadruple) => 0,
        BondPrimitive::Any => BOND_STATE_MASK_ALL,
        BondPrimitive::Ring => ring_bond_state_mask(),
    }
}

const fn primitive_direction(primitive: BondPrimitive) -> Option<Bond> {
    match primitive {
        BondPrimitive::Bond(direction @ (Bond::Up | Bond::Down)) => Some(direction),
        _ => None,
    }
}

/// Index of `states` in the spelling tables.
fn spelling_index(states: u16) -> usize {
    let mut index = 0usize;
    for (ring_offset, ring) in [(0, false), (DISTINGUISHABLE_LABELS.len(), true)] {
        for (position, label) in DISTINGUISHABLE_LABELS.into_iter().enumerate() {
            if states & bond_state_bit(label, ring) != 0 {
                index |= 1 << (position + ring_offset);
            }
        }
    }
    index
}

/// What a bond expression matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BondSemantics {
    states: u16,
    direction: Option<Bond>,
}

impl BondSemantics {
    pub(crate) fn of_expr(expr: &BondExpr) -> Self {
        match expr {
            BondExpr::Elided => Self::new(ELIDED_STATES, None),
            BondExpr::Query(tree) => Self::of_tree(tree),
        }
    }

    pub(crate) fn of_tree(tree: &BondExprTree) -> Self {
        let meaning = TreeMeaning::of(tree);
        if meaning.negated_direction {
            Self::new(0, None)
        } else {
            Self::new(meaning.states, meaning.first_direction)
        }
    }

    const fn new(states: u16, direction: Option<Bond>) -> Self {
        let direction = if states & single_like_bond_state_mask() == 0 {
            None
        } else {
            direction
        };
        Self { states, direction }
    }

    /// Matched bond states as a mask of [`bond_state_bit`] values.
    pub(crate) const fn states(self) -> u16 {
        self.states
    }

    pub(crate) const fn direction(self) -> Option<Bond> {
        self.direction
    }

    pub(crate) const fn matches_only_double_bonds(self) -> bool {
        self.states != 0 && self.states & !bond_label_state_mask(BondLabel::Double) == 0
    }

    /// Replaces the direction of a bond that already carries one.
    pub(crate) const fn with_direction(self, direction: Bond) -> Self {
        match self.direction {
            Some(_) => Self::new(self.states, Some(direction)),
            None => self,
        }
    }

    /// Shortest spelling of this meaning, the elided bond when it is the shortest.
    pub(crate) fn to_expr(self) -> BondExpr {
        if self.direction.is_none() && self.states == ELIDED_STATES {
            return BondExpr::Elided;
        }
        let index = spelling_index(self.states);
        let spelling = if self.direction.is_some() {
            spellings::UP[index]
        } else {
            spellings::PLAIN[index]
        };
        let mut tree = parse_bond_expr_text(spelling)
            .unwrap_or_else(|_| unreachable!("bond spellings are valid SMARTS"));
        if self.direction == Some(Bond::Down) {
            flip_tree_directions(&mut tree);
        }
        BondExpr::Query(tree)
    }
}

struct TreeMeaning {
    states: u16,
    first_direction: Option<Bond>,
    negated_direction: bool,
}

impl TreeMeaning {
    fn of(tree: &BondExprTree) -> Self {
        match tree {
            BondExprTree::Primitive(primitive) => Self {
                states: primitive_states(*primitive),
                first_direction: primitive_direction(*primitive),
                negated_direction: false,
            },
            BondExprTree::Not(inner) => {
                let inner = Self::of(inner);
                Self {
                    states: !inner.states & BOND_STATE_MASK_ALL,
                    first_direction: None,
                    negated_direction: inner.negated_direction || inner.first_direction.is_some(),
                }
            }
            BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => {
                Self::fold(items, BOND_STATE_MASK_ALL, |left, right| left & right)
            }
            BondExprTree::Or(items) => Self::fold(items, 0, |left, right| left | right),
        }
    }

    fn fold(items: &[BondExprTree], identity: u16, combine: fn(u16, u16) -> u16) -> Self {
        items.iter().map(Self::of).fold(
            Self {
                states: identity,
                first_direction: None,
                negated_direction: false,
            },
            |acc, item| Self {
                states: combine(acc.states, item.states),
                first_direction: acc.first_direction.or(item.first_direction),
                negated_direction: acc.negated_direction || item.negated_direction,
            },
        )
    }
}

/// Direction of a bond read from its other end.
pub(crate) const fn reversed_direction(direction: Bond) -> Bond {
    match direction {
        Bond::Up => Bond::Down,
        Bond::Down => Bond::Up,
        other => other,
    }
}

/// Swaps `/` and `\`, as reversing the atom order of a bond does.
pub(crate) fn flip_directions(expr: &mut BondExpr) {
    if let BondExpr::Query(tree) = expr {
        flip_tree_directions(tree);
    }
}

/// `expr` of a bond stored from `src` to `dst`, as written from `dst` when
/// `from_dst` holds.
pub(crate) fn bond_expr_written_from(expr: &BondExpr, from_dst: bool) -> Cow<'_, BondExpr> {
    match expr {
        BondExpr::Query(tree) if from_dst && tree_has_direction(tree) => {
            let mut flipped = expr.clone();
            flip_directions(&mut flipped);
            Cow::Owned(flipped)
        }
        _ => Cow::Borrowed(expr),
    }
}

fn tree_has_direction(tree: &BondExprTree) -> bool {
    match tree {
        BondExprTree::Primitive(primitive) => primitive_direction(*primitive).is_some(),
        BondExprTree::Not(inner) => tree_has_direction(inner),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            items.iter().any(tree_has_direction)
        }
    }
}

fn flip_tree_directions(tree: &mut BondExprTree) {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(bond)) => *bond = reversed_direction(*bond),
        BondExprTree::Primitive(_) => {}
        BondExprTree::Not(inner) => flip_tree_directions(inner),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            items.iter_mut().for_each(flip_tree_directions);
        }
    }
}
