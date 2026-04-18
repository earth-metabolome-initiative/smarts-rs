//! Target graph traits and chemistry labels used by the matcher.

use elements_rs::Element;
use smiles_parser::bond::Bond;

/// Dense atom identifier used by the matcher.
pub type AtomId = usize;

/// The chemistry label attached to one atom node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtomLabel {
    /// Chemical element identity.
    pub element: Element,
    /// Whether the atom was marked aromatic in the source representation.
    pub aromatic: bool,
    /// Exact isotope mass number when explicitly present.
    pub isotope: Option<u16>,
    /// Formal charge.
    pub formal_charge: i8,
    /// Explicit hydrogen count carried by the atom record.
    pub explicit_hydrogens: u8,
}

impl AtomLabel {
    /// Creates a minimal atom label for common tests and prototypes.
    #[inline]
    #[must_use]
    pub const fn new(element: Element) -> Self {
        Self {
            element,
            aromatic: false,
            isotope: None,
            formal_charge: 0,
            explicit_hydrogens: 0,
        }
    }
}

/// The chemistry label attached to one bond edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BondLabel {
    /// A single bond.
    Single,
    /// A double bond.
    Double,
    /// A triple bond.
    Triple,
    /// An aromatic bond.
    Aromatic,
    /// A directional up bond.
    Up,
    /// A directional down bond.
    Down,
    /// A wildcard bond.
    Any,
}

impl From<Bond> for BondLabel {
    #[inline]
    fn from(value: Bond) -> Self {
        match value {
            Bond::Single => Self::Single,
            Bond::Double => Self::Double,
            Bond::Triple => Self::Triple,
            Bond::Aromatic => Self::Aromatic,
            Bond::Up => Self::Up,
            Bond::Down => Self::Down,
            Bond::Quadruple => Self::Any,
        }
    }
}

/// A neighboring atom together with the bond used to reach it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Neighbor {
    /// Dense identifier of the neighboring atom.
    pub atom_id: AtomId,
    /// Bond label connecting the current atom to the neighbor.
    pub bond: BondLabel,
}

impl Neighbor {
    /// Creates a neighbor record.
    #[inline]
    #[must_use]
    pub const fn new(atom_id: AtomId, bond: BondLabel) -> Self {
        Self { atom_id, bond }
    }
}

/// Minimal chemistry-oriented view required by SMARTS matching.
pub trait MoleculeTarget {
    /// Iterator returned when traversing neighbors of one atom.
    type Neighbors<'a>: Iterator<Item = Neighbor>
    where
        Self: 'a;

    /// Returns the number of atoms in the target.
    fn atom_count(&self) -> usize;

    /// Returns the label for the provided atom.
    fn atom(&self, atom_id: AtomId) -> Option<&AtomLabel>;

    /// Returns the bond label between two atoms if present.
    fn bond(&self, left_atom: AtomId, right_atom: AtomId) -> Option<BondLabel>;

    /// Returns an iterator over neighbors of the provided atom.
    fn neighbors(&self, atom_id: AtomId) -> Self::Neighbors<'_>;

    /// Returns whether a bond exists between the provided atoms.
    #[inline]
    fn has_bond(&self, left_atom: AtomId, right_atom: AtomId) -> bool {
        self.bond(left_atom, right_atom).is_some()
    }

    /// Returns the degree of the provided atom.
    #[inline]
    fn degree(&self, atom_id: AtomId) -> usize {
        self.neighbors(atom_id).count()
    }

    /// Returns a dense edge identifier if the backend exposes one.
    #[inline]
    fn edge_id(&self, _left_atom: AtomId, _right_atom: AtomId) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use elements_rs::Element;
    use smiles_parser::bond::Bond;

    use super::{AtomId, AtomLabel, BondLabel, MoleculeTarget, Neighbor};

    struct DummyTarget;

    impl MoleculeTarget for DummyTarget {
        type Neighbors<'a>
            = core::iter::Copied<core::slice::Iter<'a, Neighbor>>
        where
            Self: 'a;

        fn atom_count(&self) -> usize {
            2
        }

        fn atom(&self, atom_id: AtomId) -> Option<&AtomLabel> {
            static ATOMS: [AtomLabel; 2] = [AtomLabel::new(Element::C), AtomLabel::new(Element::O)];
            ATOMS.get(atom_id)
        }

        fn bond(&self, left_atom: AtomId, right_atom: AtomId) -> Option<BondLabel> {
            matches!((left_atom, right_atom), (0, 1) | (1, 0)).then_some(BondLabel::Double)
        }

        fn neighbors(&self, atom_id: AtomId) -> Self::Neighbors<'_> {
            static FIRST: [Neighbor; 1] = [Neighbor::new(1, BondLabel::Double)];
            static SECOND: [Neighbor; 1] = [Neighbor::new(0, BondLabel::Double)];
            match atom_id {
                0 => FIRST.iter().copied(),
                1 => SECOND.iter().copied(),
                _ => [].iter().copied(),
            }
        }
    }

    #[test]
    fn atom_labels_and_neighbors_have_expected_defaults() {
        assert_eq!(
            AtomLabel::new(Element::N),
            AtomLabel {
                element: Element::N,
                aromatic: false,
                isotope: None,
                formal_charge: 0,
                explicit_hydrogens: 0,
            }
        );
        assert_eq!(Neighbor::new(7, BondLabel::Triple).atom_id, 7);
        assert_eq!(Neighbor::new(7, BondLabel::Triple).bond, BondLabel::Triple);
    }

    #[test]
    fn bond_label_conversion_covers_remaining_variants() {
        assert_eq!(BondLabel::from(Bond::Aromatic), BondLabel::Aromatic);
        assert_eq!(BondLabel::from(Bond::Down), BondLabel::Down);
        assert_eq!(BondLabel::from(Bond::Quadruple), BondLabel::Any);
    }

    #[test]
    fn default_trait_helpers_cover_has_bond_degree_and_edge_id() {
        let target = DummyTarget;

        assert!(target.has_bond(0, 1));
        assert!(!target.has_bond(0, 0));
        assert_eq!(target.degree(0), 1);
        assert_eq!(target.degree(99), 0);
        assert_eq!(target.edge_id(0, 1), None);
    }
}
