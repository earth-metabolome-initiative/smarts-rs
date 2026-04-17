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
