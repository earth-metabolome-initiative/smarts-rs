//! Molecule target backed by `geometric-traits`.
//!
//! This module wraps the raw graph crate in a chemistry-oriented surface that
//! is much closer to what a SMARTS matcher actually wants.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use geometric_traits::{
    impls::ValuedCSR2D,
    naive_structs::GenericGraph,
    traits::{
        Edges, MatrixMut, MonopartiteGraph, MonoplexGraph, RankSelectSparseMatrix, SparseMatrix2D,
        SparseMatrixMut, SparseValuedMatrix2D, VocabularyRef,
    },
};

use crate::target::{AtomId, AtomLabel, BondLabel, MoleculeTarget, Neighbor};

type RawMoleculeGraph = GenericGraph<Vec<AtomLabel>, ValuedCSR2D<usize, usize, usize, BondLabel>>;
type RawBondMatrix = ValuedCSR2D<usize, usize, usize, BondLabel>;

/// One undirected bond entry used when building a molecule graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UndirectedBond {
    /// First atom index.
    pub left_atom: AtomId,
    /// Second atom index.
    pub right_atom: AtomId,
    /// Bond label.
    pub label: BondLabel,
}

impl UndirectedBond {
    /// Creates a new undirected bond description.
    #[must_use]
    pub fn new(left_atom: AtomId, right_atom: AtomId, label: BondLabel) -> Self {
        Self {
            left_atom,
            right_atom,
            label,
        }
    }
}

/// Errors returned while assembling the molecule graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoleculeGraphError {
    /// A bond referenced an atom index that does not exist.
    AtomOutOfBounds {
        /// The invalid atom index.
        atom_index: AtomId,
        /// The number of atoms available in the graph.
        atom_count: usize,
    },
    /// The bond list could not be inserted into the sparse matrix.
    InvalidBondList {
        /// A text description of the matrix-level error.
        details: String,
    },
}

/// Neighbor iterator for [`MoleculeGraph`].
#[derive(Clone)]
pub struct MoleculeNeighbors<'a> {
    atom_ids: <RawBondMatrix as SparseMatrix2D>::SparseRow<'a>,
    bond_labels: <RawBondMatrix as SparseValuedMatrix2D>::SparseRowValues<'a>,
}

impl Iterator for MoleculeNeighbors<'_> {
    type Item = Neighbor;

    fn next(&mut self) -> Option<Self::Item> {
        self.atom_ids
            .next()
            .zip(self.bond_labels.next())
            .map(|(atom_id, bond)| Neighbor::new(atom_id, bond))
    }
}

impl DoubleEndedIterator for MoleculeNeighbors<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.atom_ids
            .next_back()
            .zip(self.bond_labels.next_back())
            .map(|(atom_id, bond)| Neighbor::new(atom_id, bond))
    }
}

/// Small wrapper around a `geometric-traits` graph storing mirrored bond rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoleculeGraph {
    raw: RawMoleculeGraph,
}

impl MoleculeGraph {
    /// Builds a molecule graph by mirroring each undirected bond into two sparse
    /// matrix entries.
    ///
    /// # Errors
    ///
    /// Returns [`MoleculeGraphError`] when a bond references a missing atom or
    /// when the sparse matrix backend rejects the mirrored bond list.
    pub fn new(
        atoms: Vec<AtomLabel>,
        bonds: impl IntoIterator<Item = UndirectedBond>,
    ) -> Result<Self, MoleculeGraphError> {
        let atom_count = atoms.len();
        let mut entries: Vec<(usize, usize, BondLabel)> = Vec::new();

        for bond in bonds {
            if bond.left_atom >= atom_count {
                return Err(MoleculeGraphError::AtomOutOfBounds {
                    atom_index: bond.left_atom,
                    atom_count,
                });
            }
            if bond.right_atom >= atom_count {
                return Err(MoleculeGraphError::AtomOutOfBounds {
                    atom_index: bond.right_atom,
                    atom_count,
                });
            }

            entries.push((bond.left_atom, bond.right_atom, bond.label));
            if bond.left_atom != bond.right_atom {
                entries.push((bond.right_atom, bond.left_atom, bond.label));
            }
        }

        entries.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.1.cmp(&right.1))
                .then(left.2.cmp(&right.2))
        });

        let mut matrix: RawBondMatrix =
            SparseMatrixMut::with_sparse_shaped_capacity((atom_count, atom_count), entries.len());
        for entry in entries {
            MatrixMut::add(&mut matrix, entry).map_err(|error| {
                MoleculeGraphError::InvalidBondList {
                    details: format!("{error:?}"),
                }
            })?;
        }

        Ok(Self {
            raw: GenericGraph::from((atoms, matrix)),
        })
    }

    /// Returns the wrapped `geometric-traits` graph.
    #[must_use]
    pub fn raw(&self) -> &RawMoleculeGraph {
        &self.raw
    }
}

impl MoleculeTarget for MoleculeGraph {
    type Neighbors<'a>
        = MoleculeNeighbors<'a>
    where
        Self: 'a;

    fn atom_count(&self) -> usize {
        self.raw.number_of_nodes()
    }

    fn atom(&self, atom_id: AtomId) -> Option<&AtomLabel> {
        self.raw.nodes_vocabulary().convert_ref(&atom_id)
    }

    fn bond(&self, left_atom: AtomId, right_atom: AtomId) -> Option<BondLabel> {
        self.raw
            .edges()
            .matrix()
            .sparse_value_at(left_atom, right_atom)
    }

    fn neighbors(&self, atom_id: AtomId) -> Self::Neighbors<'_> {
        MoleculeNeighbors {
            atom_ids: self.raw.successors(atom_id),
            bond_labels: self.raw.edges().matrix().sparse_row_values(atom_id),
        }
    }

    fn edge_id(&self, left_atom: AtomId, right_atom: AtomId) -> Option<usize> {
        let matrix = self.raw.edges().matrix();
        matrix
            .has_entry(left_atom, right_atom)
            .then(|| matrix.rank(&(left_atom, right_atom)))
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use elements_rs::Element;

    use super::{MoleculeGraph, MoleculeGraphError, UndirectedBond};
    use crate::target::{AtomLabel, BondLabel, MoleculeTarget, Neighbor};

    fn ethanol_graph() -> MoleculeGraph {
        MoleculeGraph::new(
            vec![
                AtomLabel {
                    element: Element::C,
                    aromatic: false,
                    isotope: None,
                    formal_charge: 0,
                    explicit_hydrogens: 3,
                },
                AtomLabel {
                    element: Element::C,
                    aromatic: false,
                    isotope: None,
                    formal_charge: 0,
                    explicit_hydrogens: 2,
                },
                AtomLabel {
                    element: Element::O,
                    aromatic: false,
                    isotope: None,
                    formal_charge: 0,
                    explicit_hydrogens: 1,
                },
            ],
            [
                UndirectedBond::new(0, 1, BondLabel::Single),
                UndirectedBond::new(1, 2, BondLabel::Single),
            ],
        )
        .unwrap()
    }

    #[test]
    fn builds_a_small_molecule_graph() {
        let graph = ethanol_graph();

        assert_eq!(graph.atom_count(), 3);
        assert_eq!(graph.atom(2).unwrap().element, Element::O);
        assert_eq!(graph.bond(0, 1), Some(BondLabel::Single));
        assert_eq!(graph.bond(1, 0), Some(BondLabel::Single));
        assert!(graph.has_bond(1, 2));
        assert_eq!(graph.degree(1), 2);

        let neighbors: Vec<Neighbor> = graph.neighbors(1).collect();
        assert_eq!(
            neighbors,
            vec![
                Neighbor::new(0, BondLabel::Single),
                Neighbor::new(2, BondLabel::Single)
            ]
        );
    }

    #[test]
    fn exposes_dense_edge_ids_for_mirrored_bonds() {
        let graph = ethanol_graph();

        assert_eq!(graph.edge_id(0, 1), Some(0));
        assert_eq!(graph.edge_id(1, 0), Some(1));
        assert_eq!(graph.edge_id(0, 2), None);
    }

    #[test]
    fn supports_duplicate_atom_labels_without_artificial_identity() {
        let graph = MoleculeGraph::new(
            vec![AtomLabel::new(Element::C), AtomLabel::new(Element::C)],
            [UndirectedBond::new(0, 1, BondLabel::Single)],
        )
        .unwrap();

        assert_eq!(graph.atom(0), Some(&AtomLabel::new(Element::C)));
        assert_eq!(graph.atom(1), Some(&AtomLabel::new(Element::C)));
    }

    #[test]
    fn rejects_bonds_that_reference_missing_atoms() {
        let error = MoleculeGraph::new(
            vec![AtomLabel::new(Element::C)],
            [UndirectedBond::new(0, 1, BondLabel::Single)],
        )
        .unwrap_err();

        assert_eq!(
            error,
            MoleculeGraphError::AtomOutOfBounds {
                atom_index: 1,
                atom_count: 1,
            }
        );
    }
}
