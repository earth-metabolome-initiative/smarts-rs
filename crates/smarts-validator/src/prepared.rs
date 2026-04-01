//! Prepared molecule sidecars and placeholder prepared targets.

use alloc::vec::Vec;

use crate::target::{AtomId, MoleculeTarget, TargetText};

/// Dense per-atom prepared properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeProps<T> {
    values: Vec<T>,
}

impl<T> NodeProps<T> {
    /// Builds node properties from a dense vector aligned with atom IDs.
    #[inline]
    #[must_use]
    pub fn new(values: Vec<T>) -> Self {
        Self { values }
    }

    /// Returns the property value for the provided atom.
    #[inline]
    #[must_use]
    pub fn get(&self, atom_id: AtomId) -> Option<&T> {
        self.values.get(atom_id)
    }

    /// Returns the number of stored atom properties.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether there are no stored atom properties.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Dense per-edge prepared properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeProps<T> {
    values: Vec<T>,
}

impl<T> EdgeProps<T> {
    /// Builds edge properties from a dense vector aligned with edge IDs.
    #[inline]
    #[must_use]
    pub fn new(values: Vec<T>) -> Self {
        Self { values }
    }

    /// Returns the property value for the provided edge.
    #[inline]
    #[must_use]
    pub fn get(&self, edge_id: usize) -> Option<&T> {
        self.values.get(edge_id)
    }

    /// Returns the number of stored edge properties.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether there are no stored edge properties.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Prepared molecule data ready for SMARTS matching.
///
/// This currently computes only atom degrees, but it establishes the right
/// shape for future preparation passes such as implicit H, ring data, and
/// aromaticity-derived caches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedMolecule<T> {
    target: T,
    degrees: NodeProps<usize>,
}

impl<T: MoleculeTarget> PreparedMolecule<T> {
    /// Prepares a molecule target for matching.
    #[inline]
    #[must_use]
    pub fn new(target: T) -> Self {
        let degrees = NodeProps::new(
            (0..target.atom_count())
                .map(|atom_id| target.degree(atom_id))
                .collect(),
        );
        Self { target, degrees }
    }

    /// Returns the original target.
    #[inline]
    #[must_use]
    pub fn target(&self) -> &T {
        &self.target
    }

    /// Returns the cached atom degree for the provided atom.
    #[inline]
    #[must_use]
    pub fn degree(&self, atom_id: AtomId) -> Option<usize> {
        self.degrees.get(atom_id).copied()
    }

    /// Returns the dense degree sidecar.
    #[inline]
    #[must_use]
    pub fn degrees(&self) -> &NodeProps<usize> {
        &self.degrees
    }
}

/// Target data after preparation for the placeholder string entrypoint.
///
/// This remains separate from [`PreparedMolecule`] because the string-based
/// `matches` API is still just a scaffold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedTarget {
    target: TargetText,
}

impl PreparedTarget {
    /// Wraps the placeholder target after preparation.
    #[inline]
    #[must_use]
    pub fn new(target: TargetText) -> Self {
        Self { target }
    }

    /// Returns the underlying placeholder target.
    #[inline]
    #[must_use]
    pub fn target(&self) -> &TargetText {
        &self.target
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use elements_rs::Element;

    use super::{EdgeProps, NodeProps, PreparedMolecule, PreparedTarget};
    use crate::{
        geometric_target::{MoleculeGraph, UndirectedBond},
        target::{AtomLabel, BondLabel, TargetText},
    };

    #[test]
    fn prepared_target_keeps_original_input() {
        let target = TargetText::new("NCC").unwrap();
        let prepared = PreparedTarget::new(target);
        assert_eq!(prepared.target().as_str(), "NCC");
    }

    #[test]
    fn node_props_are_dense_and_indexed_by_atom_id() {
        let props = NodeProps::new(vec![2usize, 4, 1]);

        assert_eq!(props.len(), 3);
        assert_eq!(props.get(1), Some(&4));
        assert_eq!(props.get(3), None);
    }

    #[test]
    fn edge_props_are_dense_and_indexed_by_edge_id() {
        let props = EdgeProps::new(vec!["a", "b"]);

        assert_eq!(props.len(), 2);
        assert_eq!(props.get(0), Some(&"a"));
        assert_eq!(props.get(2), None);
    }

    #[test]
    fn prepared_molecule_caches_atom_degrees() {
        let target = MoleculeGraph::new(
            vec![
                AtomLabel::new(Element::C),
                AtomLabel::new(Element::C),
                AtomLabel::new(Element::O),
            ],
            [
                UndirectedBond::new(0, 1, BondLabel::Single),
                UndirectedBond::new(1, 2, BondLabel::Single),
            ],
        )
        .unwrap();
        let prepared = PreparedMolecule::new(target);

        assert_eq!(prepared.degree(0), Some(1));
        assert_eq!(prepared.degree(1), Some(2));
        assert_eq!(prepared.degree(2), Some(1));
        assert_eq!(prepared.degrees().len(), 3);
    }
}
