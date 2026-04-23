//! Prepared molecule sidecars and prepared target caches.

use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    vec,
    vec::Vec,
};
use elements_rs::{AtomicNumber, Element};
use smiles_parser::{
    atom::{bracketed::chirality::Chirality, Atom},
    bond::{bond_edge::bond_edge_other, Bond},
    AromaticityAssignment, AromaticityPolicy, DoubleBondStereoConfig, RingMembership, Smiles,
};

use crate::target::{AtomId, BondLabel, MoleculeTarget};

type RingCaches = (
    RingMembership,
    BTreeSet<(AtomId, AtomId)>,
    NodeProps<u8>,
    NodeProps<u8>,
    NodeProps<u8>,
);
type PreparedNeighborList = Box<[(AtomId, BondLabel)]>;
type AtomIdList = Box<[AtomId]>;

#[derive(Debug, Clone)]
struct PreparedAtomCaches {
    aromatic_atoms: NodeProps<bool>,
    effective_formal_charges: NodeProps<i8>,
    hidden_attached_hydrogens: NodeProps<bool>,
    tetrahedral_chiralities: NodeProps<Option<Chirality>>,
    connected_components: NodeProps<usize>,
    degrees: NodeProps<usize>,
    connectivities: NodeProps<u8>,
    implicit_hydrogens: NodeProps<u8>,
    total_hydrogens: NodeProps<u8>,
    total_valences: NodeProps<u8>,
    hybridizations: NodeProps<u8>,
    hetero_neighbor_counts: NodeProps<u8>,
    aliphatic_hetero_neighbor_counts: NodeProps<u8>,
}

#[derive(Debug, Clone)]
struct PreparedAtomIndexes {
    atom_ids_by_atomic_number: BTreeMap<u16, AtomIdList>,
    atom_ids_by_degree: BTreeMap<usize, AtomIdList>,
    atom_ids_by_total_hydrogen: BTreeMap<u8, AtomIdList>,
    aromatic_atom_ids: AtomIdList,
    aliphatic_atom_ids: AtomIdList,
    ring_atom_ids: AtomIdList,
}

/// Dense per-atom prepared properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeProps<T> {
    values: Vec<T>,
}

impl<T> NodeProps<T> {
    /// Builds node properties from a dense vector aligned with atom IDs.
    #[inline]
    #[must_use]
    pub const fn new(values: Vec<T>) -> Self {
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
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether there are no stored atom properties.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
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
    pub const fn new(values: Vec<T>) -> Self {
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
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether there are no stored edge properties.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
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
    pub const fn target(&self) -> &T {
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
    pub const fn degrees(&self) -> &NodeProps<usize> {
        &self.degrees
    }
}

/// Prepared target data for a parsed `SMILES` molecule.
///
/// This is the first real target backend for `smarts-rs`. It keeps the
/// parsed [`Smiles`] graph plus the first cached properties required for
/// single-atom SMARTS evaluation.
#[derive(Debug, Clone)]
pub struct PreparedTarget {
    target: Smiles,
    aromaticity: AromaticityAssignment,
    aromatic_atoms: NodeProps<bool>,
    effective_formal_charges: NodeProps<i8>,
    effective_neighbors: NodeProps<PreparedNeighborList>,
    hidden_attached_hydrogens: NodeProps<bool>,
    ring_membership: RingMembership,
    ring_bonds: BTreeSet<(AtomId, AtomId)>,
    tetrahedral_chiralities: NodeProps<Option<Chirality>>,
    double_bond_stereo: BTreeMap<(AtomId, AtomId), DoubleBondStereoConfig>,
    connected_components: NodeProps<usize>,
    degrees: NodeProps<usize>,
    atom_ids_by_atomic_number: BTreeMap<u16, AtomIdList>,
    atom_ids_by_degree: BTreeMap<usize, AtomIdList>,
    atom_ids_by_total_hydrogen: BTreeMap<u8, AtomIdList>,
    aromatic_atom_ids: AtomIdList,
    aliphatic_atom_ids: AtomIdList,
    ring_atom_ids: AtomIdList,
    connectivities: NodeProps<u8>,
    implicit_hydrogens: NodeProps<u8>,
    total_hydrogens: NodeProps<u8>,
    total_valences: NodeProps<u8>,
    hybridizations: NodeProps<u8>,
    hetero_neighbor_counts: NodeProps<u8>,
    aliphatic_hetero_neighbor_counts: NodeProps<u8>,
    ring_membership_counts: NodeProps<u8>,
    ring_bond_counts: NodeProps<u8>,
    smallest_ring_sizes: NodeProps<u8>,
}

impl PreparedTarget {
    /// Prepares a parsed target molecule for SMARTS matching.
    #[must_use]
    pub fn new(target: Smiles) -> Self {
        let atom_count = target.nodes().len();
        let aromaticity = target.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        let (
            ring_membership,
            ring_bonds,
            ring_bond_counts,
            ring_membership_counts,
            smallest_ring_sizes,
        ) = ring_caches(&target);
        let atom_caches = prepared_atom_caches(&target, &aromaticity);
        let atom_indexes =
            prepared_atom_indexes(atom_count, &target, &ring_membership, &atom_caches);
        let effective_neighbors = effective_neighbor_cache(&target, &aromaticity);
        let double_bond_stereo = semantic_double_bond_stereo_configs(&target);
        let PreparedAtomCaches {
            aromatic_atoms,
            effective_formal_charges,
            hidden_attached_hydrogens,
            tetrahedral_chiralities,
            connected_components,
            degrees,
            connectivities,
            implicit_hydrogens,
            total_hydrogens,
            total_valences,
            hybridizations,
            hetero_neighbor_counts,
            aliphatic_hetero_neighbor_counts,
        } = atom_caches;
        let PreparedAtomIndexes {
            atom_ids_by_atomic_number,
            atom_ids_by_degree,
            atom_ids_by_total_hydrogen,
            aromatic_atom_ids,
            aliphatic_atom_ids,
            ring_atom_ids,
        } = atom_indexes;

        Self {
            target,
            aromaticity,
            aromatic_atoms,
            effective_formal_charges,
            effective_neighbors,
            hidden_attached_hydrogens,
            ring_membership,
            ring_bonds,
            tetrahedral_chiralities,
            double_bond_stereo,
            connected_components,
            degrees,
            atom_ids_by_atomic_number,
            atom_ids_by_degree,
            atom_ids_by_total_hydrogen,
            aromatic_atom_ids,
            aliphatic_atom_ids,
            ring_atom_ids,
            connectivities,
            implicit_hydrogens,
            total_hydrogens,
            total_valences,
            hybridizations,
            hetero_neighbor_counts,
            aliphatic_hetero_neighbor_counts,
            ring_membership_counts,
            ring_bond_counts,
            smallest_ring_sizes,
        }
    }

    /// Returns the underlying parsed target molecule.
    #[inline]
    #[must_use]
    pub const fn target(&self) -> &Smiles {
        &self.target
    }

    /// Returns the number of atoms in the prepared target.
    #[inline]
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.target.nodes().len()
    }

    /// Returns the parsed atom for the provided atom id.
    #[inline]
    #[must_use]
    pub fn atom(&self, atom_id: AtomId) -> Option<&Atom> {
        self.target.node_by_id(atom_id)
    }

    /// Returns the effective bond label between two atoms, taking the cached
    /// aromaticity assignment into account.
    #[inline]
    #[must_use]
    pub fn bond(&self, left_atom: AtomId, right_atom: AtomId) -> Option<BondLabel> {
        let left_neighbors = self.effective_neighbors.get(left_atom)?;
        let right_neighbors = self.effective_neighbors.get(right_atom)?;
        let (neighbors, target_atom) = if left_neighbors.len() <= right_neighbors.len() {
            (left_neighbors, right_atom)
        } else {
            (right_neighbors, left_atom)
        };

        neighbors
            .iter()
            .find_map(|&(neighbor_atom, label)| (neighbor_atom == target_atom).then_some(label))
    }

    /// Returns a zero-allocation iterator over neighbors of one atom together
    /// with their effective bond labels.
    #[inline]
    pub(crate) fn neighbors(
        &self,
        atom_id: AtomId,
    ) -> impl Iterator<Item = (AtomId, BondLabel)> + '_ {
        self.effective_neighbors.values[atom_id].iter().copied()
    }

    /// Returns whether the provided bond is aromatic under the prepared view.
    #[inline]
    #[must_use]
    pub fn is_aromatic_bond(&self, left_atom: AtomId, right_atom: AtomId) -> bool {
        self.aromaticity.contains_edge(left_atom, right_atom)
            || self
                .target
                .edge_for_node_pair((left_atom, right_atom))
                .is_some_and(|edge| edge.2 == Bond::Aromatic)
    }

    /// Returns whether the provided atom is aromatic under the RDKit-default
    /// aromaticity assignment.
    #[inline]
    #[must_use]
    pub fn is_aromatic(&self, atom_id: AtomId) -> bool {
        self.aromatic_atoms.get(atom_id).copied().unwrap_or(false)
    }

    /// Returns the effective formal charge used for SMARTS matching.
    #[inline]
    #[must_use]
    pub fn formal_charge(&self, atom_id: AtomId) -> Option<i8> {
        self.effective_formal_charges.get(atom_id).copied()
    }

    /// Returns whether an explicit hydrogen is hidden for SMARTS matching.
    #[inline]
    #[must_use]
    pub(crate) fn is_hidden_attached_hydrogen(&self, atom_id: AtomId) -> bool {
        self.hidden_attached_hydrogens
            .get(atom_id)
            .copied()
            .unwrap_or(false)
    }

    /// Returns whether the provided atom belongs to at least one ring.
    #[inline]
    #[must_use]
    pub fn is_ring_atom(&self, atom_id: AtomId) -> bool {
        self.ring_membership.contains_atom(atom_id)
    }

    /// Returns whether the provided bond belongs to at least one ring.
    #[inline]
    #[must_use]
    pub fn is_ring_bond(&self, left_atom: AtomId, right_atom: AtomId) -> bool {
        self.ring_bonds.contains(&edge_key(left_atom, right_atom))
    }

    /// Returns the semantic SMARTS-facing tetrahedral chirality of the atom.
    #[inline]
    #[must_use]
    pub fn tetrahedral_chirality(&self, atom_id: AtomId) -> Option<Chirality> {
        self.tetrahedral_chiralities.get(atom_id).copied().flatten()
    }

    /// Returns the semantic SMARTS-facing double-bond stereo configuration.
    #[inline]
    #[must_use]
    pub fn double_bond_stereo_config(
        &self,
        left_atom: AtomId,
        right_atom: AtomId,
    ) -> Option<DoubleBondStereoConfig> {
        self.double_bond_stereo
            .get(&edge_key(left_atom, right_atom))
            .copied()
    }

    /// Returns the connected-component identifier for the provided atom.
    #[inline]
    #[must_use]
    pub fn connected_component(&self, atom_id: AtomId) -> Option<usize> {
        self.connected_components.get(atom_id).copied()
    }

    /// Returns the cached topological degree for the provided atom.
    #[inline]
    #[must_use]
    pub fn degree(&self, atom_id: AtomId) -> Option<usize> {
        self.degrees.get(atom_id).copied()
    }

    /// Returns atom IDs with the provided atomic number.
    #[inline]
    #[must_use]
    pub(crate) fn atom_ids_with_atomic_number(&self, atomic_number: u16) -> &[AtomId] {
        self.atom_ids_by_atomic_number
            .get(&atomic_number)
            .map_or(&[], Box::as_ref)
    }

    /// Returns atom IDs with the provided topological degree.
    #[inline]
    #[must_use]
    pub(crate) fn atom_ids_with_degree(&self, degree: usize) -> &[AtomId] {
        self.atom_ids_by_degree
            .get(&degree)
            .map_or(&[], Box::as_ref)
    }

    /// Returns atom IDs with the provided total hydrogen count.
    #[inline]
    #[must_use]
    pub(crate) fn atom_ids_with_total_hydrogen(&self, count: u8) -> &[AtomId] {
        self.atom_ids_by_total_hydrogen
            .get(&count)
            .map_or(&[], Box::as_ref)
    }

    /// Returns atom IDs that are aromatic in the prepared target view.
    #[inline]
    #[must_use]
    pub(crate) fn aromatic_atom_ids(&self) -> &[AtomId] {
        &self.aromatic_atom_ids
    }

    /// Returns atom IDs that are not aromatic in the prepared target view.
    #[inline]
    #[must_use]
    pub(crate) fn aliphatic_atom_ids(&self) -> &[AtomId] {
        &self.aliphatic_atom_ids
    }

    /// Returns atom IDs that belong to at least one ring.
    #[inline]
    #[must_use]
    pub(crate) fn ring_atom_ids(&self) -> &[AtomId] {
        &self.ring_atom_ids
    }

    /// Returns the SMARTS connectivity count for the provided atom.
    ///
    /// This is the RDKit-style `X` value:
    /// topological degree plus bracket-explicit hydrogens plus implicit
    /// hydrogens. Explicit hydrogen neighbor nodes are already included in the
    /// topological degree and are not counted twice.
    #[inline]
    #[must_use]
    pub fn connectivity(&self, atom_id: AtomId) -> Option<u8> {
        self.connectivities.get(atom_id).copied()
    }

    /// Returns the cached implicit hydrogen count for the provided atom.
    #[inline]
    #[must_use]
    pub fn implicit_hydrogen_count(&self, atom_id: AtomId) -> Option<u8> {
        self.implicit_hydrogens.get(atom_id).copied()
    }

    /// Returns the cached total hydrogen count for the provided atom.
    #[inline]
    #[must_use]
    pub fn total_hydrogen_count(&self, atom_id: AtomId) -> Option<u8> {
        self.total_hydrogens.get(atom_id).copied()
    }

    /// Returns the RDKit-style total valence for the provided atom.
    #[inline]
    #[must_use]
    pub fn total_valence(&self, atom_id: AtomId) -> Option<u8> {
        self.total_valences.get(atom_id).copied()
    }

    /// Returns the RDKit-style SMARTS hybridization code for the atom.
    #[inline]
    #[must_use]
    pub fn hybridization(&self, atom_id: AtomId) -> Option<u8> {
        self.hybridizations.get(atom_id).copied()
    }

    /// Returns the number of non-carbon, non-hydrogen neighbors.
    #[inline]
    #[must_use]
    pub fn hetero_neighbor_count(&self, atom_id: AtomId) -> Option<u8> {
        self.hetero_neighbor_counts.get(atom_id).copied()
    }

    /// Returns the number of aliphatic non-carbon, non-hydrogen neighbors.
    #[inline]
    #[must_use]
    pub fn aliphatic_hetero_neighbor_count(&self, atom_id: AtomId) -> Option<u8> {
        self.aliphatic_hetero_neighbor_counts.get(atom_id).copied()
    }

    /// Returns the number of symmetrized-SSSR rings containing the atom.
    #[inline]
    #[must_use]
    pub fn ring_membership_count(&self, atom_id: AtomId) -> Option<u8> {
        self.ring_membership_counts.get(atom_id).copied()
    }

    /// Returns the number of ring bonds incident to the atom.
    #[inline]
    #[must_use]
    pub fn ring_bond_count(&self, atom_id: AtomId) -> Option<u8> {
        self.ring_bond_counts.get(atom_id).copied()
    }

    /// Returns the smallest symmetrized-SSSR ring size containing the atom.
    #[inline]
    #[must_use]
    pub fn smallest_ring_size(&self, atom_id: AtomId) -> Option<u8> {
        self.smallest_ring_sizes.get(atom_id).copied()
    }
}

fn prepared_atom_caches(
    target: &Smiles,
    aromaticity: &AromaticityAssignment,
) -> PreparedAtomCaches {
    let atom_count = target.nodes().len();
    PreparedAtomCaches {
        aromatic_atoms: node_props_by_atom(atom_count, |atom_id| {
            atom_is_aromatic_for_prepared_view(target, aromaticity, atom_id)
        }),
        effective_formal_charges: node_props_by_atom(atom_count, |atom_id| {
            effective_formal_charge(target, aromaticity, atom_id)
        }),
        hidden_attached_hydrogens: NodeProps::new(hidden_attached_hydrogen_flags(target)),
        tetrahedral_chiralities: node_props_by_atom(atom_count, |atom_id| {
            target.smarts_tetrahedral_chirality(atom_id)
        }),
        connected_components: connected_component_ids(target),
        degrees: node_props_by_atom(atom_count, |atom_id| target.edge_count_for_node(atom_id)),
        connectivities: node_props_by_atom(atom_count, |atom_id| {
            target.connectivity_count(atom_id)
        }),
        implicit_hydrogens: node_props_by_atom(atom_count, |atom_id| {
            target.implicit_hydrogen_count(atom_id)
        }),
        total_hydrogens: total_hydrogen_cache(target),
        total_valences: node_props_by_atom(atom_count, |atom_id| {
            target.smarts_total_valence(atom_id, aromaticity)
        }),
        hybridizations: node_props_by_atom(atom_count, |atom_id| {
            hybridization_code(target, aromaticity, atom_id)
        }),
        hetero_neighbor_counts: node_props_by_atom(atom_count, |atom_id| {
            hetero_neighbor_count(target, atom_id)
        }),
        aliphatic_hetero_neighbor_counts: node_props_by_atom(atom_count, |atom_id| {
            aliphatic_hetero_neighbor_count(target, aromaticity, atom_id)
        }),
    }
}

fn prepared_atom_indexes(
    atom_count: usize,
    target: &Smiles,
    ring_membership: &RingMembership,
    caches: &PreparedAtomCaches,
) -> PreparedAtomIndexes {
    PreparedAtomIndexes {
        atom_ids_by_atomic_number: atom_id_index_by(atom_count, |atom_id| {
            target
                .node_by_id(atom_id)
                .and_then(Atom::element)
                .map(|element| u16::from(element.atomic_number()))
        }),
        atom_ids_by_degree: atom_id_index_by(atom_count, |atom_id| {
            caches.degrees.get(atom_id).copied()
        }),
        atom_ids_by_total_hydrogen: atom_id_index_by(atom_count, |atom_id| {
            caches.total_hydrogens.get(atom_id).copied()
        }),
        aromatic_atom_ids: atom_ids_where(atom_count, |atom_id| {
            caches.aromatic_atoms.get(atom_id).copied().unwrap_or(false)
        }),
        aliphatic_atom_ids: atom_ids_where(atom_count, |atom_id| {
            !caches.aromatic_atoms.get(atom_id).copied().unwrap_or(false)
        }),
        ring_atom_ids: atom_ids_where(atom_count, |atom_id| ring_membership.contains_atom(atom_id)),
    }
}

fn node_props_by_atom<T>(
    atom_count: usize,
    value_for_atom: impl FnMut(AtomId) -> T,
) -> NodeProps<T> {
    NodeProps::new((0..atom_count).map(value_for_atom).collect())
}

fn total_hydrogen_cache(target: &Smiles) -> NodeProps<u8> {
    NodeProps::new(
        target
            .nodes()
            .iter()
            .enumerate()
            .map(|(atom_id, atom)| {
                let neighbor_hydrogens = explicit_hydrogen_neighbor_count(target, atom_id);
                atom.hydrogen_count()
                    .checked_add(target.implicit_hydrogen_count(atom_id))
                    .and_then(|value| value.checked_add(neighbor_hydrogens))
                    .unwrap_or(u8::MAX)
            })
            .collect(),
    )
}

fn atom_id_index_by<K: Ord>(
    atom_count: usize,
    mut key_for_atom: impl FnMut(AtomId) -> Option<K>,
) -> BTreeMap<K, AtomIdList> {
    let mut buckets: BTreeMap<K, Vec<AtomId>> = BTreeMap::new();
    for atom_id in 0..atom_count {
        if let Some(key) = key_for_atom(atom_id) {
            buckets.entry(key).or_default().push(atom_id);
        }
    }
    buckets
        .into_iter()
        .map(|(key, atoms)| (key, atoms.into_boxed_slice()))
        .collect()
}

fn atom_ids_where(atom_count: usize, mut predicate: impl FnMut(AtomId) -> bool) -> AtomIdList {
    (0..atom_count)
        .filter(|&atom_id| predicate(atom_id))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn explicit_hydrogen_neighbor_count(target: &Smiles, atom_id: usize) -> u8 {
    target
        .edges_for_node(atom_id)
        .filter_map(|edge| bond_edge_other(edge, atom_id))
        .filter(|&neighbor_id| {
            target
                .node_by_id(neighbor_id)
                .and_then(Atom::element)
                .is_some_and(|element| element == Element::H)
        })
        .count()
        .try_into()
        .unwrap_or(u8::MAX)
}

fn hidden_attached_hydrogen_flags(target: &Smiles) -> Vec<bool> {
    (0..target.nodes().len())
        .map(|atom_id| parsed_atom_is_hidden_attached_hydrogen(target, atom_id))
        .collect()
}

fn parsed_atom_is_hidden_attached_hydrogen(target: &Smiles, atom_id: usize) -> bool {
    let Some(atom) = target.node_by_id(atom_id) else {
        return false;
    };
    if atom.element() != Some(Element::H)
        || atom.isotope_mass_number().is_some()
        || atom.charge_value() != 0
        || target.edge_count_for_node(atom_id) != 1
    {
        return false;
    }

    let mut neighbors = target
        .edges_for_node(atom_id)
        .filter_map(|edge| bond_edge_other(edge, atom_id));
    let Some(neighbor_id) = neighbors.next() else {
        return false;
    };
    if neighbors.next().is_some() {
        return false;
    }

    target
        .node_by_id(neighbor_id)
        .and_then(Atom::element)
        .is_some_and(|element| element != Element::H)
}

fn hybridization_code(target: &Smiles, aromaticity: &AromaticityAssignment, atom_id: usize) -> u8 {
    let Some(atom) = target.node_by_id(atom_id) else {
        return 0;
    };
    if atom.element().is_none() {
        return 0;
    }
    if atom_is_aromatic_for_prepared_view(target, aromaticity, atom_id) {
        return 2;
    }

    let mut double_bonds = 0u8;
    let mut has_triple = false;
    for edge in target.edges_for_node(atom_id) {
        match normalized_bond(edge.2) {
            Bond::Double => double_bonds = double_bonds.saturating_add(1),
            Bond::Triple | Bond::Quadruple => has_triple = true,
            Bond::Single | Bond::Up | Bond::Down | Bond::Aromatic => {}
        }
    }

    if has_triple || double_bonds >= 2 {
        return 1;
    }
    if double_bonds == 1 {
        return 2;
    }
    if atom_has_conjugated_lone_pair(target, aromaticity, atom_id) {
        return 2;
    }
    3
}

fn atom_has_conjugated_lone_pair(
    target: &Smiles,
    aromaticity: &AromaticityAssignment,
    atom_id: usize,
) -> bool {
    let Some(element) = target.node_by_id(atom_id).and_then(Atom::element) else {
        return false;
    };
    if !matches!(element, Element::N | Element::O | Element::P | Element::S) {
        return false;
    }

    target.edges_for_node(atom_id).any(|edge| {
        normalized_bond(edge.2) == Bond::Single
            && bond_edge_other(edge, atom_id).is_some_and(|neighbor_id| {
                neighbor_supports_conjugation(target, aromaticity, atom_id, neighbor_id)
            })
    })
}

fn neighbor_supports_conjugation(
    target: &Smiles,
    aromaticity: &AromaticityAssignment,
    atom_id: usize,
    neighbor_id: usize,
) -> bool {
    atom_is_aromatic_for_prepared_view(target, aromaticity, neighbor_id)
        || target.edges_for_node(neighbor_id).any(|edge| {
            bond_edge_other(edge, neighbor_id).is_some_and(|other_id| other_id != atom_id)
                && matches!(
                    normalized_bond(edge.2),
                    Bond::Double | Bond::Triple | Bond::Quadruple
                )
        })
}

const fn normalized_bond(bond: Bond) -> Bond {
    match bond {
        Bond::Up | Bond::Down => Bond::Single,
        Bond::Single | Bond::Double | Bond::Triple | Bond::Quadruple | Bond::Aromatic => bond,
    }
}

fn effective_bond_label(
    target: &Smiles,
    aromaticity: &AromaticityAssignment,
    left_atom: AtomId,
    right_atom: AtomId,
    raw_bond: Bond,
) -> BondLabel {
    if rdkit_like_oxyhalogen_terminal_oxo_bond(target, left_atom, right_atom, raw_bond) {
        return BondLabel::Single;
    }
    if rdkit_like_phosphorus_terminal_oxo_bond(target, aromaticity, left_atom, right_atom, raw_bond)
    {
        return BondLabel::Single;
    }
    if aromaticity.contains_edge(left_atom, right_atom)
        && matches!(
            normalized_bond(raw_bond),
            Bond::Single | Bond::Double | Bond::Aromatic
        )
    {
        BondLabel::Aromatic
    } else {
        BondLabel::from(raw_bond)
    }
}

fn rdkit_like_oxyhalogen_terminal_oxo_bond(
    target: &Smiles,
    left_atom: AtomId,
    right_atom: AtomId,
    raw_bond: Bond,
) -> bool {
    if normalized_bond(raw_bond) != Bond::Double {
        return false;
    }

    let Some((oxygen_atom, halogen_atom)) =
        identify_terminal_oxyhalogen_pair(target, left_atom, right_atom)
    else {
        return false;
    };

    let mut saw_other_oxygen = false;
    let mut saw_non_oxygen_heavy_neighbor = false;

    for edge in target.edges_for_node(halogen_atom) {
        let Some(neighbor_atom) = bond_edge_other(edge, halogen_atom) else {
            continue;
        };
        if neighbor_atom == oxygen_atom {
            continue;
        }
        let Some(element) = target.node_by_id(neighbor_atom).and_then(Atom::element) else {
            continue;
        };
        if element == Element::O {
            saw_other_oxygen = true;
        } else if element != Element::H {
            saw_non_oxygen_heavy_neighbor = true;
        }
    }

    !saw_non_oxygen_heavy_neighbor
        && (saw_other_oxygen
            || target
                .node_by_id(halogen_atom)
                .is_some_and(|atom| atom.hydrogen_count() > 0))
}

fn identify_terminal_oxyhalogen_pair(
    target: &Smiles,
    left_atom: AtomId,
    right_atom: AtomId,
) -> Option<(AtomId, AtomId)> {
    let left_element = target.node_by_id(left_atom)?.element()?;
    let right_element = target.node_by_id(right_atom)?.element()?;

    if left_element == Element::O
        && target.edges_for_node(left_atom).count() == 1
        && matches!(right_element, Element::Cl | Element::Br | Element::I)
    {
        return Some((left_atom, right_atom));
    }

    if right_element == Element::O
        && target.edges_for_node(right_atom).count() == 1
        && matches!(left_element, Element::Cl | Element::Br | Element::I)
    {
        return Some((right_atom, left_atom));
    }

    None
}

fn effective_formal_charge(
    target: &Smiles,
    aromaticity: &AromaticityAssignment,
    atom_id: AtomId,
) -> i8 {
    let Some(atom) = target.node_by_id(atom_id) else {
        return 0;
    };
    let base_charge = atom.charge_value();

    if atom
        .element()
        .is_some_and(|element| matches!(element, Element::Cl | Element::Br | Element::I))
    {
        return base_charge.saturating_add(
            i8::try_from(terminal_oxyhalogen_oxo_bond_count(target, atom_id)).unwrap_or(i8::MAX),
        );
    }

    if atom.element() == Some(Element::O)
        && target.edges_for_node(atom_id).count() == 1
        && target.edges_for_node(atom_id).any(|edge| {
            bond_edge_other(edge, atom_id).is_some_and(|neighbor_atom| {
                rdkit_like_oxyhalogen_terminal_oxo_bond(target, atom_id, neighbor_atom, edge.2)
            })
        })
    {
        return base_charge.saturating_sub(1);
    }

    if atom.element() == Some(Element::P) {
        return base_charge.saturating_add(
            i8::try_from(terminal_phosphorus_oxo_bond_count(
                target,
                aromaticity,
                atom_id,
            ))
            .unwrap_or(i8::MAX),
        );
    }

    if atom.element() == Some(Element::O)
        && target.edges_for_node(atom_id).count() == 1
        && target.edges_for_node(atom_id).any(|edge| {
            bond_edge_other(edge, atom_id).is_some_and(|neighbor_atom| {
                rdkit_like_phosphorus_terminal_oxo_bond(
                    target,
                    aromaticity,
                    atom_id,
                    neighbor_atom,
                    edge.2,
                )
            })
        })
    {
        return base_charge.saturating_sub(1);
    }

    base_charge
}

fn terminal_oxyhalogen_oxo_bond_count(target: &Smiles, atom_id: AtomId) -> usize {
    target
        .edges_for_node(atom_id)
        .filter(|edge| {
            bond_edge_other(*edge, atom_id).is_some_and(|neighbor_atom| {
                rdkit_like_oxyhalogen_terminal_oxo_bond(target, atom_id, neighbor_atom, edge.2)
            })
        })
        .count()
}

fn rdkit_like_phosphorus_terminal_oxo_bond(
    target: &Smiles,
    _aromaticity: &AromaticityAssignment,
    left_atom: AtomId,
    right_atom: AtomId,
    raw_bond: Bond,
) -> bool {
    if normalized_bond(raw_bond) != Bond::Double {
        return false;
    }

    let Some((oxygen_atom, phosphorus_atom)) =
        identify_terminal_phosphorus_oxo_pair(target, left_atom, right_atom)
    else {
        return false;
    };

    if target
        .node_by_id(phosphorus_atom)
        .is_some_and(|atom| atom.charge_value() != 0)
    {
        return false;
    }

    let mut other_double_neighbor = None;
    let mut single_neighbors = Vec::new();

    for edge in target.edges_for_node(phosphorus_atom) {
        let Some(neighbor_atom) = bond_edge_other(edge, phosphorus_atom) else {
            continue;
        };
        if neighbor_atom == oxygen_atom {
            continue;
        }

        match normalized_bond(edge.2) {
            Bond::Double => {
                if other_double_neighbor.is_some() {
                    return false;
                }
                other_double_neighbor = Some(neighbor_atom);
            }
            Bond::Single => single_neighbors.push(neighbor_atom),
            Bond::Triple | Bond::Quadruple | Bond::Up | Bond::Down | Bond::Aromatic => {
                return false
            }
        }
    }

    let Some(other_double_neighbor) = other_double_neighbor else {
        return false;
    };
    let Some(other_double_element) = target
        .node_by_id(other_double_neighbor)
        .and_then(Atom::element)
    else {
        return false;
    };
    if single_neighbors.is_empty() {
        return false;
    }

    match other_double_element {
        Element::N => target.edges_for_node(other_double_neighbor).any(|edge| {
            bond_edge_other(edge, other_double_neighbor).is_some_and(|neighbor_atom| {
                neighbor_atom != phosphorus_atom
                    && target
                        .node_by_id(neighbor_atom)
                        .and_then(Atom::element)
                        .is_some_and(|element| element != Element::H)
            })
        }),
        Element::C => target.edges_for_node(other_double_neighbor).any(|edge| {
            bond_edge_other(edge, other_double_neighbor).is_some_and(|neighbor_atom| {
                neighbor_atom != phosphorus_atom
                    && target
                        .node_by_id(neighbor_atom)
                        .and_then(Atom::element)
                        .is_some_and(|element| element != Element::H)
            })
        }),
        _ => false,
    }
}

fn identify_terminal_phosphorus_oxo_pair(
    target: &Smiles,
    left_atom: AtomId,
    right_atom: AtomId,
) -> Option<(AtomId, AtomId)> {
    let left_element = target.node_by_id(left_atom)?.element()?;
    let right_element = target.node_by_id(right_atom)?.element()?;

    if left_element == Element::O
        && target.edges_for_node(left_atom).count() == 1
        && right_element == Element::P
    {
        return Some((left_atom, right_atom));
    }

    if right_element == Element::O
        && target.edges_for_node(right_atom).count() == 1
        && left_element == Element::P
    {
        return Some((right_atom, left_atom));
    }

    None
}

fn terminal_phosphorus_oxo_bond_count(
    target: &Smiles,
    aromaticity: &AromaticityAssignment,
    atom_id: AtomId,
) -> usize {
    target
        .edges_for_node(atom_id)
        .filter(|edge| {
            bond_edge_other(*edge, atom_id).is_some_and(|neighbor_atom| {
                rdkit_like_phosphorus_terminal_oxo_bond(
                    target,
                    aromaticity,
                    atom_id,
                    neighbor_atom,
                    edge.2,
                )
            })
        })
        .count()
}

fn atom_is_aromatic_for_prepared_view(
    target: &Smiles,
    aromaticity: &AromaticityAssignment,
    atom_id: usize,
) -> bool {
    aromaticity.contains_atom(atom_id) || target.node_by_id(atom_id).is_some_and(Atom::aromatic)
}

fn hetero_neighbor_count(target: &Smiles, atom_id: usize) -> u8 {
    target
        .edges_for_node(atom_id)
        .filter_map(|edge| bond_edge_other(edge, atom_id))
        .filter(|&neighbor_id| {
            target
                .node_by_id(neighbor_id)
                .and_then(Atom::element)
                .is_some_and(|element| !matches!(element, Element::C | Element::H))
        })
        .count()
        .try_into()
        .unwrap_or(u8::MAX)
}

fn aliphatic_hetero_neighbor_count(
    target: &Smiles,
    aromaticity: &AromaticityAssignment,
    atom_id: usize,
) -> u8 {
    target
        .edges_for_node(atom_id)
        .filter_map(|edge| bond_edge_other(edge, atom_id))
        .filter(|&neighbor_id| {
            target
                .node_by_id(neighbor_id)
                .and_then(Atom::element)
                .is_some_and(|element| !matches!(element, Element::C | Element::H))
                && !atom_is_aromatic_for_prepared_view(target, aromaticity, neighbor_id)
        })
        .count()
        .try_into()
        .unwrap_or(u8::MAX)
}

fn ring_bonds_from_cycles(cycles: &[Vec<usize>]) -> BTreeSet<(AtomId, AtomId)> {
    let mut ring_bonds = BTreeSet::new();
    for cycle in cycles {
        if cycle.len() < 2 {
            continue;
        }
        for edge in cycle.windows(2) {
            ring_bonds.insert(edge_key(edge[0], edge[1]));
        }
        ring_bonds.insert(edge_key(cycle[cycle.len() - 1], cycle[0]));
    }
    ring_bonds
}

fn ring_caches(target: &Smiles) -> RingCaches {
    let ring_membership = target.ring_membership();
    let symm_sssr = target.symm_sssr_result();
    let ring_bonds = ring_bonds_from_cycles(symm_sssr.cycles());
    let ring_bond_counts = NodeProps::new(
        (0..target.nodes().len())
            .map(|atom_id| ring_bond_count(&ring_bonds, atom_id))
            .collect(),
    );
    let (ring_membership_counts, smallest_ring_sizes) =
        ring_cycle_properties(target.nodes().len(), symm_sssr.cycles());
    (
        ring_membership,
        ring_bonds,
        ring_bond_counts,
        ring_membership_counts,
        smallest_ring_sizes,
    )
}

fn ring_bond_count(ring_bonds: &BTreeSet<(AtomId, AtomId)>, atom_id: usize) -> u8 {
    ring_bonds
        .iter()
        .filter(|edge| edge.0 == atom_id || edge.1 == atom_id)
        .count()
        .try_into()
        .unwrap_or(u8::MAX)
}

fn ring_cycle_properties(
    atom_count: usize,
    cycles: &[Vec<usize>],
) -> (NodeProps<u8>, NodeProps<u8>) {
    let mut ring_counts = vec![0_u8; atom_count];
    let mut smallest_sizes = vec![0_u8; atom_count];

    for cycle in cycles {
        let cycle_len = u8::try_from(cycle.len()).unwrap_or(u8::MAX);
        for &atom_id in cycle {
            ring_counts[atom_id] = ring_counts[atom_id].saturating_add(1);
            let current = &mut smallest_sizes[atom_id];
            if *current == 0 || cycle_len < *current {
                *current = cycle_len;
            }
        }
    }

    (NodeProps::new(ring_counts), NodeProps::new(smallest_sizes))
}

fn connected_component_ids(target: &Smiles) -> NodeProps<usize> {
    let atom_count = target.nodes().len();
    let mut components = vec![usize::MAX; atom_count];
    let mut next_component = 0usize;
    let mut stack = Vec::new();

    for start_atom in 0..atom_count {
        if components[start_atom] != usize::MAX {
            continue;
        }

        components[start_atom] = next_component;
        stack.push(start_atom);

        while let Some(atom_id) = stack.pop() {
            for edge in target.edges_for_node(atom_id) {
                let Some(other_atom) = bond_edge_other(edge, atom_id) else {
                    continue;
                };
                if components[other_atom] == usize::MAX {
                    components[other_atom] = next_component;
                    stack.push(other_atom);
                }
            }
        }

        next_component += 1;
    }

    NodeProps::new(components)
}

fn effective_neighbor_cache(
    target: &Smiles,
    aromaticity: &AromaticityAssignment,
) -> NodeProps<PreparedNeighborList> {
    let atom_count = target.nodes().len();
    let mut neighbors = vec![Vec::new(); atom_count];

    for atom_id in 0..atom_count {
        for edge in target.edges_for_node(atom_id) {
            let Some(other_atom) = bond_edge_other(edge, atom_id) else {
                continue;
            };
            if atom_id >= other_atom {
                continue;
            }
            let label = effective_bond_label(target, aromaticity, atom_id, other_atom, edge.2);
            neighbors[atom_id].push((other_atom, label));
            neighbors[other_atom].push((atom_id, label));
        }
    }
    for atom_neighbors in &mut neighbors {
        atom_neighbors.sort_unstable_by_key(|&(neighbor_atom, _)| neighbor_atom);
    }

    NodeProps::new(neighbors.into_iter().map(Vec::into_boxed_slice).collect())
}

fn semantic_double_bond_stereo_configs(
    target: &Smiles,
) -> BTreeMap<(AtomId, AtomId), DoubleBondStereoConfig> {
    let mut configs = BTreeMap::new();
    for atom_id in 0..target.nodes().len() {
        for edge in target.edges_for_node(atom_id) {
            let Some(other_atom) = bond_edge_other(edge, atom_id) else {
                continue;
            };
            if atom_id >= other_atom {
                continue;
            }
            if let Some(config) = target.double_bond_stereo_config(atom_id, other_atom) {
                configs.insert(edge_key(atom_id, other_atom), config);
            }
        }
    }
    configs
}

#[inline]
const fn edge_key(left_atom: AtomId, right_atom: AtomId) -> (AtomId, AtomId) {
    if left_atom <= right_atom {
        (left_atom, right_atom)
    } else {
        (right_atom, left_atom)
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::{vec, vec::Vec};
    use core::str::FromStr;
    use elements_rs::Element;
    use smiles_parser::{atom::Atom, AromaticityPolicy, Smiles};

    use super::{
        connected_component_ids, effective_formal_charge, hybridization_code,
        identify_terminal_oxyhalogen_pair, rdkit_like_oxyhalogen_terminal_oxo_bond,
        rdkit_like_phosphorus_terminal_oxo_bond, ring_bonds_from_cycles, EdgeProps, NodeProps,
        PreparedMolecule, PreparedTarget,
    };
    use crate::{
        geometric_target::{MoleculeGraph, UndirectedBond},
        target::{AtomLabel, BondLabel, MoleculeTarget},
    };

    const MACCS_KEY_17_COUNTEREXAMPLE: &str = concat!(
        "CC[C@@H]1[C@@H](N1)C(=O)N(C)CC(=O)N(C)C(=C(C)C)C(=O)N[C@H]2CC3=CC(=CC(=C3)O)",
        "C4=CC5=C(C=C4)N(C(=C5CC(COC(=O)[C@@H]6CCCN(C2=O)N6)(C)C)C7=C(N=CC#C7)[C@H](C)OC)CC"
    );
    const OXYHALOGEN_COUNTEREXAMPLE: &str = "[O-]Cl(=O)(=O)=O";
    const BROMIC_ACID_COUNTEREXAMPLE: &str = "OBr(=O)=O";

    #[test]
    fn prepared_target_keeps_smiles_and_basic_caches() {
        let target = Smiles::from_str("CCO").unwrap();
        let prepared = PreparedTarget::new(target);

        assert_eq!(prepared.atom_count(), 3);
        assert_eq!(prepared.target().to_string(), "CCO");
        assert_eq!(prepared.degree(0), Some(1));
        assert_eq!(prepared.degree(1), Some(2));
        assert_eq!(prepared.degree(2), Some(1));
        assert_eq!(prepared.implicit_hydrogen_count(0), Some(3));
        assert_eq!(prepared.implicit_hydrogen_count(1), Some(2));
        assert_eq!(prepared.implicit_hydrogen_count(2), Some(1));
        assert_eq!(prepared.total_hydrogen_count(0), Some(3));
        assert_eq!(prepared.total_hydrogen_count(1), Some(2));
        assert_eq!(prepared.total_hydrogen_count(2), Some(1));
        assert_eq!(prepared.connectivity(0), Some(4));
        assert_eq!(prepared.connectivity(1), Some(4));
        assert_eq!(prepared.connectivity(2), Some(2));
        assert_eq!(prepared.total_valence(0), Some(4));
        assert_eq!(prepared.total_valence(1), Some(4));
        assert_eq!(prepared.total_valence(2), Some(2));
        assert_eq!(prepared.connected_component(0), Some(0));
        assert_eq!(prepared.connected_component(1), Some(0));
        assert_eq!(prepared.connected_component(2), Some(0));
    }

    #[test]
    fn prepared_target_indexes_common_atom_domains() {
        let prepared = PreparedTarget::new(Smiles::from_str("CCO.c1ccccc1").unwrap());

        assert_eq!(
            prepared.atom_ids_with_atomic_number(6),
            &[0, 1, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(prepared.atom_ids_with_atomic_number(8), &[2]);
        assert!(prepared.atom_ids_with_atomic_number(9).is_empty());
        assert_eq!(prepared.atom_ids_with_degree(1), &[0, 2]);
        assert_eq!(prepared.atom_ids_with_degree(2), &[1, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            prepared.atom_ids_with_total_hydrogen(1),
            &[2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(prepared.atom_ids_with_total_hydrogen(2), &[1]);
        assert_eq!(prepared.atom_ids_with_total_hydrogen(3), &[0]);
        assert_eq!(prepared.aromatic_atom_ids(), &[3, 4, 5, 6, 7, 8]);
        assert_eq!(prepared.aliphatic_atom_ids(), &[0, 1, 2]);
        assert_eq!(prepared.ring_atom_ids(), &[3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn prepared_target_caches_connected_components() {
        let prepared = PreparedTarget::new(Smiles::from_str("CC.O").unwrap());

        assert_eq!(prepared.connected_component(0), Some(0));
        assert_eq!(prepared.connected_component(1), Some(0));
        assert_eq!(prepared.connected_component(2), Some(1));
    }

    #[test]
    fn prepared_target_uses_rdkit_default_aromaticity() {
        let target = Smiles::from_str("c1ccccc1").unwrap();
        let prepared = PreparedTarget::new(target);

        for atom_id in 0..prepared.atom_count() {
            assert!(prepared.is_aromatic(atom_id));
        }
    }

    #[test]
    fn prepared_target_caches_ring_properties() {
        let target = Smiles::from_str("c1ccc2ccccc2c1").unwrap();
        let prepared = PreparedTarget::new(target);

        assert_eq!(prepared.ring_membership_count(0), Some(1));
        assert_eq!(prepared.ring_membership_count(3), Some(2));
        assert_eq!(prepared.ring_bond_count(0), Some(2));
        assert_eq!(prepared.ring_bond_count(3), Some(3));
        assert_eq!(prepared.smallest_ring_size(0), Some(6));
        assert_eq!(prepared.smallest_ring_size(3), Some(6));
        assert!(prepared.is_ring_atom(0));
        assert!(prepared.is_ring_bond(0, 1));
        assert!(!prepared.is_ring_bond(0, 4));
        assert_eq!(prepared.connectivity(0), Some(3));
        assert_eq!(prepared.connectivity(3), Some(3));
        assert_eq!(prepared.total_valence(0), Some(4));
        assert_eq!(prepared.total_valence(3), Some(4));
    }

    #[test]
    fn prepared_target_caches_special_connectivity_and_valence_cases() {
        let ammonium = PreparedTarget::new(Smiles::from_str("[NH4+]").unwrap());
        assert_eq!(ammonium.connectivity(0), Some(4));
        assert_eq!(ammonium.total_valence(0), Some(4));

        let sodium = PreparedTarget::new(Smiles::from_str("[Na+]").unwrap());
        assert_eq!(sodium.connectivity(0), Some(0));
        assert_eq!(sodium.total_valence(0), Some(0));

        let pyrrolic = PreparedTarget::new(Smiles::from_str("[nH]1cccc1").unwrap());
        assert_eq!(pyrrolic.connectivity(0), Some(3));
        assert_eq!(pyrrolic.total_valence(0), Some(3));

        let phosphoric = PreparedTarget::new(Smiles::from_str("P(=O)(O)(O)O").unwrap());
        assert_eq!(phosphoric.connectivity(0), Some(4));
        assert_eq!(phosphoric.total_valence(0), Some(5));
    }

    #[test]
    fn prepared_target_caches_common_hybridization_codes() {
        let alkene = PreparedTarget::new(Smiles::from_str("CC=CF").unwrap());
        assert_eq!(alkene.hybridization(0), Some(3));
        assert_eq!(alkene.hybridization(1), Some(2));
        assert_eq!(alkene.hybridization(2), Some(2));
        assert_eq!(alkene.hybridization(3), Some(3));

        let nitrile = PreparedTarget::new(Smiles::from_str("CC#N").unwrap());
        assert_eq!(nitrile.hybridization(0), Some(3));
        assert_eq!(nitrile.hybridization(1), Some(1));
        assert_eq!(nitrile.hybridization(2), Some(1));

        let amide = PreparedTarget::new(Smiles::from_str("CC(=O)NC").unwrap());
        assert_eq!(amide.hybridization(0), Some(3));
        assert_eq!(amide.hybridization(1), Some(2));
        assert_eq!(amide.hybridization(2), Some(2));
        assert_eq!(amide.hybridization(3), Some(2));
        assert_eq!(amide.hybridization(4), Some(3));
    }

    #[test]
    fn prepared_target_caches_rdkit_like_hetero_neighbor_counts() {
        let aza_phenol = PreparedTarget::new(Smiles::from_str("O=C(O)c1nc(O)ccn1").unwrap());
        let expected_hetero_counts = [0_u8, 2, 0, 2, 0, 2, 0, 0, 1, 0];
        let expected_aliphatic_hetero_counts = [0_u8, 2, 0, 0, 0, 1, 0, 0, 0, 0];

        for (atom_id, expected) in expected_hetero_counts.into_iter().enumerate() {
            assert_eq!(aza_phenol.hetero_neighbor_count(atom_id), Some(expected));
        }
        for (atom_id, expected) in expected_aliphatic_hetero_counts.into_iter().enumerate() {
            assert_eq!(
                aza_phenol.aliphatic_hetero_neighbor_count(atom_id),
                Some(expected)
            );
        }

        let sulfoxide = PreparedTarget::new(Smiles::from_str("CS(=O)C").unwrap());
        for atom_id in 0..sulfoxide.atom_count() {
            assert_eq!(sulfoxide.hetero_neighbor_count(atom_id), Some(1));
            assert_eq!(sulfoxide.aliphatic_hetero_neighbor_count(atom_id), Some(1));
        }

        let sulfone = PreparedTarget::new(Smiles::from_str("CS(=O)(=O)C").unwrap());
        let expected_sulfone_hetero_counts = [1_u8, 2, 1, 1, 1];
        for (atom_id, expected) in expected_sulfone_hetero_counts.into_iter().enumerate() {
            assert_eq!(sulfone.hetero_neighbor_count(atom_id), Some(expected));
            assert_eq!(
                sulfone.aliphatic_hetero_neighbor_count(atom_id),
                Some(expected)
            );
        }

        let nitrobenzene = PreparedTarget::new(Smiles::from_str("c1cc([N+](=O)[O-])ccc1").unwrap());
        let expected_nitro_hetero_counts = [0_u8, 0, 1, 2, 1, 1, 0, 0, 0];
        for (atom_id, expected) in expected_nitro_hetero_counts.into_iter().enumerate() {
            assert_eq!(nitrobenzene.hetero_neighbor_count(atom_id), Some(expected));
            assert_eq!(
                nitrobenzene.aliphatic_hetero_neighbor_count(atom_id),
                Some(expected)
            );
        }
    }

    #[test]
    fn prepared_target_caches_semantic_stereo() {
        let left = PreparedTarget::new(Smiles::from_str("F[C@H](Cl)Br").unwrap());
        let right = PreparedTarget::new(Smiles::from_str("F[C@@H](Cl)Br").unwrap());
        let trans = PreparedTarget::new(Smiles::from_str("F/C=C/F").unwrap());
        let cis = PreparedTarget::new(Smiles::from_str("F/C=C\\F").unwrap());
        let plain = PreparedTarget::new(Smiles::from_str("FC=CF").unwrap());

        assert_ne!(
            left.tetrahedral_chirality(1),
            right.tetrahedral_chirality(1)
        );
        assert!(left.tetrahedral_chirality(1).is_some());
        assert_eq!(
            trans.double_bond_stereo_config(1, 2),
            Some(smiles_parser::DoubleBondStereoConfig::E)
        );
        assert_eq!(
            cis.double_bond_stereo_config(1, 2),
            Some(smiles_parser::DoubleBondStereoConfig::Z)
        );
        assert_eq!(plain.double_bond_stereo_config(1, 2), None);
    }

    #[test]
    fn prepared_target_handles_non_ring_non_stereo_and_missing_atoms() {
        let prepared = PreparedTarget::new(Smiles::from_str("CC").unwrap());

        assert_eq!(prepared.target().to_string(), "CC");
        assert_eq!(prepared.bond(0, 1), Some(BondLabel::Single));
        assert_eq!(prepared.bond(0, 2), None);
        assert_eq!(
            prepared.neighbors(0).collect::<Vec<_>>(),
            vec![(1, BondLabel::Single)]
        );
        assert!(!prepared.is_ring_bond(0, 1));
        assert_eq!(prepared.tetrahedral_chirality(0), None);
        assert_eq!(prepared.double_bond_stereo_config(0, 1), None);
        assert_eq!(prepared.connected_component(9), None);
    }

    #[test]
    fn prepared_target_preserves_rdkit_like_aromaticity_for_charged_thiophene() {
        let smiles = Smiles::from_str("Cc1cc(C)[s+]s1").unwrap();
        let raw_aromatic_atoms = smiles
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(atom_id, atom)| atom.aromatic().then_some(atom_id))
            .collect::<Vec<_>>();
        let prepared = PreparedTarget::new(smiles);
        let prepared_aromatic_atoms = (0..prepared.atom_count())
            .filter(|&atom_id| prepared.is_aromatic(atom_id))
            .collect::<Vec<_>>();

        assert_eq!(raw_aromatic_atoms, vec![1, 2, 3, 5, 6]);
        assert_eq!(prepared_aromatic_atoms, vec![1, 2, 3, 5, 6]);
    }

    #[test]
    fn prepared_target_preserves_rdkit_like_aromaticity_for_pyridinium() {
        let smiles = Smiles::from_str("C[n+]1ccccc1C=NO").unwrap();
        let raw_aromatic_atoms = smiles
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(atom_id, atom)| atom.aromatic().then_some(atom_id))
            .collect::<Vec<_>>();
        let prepared = PreparedTarget::new(smiles);
        let prepared_aromatic_atoms = (0..prepared.atom_count())
            .filter(|&atom_id| prepared.is_aromatic(atom_id))
            .collect::<Vec<_>>();

        assert_eq!(raw_aromatic_atoms, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(prepared_aromatic_atoms, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn prepared_target_keeps_the_maccs_key_17_counterexample_triple_bond() {
        let prepared = PreparedTarget::new(Smiles::from_str(MACCS_KEY_17_COUNTEREXAMPLE).unwrap());

        assert_eq!(prepared.atom(59).and_then(Atom::element), Some(Element::C));
        assert_eq!(prepared.atom(60).and_then(Atom::element), Some(Element::C));
        assert_eq!(prepared.bond(59, 60), Some(BondLabel::Triple));
        assert!(prepared
            .neighbors(59)
            .any(|(other_atom, bond_label)| other_atom == 60 && bond_label == BondLabel::Triple));
        assert!(prepared
            .neighbors(60)
            .any(|(other_atom, bond_label)| other_atom == 59 && bond_label == BondLabel::Triple));
    }

    #[test]
    fn prepared_target_normalizes_terminal_oxyhalogen_oxo_bonds_like_rdkit() {
        let prepared = PreparedTarget::new(Smiles::from_str(OXYHALOGEN_COUNTEREXAMPLE).unwrap());

        assert_eq!(prepared.bond(0, 1), Some(BondLabel::Single));
        assert_eq!(prepared.bond(1, 2), Some(BondLabel::Single));
        assert_eq!(prepared.bond(1, 3), Some(BondLabel::Single));
        assert_eq!(prepared.bond(1, 4), Some(BondLabel::Single));
    }

    #[test]
    fn prepared_target_normalizes_terminal_oxyhalogen_charges_like_rdkit() {
        let prepared = PreparedTarget::new(Smiles::from_str(BROMIC_ACID_COUNTEREXAMPLE).unwrap());

        assert_eq!(prepared.formal_charge(0), Some(0));
        assert_eq!(prepared.formal_charge(1), Some(2));
        assert_eq!(prepared.formal_charge(2), Some(-1));
        assert_eq!(prepared.formal_charge(3), Some(-1));
    }

    #[test]
    fn prepared_helper_paths_cover_remaining_normalization_branches() {
        let alkyl_hypochlorite = Smiles::from_str("Cl(=O)C").unwrap();
        let alkyl_hypochlorite_aromaticity =
            alkyl_hypochlorite.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        let raw_cl_o = alkyl_hypochlorite.edge_for_node_pair((0, 1)).unwrap().2;

        assert_eq!(
            identify_terminal_oxyhalogen_pair(&alkyl_hypochlorite, 0, 1),
            Some((1, 0))
        );
        assert!(!rdkit_like_oxyhalogen_terminal_oxo_bond(
            &alkyl_hypochlorite,
            0,
            1,
            raw_cl_o
        ));
        assert_eq!(
            effective_formal_charge(&alkyl_hypochlorite, &alkyl_hypochlorite_aromaticity, 99),
            0
        );

        let wildcard_cycles = ring_bonds_from_cycles(&[vec![0], vec![0, 1, 2]]);
        assert!(!wildcard_cycles.contains(&(0, 0)));
        assert!(wildcard_cycles.contains(&(0, 1)));
        assert!(wildcard_cycles.contains(&(0, 2)));

        let disconnected = connected_component_ids(&Smiles::from_str("CC.O").unwrap());
        assert_eq!(disconnected.get(0), Some(&0));
        assert_eq!(disconnected.get(2), Some(&1));

        let aromatic_assignment_target = Smiles::from_str("c1ccccc1").unwrap();
        let aromatic_assignment =
            aromatic_assignment_target.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(
            hybridization_code(&aromatic_assignment_target, &aromatic_assignment, 999),
            0
        );

        let unsupported_lone_pair = Smiles::from_str("CCl").unwrap();
        let unsupported_lone_pair_aromaticity =
            unsupported_lone_pair.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(
            hybridization_code(
                &unsupported_lone_pair,
                &unsupported_lone_pair_aromaticity,
                1
            ),
            3
        );
    }

    #[test]
    fn phosphorus_terminal_oxo_helper_covers_remaining_false_and_true_cases() {
        let duplicate_double = Smiles::from_str("COP(=O)(=N)=C").unwrap();
        let duplicate_double_aromaticity =
            duplicate_double.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        let duplicate_double_bond = duplicate_double.edge_for_node_pair((1, 2)).unwrap().2;
        assert!(!rdkit_like_phosphorus_terminal_oxo_bond(
            &duplicate_double,
            &duplicate_double_aromaticity,
            1,
            2,
            duplicate_double_bond
        ));

        let triple_neighbor = Smiles::from_str("COP(=O)#N").unwrap();
        let triple_neighbor_aromaticity =
            triple_neighbor.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        let triple_neighbor_bond = triple_neighbor.edge_for_node_pair((2, 3)).unwrap().2;
        assert!(!rdkit_like_phosphorus_terminal_oxo_bond(
            &triple_neighbor,
            &triple_neighbor_aromaticity,
            2,
            3,
            triple_neighbor_bond
        ));

        let empty_single_neighbors = Smiles::from_str("P(=O)=NC").unwrap();
        let empty_single_neighbors_aromaticity =
            empty_single_neighbors.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        let empty_single_neighbors_bond =
            empty_single_neighbors.edge_for_node_pair((0, 1)).unwrap().2;
        assert!(!rdkit_like_phosphorus_terminal_oxo_bond(
            &empty_single_neighbors,
            &empty_single_neighbors_aromaticity,
            0,
            1,
            empty_single_neighbors_bond
        ));

        let sulfur_partner = Smiles::from_str("CP(=O)(=S)C").unwrap();
        let sulfur_partner_aromaticity =
            sulfur_partner.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        let sulfur_partner_bond = sulfur_partner.edge_for_node_pair((1, 2)).unwrap().2;
        assert!(!rdkit_like_phosphorus_terminal_oxo_bond(
            &sulfur_partner,
            &sulfur_partner_aromaticity,
            1,
            2,
            sulfur_partner_bond
        ));

        let substituted_imidate = Smiles::from_str("CP(=O)(=NC)C").unwrap();
        let substituted_imidate_aromaticity =
            substituted_imidate.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        let substituted_imidate_bond = substituted_imidate.edge_for_node_pair((1, 2)).unwrap().2;
        assert!(rdkit_like_phosphorus_terminal_oxo_bond(
            &substituted_imidate,
            &substituted_imidate_aromaticity,
            1,
            2,
            substituted_imidate_bond
        ));

        let substituted_ylide = Smiles::from_str("CP(=O)(=CC)C").unwrap();
        let substituted_ylide_aromaticity =
            substituted_ylide.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        let substituted_ylide_bond = substituted_ylide.edge_for_node_pair((1, 2)).unwrap().2;
        assert!(rdkit_like_phosphorus_terminal_oxo_bond(
            &substituted_ylide,
            &substituted_ylide_aromaticity,
            1,
            2,
            substituted_ylide_bond
        ));
    }

    #[test]
    fn node_props_are_dense_and_indexed_by_atom_id() {
        let props = NodeProps::new(vec![2usize, 4, 1]);

        assert_eq!(props.len(), 3);
        assert!(!props.is_empty());
        assert_eq!(props.get(1), Some(&4));
        assert_eq!(props.get(3), None);
    }

    #[test]
    fn edge_props_are_dense_and_indexed_by_edge_id() {
        let props = EdgeProps::new(vec!["a", "b"]);

        assert_eq!(props.len(), 2);
        assert!(!props.is_empty());
        assert_eq!(props.get(0), Some(&"a"));
        assert_eq!(props.get(2), None);
    }

    #[test]
    fn empty_property_sidecars_report_empty() {
        let node_props = NodeProps::<usize>::new(vec![]);
        let edge_props = EdgeProps::<u8>::new(vec![]);

        assert!(node_props.is_empty());
        assert!(edge_props.is_empty());
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
        assert_eq!(prepared.degree(9), None);
        assert_eq!(prepared.target().atom_count(), 3);
        assert_eq!(prepared.degrees().len(), 3);
    }
}
