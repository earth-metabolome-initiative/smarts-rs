//! Integration probes for the `geometric-traits`-backed molecule target.

use elements_rs::Element;
use geometric_traits::{
    impls::{SortedVec, ValuedCSR2D},
    naive_structs::{GenericEdgesBuilder, GenericGraph, GenericVocabularyBuilder},
    traits::{
        Edges, EdgesBuilder, MonopartiteGraph, MonoplexGraph, SparseValuedMatrix2D,
        VocabularyBuilder, VocabularyRef,
    },
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct AtomLabel {
    id: usize,
    element: Element,
    aromatic: bool,
    isotope_mass_number: Option<u16>,
    formal_charge: i8,
    explicit_hydrogens: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(dead_code)]
enum BondLabel {
    Single,
    Double,
    Aromatic,
}

type MoleculeBondMatrix = ValuedCSR2D<usize, usize, usize, BondLabel>;
type MoleculeGraph = GenericGraph<SortedVec<AtomLabel>, MoleculeBondMatrix>;

fn ethanol_graph() -> MoleculeGraph {
    let atoms: SortedVec<AtomLabel> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols([
            (
                0,
                AtomLabel {
                    id: 0,
                    element: Element::C,
                    aromatic: false,
                    isotope_mass_number: None,
                    formal_charge: 0,
                    explicit_hydrogens: 3,
                },
            ),
            (
                1,
                AtomLabel {
                    id: 1,
                    element: Element::C,
                    aromatic: false,
                    isotope_mass_number: None,
                    formal_charge: 0,
                    explicit_hydrogens: 2,
                },
            ),
            (
                2,
                AtomLabel {
                    id: 2,
                    element: Element::O,
                    aromatic: false,
                    isotope_mass_number: None,
                    formal_charge: 0,
                    explicit_hydrogens: 1,
                },
            ),
        ])
        .build()
        .unwrap();

    let bonds: MoleculeBondMatrix = GenericEdgesBuilder::<_, MoleculeBondMatrix>::default()
        .expected_number_of_edges(4)
        .expected_shape((3, 3))
        .edges([
            (0, 1, BondLabel::Single),
            (1, 0, BondLabel::Single),
            (1, 2, BondLabel::Single),
            (2, 1, BondLabel::Single),
        ])
        .build()
        .unwrap();

    GenericGraph::from((atoms, bonds))
}

#[test]
fn can_build_a_small_molecule_graph_with_labeled_bonds() {
    let graph = ethanol_graph();

    let atom = graph.nodes_vocabulary().convert_ref(&1).unwrap();
    assert_eq!(atom.element, Element::C);
    assert_eq!(atom.explicit_hydrogens, 2);

    let neighbors: Vec<_> = Edges::successors(graph.edges(), 1).collect();
    assert_eq!(neighbors, vec![0, 2]);

    let bond = graph.edges().matrix().sparse_value_at(1, 2).unwrap();
    assert_eq!(bond, BondLabel::Single);
}

#[test]
fn duplicate_atom_labels_require_artificial_node_identity() {
    let graph = ethanol_graph();

    let first = graph.nodes_vocabulary().convert_ref(&0).unwrap();
    let second = graph.nodes_vocabulary().convert_ref(&1).unwrap();

    assert_eq!(first.element, second.element);
    assert_ne!(first.id, second.id);
}
