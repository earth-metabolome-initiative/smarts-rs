//! RDKit-frozen advanced Daylight SMARTS example parity tests.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count, matches, substructure_matches};

#[derive(Debug, Clone, Copy)]
struct ExampleCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
}

const CASES: &[ExampleCase] = &[
    ExampleCase {
        name: "hydrogen_bond_donor_matches_alcohol_oxygen",
        smarts: "[O,N;!H0]",
        smiles: "CCO",
        expected_count: 1,
    },
    ExampleCase {
        name: "hydrogen_bond_donor_matches_amine_nitrogen",
        smarts: "[O,N;!H0]",
        smiles: "CCN",
        expected_count: 1,
    },
    ExampleCase {
        name: "hydrogen_bond_donor_rejects_tertiary_amide",
        smarts: "[O,N;!H0]",
        smiles: "CC(=O)N(C)C",
        expected_count: 0,
    },
    ExampleCase {
        name: "hydrogen_bond_acceptor_matches_acyclic_amide_carbonyl",
        smarts: "[C,N;R0]=O",
        smiles: "CC(=O)N",
        expected_count: 1,
    },
    ExampleCase {
        name: "hydrogen_bond_acceptor_matches_formamide_nitrogen_case",
        smarts: "[C,N;R0]=O",
        smiles: "NC=O",
        expected_count: 1,
    },
    ExampleCase {
        name: "hydrogen_bond_acceptor_rejects_ring_carbonyl",
        smarts: "[C,N;R0]=O",
        smiles: "O=C1CCCC1",
        expected_count: 0,
    },
    ExampleCase {
        name: "rotatable_bond_matches_only_butane_central_bond",
        smarts: "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]",
        smiles: "CCCC",
        expected_count: 1,
    },
    ExampleCase {
        name: "rotatable_bond_rejects_ring_bonds",
        smarts: "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]",
        smiles: "C1CCC1",
        expected_count: 0,
    },
    ExampleCase {
        name: "rotatable_bond_rejects_alkyne_attachment",
        smarts: "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]",
        smiles: "CC#CC",
        expected_count: 0,
    },
    ExampleCase {
        name: "pcb_pattern_matches_dichlorobiphenyl",
        smarts: "[$(c:cCl),$(c:c:cCl),$(c:c:c:cCl)]-[$(c:cCl),$(c:c:cCl),$(c:c:c:cCl)]",
        smiles: "Clc1ccc(cc1)-c1ccccc1Cl",
        expected_count: 1,
    },
    ExampleCase {
        name: "pcb_pattern_rejects_unsubstituted_biphenyl",
        smarts: "[$(c:cCl),$(c:c:cCl),$(c:c:c:cCl)]-[$(c:cCl),$(c:c:cCl),$(c:c:c:cCl)]",
        smiles: "c1ccccc1-c1ccccc1",
        expected_count: 0,
    },
];

#[test]
fn advanced_daylight_examples_should_agree_with_rdkit() {
    for case in CASES {
        let query = QueryMol::from_str(case.smarts)
            .unwrap_or_else(|error| panic!("{}: failed to parse SMARTS: {error}", case.name));
        let count = match_count(&query, case.smiles)
            .unwrap_or_else(|error| panic!("{}: validator error: {error}", case.name));
        let matches_out = substructure_matches(&query, case.smiles)
            .unwrap_or_else(|error| panic!("{}: validator error: {error}", case.name));

        assert_eq!(
            count, case.expected_count,
            "{}: SMARTS {:?} against SMILES {:?}",
            case.name, case.smarts, case.smiles
        );
        assert_eq!(
            matches_out.len(),
            count,
            "{}: materialized match count mismatch",
            case.name
        );
        assert_eq!(
            matches(&query, case.smiles).unwrap(),
            count > 0,
            "{}: boolean/count mismatch",
            case.name
        );
    }
}
