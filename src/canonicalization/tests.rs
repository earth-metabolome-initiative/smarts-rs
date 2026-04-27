use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::str::FromStr;

use elements_rs::Element;
use smiles_parser::bond::Bond;

use super::QueryCanonicalLabeling;
use crate::{AtomExpr, BondExpr, BondExprTree, BondPrimitive, QueryAtom, QueryBond, QueryMol};

fn canonical_string(source: &str) -> String {
    QueryMol::from_str(source)
        .unwrap()
        .canonicalize()
        .to_string()
}

fn assert_canonical_roundtrips(source: &str) {
    let query = QueryMol::from_str(source).unwrap();
    let canonical = query.canonicalize();
    let recanonicalized = canonical.canonicalize();
    let rendered = canonical.to_string();
    let reparsed = QueryMol::from_str(&rendered).unwrap();
    let reparsed_canonical = reparsed.canonicalize();
    assert_eq!(canonical, recanonicalized);
    assert!(canonical.is_canonical());
    assert_eq!(canonical.atom_count(), query.atom_count());
    assert_eq!(canonical.bond_count(), query.bond_count());
    assert_eq!(canonical.component_count(), query.component_count());
    assert_eq!(rendered, query.to_canonical_smarts());
    assert_eq!(canonical, reparsed_canonical);
    assert_eq!(rendered, reparsed.to_canonical_smarts());
}

fn recursive_component_chain(depth: usize) -> String {
    let mut nested = String::from("CsO");
    for _ in 0..depth {
        nested = format!("Cs[$({nested})]O.OO");
    }
    format!("[$({nested})]O.O")
}

fn all_permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }

    let mut permutations = Vec::new();
    for (index, &item) in items.iter().enumerate() {
        let mut remaining = items.to_vec();
        remaining.remove(index);
        for mut tail in all_permutations(&remaining) {
            let mut permutation = Vec::with_capacity(items.len());
            permutation.push(item);
            permutation.append(&mut tail);
            permutations.push(permutation);
        }
    }
    permutations
}

fn permute_atoms(query: &QueryMol, atom_order: &[usize]) -> QueryMol {
    assert_eq!(atom_order.len(), query.atom_count());

    let mut new_index_of_old = vec![usize::MAX; query.atom_count()];
    for (new_id, old_id) in atom_order.iter().copied().enumerate() {
        new_index_of_old[old_id] = new_id;
    }

    let atoms = atom_order
        .iter()
        .copied()
        .enumerate()
        .map(|(new_id, old_id)| {
            let old_atom = &query.atoms()[old_id];
            QueryAtom {
                id: new_id,
                component: old_atom.component,
                expr: old_atom.expr.clone(),
            }
        })
        .collect::<Vec<_>>();

    let mut bonds = query
        .bonds()
        .iter()
        .rev()
        .map(|bond| {
            let src = new_index_of_old[bond.src];
            let dst = new_index_of_old[bond.dst];
            let expr = if src <= dst {
                bond.expr.clone()
            } else {
                super::flip_directional_bond_expr(&bond.expr)
            };
            let (src, dst) = if src <= dst { (src, dst) } else { (dst, src) };
            QueryBond {
                id: 0,
                src,
                dst,
                expr,
            }
        })
        .collect::<Vec<_>>();
    for (bond_id, bond) in bonds.iter_mut().enumerate() {
        bond.id = bond_id;
    }

    QueryMol::from_parts(
        atoms,
        bonds,
        query.component_count(),
        query.component_groups().to_vec(),
    )
}

fn permute_top_level_entries(query: &QueryMol, entry_order: &[usize]) -> QueryMol {
    let entries = super::build_top_level_component_entries(query);
    assert_eq!(entry_order.len(), entries.len());

    let reordered_entries = entry_order
        .iter()
        .copied()
        .map(|index| entries[index].clone())
        .collect::<Vec<_>>();
    let mut new_component_of_old = vec![usize::MAX; query.component_count()];
    let mut next_component_id = 0usize;
    let mut component_groups = Vec::with_capacity(query.component_count());
    let mut next_group_id = 0usize;

    for entry in reordered_entries {
        let group = (entry.grouped && entry.component_ids.len() > 1).then(|| {
            let group_id = next_group_id;
            next_group_id += 1;
            group_id
        });
        for old_component_id in entry.component_ids {
            new_component_of_old[old_component_id] = next_component_id;
            component_groups.push(group);
            next_component_id += 1;
        }
    }

    let atoms = query
        .atoms()
        .iter()
        .map(|atom| QueryAtom {
            id: atom.id,
            component: new_component_of_old[atom.component],
            expr: atom.expr.clone(),
        })
        .collect::<Vec<_>>();

    QueryMol::from_parts(
        atoms,
        query.bonds().to_vec(),
        query.component_count(),
        component_groups,
    )
}

fn assert_same_canonical_group(group: &[&str]) {
    let expected = canonical_string(group[0]);
    for source in &group[1..] {
        assert_eq!(
            expected,
            canonical_string(source),
            "group did not converge: {group:?}"
        );
    }
}

fn assert_all_atom_permutations_converge(source: &str) {
    let query = QueryMol::from_str(source).unwrap();
    let expected = query.canonicalize();
    let atom_ids = (0..query.atom_count()).collect::<Vec<_>>();
    for atom_order in all_permutations(&atom_ids) {
        let permuted = permute_atoms(&query, &atom_order);
        let canonicalized = permuted.canonicalize();
        assert_eq!(
            expected, canonicalized,
            "atom permutation changed canonical form for {source}: {atom_order:?}"
        );
        assert!(canonicalized.is_canonical());
    }
}

fn assert_all_top_level_entry_permutations_converge(source: &str) {
    let query = QueryMol::from_str(source).unwrap();
    let expected = query.canonicalize();
    let entry_ids = (0..super::build_top_level_component_entries(&query).len()).collect::<Vec<_>>();
    for entry_order in all_permutations(&entry_ids) {
        let permuted = permute_top_level_entries(&query, &entry_order);
        let canonicalized = permuted.canonicalize();
        assert_eq!(
            expected, canonicalized,
            "top-level entry permutation changed canonical form for {source}: {entry_order:?}"
        );
        assert!(canonicalized.is_canonical());
    }
}

#[test]
fn canonical_labeling_of_empty_query_is_empty() {
    let query = QueryMol::from_str("").err();
    assert!(query.is_some());

    let empty = QueryMol::from_parts(Vec::new(), Vec::new(), 0, Vec::new());
    let labeling = empty.canonical_labeling();
    assert_eq!(labeling, QueryCanonicalLabeling::new(Vec::new()));
    assert!(empty.is_canonical());
}

#[test]
fn canonicalize_parse_smarts_empty_query_roundtrips() {
    assert!(crate::parse_smarts("").is_err());
    let empty = QueryMol::from_parts(Vec::new(), Vec::new(), 0, Vec::new());
    let canonical = empty.canonicalize();

    assert!(canonical.is_canonical());
    assert_eq!(canonical.atom_count(), 0);
    assert_eq!(canonical.bond_count(), 0);
    assert_eq!(canonical.component_count(), 0);
    assert_eq!(canonical.to_string(), "");
    assert_eq!(canonical.to_canonical_smarts(), "");
    assert!(crate::parse_smarts(&canonical.to_canonical_smarts()).is_err());
    assert_eq!(
        canonical,
        QueryMol::from_parts(Vec::new(), Vec::new(), 0, Vec::new())
    );
}

#[test]
fn canonical_labeling_inverse_matches_order() {
    let query = QueryMol::from_str("OC").unwrap();
    let labeling = query.canonical_labeling();

    assert_eq!(labeling.order().len(), 2);
    assert_eq!(labeling.new_index_of_old_atom()[labeling.order()[0]], 0);
    assert_eq!(labeling.new_index_of_old_atom()[labeling.order()[1]], 1);
}

#[test]
fn canonicalize_is_idempotent() {
    let query = QueryMol::from_str("N(C)C").unwrap();
    let once = query.canonicalize();
    let twice = once.canonicalize();

    assert_eq!(once, twice);
    assert!(once.is_canonical());
}

#[test]
fn canonicalize_converges_component_permutations() {
    assert_eq!(canonical_string("C.N"), canonical_string("N.C"));
    assert_eq!(canonical_string("(N.C).O"), canonical_string("O.(C.N)"));
    assert_all_top_level_entry_permutations_converge("C.N.O");
    assert_all_top_level_entry_permutations_converge("(C.N).O");
    assert_all_top_level_entry_permutations_converge("(C.N).(O.S)");
}

#[test]
fn canonicalize_converges_noncontiguous_component_group_ids() {
    let noncontiguous = QueryMol::from_parts(
        vec![
            QueryAtom {
                id: 0,
                component: 0,
                expr: AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            },
            QueryAtom {
                id: 1,
                component: 1,
                expr: AtomExpr::Bare {
                    element: Element::O,
                    aromatic: false,
                },
            },
            QueryAtom {
                id: 2,
                component: 2,
                expr: AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            },
        ],
        Vec::new(),
        3,
        vec![Some(0), None, Some(0)],
    );

    assert_eq!(
        canonical_string("(C.C).(O)"),
        noncontiguous.canonicalize().to_string()
    );
}

#[test]
fn canonicalize_converges_linear_spellings() {
    assert_eq!(canonical_string("CO"), canonical_string("OC"));
    assert_eq!(canonical_string("OCN"), canonical_string("NCO"));
    assert_eq!(canonical_string("C(N)O"), canonical_string("C(O)N"));
    assert_all_atom_permutations_converge("CO");
    assert_all_atom_permutations_converge("CCO");
    assert_all_atom_permutations_converge("C(N)O");
}

#[test]
fn canonicalize_sorts_bracket_boolean_children() {
    assert_eq!(canonical_string("[H1;C]"), canonical_string("[C;H1]"));
    assert_eq!(canonical_string("[O,C,N]"), canonical_string("[N,O,C]"));
    assert_eq!(canonical_string("[C&X4&H2]"), canonical_string("[H2&C&X4]"));
    assert_same_canonical_group(&["[C;H1;X4]", "[X4;C;H1]", "[H1;X4;C]", "[C;X4;H1]"]);
    assert_same_canonical_group(&["[C,N,O]", "[O,N,C]", "[N,O,C]", "[C,O,N]"]);
}

#[test]
fn canonicalize_deduplicates_equivalent_bracket_boolean_children() {
    assert_eq!(canonical_string("[#6&#6]"), "[#6]");
    assert_eq!(canonical_string("[#6;#6]"), "[#6]");
    assert_eq!(canonical_string("[#6,#6]"), "[#6]");
    assert_eq!(canonical_string("[!#6&!#6]"), "[!#6]");
    assert_eq!(canonical_string("[#6;#6;#6]"), "[#6]");

    for term in [
        "#7", "C", "c", "N", "n", "R", "r6", "+0", "H0", "D3", "X4", "v4", "x2", "A", "a",
        "$([#6])",
    ] {
        let expected = canonical_string(&format!("[{term}]"));
        assert_eq!(canonical_string(&format!("[{term}&{term}]")), expected);
        assert_eq!(canonical_string(&format!("[{term};{term}]")), expected);
        assert_eq!(canonical_string(&format!("[{term},{term}]")), expected);
        let negated_expected = canonical_string(&format!("[!{term}]"));
        assert_eq!(
            canonical_string(&format!("[!{term}&!{term}]")),
            negated_expected
        );
    }
}

#[test]
fn canonicalize_simplifies_equivalent_atom_boolean_forms() {
    for (source, expected) in [
        ("[#6;R]", "[#6&R]"),
        ("[#6;#7]", "[!*]"),
        ("[#6,#6&R]", "[#6]"),
        ("[C,C&R]", "[C]"),
        ("[#6,!#6]", "*"),
        ("[A,a]", "*"),
        ("[C&#6]", "[C]"),
        ("[c&#6]", "[c]"),
        ("[N&#7]", "[N]"),
        ("[n&#7]", "[n]"),
        ("[O&#8]", "[O]"),
        ("[o&#8]", "[o]"),
        ("[C&A]", "[C]"),
        ("[c&a]", "[c]"),
        ("[#1&A]", "[#1]"),
        ("[#5&A]", "[B]"),
        ("[#5&a]", "[b]"),
        ("[#6&A]", "[C]"),
        ("[#6&a]", "[c]"),
        ("[#7&A]", "[N]"),
        ("[#7&a]", "[n]"),
        ("[#8&A]", "[O]"),
        ("[#8&a]", "[o]"),
        ("[#15&A]", "[P]"),
        ("[#15&a]", "[p]"),
        ("[#16&A]", "[S]"),
        ("[#16&a]", "[s]"),
        ("[#33&a]", "[as]"),
        ("[#34&a]", "[se]"),
        ("[#9&A]", "[F]"),
        ("[#17&A]", "[Cl]"),
        ("[#35&A]", "[Br]"),
        ("[#53&A]", "[I]"),
        ("[#6,#6&H3,#6&R]", "[#6]"),
        ("[!#6,!#6&R]", "[!#6]"),
        ("[C,#6]", "[#6]"),
        ("[c,#6]", "[#6]"),
        ("[C,A]", "[A]"),
        ("[c,a]", "[a]"),
        ("[#6,C&H3]", "[#6]"),
        ("[#6,12C&H3]", "[#6]"),
        ("[C,12C&H3]", "[C]"),
        ("[A,12C&H3]", "[A]"),
        ("[C,c]", "[#6]"),
        ("[B,b]", "[#5]"),
        ("[N,n]", "[#7]"),
        ("[O,o]", "[#8]"),
        ("[P,p]", "[#15]"),
        ("[S,s]", "[#16]"),
        ("[As,as]", "[#33]"),
        ("[Se,se]", "[#34]"),
        ("[#6&A,#6&a]", "[#6]"),
        ("[C,c,!#6]", "*"),
        ("[R,!R]", "*"),
        ("[r6,!r6]", "*"),
        ("[H0,!H0]", "*"),
        ("[#6,!C]", "*"),
        ("[#6,!c]", "*"),
        ("[12*,!12C]", "*"),
        ("[D{2-4},!D3]", "*"),
        ("[#6&A,!C]", "*"),
        ("[12*&C,!12C]", "*"),
        ("[D{2-4}&D3,!D3]", "*"),
        ("[#6;R;H1]", "[#6&H&R]"),
        ("[#6&R,#6&!R]", "[#6]"),
        ("[C&R,C&!R]", "[C]"),
        ("[#6&+0,#6&!+0]", "[#6]"),
        ("[#6&H0,#6&!H0]", "[#6]"),
        ("[#6&D3,#6&!D3]", "[#6]"),
        ("[#6&r6,#6&!r6]", "[#6]"),
        ("[D2&R,D2&!R]", "[D2]"),
        ("[#6&D2,#6&!D2]", "[#6]"),
        ("[#6&12*,#6&!12*]", "[#6]"),
        ("[C&12*,C&!12*]", "[C]"),
        ("[12C&R,12C&!R]", "[12C]"),
        ("[!#6&R,!#6&!R]", "[!#6]"),
        ("[!#6&+0,!#6&!+0]", "[!#6]"),
        ("[#6&R,#6&!R,#7]", "[#6,#7]"),
        ("[#6&12*&!12C]", "[12c]"),
        ("[#6&R,#6&R&X4]", "[#6&R]"),
        ("[#6&R,#6&R&X4,#7]", "[#6&R,#7]"),
        ("[$([#6])&R,$([#6])&!R]", "[#6]"),
        ("[C&R,c&R]", "[#6&R]"),
        ("[#6&R&C,#6&R&!C]", "[#6&R]"),
        ("[#6&+0&C,#6&+0&!C]", "[#6&+0]"),
        ("[#6&H0&C,#6&H0&!C]", "[#6&H0]"),
        ("[#7&R&N,#7&R&!N]", "[#7&R]"),
        ("[$([#6])&#6&C,$([#6])&#6&!C]", "[#6]"),
        ("[#6&A,!#6&A]", "[A]"),
        ("[#6&a,!#6&a]", "[a]"),
        ("[#7&A,!#7&A]", "[A]"),
        ("[#7&a,!#7&a]", "[a]"),
        ("[C,A&!#6]", "[A]"),
        ("[c,a&!#6]", "[a]"),
        ("[N,A&!#7]", "[A]"),
        ("[n,a&!#7]", "[a]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_complement_partition_atom_forms() {
    for (source, expected) in [
        ("[!a]", "[A]"),
        ("[!A]", "[a]"),
        ("[A,!a]", "[A]"),
        ("[a,!A]", "[a]"),
        ("[A;!a,R]", "[A]"),
        ("[a;!A,R]", "[a]"),
        ("[A&R,!a&R]", "[A&R]"),
        ("[a&R,!A&R]", "[R&a]"),
        ("[#6&!C]", "[c]"),
        ("[#6&!c]", "[C]"),
        ("[#6&!A]", "[c]"),
        ("[#6&!a]", "[C]"),
        ("[#7&!N]", "[n]"),
        ("[#7&!n]", "[N]"),
        ("[#6&12*&!12c]", "[12C]"),
        ("[A,!#6]", "[!c]"),
        ("[a,!#6]", "[!C]"),
        ("[A,!#7]", "[!n]"),
        ("[a,!#7]", "[!N]"),
        ("[A,!#11]", "*"),
        ("[a,!#11]", "[!Na]"),
        ("[#6,!C&R]", "[#6,R]"),
        ("[#6,!c&R]", "[#6,R]"),
        ("[#6&X4,!C&X4]", "[X4]"),
        ("[#6&X4,!c&X4]", "[X4]"),
        ("[C&R,A&!#6,A&!R]", "[A]"),
        ("[c&R,a&!#6,a&!R]", "[a]"),
        ("[#6&A&R,A&!#6,A&!R]", "[A]"),
        ("[#6&a&R,a&!#6,a&!R]", "[a]"),
        ("[C&R&H0,A&H0&!#6,A&H0&!R]", "[A&H0]"),
        ("[C&R&H0,A&H0&!#6,A&H0&!R,A&!H0]", "[A]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_negated_general_atom_boolean_forms() {
    for (source, expected) in [
        ("[C,!#6]", "[!c]"),
        ("[c,!#6]", "[!C]"),
        ("[D3,!D{2-4}]", "[!D2&!D4]"),
        ("[H0,!H{0-1}]", "[!H]"),
        ("[R1,!R{1-2}]", "[!R2]"),
        ("[#6&R,!R]", "[!R,#6]"),
        ("[#6&R,!#6]", "[!#6,R]"),
        ("[#6&R,!#6&H0,R&H0]", "[!#6&H0,#6&R]"),
        ("[#6&R&X4,!#6&H0&X4,R&H0&X4]", "[!#6&H0,#6&R;X4]"),
        ("[#6&R&H0,!#6&X4,!R&X4,!H0&X4]", "[#6&H0&R,X4]"),
        ("[$([#6])&R&H0,!$([#6])&X4,!R&X4,!H0&X4]", "[#6&H0&R,X4]"),
        ("[#6,!C&!c]", "*"),
        ("[#7,!N&!n]", "*"),
        ("[#6&12*,!12C&!12c]", "*"),
        ("[#6&X4,!C&X4,!c&X4]", "[X4]"),
        ("[D{2-4},!D2&!D3&!D4&R]", "[D{2-4},R]"),
        ("[D{2-5},!D{2-3}&!D{4-5}&R&H0]", "[D{2-5},H0&R]"),
        ("[#6&12*,!12C&!12c&R]", "[#6&12*,R]"),
        ("[#7&15*,!15N&!15n&H0]", "[#7&15*,H0]"),
        ("[#6,!C&R&H0]", "[#6,H0&R]"),
        ("[#6,!c&R&H0]", "[#6,H0&R]"),
        ("[#6&X4,!C&X4&R]", "[#6,R;X4]"),
        ("[#6&X4,!c&X4&R]", "[#6,R;X4]"),
        ("[#6&12*,!12C&R]", "[#6&12*,R]"),
        ("[#6&12*,!12c&R]", "[#6&12*,R]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_low_precedence_complement_residual_forms() {
    for (source, expected) in [
        ("[#6;!C,R]", "[#6&R,c]"),
        ("[#6&X4;!C,R]", "[#6&R,c;X4]"),
        ("[#6;!c,R]", "[#6&R,C]"),
        ("[#7;!N,H0]", "[#7&H0,n]"),
        ("[A&X4;!C,D3]", "[!C,D3;A;X4]"),
        ("[#6&12*;!12C,R]", "[#6&R,c;12*]"),
        ("[#6&12*;!12c,R]", "[#6&R,C;12*]"),
        ("[D{2-4};!D3,R]", "[D2,D4,D{2-4}&R]"),
        ("[X{2-4};!X{2-3},H0]", "[H0&X{2-4},X4]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_negated_mutually_exclusive_atom_forms() {
    for (source, expected) in [
        ("[!C,!c]", "*"),
        ("[!#6,!#7]", "*"),
        ("[!A,!a]", "*"),
        ("[!D1,!D2]", "*"),
        ("[!R1,!R2]", "*"),
        ("[!+0,!+]", "*"),
        ("[#6,!C,!c]", "*"),
        ("[!12C,!12c]", "*"),
        ("[#6,!#6&!#7]", "[!#7]"),
        ("[!#6&!C]", "[!#6]"),
        ("[!D{2-4}&!D3]", "[!D{2-4}]"),
        ("[!#6,!C]", "[!C]"),
        ("[!D{2-4},!D3]", "[!D3]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_false_atom_boolean_forms() {
    for (source, expected) in [
        ("[!*]", "[!*]"),
        ("[#6&!#6]", "[!*]"),
        ("[C&!C]", "[!*]"),
        ("[#6&#7]", "[!*]"),
        ("[C&c]", "[!*]"),
        ("[A&a]", "[!*]"),
        ("[12C&13C]", "[!*]"),
        ("[12C&13*]", "[!*]"),
        ("[+0&+]", "[!*]"),
        ("[D1&D2]", "[!*]"),
        ("[D{1-2}&D{3-4}]", "[!*]"),
        ("[R0&R]", "[!*]"),
        ("[r0&r]", "[!*]"),
        ("[C&!#6]", "[!*]"),
        ("[c&!#6]", "[!*]"),
        ("[12C&!12*]", "[!*]"),
        ("[D&!D1]", "[!*]"),
        ("[D3&!D{2-4}]", "[!*]"),
        ("[R&!R{1-}]", "[!*]"),
        ("[#6&A&!C]", "[!*]"),
        ("[#6&a&!c]", "[!*]"),
        ("[#6&!C&!c]", "[!*]"),
        ("[!A&!a]", "[!*]"),
        ("[!D{0-2};!D{3-}]", "[!*]"),
        ("[!D{0-2}&!D{3-}]", "[!*]"),
        ("[!R&!R0]", "[!*]"),
        ("[!r&!r0]", "[!*]"),
        ("[!x&!x0]", "[!*]"),
        ("[!H&!H0]", "[H{2-}]"),
        ("[!D{0-2}&!D{4-}]", "[D3]"),
        ("[D{1-2}&!D1&!D2]", "[!*]"),
        ("[D{2-4}&!D2&!D3&!D4]", "[!*]"),
        ("[#6&12*&!12C&!12c]", "[!*]"),
        ("[12*&C&!12C]", "[!*]"),
        ("[D{2-4}&D3&!D3]", "[!*]"),
        ("[!*,#6]", "[#6]"),
        ("[!*&#6]", "[!*]"),
        ("[!*,!*]", "[!*]"),
        ("[#6&!#6,#7]", "[#7]"),
        ("[#6&!#6,#7&!#7]", "[!*]"),
        ("[#6,!*]", "[#6]"),
        ("[#6,!#6&!*]", "[#6]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_isotope_and_numeric_atom_forms() {
    for (source, expected) in [
        ("[12C&#6]", "[12C]"),
        ("[12C&C]", "[12C]"),
        ("[12C&A]", "[12C]"),
        ("[12c&#6]", "[12c]"),
        ("[12c&c]", "[12c]"),
        ("[12c&a]", "[12c]"),
        ("[2H&#1]", "[2H]"),
        ("[12C,#6]", "[#6]"),
        ("[12C,C]", "[C]"),
        ("[12C,A]", "[A]"),
        ("[12c,#6]", "[#6]"),
        ("[12c,c]", "[c]"),
        ("[12c,a]", "[a]"),
        ("[12C&12*]", "[12C]"),
        ("[12C,12*]", "[12*]"),
        ("[12*&C]", "[12C]"),
        ("[12*&c]", "[12c]"),
        ("[12*&#6&A]", "[12C]"),
        ("[12C,12c]", "[#6&12*]"),
        ("[12*&C,12*&c]", "[#6&12*]"),
        ("[12*&C,12*]", "[12*]"),
        ("[12C,12c,12*]", "[12*]"),
        ("[12C,12c,#6]", "[#6]"),
        ("[12C,12c,!#6]", "[!#6,12*]"),
        ("[D1]", "[D]"),
        ("[D{1-1}]", "[D]"),
        ("[D{-0}]", "[D0]"),
        ("[D{0-}]", "*"),
        ("[X{0-}]", "*"),
        ("[v{0-}]", "*"),
        ("[x{0-}]", "*"),
        ("[H{0-}]", "*"),
        ("[h{0-}]", "*"),
        ("[R{0-}]", "*"),
        ("[r{0-}]", "*"),
        ("[z{0-}]", "*"),
        ("[Z{0-}]", "*"),
        ("[R{1-}]", "[R]"),
        ("[r{1-}]", "[r]"),
        ("[x{1-}]", "[x]"),
        ("[R&R{1-}]", "[R]"),
        ("[R,R{1-}]", "[R]"),
        ("[#6&D{0-}]", "[#6]"),
        ("[#6,D{0-}]", "*"),
        ("[X1]", "[X]"),
        ("[v1]", "[v]"),
        ("[H1]", "[#1]"),
        ("[#6&H1]", "[#6&H]"),
        ("[h1]", "[h]"),
        ("[z1]", "[z]"),
        ("[Z1]", "[Z]"),
        ("[!D0]", "[D{1-}]"),
        ("[!H0]", "[H{1-}]"),
        ("[!R0]", "[R]"),
        ("[!^0]", "[^{1-}]"),
        ("[R{1-1}]", "[R1]"),
        ("[r{5-5}]", "[r5]"),
        ("[x{1-1}]", "[x1]"),
        ("[D&D1]", "[D]"),
        ("[D,D1]", "[D]"),
        ("[R1&R]", "[R1]"),
        ("[R1,R]", "[R]"),
        ("[r5&r]", "[r5]"),
        ("[r5,r]", "[r]"),
        ("[D3&D{2-4}]", "[D3]"),
        ("[D3,D{2-4}]", "[D{2-4}]"),
        ("[D{2-3}&D{1-4}]", "[D{2-3}]"),
        ("[D{2-3},D{1-4}]", "[D{1-4}]"),
        ("[D{2-4}&D{3-5}]", "[D{3-4}]"),
        ("[D{2-4},D{3-5}]", "[D{2-5}]"),
        ("[R1&R{1-3}]", "[R1]"),
        ("[R1,R{2-3}]", "[R{1-3}]"),
        ("[r5&r{4-6}]", "[r5]"),
        ("[r4,r{5-6}]", "[r{4-6}]"),
        ("[D,D{2-}]", "[D{1-}]"),
        ("[D0,D{1-}]", "*"),
        ("[D{0-2},D{3-}]", "*"),
        ("[R,R0]", "*"),
        ("[R0,R{1-}]", "*"),
        ("[r,r0]", "*"),
        ("[x,x0]", "*"),
        ("[z,z0]", "[z{-1}]"),
        ("[Z,Z0]", "[Z{-1}]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_equivalent_bond_boolean_forms() {
    for (source, expected) in [
        ("[#6]-&~[#7]", "[#6]-[#7]"),
        ("[#6]-,~[#7]", "[#6]~[#7]"),
        ("[#6]@&~[#7]", "[#6]@[#7]"),
        ("[#6]-;@[#7]", "[#6]-&@[#7]"),
        ("[#6]-&!~[#7]", "[#6]!~[#7]"),
        ("[#6]-&!-[#7]", "[#6]!~[#7]"),
        ("[#6]~&!-&!:[#7]", "[#6]!-&!:[#7]"),
        ("[#6]@,!@[#7]", "[#6]~[#7]"),
        ("[#6]@&!@[#7]", "[#6]!~[#7]"),
        ("[#6]~&!@[#7]", "[#6]!@[#7]"),
        ("[#6]~&!~[#7]", "[#6]!~[#7]"),
        ("[#6]-,!-[#7]", "[#6]~[#7]"),
        ("[#6]-,!~[#7]", "[#6]-[#7]"),
        ("[#6]-,@&-[#7]", "[#6]-[#7]"),
        ("[#6]@,@&-[#7]", "[#6]@[#7]"),
        ("[#6]-&~&@[#7]", "[#6]-&@[#7]"),
        ("[#6]-;~;@[#7]", "[#6]-&@[#7]"),
        ("[#6]-&-&~[#7]", "[#6]-[#7]"),
        ("[#6]-,=,-[#7]", "[#6]-,=[#7]"),
        ("[#6]~,!~[#7]", "[#6]~[#7]"),
        ("[#6]-&@,-&!@[#7]", "[#6]-[#7]"),
        ("[#6]@&-,@&!-[#7]", "[#6]@[#7]"),
        ("[#6]-&=,-&!=[#7]", "[#6]-[#7]"),
        ("[#6]-;@,!@[#7]", "[#6]-[#7]"),
        ("[#6]@;-,!-[#7]", "[#6]@[#7]"),
        ("[#6]-;=,!=[#7]", "[#6]-[#7]"),
        ("[#6]@;!@,-[#7]", "[#6]-&@[#7]"),
        ("[#6]-&@,-&@&~[#7]", "[#6]-&@[#7]"),
        ("[#6]-&@&~,-&@[#7]", "[#6]-&@[#7]"),
        ("[#6]-&@,!@[#7]", "[#6]!@,-[#7]"),
        ("[#6]-&@,!~[#7]", "[#6]-&@[#7]"),
        ("[#6]-&@,!-&:,@&:[#7]", "[#6]-&@,:[#7]"),
        ("[#6]-,!-&@[#7]", "[#6]-,@[#7]"),
        ("[#6]=,!=&@[#7]", "[#6]=,@[#7]"),
        ("[#6]-,=;!-;!=[#7]", "[#6]!~[#7]"),
        ("[#6]=,:;!=;!:[#7]", "[#6]!~[#7]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_low_precedence_bond_complement_forms() {
    for (source, expected) in [
        ("[#6]~;!-,@[#7]", "[#6]!-,@[#7]"),
        ("[#6]~;!=,@[#7]", "[#6]!=,@[#7]"),
        ("[#6]~;!#,@[#7]", "[#6]!#,@[#7]"),
        ("[#6]~;!:,@[#7]", "[#6]!:,@[#7]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_low_precedence_absorption_forms() {
    for (source, expected) in [
        ("[#6;#6,#7]", "[#6]"),
        ("[#6;!#6,#7]", "[!*]"),
        ("[#6,#7;#6]", "[#6]"),
        ("[#6,!#6;#7]", "[#7]"),
        ("[#6,#7;!#6]", "[#7]"),
        ("[#6,!#6;!#7]", "[!#7]"),
        ("[#6;#7,#8]", "[!*]"),
        ("[#6,#7;#8]", "[!*]"),
        ("[#6,#7;!#8]", "[#6,#7]"),
        ("[#6,!#6;#6]", "[#6]"),
        ("[#6;R,#6]", "[#6]"),
        ("[C;R,C]", "[C]"),
        ("[R;#6,R]", "[R]"),
        ("[!#6;R,!#6]", "[!#6]"),
        ("[#6;!#6,R]", "[#6&R]"),
        ("[D2,D3;D{2-4}]", "[D{2-3}]"),
        ("[D{2-4};D2,D5]", "[D2]"),
        ("[D2,D3;!D2]", "[D3]"),
        ("[D2,D3;!D{2-3}]", "[!*]"),
        ("[C,c;#6]", "[#6]"),
        ("[C,c;A]", "[C]"),
        ("[C,c;!C]", "[c]"),
        ("[12C,13C;#6]", "[12*,13*;C]"),
        ("[12C,13C;12*]", "[12C]"),
        ("[12C,13C;!12*]", "[13C]"),
        ("[12C,12c;12*]", "[#6&12*]"),
        ("[#6]-;-,=[#7]", "[#6]-[#7]"),
        ("[#6]-;!-,=[#7]", "[#6]!~[#7]"),
        ("[#6]-,=;-[#7]", "[#6]-[#7]"),
        ("[#6]-,!-;=[#7]", "[#6]=[#7]"),
        ("[#6]-,=;!-[#7]", "[#6]=[#7]"),
        ("[#6]-,=;!:[#7]", "[#6]-,=[#7]"),
        ("[#6]-;=,#[#7]", "[#6]!~[#7]"),
        ("[#6]-,=;#[#7]", "[#6]!~[#7]"),
        ("[#6]-;@,-[#7]", "[#6]-[#7]"),
        ("[#6]@;-,@[#7]", "[#6]@[#7]"),
        ("[#6]-;!-,@[#7]", "[#6]-&@[#7]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }
}

#[test]
fn canonicalize_simplifies_low_precedence_distribution_forms() {
    for (source, expected) in [
        ("[#6,R;#6,H0]", "[#6,H0&R]"),
        ("[#6,R;#6,H0;#6,X4]", "[#6,H0&R&X4]"),
        ("[#6,R;#6,R,H0]", "[#6,R]"),
        ("[#6,R;#6,!R]", "[#6]"),
        ("[#6,R;!#6,R]", "[R]"),
        ("[#6,R;!#6,H0;R,H0]", "[!#6,H0;#6,R]"),
        ("[#6,R;!#6,H0;!R,H0]", "[#6,R;H0]"),
        ("[D{1-3};!D4;#6,D2]", "[#6&D{1-3},D2]"),
        ("[D{2-4};!D2&!D3&!D4,R]", "[D{2-4}&R]"),
        ("[R{1-3};!R1&!R2&!R3,#6]", "[#6&R{1-3}]"),
        ("[#6&12*;!12C&!12c,R]", "[#6&12*&R]"),
        ("[C,R;!C,H0;!R,H0]", "[C,R;H0]"),
        ("[#7,D2;!#7,X4;!D2,X4]", "[#7,D2;X4]"),
        ("[$([#6]),R;!$([#6]),H0;!R,H0]", "[#6,R;H0]"),
        ("[C,R;C,H0]", "[C,H0&R]"),
        ("[D2,R;D2,X4]", "[D2,R&X4]"),
        ("[$([#6]),R;$([#6]),H0]", "[#6,H0&R]"),
        ("[#6]-,@;-,=[#7]", "[#6]-,=&@[#7]"),
        ("[#6]-,@;-,!@[#7]", "[#6]-[#7]"),
        ("[#6]-,@;!-,@[#7]", "[#6]@[#7]"),
        ("[#6]-,@;-,:[#7]", "[#6]-,:&@[#7]"),
        ("[#6]=,@;=,#[#7]", "[#6]#&@,=[#7]"),
        ("[#6]-,=;!-&!=,@[#7]", "[#6]-,=;@[#7]"),
        ("[#6]-,@;!-&!@,=[#7]", "[#6]=&@[#7]"),
    ] {
        assert_eq!(canonical_string(source), expected, "{source}");
    }

    assert_same_canonical_group(&["[#6]-&@&=,:[#7]", "[#6]-&@&=,!-&:,!@&:,!=&:[#7]"]);
}

#[test]
fn canonicalize_sorts_bond_boolean_children() {
    assert_eq!(canonical_string("C-,=N"), canonical_string("C=,-N"));
    assert_eq!(canonical_string("C-;@N"), canonical_string("C@;-N"));
    assert_same_canonical_group(&["C-,=N", "C=,-N", "C=,-N"]);
    assert_same_canonical_group(&["C-;@N", "C@;-N"]);
}

#[test]
fn canonicalize_recursively_canonicalizes_nested_queries() {
    assert_eq!(canonical_string("[$(OC)]"), canonical_string("[$(CO)]"));
    assert_eq!(
        canonical_string("[$([H1;C])]"),
        canonical_string("[$([C;H1])]")
    );
    assert_same_canonical_group(&["[$(OC)]", "[$(CO)]"]);
    assert_eq!(canonical_string("[$((CO))]"), "[$(CO)]");
    assert_eq!(canonical_string("[$(*)]"), "*");
    assert_eq!(canonical_string("[$([#6])]"), "[#6]");
    assert_eq!(canonical_string("[$([C])]"), "[C]");
    assert_eq!(canonical_string("[$([#6])&R]"), "[#6&R]");
    assert_eq!(canonical_string("[#6&$([#6])]"), "[#6]");
    assert_eq!(canonical_string("[$([#6:1])]"), "[$([#6:1])]");
    assert_eq!(canonical_string("[!$([#6&-])]"), "[!#6,!-]");
    assert_eq!(canonical_string("[!$([#6,-])]"), "[!#6&!-]");
    assert_eq!(canonical_string("[!$([#6;R])&A]"), "[!#6,!R;A]");
    assert_eq!(canonical_string("[!$([#6;R])&a]"), "[!#6,!R;a]");
    assert_eq!(canonical_string("[!$([C;R])&A]"), "[!C,!R;A]");
    assert_eq!(canonical_string("[$([#6;R])&A,!$([#6;R])&A]"), "[A]");
    assert_eq!(
        canonical_string("[$([#6;R])&A&H0,!$([#6;R])&A&H0]"),
        "[A&H0]"
    );
    assert_eq!(canonical_string("[$([#6;R,H0])&A,!$([#6;R,H0])&A]"), "[A]");
    assert_eq!(canonical_string("[$([C;R,H0])&A,!$([C;R,H0])&A]"), "[A]");
    assert_eq!(canonical_string("[$([#6;R])&a,!$([#6;R])&a]"), "[a]");
    assert_eq!(canonical_string("[$([C;R])&A,!$([C;R])&A]"), "[A]");
    assert_eq!(canonical_string("[$([#6]),$([#6;R])]"), "[#6]");
    assert_eq!(canonical_string("[$([#6]),$([#6,#7;R])]"), "[#6,#7&R]");
    assert_eq!(canonical_string("[$([#6]),$([C])]"), "[#6]");
    assert_eq!(canonical_string("[$([C]),$([C,N;R])]"), "[C,N&R]");
    assert_eq!(canonical_string("[$([C]),$([C;R])]"), "[C]");
    assert_eq!(canonical_string("[$([#6&R]),$([#6&R;X4])]"), "[#6&R]");
    assert_all_atom_permutations_converge("[$(CO)]N");
    assert_all_atom_permutations_converge("C[$([O,N])]");
}

#[test]
fn canonicalize_runs_expression_simplifications_through_recursion_and_bonds() {
    assert_eq!(canonical_string("[!!#6]"), "[#6]");
    assert_eq!(canonical_string("[!!-2]"), "[-2]");
    assert_eq!(canonical_string("[!!h4&#51]"), "[#51&h4]");
    assert_eq!(canonical_string("[$([!!#6;*])]"), "[#6]");

    let query = QueryMol::from_str("C-N").unwrap();
    let mut bonds = query.bonds().to_vec();
    bonds[0].expr = BondExpr::Query(BondExprTree::Not(Box::new(BondExprTree::Not(Box::new(
        BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
    )))));
    let query = QueryMol::from_parts(
        query.atoms().to_vec(),
        bonds,
        query.component_count(),
        query.component_groups().to_vec(),
    );

    assert_eq!(query.canonicalize().to_string(), "C-N");

    let query = QueryMol::from_str("C-N").unwrap();
    let mut bonds = query.bonds().to_vec();
    let single = BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single));
    let ring = BondExprTree::Primitive(BondPrimitive::Ring);
    let aromatic = BondExprTree::Primitive(BondPrimitive::Bond(Bond::Aromatic));
    bonds[0].expr = BondExpr::Query(BondExprTree::Or(vec![
        single.clone(),
        BondExprTree::LowAnd(vec![BondExprTree::Or(vec![single, ring]), aromatic]),
    ]));
    let query = QueryMol::from_parts(
        query.atoms().to_vec(),
        bonds,
        query.component_count(),
        query.component_groups().to_vec(),
    );

    assert_eq!(query.canonicalize().to_string(), "C-,:&@N");
}

#[test]
fn canonicalize_handles_ga_reported_recursive_query() {
    let source = "*~[!R&v11:62086](/[!!h4&#51&-3&@SP3&X,X10,a&^3&v5]~[!$([Fe]~[#8])&R:1432](*=[@SP1:39279]:[!#6&$(*)&H0])/[#6:42118]~[#6&@TH2&z{-16}:55558])=[#6:55558]~[#6&$(*):55558].*([!$([#8])&R:1432]~[@][R{-16}&h&z{-12}:24468])=[@SP1:39279]~[!R&z]=[@]";
    let canonical = canonical_string(source);

    assert!(!canonical.contains("!!"), "{canonical}");
    let reparsed = QueryMol::from_str(&canonical).unwrap();
    assert_eq!(reparsed.canonicalize().to_string(), canonical);
}

#[test]
fn canonicalize_handles_cytosporins_slow_reports() {
    for (source, expected) in [
        (
            "*@[B,X{16-}].[!R,b].([!#16]~[!#8].[!107*&r6]-[#6&$([#6,#7])&H]=&@[D3])",
            "*@[B,X{16-}].[!R,b].([!#16]~[!#8].[!107*&r6]-[#6&H]=&@[D3])",
        ),
        (
            "*-,:,=C.(*(!=[!-]-&@1)!-[#6]-&@1~[!#6&H]-[!Cl].[!#6&H]-[!Cl]-,/&@[#6&R])",
            "*-,:,=C.(*(!=[!-]-&@1)!-[#6]-&@1~[!#6&H]-[!Cl].[!#6&H]-[!Cl]-,/&@[#6&R])",
        ),
    ] {
        let canonical = canonical_string(source);
        assert_eq!(canonical, expected, "{source}");
        let reparsed = QueryMol::from_str(&canonical).unwrap();
        assert_eq!(reparsed.canonicalize().to_string(), canonical);
    }
}

#[test]
fn canonicalize_handles_matching_timeout_report() {
    let source = "*:[D3&$([#6,#7])&H0;C:8481](-;@[R&v2;H0;C]).[!R:28527][!R,H0&R&v2;-2;!R].[*;$([!#6,#7]!-[$([#6,#7])])](#[r6,D3,X3;!R&X3&$([#6,#7])]).[$([#6,#7])&X16&!R;H0]:[A,D3&!C;v2;C](=[N])";
    let expected = "*:[C&D3&H0:8481]-&@[C&H0&R&v2].[!R&-2][!R:28527].[!R;#6,#7;H0;X16]:[C&v2]=[N].[!R;#6,#7;X3]#[$([!#6]!-[#6,#7])]";
    let canonical = canonical_string(source);

    assert_eq!(canonical, expected);
    let reparsed = QueryMol::from_str(&canonical).unwrap();
    assert_eq!(reparsed.canonicalize().to_string(), canonical);
}

#[test]
fn canonicalize_preserves_group_structure_but_sorts_group_contents() {
    assert_eq!(canonical_string("(N.C)"), canonical_string("(C.N)"));
    assert_ne!(canonical_string("(C.N)"), canonical_string("C.N"));
}

#[test]
fn canonicalize_converges_ring_and_symmetric_permutations() {
    assert_same_canonical_group(&["C1CC1O", "OC1CC1", "C1(O)CC1"]);
    assert_all_atom_permutations_converge("C1CC1");
    assert_all_atom_permutations_converge("C1CC1O");
    assert_all_atom_permutations_converge("C1NC1O");
    assert_all_atom_permutations_converge("C1=CC=C1");
}

#[test]
fn canonicalize_handles_parallel_bonds_between_the_same_atoms() {
    let query = QueryMol::from_str("C1C1").unwrap();

    let canonical = query.canonicalize();
    assert!(canonical.is_canonical());
    assert_eq!(canonical.atom_count(), 2);
    assert_eq!(canonical.bond_count(), 2);
    assert_eq!(canonical.to_string(), canonical.canonicalize().to_string());
    assert_eq!(
        canonical,
        QueryMol::from_str(&canonical.to_string())
            .unwrap()
            .canonicalize()
    );
}

#[test]
fn canonicalize_makes_atomic_hydrogen_unambiguous_in_recursive_queries() {
    let query = QueryMol::from_str("[!$([H-])]").unwrap();

    let canonical = query.canonicalize();
    let rendered = canonical.to_string();
    let reparsed = QueryMol::from_str(&rendered).unwrap();

    assert!(canonical.is_canonical());
    assert_eq!(rendered, "[!#1,!-]");
    assert_eq!(canonical, reparsed.canonicalize());
}

#[test]
fn canonicalize_handles_disconnected_sulfur_rich_fuzz_artifact() {
    assert_canonical_roundtrips(
        "CCCC.OCCCSSSSSSSSSSCC.CCC.OCCCC.CCC.OCCCSSSSSSSSSSCCSOSSSSSSSSSCC.OCOC",
    );
}

#[test]
fn canonicalize_handles_recursive_high_and_fuzz_artifact() {
    assert_canonical_roundtrips("[$(COcccccccc)]([RRRRRRRRRSmRR+])");
}

#[test]
fn canonicalize_handles_recursive_oom_fuzz_artifact() {
    let source = String::from_utf8(vec![
        67, 67, 67, 67, 91, 36, 40, 67, 115, 91, 36, 40, 42, 65, 65, 70, 80, 80, 80, 80, 80, 80,
        80, 80, 80, 80, 67, 80, 80, 80, 80, 80, 64, 67, 45, 67, 65, 99, 42, 79, 46, 98, 42, 42, 42,
        42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        42, 42, 42, 42, 42, 42, 42, 42, 42, 91, 43, 72, 93, 42, 42, 42, 42, 42, 42, 46, 115, 42,
        42, 42, 42, 42, 65, 67, 65, 80, 80, 80, 79, 41, 93, 79, 110, 79, 79, 41, 93, 79, 46, 79,
        80, 64, 65, 65, 67, 67, 65, 65, 65, 65,
    ])
    .unwrap();
    assert_canonical_roundtrips(&source);
}

#[test]
fn canonicalize_handles_recursive_component_oom_fuzz_artifact() {
    assert_canonical_roundtrips("[$(Cs[$(*.sO)]O.OO)]O.O");
}

#[test]
fn canonicalize_handles_recursive_expression_oom_fuzz_artifact() {
    let source = String::from_utf8(vec![
        0x5b, 0x21, 0x24, 0x28, 0x5b, 0x43, 0x21, 0x24, 0x28, 0x5b, 0x42, 0x72, 0x4e, 0x40, 0x48,
        0x2d, 0x21, 0x24, 0x28, 0x5b, 0x43, 0x2c, 0x4e, 0x40, 0x48, 0x2d, 0x42, 0x72, 0x5d, 0x29,
        0x42, 0x72, 0x5d, 0x29, 0x4e, 0x41, 0x2d, 0x26, 0x42, 0x72, 0x5d, 0x29, 0x5d, 0x0a,
    ])
    .unwrap();
    assert_canonical_roundtrips(source.trim_end());
}

#[test]
fn canonicalize_handles_recursive_disconnected_triple_oom_fuzz_artifact() {
    assert_canonical_roundtrips("[$(C#[$(CsO)].OO)]O.O");
}

#[test]
fn canonicalize_handles_duplicate_wildcard_bracket_fuzz_artifact() {
    let source = String::from_utf8(vec![0x5b, 0x2a, 0x2a, 0x5d, 0x0c, 0x43]).unwrap();
    assert_canonical_roundtrips(&source);
}

#[test]
fn canonicalize_handles_unpaired_directional_single_bond_fuzz_artifact() {
    assert_canonical_roundtrips("C/C");
}

#[test]
fn canonicalize_handles_repeated_directional_single_bond_fuzz_artifact() {
    assert_canonical_roundtrips("CC//C");
}

#[test]
fn canonicalize_handles_directional_single_bond_conjunction_fuzz_artifact() {
    assert_canonical_roundtrips(r"C\;-O\O/n");
}

#[test]
fn canonicalize_handles_lone_bracket_hydrogen_fuzz_artifact() {
    assert_canonical_roundtrips("C. [*H].O");
}

#[test]
fn canonicalize_handles_repeated_ring_bond_conjunction_fuzz_artifact() {
    assert_canonical_roundtrips(r"CC\--@@@@NCC");
}

#[test]
fn canonicalize_handles_negated_directional_single_bond_fuzz_artifact() {
    assert_canonical_roundtrips("ACPaOCCCCCC!/CC");
}

#[test]
fn canonicalize_handles_recursive_hydrogen_bundle_fuzz_artifact() {
    assert_canonical_roundtrips("[!$([N;Tm*H-;H-])]");
}

#[test]
fn canonicalize_handles_recursive_oom_bundle_fuzz_artifact() {
    assert_canonical_roundtrips("[$(Cs[$(CsO[$(Cs[$(CsO)]O.OO)]O.O)]O.OO)]O.O");
}

#[test]
fn canonicalize_handles_deep_recursive_component_chain_regression() {
    let source = recursive_component_chain(15);
    let query = QueryMol::from_str(&source).unwrap();
    // The historical failure was explosive repeated work on this small top-level query.
    let rendered = query.to_string();
    let reparsed = QueryMol::from_str(&rendered).unwrap();

    assert_eq!(query.atom_count(), 3);
    assert_eq!(query.bond_count(), 1);
    assert_eq!(query.component_count(), 2);
    assert_eq!(query.atom_count(), reparsed.atom_count());
    assert_eq!(query.bond_count(), reparsed.bond_count());
    assert_eq!(query.component_count(), reparsed.component_count());
    assert_canonical_roundtrips(&source);
}

#[test]
fn canonicalize_handles_repeated_ring_topology_bundle_fuzz_artifact() {
    assert_canonical_roundtrips("C-;@;@@CCC-;@CC--;@CP-;@CC@a");
}

#[test]
fn canonicalize_handles_directional_bond_bundle_fuzz_artifact() {
    assert_canonical_roundtrips(r"C#[R0]\\;\\\\,\\\\\\\\\\C\\\\\\\\\\\\\\C\:CCC");
}

#[test]
fn canonicalize_handles_directional_bond_bundle_2_fuzz_artifact() {
    assert_canonical_roundtrips(r"CC:CC\\\\\\,\\\\\\\\#\\\\\\\\\\\\\\\\\\\\\\\\C:CCACC");
}

#[test]
fn canonicalize_handles_low_and_disjunction_display_fuzz_artifact() {
    assert_canonical_roundtrips("F[!r12rSr,b]");
}

#[test]
fn canonicalize_handles_directional_disjunction_roundtrip_fuzz_artifact() {
    assert_canonical_roundtrips("A@@/;@;/@~~~~~~,~~~~~~A:::~~,~~~~~-~,~~@/&\\,@/;/&\\,@I\n");
}

#[test]
fn canonicalize_handles_directional_low_and_relabel_fuzz_artifact() {
    for source in [
        "A/,:!/;!@,!/;!@,/N",
        "A/,!/@@,:!/;!@,:!/;!@,/N",
        "A:/,!@,!/@!:;@,:!@,-!:,!:/N",
    ] {
        assert_canonical_roundtrips(source);
        assert_all_atom_permutations_converge(source);
    }
}

#[test]
fn canonicalize_handles_directional_low_and_stack_fuzz_artifact() {
    let source = "A@@/;/@/\\&@/@;~~~@\\&@,@/@;~~~~\\,@/;~~~~/;~/-I\n";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_covered_bond_disjunction_relabel_fuzz_artifact() {
    for source in [
        "N!@=,/,:;!@,:N@:A",
        "A/,!/@;@@@,:!!@@,:!!:,=!@,:!/;!@,::,:!@,:,:!!:,=!@,:!/;!@,::,:!@,:!/;!@,::,:!!!@@,:!!:,=!@,:!/;!@,::,:@,:!/;!@,:!/:,:!/;!@,:!/::N/,@@,:!!:,=!@,:!/;!@,::,:@,:!/;!@,:!/:,:!/;!@,:!/::,/N",
    ] {
        assert_canonical_roundtrips(source);
        assert_all_atom_permutations_converge(source);
    }
}

#[test]
fn canonicalize_handles_complementary_bond_disjunction_stack_fuzz_artifact() {
    let source = "A!/@;@@,!/;!@,::,:C@,:!/Cl@,:!//,!/@;::,:C@,:,!/;!@,!/::,/N";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_pruned_bond_disjunction_product_fuzz_artifact() {
    for source in ["N!@-,:;/,@A", "N!-@,!@:;!/@,:A"] {
        assert_canonical_roundtrips(source);
        assert_all_atom_permutations_converge(source);
    }
}

#[test]
fn canonicalize_handles_rotated_bond_disjunction_cover_fuzz_artifact() {
    let source = "A@@o@~~~~~~,~~~~-~/,@@/;@/@~~~~~~,~~~~-~~,~~~~-@~Ac~@@~~-///:::::/I\n";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_negated_directional_ring_bundle_fuzz_artifact() {
    let source = "A:,-!:!@,-!/,:!//;@,::,/,=!@,:!/;!@,:@,-!@,!:~!/;!@,::,-!:!@,-!/,:!/;@,::,/N";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_directional_low_and_distribution_fuzz_artifact() {
    let source = "A/@,:!@,-!:,~~~/@,:!@,-!:!:&!@!:!:&!@!/N!/;!@@,!!~~~/@,::,-!:,~~~/@,:!@,-!:!!/;!@@,!!~~~!@,-!/N!@N";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_directional_low_and_product_fuzz_artifact() {
    let source = "A/,!/@,:!/:,:!!/;!@!@,:::@,-!:~~~/@,:!@,-!:!@,-!/,:!/;!@,::,/N";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_directional_low_and_term_distribution_fuzz_artifact() {
    let source = "A/,!@,:!/;!@,:-!@,:!/;!@!/;!@,:!/::/;@@!@,:!!@@,:!!:,=!@,:!:-:!@,:!!:,-!@,:!/;!@,=!@,::,/N";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_partitioned_directional_ring_fuzz_artifact() {
    let source = "A/@,@,/,@,@,/,:!/;!!@,:!/;!:,@,/,@,@,/,:!/;!!@,!/;!!@,:!/;!@,!/N";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_partitioned_directional_single_fuzz_artifact() {
    let source = "A/,!/@,-!:!@,-!/,:!/;!@,!:~!@,:!@,-!/,:!/;!@,!:~@,:@,=!:~!/;!@,:!@,-!/,:!::,/N";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_partition_relaxation_cycle_fuzz_artifact() {
    let source = "A/@,:@,!@,!:!/;/N";
    assert_canonical_roundtrips(source);
    assert_all_atom_permutations_converge(source);
}

#[test]
fn canonicalize_handles_unbounded_hybridization_range_fuzz_artifact() {
    assert_canonical_roundtrips("[^{0-}]");
    assert_eq!(canonical_string("[^{0-}]"), "*");
}

#[test]
fn canonicalize_handles_chiral_directional_timeout_fuzz_artifact() {
    assert_canonical_roundtrips(r"C\[Cn@@@@@B@A@B@A@@N@]\\C");
}

#[test]
fn canonicalize_handles_trigonal_bipyramidal_parallel_bond_fuzz_artifact() {
    assert_canonical_roundtrips(r"[@PasBP]17O[@TBP]17O");
}

#[test]
fn canonicalize_handles_underconstrained_ring_chirality_fuzz_artifact() {
    assert_canonical_roundtrips("C[No]1[21Al43AlB]2[21AlBNo@@]1[21Al24AlB]2[21Al]");
}

#[test]
fn canonicalize_handles_underconstrained_ring_chirality_with_h_fuzz_artifact() {
    assert_canonical_roundtrips("C[No]1[21Al43AlB]2[21AlBNo@H]1[21Al40AlB]2[21Al]");
}

#[test]
fn canonicalize_handles_fused_ring_generic_chirality_fuzz_artifact() {
    assert_canonical_roundtrips("c[@]8[C@@]71cOC7=8s1");
}

#[test]
fn canonicalize_is_stable_under_manual_graph_reordering() {
    let query = QueryMol::from_str("[$(CO)]C1NC1.O").unwrap();
    let canonical = query.canonicalize();

    let atom_order = vec![4, 2, 0, 3, 1];
    let permuted_atoms = permute_atoms(&query, &atom_order);
    let permuted_both = permute_top_level_entries(&permuted_atoms, &[1, 0]);

    assert_eq!(canonical, permuted_atoms.canonicalize());
    assert_eq!(canonical, permuted_both.canonicalize());
}

#[test]
fn canonicalize_preserves_distinctions_between_non_equivalent_queries() {
    assert_ne!(canonical_string("C"), canonical_string("[#6]"));
    assert_ne!(canonical_string("C-N"), canonical_string("C=N"));
    assert_ne!(canonical_string("C.N"), canonical_string("(C.N)"));
    assert_ne!(canonical_string("[$(CO)]"), canonical_string("[$(CN)]"));
    assert_ne!(canonical_string("[C;H1]"), canonical_string("[C;H2]"));
    assert_ne!(canonical_string("C1CC1"), canonical_string("CCC"));
}

#[test]
fn canonicalize_recursive_charge_bundle_subcase_redundant_wildcard_charge_converges() {
    assert_eq!(
        canonical_string("[$([*+]~[*-])]"),
        canonical_string("[$([+]~[-])]")
    );
}

#[test]
fn canonicalize_recursive_charge_bundle_subcase_x4_charge_rooting_converges() {
    assert_eq!(
        canonical_string("[$([OX1-,OH1][#7X4+]([*])([*])([*]))]"),
        canonical_string("[$(*[#7&+&X4](*)(*)[-&O&X1,H1&O])]")
    );
}

#[test]
fn canonicalize_recursive_charge_bundle_subcase_x4v5_rooting_converges() {
    assert_eq!(
        canonical_string("[$([OX1]=[#7X4v5]([*])([*])([*]))]"),
        canonical_string("[$(*[#7&X4&v5](*)(*)=[O&X1])]")
    );
}
