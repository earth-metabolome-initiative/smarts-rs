#![no_std]
#![doc = include_str!("../README.md")]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

#[cfg(feature = "serde")]
use alloc::string::ToString;
use core::str::FromStr;

mod bracket;
mod canonicalization;
mod edit;
mod match_error;
mod parse;
mod parse_error;
pub(crate) mod query;
mod validate;

/// Parse and match error types.
pub mod error {
    pub use crate::match_error::SmartsMatchError;
    pub use crate::parse_error::{SmartsParseError, SmartsParseErrorKind, UnsupportedFeature};
}
/// Target graph adapters backed by `geometric-traits`.
pub mod geometric_target;
/// SMARTS matching methods and compiled-query state.
pub mod matching;
/// Prepared target sidecars and cached molecule properties.
pub mod prepared;
/// Conservative screening summaries for many-query workflows.
pub mod screening;
/// Target graph traits and simple target labels.
pub mod target;

#[doc(hidden)]
pub use bracket::{
    parse_bracket_text as fuzz_parse_bracket_text, BracketParseError as FuzzBracketParseError,
    BracketParseErrorKind as FuzzBracketParseErrorKind,
};
pub use canonicalization::QueryCanonicalLabeling;
pub use edit::{
    add_atom_primitive, add_bond_primitive, normalize_bond_tree, normalize_bracket_tree,
    remove_atom_primitive, remove_bond_primitive, replace_atom_primitive, replace_bond_primitive,
    EditError, EditableQueryMol, ExprPath, ExprPathSegment,
};
pub use error::{SmartsMatchError, SmartsParseError, SmartsParseErrorKind, UnsupportedFeature};
pub use matching::{CompiledQuery, MatchLimitResult, MatchScratch};
pub use parse::parse_smarts;
pub use prepared::{EdgeProps, NodeProps, PreparedMolecule, PreparedTarget};
pub use query::{
    AtomExpr, AtomId, AtomPrimitive, BondExpr, BondExprTree, BondId, BondPrimitive, BracketExpr,
    BracketExprTree, ComponentGroupId, ComponentId, HydrogenKind, NumericQuery, NumericRange,
    QueryAtom, QueryBond, QueryMol,
};
pub use screening::{
    QueryScreen, QueryScreenFeatureStats, TargetCandidateSet, TargetCorpusIndex,
    TargetCorpusIndexStats, TargetCorpusScratch, TargetScreen,
};
pub use smiles_parser::atom::bracketed::chirality::Chirality;
pub use target::{AtomLabel, BondLabel, MoleculeTarget, Neighbor};
pub use validate::{recursive_depth, validate_recursive_depth, QueryValidationError};

impl FromStr for QueryMol {
    type Err = SmartsParseError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_smarts(s)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for QueryMol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for QueryMol {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let smarts = <alloc::string::String as serde::Deserialize>::deserialize(deserializer)?;
        smarts.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{boxed::Box, collections::BTreeMap, format, string::ToString, vec, vec::Vec};
    use elements_rs::{Element, Isotope};
    use smiles_parser::bond::Bond;

    fn assert_same_query_structure(left: &QueryMol, right: &QueryMol) {
        assert_eq!(left.atom_count(), right.atom_count());
        assert_eq!(left.bond_count(), right.bond_count());
        assert_eq!(left.component_count(), right.component_count());
        assert_eq!(left.component_groups(), right.component_groups());

        let left_atoms = left
            .atoms()
            .iter()
            .map(|atom| (atom.component, atom.expr.to_string()))
            .collect::<Vec<_>>();
        let right_atoms = right
            .atoms()
            .iter()
            .map(|atom| (atom.component, atom.expr.to_string()))
            .collect::<Vec<_>>();
        assert_eq!(left_atoms, right_atoms);

        let left_bonds = left
            .bonds()
            .iter()
            .map(|bond| {
                (
                    bond.src.min(bond.dst),
                    bond.src.max(bond.dst),
                    bond.expr.clone(),
                )
            })
            .collect::<Vec<_>>();
        let right_bonds = right
            .bonds()
            .iter()
            .map(|bond| {
                (
                    bond.src.min(bond.dst),
                    bond.src.max(bond.dst),
                    bond.expr.clone(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(multiset_counts(&left_bonds), multiset_counts(&right_bonds));
    }

    fn multiset_counts<T>(items: &[T]) -> BTreeMap<T, usize>
    where
        T: Ord + Clone,
    {
        let mut counts = BTreeMap::new();
        for item in items {
            *counts.entry(item.clone()).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn rejects_empty_input() {
        let err = parse_smarts("").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::EmptyInput);
    }

    #[test]
    fn query_mol_implements_from_str() {
        let parsed = "C=C".parse::<QueryMol>().unwrap();
        let direct = parse_smarts("C=C").unwrap();
        assert_same_query_structure(&parsed, &direct);
        assert_eq!(parsed.to_string(), "C=C");
    }

    #[test]
    fn parses_single_atom() {
        let query = parse_smarts("C").unwrap();
        assert_eq!(query.to_string(), "C");
        assert_eq!(query.atom_count(), 1);
        assert_eq!(query.bond_count(), 0);
        assert_eq!(query.component_count(), 1);
    }

    #[test]
    fn parses_linear_chain_into_bonds() {
        let query = parse_smarts("CC").unwrap();
        assert_eq!(query.atom_count(), 2);
        assert_eq!(query.bond_count(), 1);
        assert_eq!(query.component_count(), 1);
        assert_eq!(query.bonds()[0].expr, BondExpr::Elided);
    }

    #[test]
    fn parses_explicit_double_bond() {
        let query = parse_smarts("C=C").unwrap();
        assert_eq!(query.atom_count(), 2);
        assert_eq!(query.bond_count(), 1);
        assert_eq!(
            query.bonds()[0].expr,
            BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)))
        );
    }

    #[test]
    fn parses_multiple_components() {
        let query = parse_smarts("C.C").unwrap();
        assert_eq!(query.atom_count(), 2);
        assert_eq!(query.bond_count(), 0);
        assert_eq!(query.component_count(), 2);
    }

    #[test]
    fn preserves_recursive_smarts_in_brackets() {
        let query = parse_smarts("[$(CO)]CO").unwrap();
        let atom = &query.atoms()[0];

        match &atom.expr {
            AtomExpr::Bracket(expr) => {
                let BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(query)) = &expr.tree
                else {
                    panic!("expected lowered recursive SMARTS, got {:?}", expr.tree);
                };
                assert_eq!(query.to_string(), "CO");
                assert_eq!(query.atom_count(), 2);
                assert_eq!(query.bond_count(), 1);
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn preserves_recursive_smarts_with_nested_brackets() {
        let query = parse_smarts("[$([OH])]").unwrap();
        let atom = &query.atoms()[0];

        match &atom.expr {
            AtomExpr::Bracket(expr) => {
                let BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(query)) = &expr.tree
                else {
                    panic!("expected lowered recursive SMARTS, got {:?}", expr.tree);
                };
                assert_eq!(query.to_string(), "[O&H]");
                assert_eq!(query.atom_count(), 1);
                assert_eq!(query.bond_count(), 0);
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn parses_bracket_low_and() {
        let query = parse_smarts("[C;H1]").unwrap();
        let atom = &query.atoms()[0];
        match &atom.expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::LowAnd(vec![
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::C,
                            aromatic: false,
                        }),
                        BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                            HydrogenKind::Total,
                            Some(NumericQuery::Exact(1)),
                        )),
                    ])
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn parses_atom_maps_at_end_of_bracket_atoms() {
        let query = parse_smarts("[C:1]").unwrap();
        match &query.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Primitive(AtomPrimitive::Symbol {
                        element: Element::C,
                        aromatic: false,
                    })
                );
                assert_eq!(expr.atom_map, Some(1));
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        let query = parse_smarts("[C,N,O:12]").unwrap();
        match &query.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(expr.atom_map, Some(12));
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Or(vec![
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::C,
                            aromatic: false,
                        }),
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::N,
                            aromatic: false,
                        }),
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::O,
                            aromatic: false,
                        }),
                    ])
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        assert_eq!(query.to_string(), "[C,N,O:12]");
    }

    #[test]
    fn parses_bracket_precedence() {
        let query = parse_smarts("[c,n&H1]").unwrap();
        let atom = &query.atoms()[0];
        match &atom.expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Or(vec![
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::C,
                            aromatic: true,
                        }),
                        BracketExprTree::HighAnd(vec![
                            BracketExprTree::Primitive(AtomPrimitive::Symbol {
                                element: Element::N,
                                aromatic: true,
                            }),
                            BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                                HydrogenKind::Total,
                                Some(NumericQuery::Exact(1)),
                            )),
                        ]),
                    ])
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn parses_multiletter_bracket_elements_that_overlap_primitives() {
        let query = parse_smarts("[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]").unwrap();
        assert_eq!(
            query.to_string(),
            "[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]"
        );

        let AtomExpr::Bracket(expr) = &query.atoms()[0].expr else {
            panic!("expected bracket atom");
        };
        let BracketExprTree::Or(items) = &expr.tree else {
            panic!("expected bracket OR expression");
        };

        assert!(items.iter().any(|item| {
            *item
                == BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::Dy,
                    aromatic: false,
                })
        }));
    }

    #[test]
    fn parses_negated_atomic_number() {
        let query = parse_smarts("[!#1]").unwrap();
        let atom = &query.atoms()[0];
        match &atom.expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Not(Box::new(BracketExprTree::Primitive(
                        AtomPrimitive::AtomicNumber(1),
                    )))
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn parses_simple_ring_closure() {
        let query = parse_smarts("C1CC1").unwrap();
        assert_eq!(query.atom_count(), 3);
        assert_eq!(query.bond_count(), 3);
        assert_eq!(query.component_count(), 1);
    }

    #[test]
    fn parses_branching_substructure() {
        let query = parse_smarts("C(C)C").unwrap();
        assert_eq!(query.atom_count(), 3);
        assert_eq!(query.bond_count(), 2);
        assert_eq!(query.component_count(), 1);
    }

    #[test]
    fn parses_percent_ring_closure() {
        let query = parse_smarts("C%12CC%12").unwrap();
        assert_eq!(query.atom_count(), 3);
        assert_eq!(query.bond_count(), 3);
        assert_eq!(query.component_count(), 1);
    }

    #[test]
    fn rejects_self_loop_ring_closures() {
        for input in ["C00", "C11"] {
            let err = parse_smarts(input).unwrap_err();
            assert_eq!(err.kind(), SmartsParseErrorKind::SelfLoopRingClosure);
        }
    }

    #[test]
    fn rejects_ring_closures_across_disconnected_components() {
        for input in ["C9C98.C8", "[$(C9C98.C8)]"] {
            let err = parse_smarts(input).unwrap_err();
            assert_eq!(err.kind(), SmartsParseErrorKind::CrossComponentRingClosure);
            assert_eq!(err.code(), "cross_component_ring_closure");
        }
    }

    #[test]
    fn roundtrips_parse_display_through_ambiguous_bare_element_boundary() {
        let query = parse_smarts("C(ss\nCCNNCC)").unwrap();
        let rendered = query.to_string();
        let reparsed = parse_smarts(&rendered).unwrap();
        assert_eq!(rendered, reparsed.to_string());
        assert_same_query_structure(&query, &reparsed);
    }

    #[test]
    fn roundtrips_recursive_queries_through_ambiguous_bare_element_boundary() {
        let query = parse_smarts("C(oOO)").unwrap();
        let rendered = query.to_string();
        let reparsed = parse_smarts(&rendered).unwrap();
        assert_eq!(rendered, reparsed.to_string());
        assert_same_query_structure(&query, &reparsed);
    }

    #[test]
    fn roundtrips_wrapped_bracket_text_with_multiring_recursive_payload() {
        let expr = fuzz_parse_bracket_text("r$(C7(C7%(88))%(88))S").unwrap();
        let rendered_expr = expr.to_string();
        let query = parse_smarts(&format!("[{rendered_expr}]")).unwrap();
        let rendered_query = query.to_string();
        let reparsed = parse_smarts(&rendered_query).unwrap();
        assert_eq!(rendered_query, reparsed.to_string());
        assert_same_query_structure(&query, &reparsed);
    }

    #[test]
    fn parses_isotope_and_degree_primitives() {
        let isotope = parse_smarts("[12C]").unwrap();
        let isotope_wildcard = parse_smarts("[89*]").unwrap();
        let degree = parse_smarts("[D3]").unwrap();
        let ring_connectivity = parse_smarts("[x2]").unwrap();

        match &isotope.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Primitive(AtomPrimitive::Isotope {
                        isotope: Isotope::try_from((Element::C, 12_u16)).unwrap(),
                        aromatic: false,
                    })
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        match &degree.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Primitive(AtomPrimitive::Degree(
                        Some(NumericQuery::Exact(3),)
                    ))
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        match &isotope_wildcard.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Primitive(AtomPrimitive::IsotopeWildcard(89))
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        match &ring_connectivity.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Primitive(AtomPrimitive::RingConnectivity(Some(
                        NumericQuery::Exact(2),
                    )))
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn parses_atom_stereo_queries() {
        let clockwise = parse_smarts("[C@H]").unwrap();
        let counter = parse_smarts("[C@@H]").unwrap();
        let tetrahedral = parse_smarts("[C@TH1]").unwrap();
        let aromatic = parse_smarts("[n@H]").unwrap();

        match &clockwise.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::HighAnd(vec![
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::C,
                            aromatic: false,
                        }),
                        BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::At)),
                        BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                            HydrogenKind::Total,
                            None,
                        )),
                    ])
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        match &counter.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::HighAnd(vec![
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::C,
                            aromatic: false,
                        }),
                        BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::AtAt)),
                        BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                            HydrogenKind::Total,
                            None,
                        )),
                    ])
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        match &tetrahedral.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::HighAnd(vec![
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::C,
                            aromatic: false,
                        }),
                        BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::TH(1))),
                    ])
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        match &aromatic.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::HighAnd(vec![
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::N,
                            aromatic: true,
                        }),
                        BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::At)),
                        BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                            HydrogenKind::Total,
                            None,
                        )),
                    ])
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn parses_bracketed_non_organic_elements() {
        let sodium = parse_smarts("[Na]").unwrap();
        let platinum = parse_smarts("[Pt]").unwrap();
        let oganesson = parse_smarts("[Og]").unwrap();

        for (query, element) in [
            (&sodium, Element::Na),
            (&platinum, Element::Pt),
            (&oganesson, Element::Og),
        ] {
            match &query.atoms()[0].expr {
                AtomExpr::Bracket(expr) => {
                    assert_eq!(
                        expr.tree,
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element,
                            aromatic: false,
                        })
                    );
                }
                other => panic!("expected bracket atom, got {other:?}"),
            }
        }
    }

    #[test]
    fn parses_isotopes_for_non_organic_and_hydrogen_elements() {
        let platinum = parse_smarts("[195Pt]").unwrap();
        let deuterium = parse_smarts("[2H]").unwrap();

        match &platinum.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Primitive(AtomPrimitive::Isotope {
                        isotope: Isotope::try_from((Element::Pt, 195_u16)).unwrap(),
                        aromatic: false,
                    })
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        match &deuterium.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Primitive(AtomPrimitive::Isotope {
                        isotope: Isotope::try_from((Element::H, 2_u16)).unwrap(),
                        aromatic: false,
                    })
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn parses_atomic_hydrogen_as_element_symbol() {
        let hydrogen = parse_smarts("[H]").unwrap();
        let charged = parse_smarts("[H+]").unwrap();

        match &hydrogen.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::Primitive(AtomPrimitive::Symbol {
                        element: Element::H,
                        aromatic: false,
                    })
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        match &charged.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                assert_eq!(
                    expr.tree,
                    BracketExprTree::HighAnd(vec![
                        BracketExprTree::Primitive(AtomPrimitive::Symbol {
                            element: Element::H,
                            aromatic: false,
                        }),
                        BracketExprTree::Primitive(AtomPrimitive::Charge(1)),
                    ])
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }

        let rendered = charged.to_string();
        let reparsed = parse_smarts(&rendered).unwrap();
        assert_eq!(rendered, "[#1&+]");
        assert_eq!(
            reparsed.to_canonical_smarts(),
            charged.to_canonical_smarts()
        );
    }

    #[test]
    fn parses_bracketed_elements_starting_with_h() {
        for smarts in ["[He]", "[Hf]", "[Hg]"] {
            assert!(parse_smarts(smarts).is_ok(), "{smarts}");
        }
    }

    #[test]
    fn parses_bracketed_elements_that_overlap_atom_primitives() {
        for smarts in ["[Al]", "[At]", "[Ag]", "[Au]", "[Re]"] {
            assert!(parse_smarts(smarts).is_ok(), "{smarts}");
        }
    }

    #[test]
    fn parses_bare_aliphatic_and_aromatic_any_atoms() {
        assert!(parse_smarts("A").is_ok());
        assert!(parse_smarts("a").is_ok());
    }

    #[test]
    fn parses_bond_expression_or() {
        let query = parse_smarts("C-,=N").unwrap();

        assert_eq!(query.atom_count(), 2);
        assert_eq!(query.bond_count(), 1);
        assert_eq!(
            query.bonds()[0].expr,
            BondExpr::Query(BondExprTree::Or(vec![
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)),
            ]))
        );
    }

    #[test]
    fn parses_bond_expression_not_and_implicit_and() {
        let negated_ring = parse_smarts("C!@N").unwrap();
        let single_ring = parse_smarts("C-@N").unwrap();

        assert_eq!(
            negated_ring.bonds()[0].expr,
            BondExpr::Query(BondExprTree::Not(Box::new(BondExprTree::Primitive(
                BondPrimitive::Ring,
            ))))
        );
        assert_eq!(
            single_ring.bonds()[0].expr,
            BondExpr::Query(BondExprTree::HighAnd(vec![
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
                BondExprTree::Primitive(BondPrimitive::Ring),
            ]))
        );
    }

    #[test]
    fn parses_valid_corner_case_bond_expressions() {
        let constrained = parse_smarts("C-;!@N").unwrap();
        let negated_single = parse_smarts("C!-N").unwrap();
        let mixed = parse_smarts("C-@,=N").unwrap();

        assert_eq!(
            constrained.bonds()[0].expr,
            BondExpr::Query(BondExprTree::LowAnd(vec![
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
                BondExprTree::Not(Box::new(BondExprTree::Primitive(BondPrimitive::Ring))),
            ]))
        );
        assert_eq!(
            negated_single.bonds()[0].expr,
            BondExpr::Query(BondExprTree::Not(Box::new(BondExprTree::Primitive(
                BondPrimitive::Bond(Bond::Single),
            ))))
        );
        assert_eq!(
            mixed.bonds()[0].expr,
            BondExpr::Query(BondExprTree::Or(vec![
                BondExprTree::HighAnd(vec![
                    BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
                    BondExprTree::Primitive(BondPrimitive::Ring),
                ]),
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)),
            ]))
        );
    }

    #[test]
    fn parses_zero_level_component_grouping() {
        let grouped = parse_smarts("(C.C)").unwrap();
        let separated = parse_smarts("(C).(C).C").unwrap();

        assert_eq!(grouped.component_count(), 2);
        assert_eq!(grouped.component_groups(), &[Some(0), Some(0)]);
        assert_eq!(grouped.to_string(), "(C.C)");

        assert_eq!(separated.component_count(), 3);
        assert_eq!(separated.component_groups(), &[Some(0), Some(1), None]);
        assert_eq!(separated.to_string(), "(C).(C).C");
    }

    #[test]
    fn canonicalizes_nested_zero_level_groups() {
        let query = parse_smarts("((C.C))").unwrap();

        assert_eq!(query.component_groups(), &[Some(0), Some(0)]);
        assert_eq!(query.to_string(), "(C.C)");
    }

    #[test]
    fn rejects_invalid_bond_expression_corner_cases() {
        for (smarts, expected) in [
            ("C!!N", SmartsParseErrorKind::UnexpectedCharacter('N')),
            ("C-;N", SmartsParseErrorKind::UnexpectedCharacter('N')),
            ("C-,,=N", SmartsParseErrorKind::UnexpectedCharacter(',')),
            ("C!;@N", SmartsParseErrorKind::UnexpectedCharacter(';')),
            ("C/?C", SmartsParseErrorKind::UnexpectedCharacter('?')),
            ("C\\?C", SmartsParseErrorKind::UnexpectedCharacter('?')),
        ] {
            let err = parse_smarts(smarts).unwrap_err();
            assert_eq!(err.kind(), expected, "unexpected parse result for {smarts}");
        }
    }

    #[test]
    fn rejects_invalid_atom_stereo_queries() {
        for input in ["[C@?]", "[C@TH0]", "[C@SP4]", "[C@TB21]", "[C@OH31]"] {
            let err = parse_smarts(input).unwrap_err();
            assert!(matches!(
                err.kind(),
                SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::AtomPrimitive)
                    | SmartsParseErrorKind::UnexpectedCharacter(_)
            ));
        }
    }

    #[test]
    fn rejects_missing_separator_after_zero_level_group() {
        let err = parse_smarts("(C)C").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter('C'));
    }

    #[test]
    fn rejects_trailing_dot_inside_zero_level_group() {
        let err = parse_smarts("(C.)").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter(')'));
    }

    #[test]
    fn parses_rdkit_accepted_and_rejected_bare_elements() {
        for smarts in ["Na", "As", "Cn"] {
            assert!(parse_smarts(smarts).is_ok(), "{smarts}");
        }

        for (smarts, expected) in [
            ("Pt", SmartsParseErrorKind::UnexpectedCharacter('t')),
            ("Og", SmartsParseErrorKind::UnexpectedCharacter('g')),
            ("Se", SmartsParseErrorKind::UnexpectedCharacter('e')),
            ("At", SmartsParseErrorKind::UnexpectedCharacter('t')),
            ("Al", SmartsParseErrorKind::UnexpectedCharacter('l')),
        ] {
            let err = parse_smarts(smarts).unwrap_err();
            assert_eq!(err.kind(), expected, "{smarts}");
        }
    }

    #[test]
    fn parses_recursive_smarts_primitive() {
        let query = parse_smarts("[$(CO)]CO").unwrap();
        assert_eq!(query.atom_count(), 3);
        assert_eq!(query.bond_count(), 2);

        match &query.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                let BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) = &expr.tree
                else {
                    panic!("expected lowered recursive SMARTS, got {:?}", expr.tree);
                };
                assert_eq!(nested.to_string(), "CO");
                assert_eq!(nested.atom_count(), 2);
                assert_eq!(nested.bond_count(), 1);
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_recursive_smarts_payload() {
        let err = parse_smarts("[$(1)]").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter('1'));
    }

    #[test]
    fn rejects_whitespace_inside_bracket_atoms() {
        let err = parse_smarts("[C ; H1]").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter(' '));

        let err = parse_smarts("[C H1]").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter(' '));

        let err = parse_smarts("[$(CO) ]").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter(' '));

        let err = parse_smarts("[$(C O)]").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter(' '));

        let err = parse_smarts("[$(C\tO)]").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter('\t'));

        let err = parse_smarts("[$(C\nO)]").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter('\n'));
    }

    #[test]
    fn rejects_rdkit_incompatible_percent_ring_labels() {
        let err = parse_smarts("C%1CC%1").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter('C'));

        let err = parse_smarts("C%00CC%00").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter('0'));

        let err = parse_smarts("C%123CC%123").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter('3'));

        let err = parse_smarts("C%(100000)CC%(100000)").unwrap_err();
        assert_eq!(err.kind(), SmartsParseErrorKind::UnexpectedCharacter('0'));
    }

    #[test]
    fn allows_colons_inside_recursive_smarts_payloads() {
        let query = parse_smarts("[$(c:c)]").unwrap();

        match &query.atoms()[0].expr {
            AtomExpr::Bracket(expr) => {
                let BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) = &expr.tree
                else {
                    panic!("expected lowered recursive SMARTS, got {:?}", expr.tree);
                };
                assert_eq!(nested.to_string(), "c:c");
                assert_eq!(nested.atom_count(), 2);
                assert_eq!(nested.bond_count(), 1);
                assert_eq!(
                    nested.bonds()[0].expr,
                    BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Aromatic)))
                );
            }
            other => panic!("expected bracket atom, got {other:?}"),
        }
    }

    #[test]
    fn parses_mixed_directional_bond_sequence() {
        let directional = parse_smarts("C/C=C\\C").unwrap();
        let ring_bond = parse_smarts("C@C").unwrap();

        assert_eq!(directional.bond_count(), 3);
        assert_eq!(
            directional.bonds()[0].expr,
            BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Up)))
        );
        assert_eq!(
            directional.bonds()[1].expr,
            BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)))
        );
        assert_eq!(
            directional.bonds()[2].expr,
            BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Down)))
        );

        assert_eq!(ring_bond.bond_count(), 1);
        assert_eq!(
            ring_bond.bonds()[0].expr,
            BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Ring))
        );
    }

    #[test]
    fn display_preserves_original_source() {
        let query = parse_smarts("[$(CO)]C/C=C\\C").unwrap();
        assert_eq!(query.to_string(), "[$(CO)]C/C=C\\C");
    }

    #[test]
    fn display_roundtrips_supported_subset() {
        let query = parse_smarts("CC(C)").unwrap();
        let rendered = query.to_string();
        let reparsed = parse_smarts(&rendered).unwrap();

        assert_eq!(rendered, reparsed.to_string());
        assert_same_query_structure(&query, &reparsed);
    }

    #[test]
    fn rejects_conflicting_ring_bonds() {
        let err = parse_smarts("C=1CCCCC#1").unwrap_err();
        assert_eq!(err.code(), "conflicting_ring_closure_bond");
    }

    #[test]
    fn rejects_unterminated_parenthesized_percent_ring_label() {
        let err = parse_smarts("B%(1").unwrap_err();
        assert_eq!(err.code(), "unexpected_end_of_input");
    }

    #[test]
    fn display_preserves_supported_smarts_roundtrip() {
        let source = "[$([OH])]/C=C\\C.C%12CC%12";
        let query = parse_smarts(source).unwrap();
        let rendered = query.to_string();
        let reparsed = parse_smarts(&rendered).unwrap();

        assert_eq!(reparsed.to_string(), rendered);
        assert_same_query_structure(&query, &reparsed);
    }

    #[test]
    fn display_renders_recursive_nested_ir() {
        let query = parse_smarts("[$([OH])]C").unwrap();
        assert_eq!(query.to_string(), "[$([O&H])]C");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn query_mol_serializes_and_deserializes_as_smarts_text() {
        let query = parse_smarts("[Na+].[Cl-]").unwrap();
        let rendered = query.to_string();
        let encoded = serde_json::to_string(&query).unwrap();
        assert_eq!(encoded, serde_json::to_string(&rendered).unwrap());

        let decoded: QueryMol = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.to_string(), rendered);
        assert_same_query_structure(&query, &decoded);
    }
}
