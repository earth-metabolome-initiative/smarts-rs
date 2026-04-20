use alloc::{collections::BTreeSet, vec};
use thiserror::Error;

use crate::query::{
    AtomExpr, AtomPrimitive, BondExprTree, BracketExprTree, ComponentGroupId, QueryMol,
};

/// Structured validation error for parsed or edited SMARTS queries.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum QueryValidationError {
    /// The query contains no atoms.
    #[error("query must contain at least one atom")]
    EmptyQuery,
    /// One or more atom ids are not dense or do not match their position.
    #[error("atom ids must be dense and match their position")]
    NonDenseAtomIds,
    /// One or more bond ids are not dense or do not match their position.
    #[error("bond ids must be dense and match their position")]
    NonDenseBondIds,
    /// One atom references a component id outside the query range.
    #[error("atom references an invalid component id")]
    InvalidAtomComponent,
    /// The component-group table length does not match the component count.
    #[error("component group table length must match component count")]
    InvalidComponentGroupTable,
    /// The component-group ids are not dense.
    #[error("component group ids must be dense")]
    NonDenseComponentGroupIds,
    /// At least one component id has no atoms assigned to it.
    #[error("each component must contain at least one atom")]
    EmptyComponent,
    /// One bond references a missing atom.
    #[error("bond references a missing atom")]
    BondEndpointOutOfRange,
    /// One bond crosses between two different components.
    #[error("bond endpoints must stay within one component")]
    CrossComponentBond,
    /// One nested recursive query is structurally invalid.
    #[error("nested recursive SMARTS query is invalid")]
    InvalidRecursiveQuery,
    /// The query exceeds one caller-provided recursive SMARTS depth limit.
    #[error("recursive SMARTS depth {depth} exceeds maximum {max_depth}")]
    RecursiveDepthExceeded {
        /// Observed maximum recursive nesting depth.
        depth: usize,
        /// Caller-provided inclusive depth limit.
        max_depth: usize,
    },
}

impl QueryMol {
    /// Performs a structural validity check on the query graph.
    ///
    /// # Errors
    ///
    /// Returns [`QueryValidationError`] if the graph contains invalid dense ids,
    /// dangling references, inconsistent components, or invalid recursive queries.
    pub fn validate(&self) -> Result<(), QueryValidationError> {
        if self.is_empty() {
            return Err(QueryValidationError::EmptyQuery);
        }

        if self.component_groups().len() != self.component_count() {
            return Err(QueryValidationError::InvalidComponentGroupTable);
        }

        for (expected_id, atom) in self.atoms().iter().enumerate() {
            if atom.id != expected_id {
                return Err(QueryValidationError::NonDenseAtomIds);
            }
            if atom.component >= self.component_count() {
                return Err(QueryValidationError::InvalidAtomComponent);
            }
            validate_atom_expr_recursive(&atom.expr)?;
        }

        let mut atoms_per_component = vec![0usize; self.component_count()];
        for atom in self.atoms() {
            atoms_per_component[atom.component] += 1;
        }
        if atoms_per_component.contains(&0) {
            return Err(QueryValidationError::EmptyComponent);
        }

        for (expected_id, bond) in self.bonds().iter().enumerate() {
            if bond.id != expected_id {
                return Err(QueryValidationError::NonDenseBondIds);
            }

            let Some(src_atom) = self.atom(bond.src) else {
                return Err(QueryValidationError::BondEndpointOutOfRange);
            };
            let Some(dst_atom) = self.atom(bond.dst) else {
                return Err(QueryValidationError::BondEndpointOutOfRange);
            };
            if src_atom.component != dst_atom.component {
                return Err(QueryValidationError::CrossComponentBond);
            }
        }

        validate_component_groups_dense(self.component_groups())?;

        Ok(())
    }
}

/// Returns the maximum recursive SMARTS nesting depth inside one query.
///
/// A query without any `$(...)` primitive has depth `0`. A query containing a
/// single non-nested recursive SMARTS has depth `1`.
#[must_use]
pub fn recursive_depth(query: &QueryMol) -> usize {
    query
        .atoms()
        .iter()
        .map(|atom| recursive_depth_atom_expr(&atom.expr))
        .max()
        .unwrap_or(0)
}

/// Validates that one query stays within a caller-provided recursive depth cap.
///
/// # Errors
///
/// Returns [`QueryValidationError::RecursiveDepthExceeded`] if the maximum
/// recursive SMARTS nesting depth is larger than `max_depth`.
pub fn validate_recursive_depth(
    query: &QueryMol,
    max_depth: usize,
) -> Result<(), QueryValidationError> {
    let depth = recursive_depth(query);
    if depth > max_depth {
        Err(QueryValidationError::RecursiveDepthExceeded { depth, max_depth })
    } else {
        Ok(())
    }
}

fn validate_component_groups_dense(
    component_groups: &[Option<ComponentGroupId>],
) -> Result<(), QueryValidationError> {
    let mut observed = BTreeSet::new();
    for group_id in component_groups.iter().flatten().copied() {
        observed.insert(group_id);
    }

    for (expected, actual) in observed.into_iter().enumerate() {
        if actual != expected {
            return Err(QueryValidationError::NonDenseComponentGroupIds);
        }
    }

    Ok(())
}

fn validate_atom_expr_recursive(expr: &AtomExpr) -> Result<(), QueryValidationError> {
    match expr {
        AtomExpr::Bracket(bracket) => validate_bracket_expr_tree(&bracket.tree),
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => Ok(()),
    }
}

fn recursive_depth_atom_expr(expr: &AtomExpr) -> usize {
    match expr {
        AtomExpr::Bracket(bracket) => recursive_depth_bracket_expr_tree(&bracket.tree),
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => 0,
    }
}

fn recursive_depth_bracket_expr_tree(tree: &BracketExprTree) -> usize {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(query)) => {
            1 + recursive_depth(query)
        }
        BracketExprTree::Primitive(_) => 0,
        BracketExprTree::Not(inner) => recursive_depth_bracket_expr_tree(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => items
            .iter()
            .map(recursive_depth_bracket_expr_tree)
            .max()
            .unwrap_or(0),
    }
}

fn validate_bracket_expr_tree(tree: &BracketExprTree) -> Result<(), QueryValidationError> {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(query)) => query
            .validate()
            .map_err(|_| QueryValidationError::InvalidRecursiveQuery),
        BracketExprTree::Primitive(_) => Ok(()),
        BracketExprTree::Not(inner) => validate_bracket_expr_tree(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            for item in items {
                validate_bracket_expr_tree(item)?;
            }
            Ok(())
        }
    }
}

#[allow(dead_code)]
fn validate_bond_expr_tree(tree: &BondExprTree) -> Result<(), QueryValidationError> {
    match tree {
        BondExprTree::Primitive(_) => Ok(()),
        BondExprTree::Not(inner) => validate_bond_expr_tree(inner),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            for item in items {
                validate_bond_expr_tree(item)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, string::ToString, vec, vec::Vec};
    use elements_rs::Element;

    use crate::query::{
        AtomExpr, AtomPrimitive, BondExpr, BracketExpr, BracketExprTree, ComponentGroupId,
        QueryAtom, QueryBond, QueryMol,
    };

    use super::{recursive_depth, validate_recursive_depth, QueryValidationError};

    fn atom(id: usize, component: usize, expr: AtomExpr) -> QueryAtom {
        QueryAtom {
            id,
            component,
            expr,
        }
    }

    fn bond(id: usize, src: usize, dst: usize, expr: BondExpr) -> QueryBond {
        QueryBond { id, src, dst, expr }
    }

    #[test]
    fn query_validation_accepts_simple_valid_queries() {
        let query = "CC".parse::<QueryMol>().unwrap();
        assert_eq!(query.validate(), Ok(()));
    }

    #[test]
    fn query_validation_rejects_dense_id_and_component_errors() {
        let not_dense_atoms = QueryMol::from_parts(
            vec![atom(
                1,
                0,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            )],
            Vec::new(),
            1,
            vec![None],
        );
        assert_eq!(
            not_dense_atoms.validate(),
            Err(QueryValidationError::NonDenseAtomIds)
        );

        let bad_component = QueryMol::from_parts(
            vec![atom(
                0,
                2,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            )],
            Vec::new(),
            1,
            vec![None],
        );
        assert_eq!(
            bad_component.validate(),
            Err(QueryValidationError::InvalidAtomComponent)
        );

        let empty_component = QueryMol::from_parts(
            vec![atom(
                0,
                0,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            )],
            Vec::new(),
            2,
            vec![None, None],
        );
        assert_eq!(
            empty_component.validate(),
            Err(QueryValidationError::EmptyComponent)
        );
    }

    #[test]
    fn query_validation_rejects_bond_and_group_inconsistencies() {
        let bad_bond = QueryMol::from_parts(
            vec![
                atom(
                    0,
                    0,
                    AtomExpr::Bare {
                        element: Element::C,
                        aromatic: false,
                    },
                ),
                atom(
                    1,
                    1,
                    AtomExpr::Bare {
                        element: Element::C,
                        aromatic: false,
                    },
                ),
            ],
            vec![bond(0, 0, 1, BondExpr::Elided)],
            2,
            vec![None, None],
        );
        assert_eq!(
            bad_bond.validate(),
            Err(QueryValidationError::CrossComponentBond)
        );

        let missing_endpoint = QueryMol::from_parts(
            vec![atom(
                0,
                0,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            )],
            vec![bond(0, 0, 2, BondExpr::Elided)],
            1,
            vec![None],
        );
        assert_eq!(
            missing_endpoint.validate(),
            Err(QueryValidationError::BondEndpointOutOfRange)
        );

        let bad_groups = QueryMol::from_parts(
            vec![atom(
                0,
                0,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            )],
            Vec::new(),
            1,
            vec![Some(2 as ComponentGroupId)],
        );
        assert_eq!(
            bad_groups.validate(),
            Err(QueryValidationError::NonDenseComponentGroupIds)
        );
    }

    #[test]
    fn query_validation_rejects_invalid_recursive_queries() {
        let invalid_nested = QueryMol::from_parts(Vec::new(), Vec::new(), 0, Vec::new());
        let query = QueryMol::from_parts(
            vec![atom(
                0,
                0,
                AtomExpr::Bracket(BracketExpr {
                    tree: BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(Box::new(
                        invalid_nested,
                    ))),
                    atom_map: None,
                }),
            )],
            Vec::new(),
            1,
            vec![None],
        );

        assert_eq!(
            query.validate(),
            Err(QueryValidationError::InvalidRecursiveQuery)
        );
    }

    #[test]
    fn recursive_helpers_expose_and_replace_recursive_query_payloads() {
        let mut primitive =
            AtomPrimitive::RecursiveQuery(Box::new("CC".parse::<QueryMol>().unwrap()));
        assert_eq!(primitive.recursive_query().unwrap().to_string(), "CC");

        *primitive.recursive_query_mut().unwrap() = "N".parse::<QueryMol>().unwrap();
        assert_eq!(primitive.recursive_query().unwrap().to_string(), "N");

        primitive.set_recursive_query("O".parse::<QueryMol>().unwrap());
        assert_eq!(primitive.recursive_query().unwrap().to_string(), "O");
    }

    #[test]
    fn recursive_depth_helpers_cover_nested_recursive_queries() {
        let leaf = "C".parse::<QueryMol>().unwrap();
        let nested = QueryMol::from_parts(
            vec![atom(
                0,
                0,
                AtomExpr::Bracket(BracketExpr {
                    tree: BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(Box::new(
                        leaf.clone(),
                    ))),
                    atom_map: None,
                }),
            )],
            Vec::new(),
            1,
            vec![None],
        );
        let doubly_nested = QueryMol::from_parts(
            vec![atom(
                0,
                0,
                AtomExpr::Bracket(BracketExpr {
                    tree: BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(Box::new(
                        nested.clone(),
                    ))),
                    atom_map: None,
                }),
            )],
            Vec::new(),
            1,
            vec![None],
        );

        assert_eq!(recursive_depth(&leaf), 0);
        assert_eq!(recursive_depth(&nested), 1);
        assert_eq!(recursive_depth(&doubly_nested), 2);
        assert_eq!(validate_recursive_depth(&doubly_nested, 2), Ok(()));
        assert_eq!(
            validate_recursive_depth(&doubly_nested, 1),
            Err(QueryValidationError::RecursiveDepthExceeded {
                depth: 2,
                max_depth: 1,
            })
        );
    }
}
