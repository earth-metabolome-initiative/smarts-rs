use alloc::{boxed::Box, collections::BTreeMap, string::String, string::ToString};
use core::cell::RefCell;

use elements_rs::{AtomicNumber, ElementVariant, MassNumber};
use smarts_parser::{
    AtomExpr, AtomId as QueryAtomId, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive,
    BracketExprTree, HydrogenKind, NumericQuery, QueryMol,
};
use smiles_parser::{
    atom::{bracketed::chirality::Chirality, Atom},
    bond::Bond,
    DoubleBondStereoConfig, Smiles,
};

use crate::{error::SmartsMatchError, prepared::PreparedTarget, target::BondLabel};

type RecursiveMatchCache = BTreeMap<(usize, usize), bool>;
type QueryNeighbors = alloc::vec::Vec<alloc::vec::Vec<(QueryAtomId, usize)>>;
type CompiledQueryParts = (QueryNeighbors, alloc::vec::Vec<usize>, QueryStereoPlan);
type AtomStereoCache = BTreeMap<AtomStereoCacheKey, Option<Chirality>>;
type UniqueMatches = BTreeMap<Box<[usize]>, Box<[usize]>>;

struct SearchContext<'a> {
    query_neighbors: &'a [alloc::vec::Vec<(QueryAtomId, usize)>],
    query_degrees: &'a [usize],
    query_to_target: &'a mut [Option<usize>],
    used_target_atoms: &'a mut [bool],
    stereo_plan: &'a QueryStereoPlan,
    atom_stereo_cache: &'a RefCell<AtomStereoCache>,
    recursive_cache: &'a mut RecursiveMatchCache,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct QueryStereoPlan {
    directional_bond_ids: alloc::vec::Vec<bool>,
    double_bond_constraints: alloc::vec::Vec<QueryDoubleBondStereoConstraint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryDoubleBondStereoConstraint {
    left_atom: QueryAtomId,
    right_atom: QueryAtomId,
    config: DoubleBondStereoConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryDirectionalSubstituent {
    bond_id: usize,
    direction: Bond,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryLocalSubstituent {
    bond_id: usize,
    direction: Option<Bond>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AtomStereoCacheKey {
    query_atom_id: QueryAtomId,
    neighbor_tokens: alloc::vec::Vec<String>,
}

/// Reusable compiled SMARTS query state for repeated matching.
#[derive(Debug, Clone)]
pub struct CompiledQuery {
    query: QueryMol,
    query_neighbors: QueryNeighbors,
    query_degrees: alloc::vec::Vec<usize>,
    stereo_plan: QueryStereoPlan,
    atom_stereo_cache: RefCell<AtomStereoCache>,
}

impl CompiledQuery {
    /// Compile one parsed SMARTS query into reusable matcher state.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
    ///   atom primitive outside the current validator slice
    /// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a
    ///   bond primitive outside the current validator slice
    pub fn new(query: QueryMol) -> Result<Self, SmartsMatchError> {
        let (query_neighbors, query_degrees, stereo_plan) = compile_query_parts(&query)?;
        Ok(Self {
            query,
            query_neighbors,
            query_degrees,
            stereo_plan,
            atom_stereo_cache: RefCell::new(AtomStereoCache::new()),
        })
    }

    /// Borrow the underlying parsed SMARTS query.
    #[inline]
    #[must_use]
    pub const fn query(&self) -> &QueryMol {
        &self.query
    }
}

/// Match a compiled SMARTS query against a target `SMILES` string.
///
/// The current validator slice supports disconnected SMARTS queries,
/// including zero-level component grouping, using the already-supported atom
/// subset plus bond predicates for:
///
/// - elided bonds
/// - `-`
/// - `=`
/// - `#`
/// - `:`
/// - `~`
/// - bond boolean logic `!`, `&`, `,`, `;`
///
/// Target molecules are parsed through `smiles-parser` and prepared with
/// RDKit-default aromaticity, degree, implicit-hydrogen, total-hydrogen, and
/// effective bond-label caches before matching.
///
/// # Errors
///
/// Returns:
/// - [`SmartsMatchError::EmptyTarget`] for empty target strings
/// - [`SmartsMatchError::InvalidTargetSmiles`] when the target is not valid
///   `SMILES`
/// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
///   atom primitive outside the current validator slice
/// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a bond
///   primitive outside the current validator slice
pub fn matches(query: &QueryMol, target: &str) -> Result<bool, SmartsMatchError> {
    if target.is_empty() {
        return Err(SmartsMatchError::EmptyTarget);
    }

    let target =
        target
            .parse::<Smiles>()
            .map_err(|error| SmartsMatchError::InvalidTargetSmiles {
                source: error.smiles_error(),
                start: error.start(),
                end: error.end(),
            })?;
    let prepared = PreparedTarget::new(target);

    matches_prepared(query, &prepared)
}

/// Match a compiled SMARTS query against a prepared target molecule.
///
/// This convenience entrypoint still derives reusable query-side state for
/// each call. For repeated matching of one SMARTS against many targets, prefer
/// [`CompiledQuery`] plus [`matches_compiled`].
///
/// # Errors
///
/// Returns:
/// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
///   atom primitive outside the current validator slice
/// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a bond
///   primitive outside the current validator slice
pub fn matches_prepared(
    query: &QueryMol,
    target: &PreparedTarget,
) -> Result<bool, SmartsMatchError> {
    let compiled = CompiledQuery::new(query.clone())?;
    Ok(matches_compiled(&compiled, target))
}

/// Match one reusable compiled SMARTS query against a prepared target.
#[inline]
#[must_use]
pub fn matches_compiled(query: &CompiledQuery, target: &PreparedTarget) -> bool {
    query_matches(query, target)
}

/// Collect all unique accepted substructure matches for one SMARTS query
/// against one target `SMILES` string.
///
/// Each inner match lists target atom ids in query atom order.
///
/// # Errors
///
/// Returns:
/// - [`SmartsMatchError::EmptyTarget`] for empty target strings
/// - [`SmartsMatchError::InvalidTargetSmiles`] when the target is not valid
///   `SMILES`
/// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
///   atom primitive outside the current validator slice
/// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a bond
///   primitive outside the current validator slice
pub fn substructure_matches(
    query: &QueryMol,
    target: &str,
) -> Result<Box<[Box<[usize]>]>, SmartsMatchError> {
    if target.is_empty() {
        return Err(SmartsMatchError::EmptyTarget);
    }

    let target =
        target
            .parse::<Smiles>()
            .map_err(|error| SmartsMatchError::InvalidTargetSmiles {
                source: error.smiles_error(),
                start: error.start(),
                end: error.end(),
            })?;
    let prepared = PreparedTarget::new(target);

    substructure_matches_prepared(query, &prepared)
}

/// Collect all unique accepted substructure matches for one SMARTS query
/// against one prepared target molecule.
///
/// Each inner match lists target atom ids in query atom order.
///
/// # Errors
///
/// Returns:
/// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
///   atom primitive outside the current validator slice
/// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a bond
///   primitive outside the current validator slice
pub fn substructure_matches_prepared(
    query: &QueryMol,
    target: &PreparedTarget,
) -> Result<Box<[Box<[usize]>]>, SmartsMatchError> {
    let compiled = CompiledQuery::new(query.clone())?;
    Ok(substructure_matches_compiled(&compiled, target))
}

/// Collect all unique accepted substructure matches for one compiled SMARTS
/// query against one prepared target.
///
/// Each inner match lists target atom ids in query atom order.
#[must_use]
pub fn substructure_matches_compiled(
    query: &CompiledQuery,
    target: &PreparedTarget,
) -> Box<[Box<[usize]>]> {
    query_substructure_matches(query, target)
}

/// Count all unique accepted substructure matches for one SMARTS query
/// against one target `SMILES` string.
///
/// # Errors
///
/// Returns:
/// - [`SmartsMatchError::EmptyTarget`] for empty target strings
/// - [`SmartsMatchError::InvalidTargetSmiles`] when the target is not valid
///   `SMILES`
/// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
///   atom primitive outside the current validator slice
/// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a bond
///   primitive outside the current validator slice
pub fn match_count(query: &QueryMol, target: &str) -> Result<usize, SmartsMatchError> {
    substructure_matches(query, target).map(|matches| matches.len())
}

/// Count all unique accepted substructure matches for one SMARTS query
/// against one prepared target.
///
/// # Errors
///
/// Returns:
/// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
///   atom primitive outside the current validator slice
/// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a bond
///   primitive outside the current validator slice
pub fn match_count_prepared(
    query: &QueryMol,
    target: &PreparedTarget,
) -> Result<usize, SmartsMatchError> {
    substructure_matches_prepared(query, target).map(|matches| matches.len())
}

/// Count all unique accepted substructure matches for one compiled SMARTS
/// query against one prepared target.
#[inline]
#[must_use]
pub fn match_count_compiled(query: &CompiledQuery, target: &PreparedTarget) -> usize {
    substructure_matches_compiled(query, target).len()
}

fn ensure_supported_query(
    query: &QueryMol,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    for atom in query.atoms() {
        ensure_supported_atom_expr(&atom.expr, stereo_plan)?;
    }
    for bond in query.bonds() {
        ensure_supported_bond_expr(bond.id, &bond.expr, stereo_plan)?;
    }
    Ok(())
}

fn ensure_supported_bond_expr(
    bond_id: usize,
    expr: &BondExpr,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    if stereo_plan
        .directional_bond_ids
        .get(bond_id)
        .copied()
        .unwrap_or(false)
    {
        return Ok(());
    }
    match expr {
        BondExpr::Elided => Ok(()),
        BondExpr::Query(tree) => ensure_supported_bond_tree(tree),
    }
}

fn query_bond_is_directional(stereo_plan: &QueryStereoPlan, bond_id: usize) -> bool {
    stereo_plan
        .directional_bond_ids
        .get(bond_id)
        .copied()
        .unwrap_or(false)
}

fn ensure_supported_bond_tree(tree: &BondExprTree) -> Result<(), SmartsMatchError> {
    match tree {
        BondExprTree::Primitive(primitive) => ensure_supported_bond_primitive(*primitive),
        BondExprTree::Not(inner) => ensure_supported_bond_tree(inner),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            for item in items {
                ensure_supported_bond_tree(item)?;
            }
            Ok(())
        }
    }
}

const fn ensure_supported_bond_primitive(primitive: BondPrimitive) -> Result<(), SmartsMatchError> {
    match primitive {
        BondPrimitive::Bond(
            Bond::Single | Bond::Double | Bond::Triple | Bond::Aromatic | Bond::Up | Bond::Down,
        )
        | BondPrimitive::Any
        | BondPrimitive::Ring => Ok(()),
        BondPrimitive::Bond(Bond::Quadruple) => Err(unsupported_bond_primitive("$")),
    }
}

const fn unsupported_bond_primitive(primitive: &'static str) -> SmartsMatchError {
    SmartsMatchError::UnsupportedBondPrimitive { primitive }
}

fn ensure_supported_atom_expr(
    expr: &AtomExpr,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => Ok(()),
        AtomExpr::Bracket(expr) => ensure_supported_bracket_tree(&expr.tree, stereo_plan),
    }
}

fn ensure_supported_bracket_tree(
    tree: &BracketExprTree,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    match tree {
        BracketExprTree::Primitive(primitive) => {
            ensure_supported_atom_primitive(primitive, stereo_plan)
        }
        BracketExprTree::Not(inner) => ensure_supported_bracket_tree(inner, stereo_plan),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            for item in items {
                ensure_supported_bracket_tree(item, stereo_plan)?;
            }
            Ok(())
        }
    }
}

fn ensure_supported_atom_primitive(
    primitive: &AtomPrimitive,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    match primitive {
        AtomPrimitive::Wildcard
        | AtomPrimitive::AliphaticAny
        | AtomPrimitive::AromaticAny
        | AtomPrimitive::Symbol { .. }
        | AtomPrimitive::Isotope { .. }
        | AtomPrimitive::IsotopeWildcard(_)
        | AtomPrimitive::Degree(_)
        | AtomPrimitive::Connectivity(_)
        | AtomPrimitive::Valence(_)
        | AtomPrimitive::Hydrogen(_, _)
        | AtomPrimitive::RingMembership(_)
        | AtomPrimitive::RingSize(_)
        | AtomPrimitive::RingConnectivity(_)
        | AtomPrimitive::AtomicNumber(_)
        | AtomPrimitive::Charge(_)
        | AtomPrimitive::Chirality(
            Chirality::At
            | Chirality::AtAt
            | Chirality::TH(1 | 2)
            | Chirality::AL(1 | 2)
            | Chirality::SP(1..=3)
            | Chirality::TB(1..=20)
            | Chirality::OH(1..=30),
        ) => Ok(()),
        AtomPrimitive::RecursiveQuery(query) => {
            let query_neighbors = build_query_neighbors(query);
            let recursive_stereo_plan = build_query_stereo_plan(query, &query_neighbors)?;
            ensure_supported_query(query, &recursive_stereo_plan)
        }
        AtomPrimitive::Chirality(_) => {
            let _ = stereo_plan;
            Err(unsupported_atom_primitive("@"))
        }
    }
}

const fn unsupported_atom_primitive(primitive: &'static str) -> SmartsMatchError {
    SmartsMatchError::UnsupportedAtomPrimitive { primitive }
}

fn compile_query_parts(query: &QueryMol) -> Result<CompiledQueryParts, SmartsMatchError> {
    let query_neighbors = build_query_neighbors(query);
    let stereo_plan = build_query_stereo_plan(query, &query_neighbors)?;
    ensure_supported_query(query, &stereo_plan)?;
    let query_degrees = query_neighbors
        .iter()
        .map(alloc::vec::Vec::len)
        .collect::<alloc::vec::Vec<_>>();
    Ok((query_neighbors, query_degrees, stereo_plan))
}

fn query_matches(query: &CompiledQuery, target: &PreparedTarget) -> bool {
    let mut recursive_cache = RecursiveMatchCache::new();
    query_matches_with_mapping(query, target, &mut recursive_cache, None, None)
}

fn query_substructure_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
) -> Box<[Box<[usize]>]> {
    let mut recursive_cache = RecursiveMatchCache::new();
    query_substructure_matches_with_mapping(query, target, &mut recursive_cache, None, None)
}

fn query_matches_with_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_query_atom: Option<QueryAtomId>,
    initial_target_atom: Option<usize>,
) -> bool {
    let mut query_to_target = alloc::vec![None; query.query.atom_count()];
    let mut used_target_atoms = alloc::vec![false; target.atom_count()];
    let mut context = SearchContext {
        query_neighbors: &query.query_neighbors,
        query_degrees: &query.query_degrees,
        query_to_target: &mut query_to_target,
        used_target_atoms: &mut used_target_atoms,
        stereo_plan: &query.stereo_plan,
        atom_stereo_cache: &query.atom_stereo_cache,
        recursive_cache,
    };
    let mapped_count =
        if let (Some(query_atom), Some(target_atom)) = (initial_query_atom, initial_target_atom) {
            if target
                .degree(target_atom)
                .is_some_and(|degree| degree < query.query_degrees[query_atom])
            {
                return false;
            }
            if !component_constraints_match(
                query_atom,
                target_atom,
                &query.query,
                target,
                context.query_to_target,
            ) {
                return false;
            }
            if !atom_expr_matches(
                &query.query.atoms()[query_atom].expr,
                query_atom,
                context.stereo_plan,
                target,
                target_atom,
                context.recursive_cache,
            ) {
                return false;
            }
            context.query_to_target[query_atom] = Some(target_atom);
            context.used_target_atoms[target_atom] = true;
            1
        } else {
            0
        };

    search_mapping(&query.query, target, &mut context, mapped_count)
}

fn query_substructure_matches_with_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_query_atom: Option<QueryAtomId>,
    initial_target_atom: Option<usize>,
) -> Box<[Box<[usize]>]> {
    let mut query_to_target = alloc::vec![None; query.query.atom_count()];
    let mut used_target_atoms = alloc::vec![false; target.atom_count()];
    let mut context = SearchContext {
        query_neighbors: &query.query_neighbors,
        query_degrees: &query.query_degrees,
        query_to_target: &mut query_to_target,
        used_target_atoms: &mut used_target_atoms,
        stereo_plan: &query.stereo_plan,
        atom_stereo_cache: &query.atom_stereo_cache,
        recursive_cache,
    };
    let mapped_count =
        if let (Some(query_atom), Some(target_atom)) = (initial_query_atom, initial_target_atom) {
            if target
                .degree(target_atom)
                .is_some_and(|degree| degree < query.query_degrees[query_atom])
            {
                return Box::default();
            }
            if !component_constraints_match(
                query_atom,
                target_atom,
                &query.query,
                target,
                context.query_to_target,
            ) {
                return Box::default();
            }
            if !atom_expr_matches(
                &query.query.atoms()[query_atom].expr,
                query_atom,
                context.stereo_plan,
                target,
                target_atom,
                context.recursive_cache,
            ) {
                return Box::default();
            }
            context.query_to_target[query_atom] = Some(target_atom);
            context.used_target_atoms[target_atom] = true;
            1
        } else {
            0
        };

    let mut matches = UniqueMatches::new();
    collect_mappings(
        &query.query,
        target,
        &mut context,
        mapped_count,
        &mut matches,
    );
    matches
        .into_values()
        .collect::<alloc::vec::Vec<_>>()
        .into_boxed_slice()
}

fn build_query_neighbors(
    query: &QueryMol,
) -> alloc::vec::Vec<alloc::vec::Vec<(QueryAtomId, usize)>> {
    let mut neighbors = alloc::vec![alloc::vec::Vec::new(); query.atom_count()];
    for (bond_index, bond) in query.bonds().iter().enumerate() {
        neighbors[bond.src].push((bond.dst, bond_index));
        neighbors[bond.dst].push((bond.src, bond_index));
    }
    neighbors
}

fn build_query_stereo_plan(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
) -> Result<QueryStereoPlan, SmartsMatchError> {
    let mut directional_bond_ids = alloc::vec![false; query.bond_count()];
    let mut double_bond_constraints = alloc::vec::Vec::new();

    for bond in query.bonds() {
        if !is_simple_double_bond_expr(&bond.expr) {
            continue;
        }

        let left_directional =
            directional_substituents(query, query_neighbors, bond.src, bond.id, bond.dst)?;
        let right_directional =
            directional_substituents(query, query_neighbors, bond.dst, bond.id, bond.src)?;

        if left_directional.is_empty() || right_directional.is_empty() {
            continue;
        }

        let left_local = local_substituents(query, query_neighbors, bond.src, bond.id, bond.dst)?;
        let right_local = local_substituents(query, query_neighbors, bond.dst, bond.id, bond.src)?;
        let (fragment, left_atom_id, right_atom_id) =
            build_local_double_bond_fragment(&left_local, &right_local)?;
        let local_smiles = fragment
            .parse::<Smiles>()
            .map_err(|_| unsupported_bond_primitive("/"))?;
        let config = local_smiles
            .double_bond_stereo_config(left_atom_id, right_atom_id)
            .ok_or_else(|| unsupported_bond_primitive("/"))?;

        for substituent in left_directional.iter().chain(right_directional.iter()) {
            directional_bond_ids[substituent.bond_id] = true;
        }
        double_bond_constraints.push(QueryDoubleBondStereoConstraint {
            left_atom: bond.src,
            right_atom: bond.dst,
            config,
        });
    }

    Ok(QueryStereoPlan {
        directional_bond_ids,
        double_bond_constraints,
    })
}

const fn is_supported_query_chirality(chirality: Chirality) -> bool {
    matches!(
        chirality,
        Chirality::At
            | Chirality::AtAt
            | Chirality::TH(1 | 2)
            | Chirality::AL(1 | 2)
            | Chirality::SP(1..=3)
            | Chirality::TB(1..=20)
            | Chirality::OH(1..=30)
    )
}

const fn query_chirality_requires_tetrahedral_match(chirality: Chirality) -> bool {
    matches!(
        chirality,
        Chirality::At | Chirality::AtAt | Chirality::TH(1 | 2)
    )
}

fn extract_query_chirality(expr: &AtomExpr) -> Option<Chirality> {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => None,
        AtomExpr::Bracket(expr) => extract_chirality_from_bracket_tree(&expr.tree),
    }
}

fn extract_chirality_from_bracket_tree(tree: &BracketExprTree) -> Option<Chirality> {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Chirality(chirality)) => Some(*chirality),
        BracketExprTree::Primitive(_) | BracketExprTree::Not(_) => None,
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            items.iter().find_map(extract_chirality_from_bracket_tree)
        }
    }
}

fn build_local_stereo_fragment(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    atom_id: QueryAtomId,
    neighbor_tokens: &[String],
) -> Result<(String, usize), SmartsMatchError> {
    let incident = &query_neighbors[atom_id];
    let Some(query_chirality) = extract_query_chirality(&query.atoms()[atom_id].expr) else {
        return Err(unsupported_atom_primitive("@"));
    };
    let preserve_incoming_neighbor = query_has_total_hydrogen(&query.atoms()[atom_id].expr);
    let prefix_neighbor = preserve_incoming_neighbor
        .then(|| {
            incident
                .iter()
                .enumerate()
                .find_map(|(index, (neighbor_atom, _))| (*neighbor_atom < atom_id).then_some(index))
        })
        .flatten();
    let remaining_count = incident
        .len()
        .saturating_sub(usize::from(prefix_neighbor.is_some()));

    let mut fragment = prefix_neighbor.map_or_else(
        || render_local_stereo_center(query_chirality, preserve_incoming_neighbor),
        |prefix_index| {
            let (_, bond_id) = incident[prefix_index];
            let mut fragment = neighbor_tokens[prefix_index].clone();
            fragment.push_str(local_stereo_bond_text(&query.bonds()[bond_id].expr));
            fragment.push_str(&render_local_stereo_center(
                query_chirality,
                preserve_incoming_neighbor,
            ));
            fragment
        },
    );

    let mut rendered_remaining = 0usize;
    for (index, (_, bond_id)) in incident.iter().enumerate() {
        if Some(index) == prefix_neighbor {
            continue;
        }
        rendered_remaining += 1;
        let atom_text = &neighbor_tokens[index];
        let bond_text = local_stereo_bond_text(&query.bonds()[*bond_id].expr);
        if rendered_remaining == remaining_count {
            fragment.push_str(bond_text);
            fragment.push_str(atom_text);
        } else {
            fragment.push('(');
            fragment.push_str(bond_text);
            fragment.push_str(atom_text);
            fragment.push(')');
        }
    }
    Ok((fragment, usize::from(prefix_neighbor.is_some())))
}

fn local_neighbor_tokens_for_mapping(
    incident: &[(QueryAtomId, usize)],
    query_to_target: &[Option<usize>],
    target: &PreparedTarget,
) -> Result<alloc::vec::Vec<String>, SmartsMatchError> {
    const PLACEHOLDER_ATOMS: [&str; 8] = ["F", "Cl", "Br", "I", "N", "O", "P", "S"];

    let target_atoms_in_incident_order = incident
        .iter()
        .map(|(neighbor_atom, _)| {
            query_to_target[*neighbor_atom].ok_or_else(|| unsupported_atom_primitive("@"))
        })
        .collect::<Result<alloc::vec::Vec<_>, _>>()?;
    let mut target_atoms = target_atoms_in_incident_order.clone();
    target_atoms.sort_unstable();
    target_atoms.dedup();

    if target_atoms.len() > PLACEHOLDER_ATOMS.len() {
        return Err(unsupported_atom_primitive("@"));
    }

    let rendered_atoms = target_atoms
        .iter()
        .map(|target_atom_id| render_target_stereo_atom(target, *target_atom_id))
        .collect::<alloc::vec::Vec<_>>();
    let mut assigned_tokens = alloc::vec::Vec::with_capacity(target_atoms.len());
    let mut placeholder_index = 0usize;

    for (index, target_atom_id) in target_atoms.into_iter().enumerate() {
        let rendered_atom = rendered_atoms[index].clone();
        let rendered_atom_is_unique = rendered_atom.as_ref().is_some_and(|rendered_atom| {
            rendered_atoms
                .iter()
                .filter(|other_rendered_atom| other_rendered_atom.as_ref() == Some(rendered_atom))
                .count()
                == 1
        });

        if rendered_atom_is_unique {
            assigned_tokens.push((
                target_atom_id,
                rendered_atom.expect("checked that the rendered atom exists"),
            ));
            continue;
        }

        while PLACEHOLDER_ATOMS
            .get(placeholder_index)
            .is_some_and(|placeholder_atom| {
                assigned_tokens
                    .iter()
                    .any(|(_, assigned_token)| assigned_token == placeholder_atom)
            })
        {
            placeholder_index += 1;
        }
        let Some(placeholder_atom) = PLACEHOLDER_ATOMS.get(placeholder_index) else {
            return Err(unsupported_atom_primitive("@"));
        };
        assigned_tokens.push((target_atom_id, String::from(*placeholder_atom)));
        placeholder_index += 1;
    }

    target_atoms_in_incident_order
        .into_iter()
        .map(|target_atom_id| {
            assigned_tokens
                .iter()
                .find_map(|(mapped_target_atom_id, placeholder_atom)| {
                    (*mapped_target_atom_id == target_atom_id).then_some(placeholder_atom.clone())
                })
                .ok_or_else(|| unsupported_atom_primitive("@"))
        })
        .collect()
}

fn render_target_stereo_atom(target: &PreparedTarget, atom_id: usize) -> Option<String> {
    let atom = target.atom(atom_id)?;
    let mut symbol = atom.element()?.to_string();
    if target.is_aromatic(atom_id) {
        symbol.make_ascii_lowercase();
    }

    if atom.isotope_mass_number().is_none()
        && atom.charge_value() == 0
        && atom.element() != Some(elements_rs::Element::H)
    {
        return Some(symbol);
    }

    let mut rendered = String::from("[");
    if let Some(isotope_mass_number) = atom.isotope_mass_number() {
        rendered.push_str(&isotope_mass_number.to_string());
    }
    rendered.push_str(&symbol);
    if atom.charge_value() != 0 {
        rendered.push_str(&render_charge(atom.charge_value()));
    }
    rendered.push(']');
    Some(rendered)
}

fn query_has_total_hydrogen(expr: &AtomExpr) -> bool {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => false,
        AtomExpr::Bracket(expr) => bracket_tree_has_total_hydrogen(&expr.tree),
    }
}

fn bracket_tree_has_total_hydrogen(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Hydrogen(HydrogenKind::Total, _)) => true,
        BracketExprTree::Primitive(_) | BracketExprTree::Not(_) => false,
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => items.iter().any(bracket_tree_has_total_hydrogen),
    }
}

fn render_local_stereo_center(chirality: Chirality, has_hydrogen: bool) -> String {
    let mut rendered = String::from("[C");
    rendered.push_str(&render_local_chirality(chirality, has_hydrogen));
    if has_hydrogen {
        rendered.push('H');
    }
    rendered.push(']');
    rendered
}

fn render_charge(charge: i8) -> String {
    match charge.cmp(&0) {
        core::cmp::Ordering::Less => {
            let magnitude = charge.unsigned_abs();
            if magnitude == 1 {
                String::from("-")
            } else {
                alloc::format!("-{magnitude}")
            }
        }
        core::cmp::Ordering::Equal => String::new(),
        core::cmp::Ordering::Greater => {
            let magnitude = u8::try_from(charge).unwrap_or(u8::MAX);
            if magnitude == 1 {
                String::from("+")
            } else {
                alloc::format!("+{magnitude}")
            }
        }
    }
}

const fn local_stereo_bond_text(expr: &BondExpr) -> &'static str {
    match expr {
        BondExpr::Elided => "",
        BondExpr::Query(_) => "-",
    }
}

fn numeric_query_matches_u16(query: NumericQuery, actual: u16) -> bool {
    match query {
        NumericQuery::Exact(expected) => actual == expected,
        NumericQuery::Range(range) => numeric_range_matches(range, actual),
    }
}

fn numeric_range_matches(range: smarts_parser::NumericRange, actual: u16) -> bool {
    range.min.is_none_or(|min| actual >= min) && range.max.is_none_or(|max| actual <= max)
}

fn count_query_matches_u16(query: Option<NumericQuery>, actual: u16, omitted_default: u16) -> bool {
    query.map_or(actual == omitted_default, |query| {
        numeric_query_matches_u16(query, actual)
    })
}

fn ring_query_matches_u16(query: Option<NumericQuery>, actual: u16) -> bool {
    query.map_or(actual > 0, |query| numeric_query_matches_u16(query, actual))
}

fn render_local_chirality(chirality: Chirality, has_hydrogen: bool) -> String {
    match (chirality, has_hydrogen) {
        (Chirality::At, true) => String::from("@"),
        (Chirality::AtAt, true) => String::from("@@"),
        (Chirality::At, false) => Chirality::TH(1).to_string(),
        (Chirality::AtAt, false) => Chirality::TH(2).to_string(),
        _ => chirality.to_string(),
    }
}

fn directional_substituents(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    endpoint: QueryAtomId,
    double_bond_id: usize,
    opposite_endpoint: QueryAtomId,
) -> Result<alloc::vec::Vec<QueryDirectionalSubstituent>, SmartsMatchError> {
    let mut directional = alloc::vec::Vec::new();
    for (neighbor_atom, bond_id) in &query_neighbors[endpoint] {
        if *bond_id == double_bond_id || *neighbor_atom == opposite_endpoint {
            continue;
        }
        let Some(direction) = first_supported_directional_bond(&query.bonds()[*bond_id].expr)?
        else {
            continue;
        };
        directional.push(QueryDirectionalSubstituent {
            bond_id: *bond_id,
            direction,
        });
    }
    Ok(directional)
}

fn local_substituents(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    endpoint: QueryAtomId,
    double_bond_id: usize,
    opposite_endpoint: QueryAtomId,
) -> Result<alloc::vec::Vec<QueryLocalSubstituent>, SmartsMatchError> {
    let mut substituents = alloc::vec::Vec::new();
    for (neighbor_atom, bond_id) in &query_neighbors[endpoint] {
        if *bond_id == double_bond_id || *neighbor_atom == opposite_endpoint {
            continue;
        }
        let expr = &query.bonds()[*bond_id].expr;
        let direction = first_supported_directional_bond(expr)?;
        substituents.push(QueryLocalSubstituent {
            bond_id: *bond_id,
            direction,
        });
    }
    Ok(substituents)
}

fn build_local_double_bond_fragment(
    left_substituents: &[QueryLocalSubstituent],
    right_substituents: &[QueryLocalSubstituent],
) -> Result<(String, usize, usize), SmartsMatchError> {
    const PLACEHOLDER_ATOMS: [&str; 8] = ["F", "Cl", "Br", "I", "N", "O", "P", "S"];

    let total_substituents = left_substituents.len() + right_substituents.len();
    if total_substituents > PLACEHOLDER_ATOMS.len() {
        return Err(unsupported_bond_primitive("/"));
    }

    let mut next_placeholder = 0usize;
    let mut next_atom_id = 0usize;
    let mut fragment = String::new();

    let left_prefix_index = left_substituents
        .iter()
        .position(|substituent| substituent.direction.is_some())
        .or_else(|| (!left_substituents.is_empty()).then_some(0));

    if let Some(prefix_index) = left_prefix_index {
        fragment.push_str(PLACEHOLDER_ATOMS[next_placeholder]);
        fragment.push_str(local_double_bond_direction_text(
            left_substituents[prefix_index].direction,
        ));
        next_placeholder += 1;
        next_atom_id += 1;
    }

    let left_center_atom_id = next_atom_id;
    fragment.push('C');
    next_atom_id += 1;

    for (index, substituent) in left_substituents.iter().enumerate() {
        if Some(index) == left_prefix_index {
            continue;
        }
        fragment.push('(');
        fragment.push_str(local_double_bond_direction_text(substituent.direction));
        fragment.push_str(PLACEHOLDER_ATOMS[next_placeholder]);
        fragment.push(')');
        next_placeholder += 1;
        next_atom_id += 1;
    }

    fragment.push('=');
    let right_center_atom_id = next_atom_id;
    fragment.push('C');
    next_atom_id += 1;

    if let Some(continuation_index) = right_continuation_index(right_substituents) {
        for (index, substituent) in right_substituents.iter().enumerate() {
            if index == continuation_index {
                continue;
            }
            fragment.push('(');
            fragment.push_str(local_double_bond_direction_text(substituent.direction));
            fragment.push_str(PLACEHOLDER_ATOMS[next_placeholder]);
            fragment.push(')');
            next_placeholder += 1;
            next_atom_id += 1;
        }

        let continuation = right_substituents[continuation_index];
        fragment.push_str(local_double_bond_direction_text(continuation.direction));
        fragment.push_str(PLACEHOLDER_ATOMS[next_placeholder]);
        let _ = next_atom_id;
    }

    Ok((fragment, left_center_atom_id, right_center_atom_id))
}

fn right_continuation_index(substituents: &[QueryLocalSubstituent]) -> Option<usize> {
    substituents
        .iter()
        .position(|substituent| substituent.direction.is_none())
        .or_else(|| (!substituents.is_empty()).then(|| substituents.len() - 1))
}

const fn local_double_bond_direction_text(direction: Option<Bond>) -> &'static str {
    match direction {
        Some(Bond::Up) => "/",
        Some(Bond::Down) => "\\",
        _ => "",
    }
}

fn first_supported_directional_bond(expr: &BondExpr) -> Result<Option<Bond>, SmartsMatchError> {
    match expr {
        BondExpr::Elided => Ok(None),
        BondExpr::Query(tree) => first_supported_directional_bond_tree(tree),
    }
}

fn bond_expr_contains_negated_directional_primitive(expr: &BondExpr) -> bool {
    match expr {
        BondExpr::Elided => false,
        BondExpr::Query(tree) => bond_tree_contains_negated_directional_primitive(tree),
    }
}

fn bond_tree_contains_negated_directional_primitive(tree: &BondExprTree) -> bool {
    match tree {
        BondExprTree::Primitive(_) => false,
        BondExprTree::Not(inner) => bond_tree_contains_directional_primitive(inner),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            let mut index = 0usize;
            while index < items.len() {
                if bond_tree_contains_negated_directional_primitive(&items[index]) {
                    return true;
                }
                index += 1;
            }
            false
        }
    }
}

fn first_supported_directional_bond_tree(
    tree: &BondExprTree,
) -> Result<Option<Bond>, SmartsMatchError> {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(Bond::Up | Bond::Down)) => match tree {
            BondExprTree::Primitive(BondPrimitive::Bond(bond)) => Ok(Some(*bond)),
            _ => Ok(None),
        },
        BondExprTree::Primitive(_) | BondExprTree::Not(_) => Ok(None),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            for item in items {
                if let Some(direction) = first_supported_directional_bond_tree(item)? {
                    return Ok(Some(direction));
                }
            }
            Ok(None)
        }
    }
}

const fn is_simple_double_bond_expr(expr: &BondExpr) -> bool {
    matches!(
        expr,
        BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)))
    )
}

fn bond_tree_contains_directional_primitive(tree: &BondExprTree) -> bool {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(Bond::Up | Bond::Down)) => true,
        BondExprTree::Primitive(_) => false,
        BondExprTree::Not(inner) => bond_tree_contains_directional_primitive(inner),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            items.iter().any(bond_tree_contains_directional_primitive)
        }
    }
}

fn search_mapping(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_>,
    mapped_count: usize,
) -> bool {
    if mapped_count == query.atom_count() {
        return stereo_constraints_match(
            query,
            context.stereo_plan,
            context.query_neighbors,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        );
    }

    let next_query_atom = select_next_query_atom(context.query_neighbors, context.query_to_target);
    let query_atom = &query.atoms()[next_query_atom];
    let candidate_target_atoms = candidate_target_atoms(
        next_query_atom,
        query,
        context.stereo_plan,
        target,
        context.query_neighbors,
        context.query_to_target,
    );

    for target_atom in candidate_target_atoms {
        if context.used_target_atoms[target_atom] {
            continue;
        }
        if target
            .degree(target_atom)
            .is_some_and(|degree| degree < context.query_degrees[next_query_atom])
        {
            continue;
        }
        if !component_constraints_match(
            next_query_atom,
            target_atom,
            query,
            target,
            context.query_to_target,
        ) {
            continue;
        }
        if !atom_expr_matches(
            &query_atom.expr,
            next_query_atom,
            context.stereo_plan,
            target,
            target_atom,
            context.recursive_cache,
        ) {
            continue;
        }
        if !mapped_bonds_match(
            next_query_atom,
            target_atom,
            query,
            context.stereo_plan,
            target,
            context.query_neighbors,
            context.query_to_target,
        ) {
            continue;
        }

        context.query_to_target[next_query_atom] = Some(target_atom);
        context.used_target_atoms[target_atom] = true;
        if search_mapping(query, target, context, mapped_count + 1) {
            return true;
        }
        context.used_target_atoms[target_atom] = false;
        context.query_to_target[next_query_atom] = None;
    }

    false
}

fn collect_mappings(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_>,
    mapped_count: usize,
    matches: &mut UniqueMatches,
) {
    if mapped_count == query.atom_count() {
        if stereo_constraints_match(
            query,
            context.stereo_plan,
            context.query_neighbors,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        ) {
            let mapping = current_query_order_mapping(context.query_to_target);
            matches
                .entry(canonical_match_key(&mapping))
                .and_modify(|existing| {
                    if mapping < *existing {
                        existing.clone_from(&mapping);
                    }
                })
                .or_insert(mapping);
        }
        return;
    }

    let next_query_atom = select_next_query_atom(context.query_neighbors, context.query_to_target);
    let query_atom = &query.atoms()[next_query_atom];
    let candidate_target_atoms = candidate_target_atoms(
        next_query_atom,
        query,
        context.stereo_plan,
        target,
        context.query_neighbors,
        context.query_to_target,
    );

    for target_atom in candidate_target_atoms {
        if context.used_target_atoms[target_atom] {
            continue;
        }
        if target
            .degree(target_atom)
            .is_some_and(|degree| degree < context.query_degrees[next_query_atom])
        {
            continue;
        }
        if !component_constraints_match(
            next_query_atom,
            target_atom,
            query,
            target,
            context.query_to_target,
        ) {
            continue;
        }
        if !atom_expr_matches(
            &query_atom.expr,
            next_query_atom,
            context.stereo_plan,
            target,
            target_atom,
            context.recursive_cache,
        ) {
            continue;
        }
        if !mapped_bonds_match(
            next_query_atom,
            target_atom,
            query,
            context.stereo_plan,
            target,
            context.query_neighbors,
            context.query_to_target,
        ) {
            continue;
        }

        context.query_to_target[next_query_atom] = Some(target_atom);
        context.used_target_atoms[target_atom] = true;
        collect_mappings(query, target, context, mapped_count + 1, matches);
        context.used_target_atoms[target_atom] = false;
        context.query_to_target[next_query_atom] = None;
    }
}

fn current_query_order_mapping(query_to_target: &[Option<usize>]) -> Box<[usize]> {
    query_to_target
        .iter()
        .map(|target_atom| target_atom.expect("complete mappings must bind every query atom"))
        .collect::<alloc::vec::Vec<_>>()
        .into_boxed_slice()
}

fn canonical_match_key(mapping: &[usize]) -> Box<[usize]> {
    let mut canonical = mapping.to_vec();
    canonical.sort_unstable();
    canonical.into_boxed_slice()
}

fn component_constraints_match(
    query_atom: QueryAtomId,
    target_atom: usize,
    query: &QueryMol,
    target: &PreparedTarget,
    query_to_target: &[Option<usize>],
) -> bool {
    let query_component = query.atoms()[query_atom].component;
    let Some(query_group) = query.component_group(query_component) else {
        return true;
    };
    let Some(target_component) = target.connected_component(target_atom) else {
        return false;
    };

    for (other_query_atom, mapped_target_atom) in query_to_target.iter().enumerate() {
        let Some(mapped_target_atom) = mapped_target_atom else {
            continue;
        };
        let other_component = query.atoms()[other_query_atom].component;
        if other_component == query_component {
            continue;
        }

        let Some(other_target_component) = target.connected_component(*mapped_target_atom) else {
            return false;
        };

        match query.component_group(other_component) {
            Some(other_group)
                if other_group == query_group && other_target_component != target_component =>
            {
                return false;
            }
            Some(other_group)
                if other_group != query_group && other_target_component == target_component =>
            {
                return false;
            }
            _ => {}
        }
    }

    true
}

fn select_next_query_atom(
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    query_to_target: &[Option<usize>],
) -> QueryAtomId {
    let mut best_atom = None;
    let mut best_key = (0usize, 0usize, core::cmp::Reverse(usize::MAX));

    for (query_atom, neighbors) in query_neighbors.iter().enumerate() {
        if query_to_target[query_atom].is_some() {
            continue;
        }
        let mapped_neighbors = neighbors
            .iter()
            .filter(|(neighbor, _)| query_to_target[*neighbor].is_some())
            .count();
        let key = (
            mapped_neighbors,
            neighbors.len(),
            core::cmp::Reverse(query_atom),
        );
        if best_atom.is_none() || key > best_key {
            best_atom = Some(query_atom);
            best_key = key;
        }
    }

    best_atom.expect("at least one unmapped query atom must remain")
}

fn candidate_target_atoms(
    query_atom: QueryAtomId,
    query: &QueryMol,
    stereo_plan: &QueryStereoPlan,
    target: &PreparedTarget,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    query_to_target: &[Option<usize>],
) -> alloc::vec::Vec<usize> {
    let mapped_neighbors = query_neighbors[query_atom]
        .iter()
        .filter_map(|(neighbor, bond_index)| {
            Some((
                query_to_target[*neighbor]?,
                *bond_index,
                &query.bonds()[*bond_index],
            ))
        })
        .collect::<alloc::vec::Vec<_>>();

    if mapped_neighbors.is_empty() {
        return (0..target.atom_count()).collect();
    }

    let (seed_target_atom, seed_bond_id, seed_query_bond) = mapped_neighbors[0];
    let mut candidates = target
        .neighbors(seed_target_atom)
        .filter_map(|(neighbor_atom, bond_label)| {
            query_bond_matches(
                seed_bond_id,
                &seed_query_bond.expr,
                stereo_plan,
                target,
                seed_target_atom,
                neighbor_atom,
                bond_label,
            )
            .then_some(neighbor_atom)
        })
        .collect::<alloc::vec::Vec<_>>();

    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn mapped_bonds_match(
    query_atom: QueryAtomId,
    target_atom: usize,
    query: &QueryMol,
    stereo_plan: &QueryStereoPlan,
    target: &PreparedTarget,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    query_to_target: &[Option<usize>],
) -> bool {
    query_neighbors[query_atom]
        .iter()
        .filter_map(|(neighbor_atom, bond_index)| {
            Some((
                query_to_target[*neighbor_atom]?,
                *bond_index,
                &query.bonds()[*bond_index],
            ))
        })
        .all(|(mapped_target_atom, query_bond_id, query_bond)| {
            target
                .bond(target_atom, mapped_target_atom)
                .is_some_and(|bond| {
                    query_bond_matches(
                        query_bond_id,
                        &query_bond.expr,
                        stereo_plan,
                        target,
                        target_atom,
                        mapped_target_atom,
                        bond,
                    )
                })
        })
}

fn query_bond_matches(
    bond_id: usize,
    expr: &BondExpr,
    stereo_plan: &QueryStereoPlan,
    target: &PreparedTarget,
    left_atom: usize,
    right_atom: usize,
    target_bond: BondLabel,
) -> bool {
    if bond_expr_contains_negated_directional_primitive(expr) {
        return false;
    }
    if query_bond_is_directional(stereo_plan, bond_id) {
        return matches!(
            target_bond,
            BondLabel::Single | BondLabel::Up | BondLabel::Down
        );
    }
    bond_expr_matches(expr, target, left_atom, right_atom, target_bond)
}

fn bond_expr_matches(
    expr: &BondExpr,
    target: &PreparedTarget,
    left_atom: usize,
    right_atom: usize,
    target_bond: BondLabel,
) -> bool {
    match expr {
        BondExpr::Elided => matches!(
            target_bond,
            BondLabel::Single | BondLabel::Aromatic | BondLabel::Up | BondLabel::Down
        ),
        BondExpr::Query(tree) => {
            bond_tree_matches(tree, target, left_atom, right_atom, target_bond)
        }
    }
}

fn bond_tree_matches(
    tree: &BondExprTree,
    target: &PreparedTarget,
    left_atom: usize,
    right_atom: usize,
    target_bond: BondLabel,
) -> bool {
    match tree {
        BondExprTree::Primitive(primitive) => {
            bond_primitive_matches(*primitive, target, left_atom, right_atom, target_bond)
        }
        BondExprTree::Not(inner) => {
            !bond_tree_matches(inner, target, left_atom, right_atom, target_bond)
        }
        BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => items
            .iter()
            .all(|item| bond_tree_matches(item, target, left_atom, right_atom, target_bond)),
        BondExprTree::Or(items) => items
            .iter()
            .any(|item| bond_tree_matches(item, target, left_atom, right_atom, target_bond)),
    }
}

fn bond_primitive_matches(
    primitive: BondPrimitive,
    target: &PreparedTarget,
    left_atom: usize,
    right_atom: usize,
    target_bond: BondLabel,
) -> bool {
    match primitive {
        BondPrimitive::Bond(Bond::Single) => {
            matches!(
                target_bond,
                BondLabel::Single | BondLabel::Up | BondLabel::Down
            )
        }
        BondPrimitive::Bond(Bond::Double) => target_bond == BondLabel::Double,
        BondPrimitive::Bond(Bond::Triple) => target_bond == BondLabel::Triple,
        BondPrimitive::Bond(Bond::Aromatic) => target_bond == BondLabel::Aromatic,
        BondPrimitive::Bond(Bond::Up | Bond::Down) => {
            matches!(
                target_bond,
                BondLabel::Single | BondLabel::Up | BondLabel::Down
            )
        }
        BondPrimitive::Bond(Bond::Quadruple) => false,
        BondPrimitive::Any => true,
        BondPrimitive::Ring => target.is_ring_bond(left_atom, right_atom),
    }
}

fn atom_expr_matches(
    expr: &AtomExpr,
    query_atom_id: QueryAtomId,
    stereo_plan: &QueryStereoPlan,
    target: &PreparedTarget,
    atom_id: usize,
    recursive_cache: &mut RecursiveMatchCache,
) -> bool {
    match expr {
        AtomExpr::Wildcard => true,
        AtomExpr::Bare { element, aromatic } => atom_matches_symbol(
            target.atom(atom_id),
            target,
            atom_id,
            target.is_aromatic(atom_id),
            *element,
            *aromatic,
        ),
        AtomExpr::Bracket(expr) => bracket_tree_matches(
            &expr.tree,
            query_atom_id,
            stereo_plan,
            target,
            atom_id,
            recursive_cache,
        ),
    }
}

fn bracket_tree_matches(
    tree: &BracketExprTree,
    query_atom_id: QueryAtomId,
    stereo_plan: &QueryStereoPlan,
    target: &PreparedTarget,
    atom_id: usize,
    recursive_cache: &mut RecursiveMatchCache,
) -> bool {
    match tree {
        BracketExprTree::Primitive(primitive) => atom_primitive_matches(
            primitive,
            query_atom_id,
            stereo_plan,
            target,
            atom_id,
            recursive_cache,
        ),
        BracketExprTree::Not(inner) => !bracket_tree_matches(
            inner,
            query_atom_id,
            stereo_plan,
            target,
            atom_id,
            recursive_cache,
        ),
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => {
            items.iter().all(|item| {
                bracket_tree_matches(
                    item,
                    query_atom_id,
                    stereo_plan,
                    target,
                    atom_id,
                    recursive_cache,
                )
            })
        }
        BracketExprTree::Or(items) => items.iter().any(|item| {
            bracket_tree_matches(
                item,
                query_atom_id,
                stereo_plan,
                target,
                atom_id,
                recursive_cache,
            )
        }),
    }
}

fn atom_primitive_matches(
    primitive: &AtomPrimitive,
    _query_atom_id: QueryAtomId,
    _stereo_plan: &QueryStereoPlan,
    target: &PreparedTarget,
    atom_id: usize,
    recursive_cache: &mut RecursiveMatchCache,
) -> bool {
    let atom = target.atom(atom_id);
    let aromatic = target.is_aromatic(atom_id);

    match primitive {
        AtomPrimitive::Wildcard | AtomPrimitive::Chirality(_) => true,
        AtomPrimitive::AliphaticAny => atom.and_then(Atom::element).is_some() && !aromatic,
        AtomPrimitive::AromaticAny => atom.and_then(Atom::element).is_some() && aromatic,
        AtomPrimitive::Symbol { element, aromatic } => atom_matches_symbol(
            atom,
            target,
            atom_id,
            target.is_aromatic(atom_id),
            *element,
            *aromatic,
        ),
        AtomPrimitive::Isotope { isotope, aromatic } => {
            atom_matches_isotope(atom, target.is_aromatic(atom_id), *isotope, *aromatic)
        }
        AtomPrimitive::IsotopeWildcard(mass_number) => atom.is_some_and(|atom| {
            if *mass_number == 0 {
                atom.isotope_mass_number().is_none()
            } else {
                atom.isotope_mass_number() == Some(*mass_number)
            }
        }),
        AtomPrimitive::AtomicNumber(atomic_number) => {
            if *atomic_number == 1 && is_hidden_attached_hydrogen(target, atom_id) {
                false
            } else {
                atom.and_then(Atom::element)
                    .is_some_and(|element| u16::from(element.atomic_number()) == *atomic_number)
            }
        }
        AtomPrimitive::Degree(expected) => target.degree(atom_id).is_some_and(|degree| {
            let actual = u16::try_from(degree).unwrap_or(u16::MAX);
            count_query_matches_u16(*expected, actual, 1)
        }),
        AtomPrimitive::Connectivity(expected) => target
            .connectivity(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::Valence(expected) => target
            .total_valence(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::Hydrogen(HydrogenKind::Total, expected) => target
            .total_hydrogen_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::Hydrogen(HydrogenKind::Implicit, expected) => target
            .implicit_hydrogen_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::RingMembership(expected) => target
            .ring_membership_count(atom_id)
            .is_some_and(|count| ring_query_matches_u16(*expected, u16::from(count))),
        AtomPrimitive::RingSize(expected) => target
            .smallest_ring_size(atom_id)
            .is_some_and(|size| ring_query_matches_u16(*expected, u16::from(size))),
        AtomPrimitive::RingConnectivity(expected) => target
            .ring_bond_count(atom_id)
            .is_some_and(|count| ring_query_matches_u16(*expected, u16::from(count))),
        AtomPrimitive::Charge(expected) => {
            atom.is_some_and(|atom| atom.charge_value() == *expected)
        }
        AtomPrimitive::RecursiveQuery(query) => {
            recursive_query_matches(query, target, atom_id, recursive_cache)
        }
    }
}

fn atom_chirality_matches(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    query_atom_id: QueryAtomId,
    target: &PreparedTarget,
    query_to_target: &[Option<usize>],
    atom_stereo_cache: &RefCell<AtomStereoCache>,
) -> bool {
    let Some(query_chirality) = extract_query_chirality(&query.atoms()[query_atom_id].expr) else {
        return true;
    };
    if !is_supported_query_chirality(query_chirality) {
        return false;
    }
    if !query_chirality_requires_tetrahedral_match(query_chirality) {
        return true;
    }

    let incident = &query_neighbors[query_atom_id];
    if incident.len() < 3 || incident.len() > 4 {
        return true;
    }

    let Some(target_atom_id) = query_to_target[query_atom_id] else {
        return false;
    };
    let Some(actual_target_chirality) = target.tetrahedral_chirality(target_atom_id) else {
        return false;
    };
    let Ok(neighbor_tokens) = local_neighbor_tokens_for_mapping(incident, query_to_target, target)
    else {
        return false;
    };
    let cache_key = AtomStereoCacheKey {
        query_atom_id,
        neighbor_tokens,
    };
    let expected_chirality = {
        let cache = atom_stereo_cache.borrow();
        cache.get(&cache_key).copied()
    };
    let expected_chirality = expected_chirality.unwrap_or_else(|| {
        let computed = build_local_stereo_fragment(
            query,
            query_neighbors,
            query_atom_id,
            &cache_key.neighbor_tokens,
        )
        .ok()
        .and_then(|(local_fragment, local_center_atom_id)| {
            local_fragment
                .parse::<Smiles>()
                .ok()
                .and_then(|local_smiles| {
                    local_smiles.smarts_tetrahedral_chirality(local_center_atom_id)
                })
        });
        atom_stereo_cache
            .borrow_mut()
            .insert(cache_key.clone(), computed);
        computed
    });
    expected_chirality == Some(actual_target_chirality)
}

fn stereo_constraints_match(
    query: &QueryMol,
    stereo_plan: &QueryStereoPlan,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    target: &PreparedTarget,
    query_to_target: &[Option<usize>],
    atom_stereo_cache: &RefCell<AtomStereoCache>,
) -> bool {
    stereo_plan
        .double_bond_constraints
        .iter()
        .all(|constraint| {
            let Some(left_target) = query_to_target[constraint.left_atom] else {
                return false;
            };
            let Some(right_target) = query_to_target[constraint.right_atom] else {
                return false;
            };
            target.double_bond_stereo_config(left_target, right_target) == Some(constraint.config)
        })
        && query.atoms().iter().all(|atom| {
            atom_chirality_matches(
                query,
                query_neighbors,
                atom.id,
                target,
                query_to_target,
                atom_stereo_cache,
            )
        })
}

fn recursive_query_matches(
    query: &QueryMol,
    target: &PreparedTarget,
    atom_id: usize,
    recursive_cache: &mut RecursiveMatchCache,
) -> bool {
    let cache_key = (core::ptr::from_ref(query) as usize, atom_id);
    if let Some(cached) = recursive_cache.get(&cache_key) {
        return *cached;
    }

    if query.is_empty() {
        recursive_cache.insert(cache_key, false);
        return false;
    }

    let Ok(compiled) = CompiledQuery::new(query.clone()) else {
        recursive_cache.insert(cache_key, false);
        return false;
    };
    let anchored =
        query_matches_with_mapping(&compiled, target, recursive_cache, Some(0), Some(atom_id));
    recursive_cache.insert(cache_key, anchored);
    anchored
}

fn atom_matches_symbol(
    atom: Option<&Atom>,
    target: &PreparedTarget,
    atom_id: usize,
    target_is_aromatic: bool,
    expected_element: elements_rs::Element,
    expected_aromatic: bool,
) -> bool {
    if expected_element == elements_rs::Element::H
        && !expected_aromatic
        && is_hidden_attached_hydrogen(target, atom_id)
    {
        return false;
    }
    atom.and_then(Atom::element).is_some_and(|element| {
        element == expected_element && target_is_aromatic == expected_aromatic
    })
}

fn atom_matches_isotope(
    atom: Option<&Atom>,
    target_is_aromatic: bool,
    expected_isotope: elements_rs::Isotope,
    expected_aromatic: bool,
) -> bool {
    atom.is_some_and(|atom| {
        atom.element().is_some_and(|element| {
            element == expected_isotope.element()
                && atom.isotope_mass_number() == Some(expected_isotope.mass_number())
                && target_is_aromatic == expected_aromatic
        })
    })
}

fn is_hidden_attached_hydrogen(target: &PreparedTarget, atom_id: usize) -> bool {
    let Some(atom) = target.atom(atom_id) else {
        return false;
    };
    if atom.element() != Some(elements_rs::Element::H)
        || atom.isotope_mass_number().is_some()
        || atom.charge_value() != 0
        || target.degree(atom_id) != Some(1)
    {
        return false;
    }

    let mut neighbors = target.neighbors(atom_id);
    let Some((neighbor_id, _)) = neighbors.next() else {
        return false;
    };
    if neighbors.next().is_some() {
        return false;
    }

    target
        .atom(neighbor_id)
        .and_then(Atom::element)
        .is_some_and(|element| element != elements_rs::Element::H)
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use smiles_parser::Smiles;

    use super::{
        component_constraints_match, match_count, match_count_compiled, matches, matches_compiled,
        query_matches, substructure_matches, CompiledQuery,
    };
    use crate::error::SmartsMatchError;
    use crate::prepared::PreparedTarget;
    use smarts_parser::QueryMol;

    #[test]
    fn rejects_empty_target() {
        let query = QueryMol::from_str("C").unwrap();
        let error = matches(&query, "").unwrap_err();
        assert!(matches!(error, SmartsMatchError::EmptyTarget));
    }

    #[test]
    fn rejects_invalid_target_smiles() {
        let query = QueryMol::from_str("C").unwrap();
        let error = matches(&query, "C)").unwrap_err();
        assert!(matches!(
            error,
            SmartsMatchError::InvalidTargetSmiles { .. }
        ));
    }

    #[test]
    fn negated_directional_bond_forms_match_nothing_in_current_rdkit_slice() {
        assert!(!matches(&QueryMol::from_str("F/!\\C=C/F").unwrap(), "F/C=C/F").unwrap());
        assert!(!matches(&QueryMol::from_str("F/!\\C=C/F").unwrap(), "F\\C=C\\F").unwrap());
        assert!(!matches(&QueryMol::from_str("F\\!/C=C/F").unwrap(), "F/C=C\\F").unwrap());
        assert!(!matches(&QueryMol::from_str("F\\!/C=C/F").unwrap(), "FC=CF").unwrap());
    }

    #[test]
    fn single_atom_supported_query_matches() {
        let query = QueryMol::from_str("[O;H1]").unwrap();
        assert!(matches(&query, "CCO").unwrap());
    }

    #[test]
    fn compiled_query_matches_existing_prepared_path() {
        let compiled = CompiledQuery::new(QueryMol::from_str("F/C=C/F").unwrap()).unwrap();
        let prepared = PreparedTarget::new(Smiles::from_str("F/C=C/F").unwrap());
        assert!(matches_compiled(&compiled, &prepared));
        assert!(matches(compiled.query(), "F/C=C/F").unwrap());
    }

    #[test]
    fn counted_matches_follow_rdkit_uniquify_for_symmetric_queries() {
        let query = QueryMol::from_str("CC").unwrap();
        assert_eq!(match_count(&query, "CC").unwrap(), 1);
        assert_eq!(
            substructure_matches(&query, "CC").unwrap().as_ref(),
            &[alloc::boxed::Box::<[usize]>::from([0, 1])]
        );
    }

    #[test]
    fn counted_matches_include_overlapping_embeddings() {
        let query = QueryMol::from_str("CC").unwrap();
        assert_eq!(match_count(&query, "CCC").unwrap(), 2);
        assert_eq!(
            substructure_matches(&query, "CCC").unwrap().as_ref(),
            &[
                alloc::boxed::Box::<[usize]>::from([0, 1]),
                alloc::boxed::Box::<[usize]>::from([1, 2]),
            ]
        );
    }

    #[test]
    fn counted_matches_choose_a_canonical_query_order_representative() {
        let query = QueryMol::from_str("C~C~C~C").unwrap();
        assert_eq!(match_count(&query, "C1CCC=1").unwrap(), 1);
        assert_eq!(
            substructure_matches(&query, "C1CCC=1").unwrap().as_ref(),
            &[alloc::boxed::Box::<[usize]>::from([0, 1, 2, 3])]
        );
    }

    #[test]
    fn counted_matches_handle_simple_single_atom_cases() {
        assert_eq!(
            match_count(&QueryMol::from_str("C").unwrap(), "CCC").unwrap(),
            3
        );
        assert_eq!(
            match_count(&QueryMol::from_str("[#8]").unwrap(), "O=CO").unwrap(),
            2
        );
        assert_eq!(
            match_count(&QueryMol::from_str("[#8]").unwrap(), "CC.O").unwrap(),
            1
        );
    }

    #[test]
    fn counted_matches_agree_with_boolean_matching() {
        let query = QueryMol::from_str("C.C").unwrap();
        let prepared = PreparedTarget::new(Smiles::from_str("CC").unwrap());
        let compiled = CompiledQuery::new(query).unwrap();
        assert_eq!(
            matches_compiled(&compiled, &prepared),
            match_count_compiled(&compiled, &prepared) > 0
        );
    }

    #[test]
    fn low_level_boolean_tree_is_respected() {
        let query = QueryMol::from_str("[!O]").unwrap();
        assert!(matches(&query, "C").unwrap());
        assert!(!matches(&query, "O").unwrap());
    }

    #[test]
    fn bracket_or_is_respected() {
        let query = QueryMol::from_str("[N,O]").unwrap();
        assert!(matches(&query, "CCO").unwrap());
    }

    #[test]
    fn connected_query_does_not_reuse_target_atoms() {
        let query = QueryMol::from_str("CCC").unwrap();
        assert!(!matches(&query, "CC").unwrap());
    }

    #[test]
    fn connected_query_matches_simple_cycle() {
        let query = QueryMol::from_str("C1CC1").unwrap();
        assert!(matches(&query, "C1CC1").unwrap());
        assert!(!matches(&query, "CCC").unwrap());
    }

    #[test]
    fn disconnected_query_matches_without_grouping() {
        let query = QueryMol::from_str("C.C").unwrap();
        assert!(matches(&query, "CC").unwrap());
        assert!(matches(&query, "C.C").unwrap());
        assert!(!matches(&query, "C").unwrap());
    }

    #[test]
    fn bond_boolean_logic_is_respected() {
        let query = QueryMol::from_str("C!#N").unwrap();
        assert!(matches(&query, "CN").unwrap());
        assert!(!matches(&query, "CC#N").unwrap());
    }

    #[test]
    fn connectivity_and_valence_primitives_are_respected() {
        assert!(matches(&QueryMol::from_str("[X4]").unwrap(), "C").unwrap());
        assert!(!matches(&QueryMol::from_str("[X3]").unwrap(), "C").unwrap());
        assert!(matches(&QueryMol::from_str("[N&X]").unwrap(), "C#N").unwrap());

        assert!(matches(&QueryMol::from_str("[v4]").unwrap(), "c1ccccc1").unwrap());
        assert!(!matches(&QueryMol::from_str("[v3]").unwrap(), "c1ccccc1").unwrap());
        assert!(matches(&QueryMol::from_str("[Cl&v]").unwrap(), "CCCl").unwrap());
    }

    #[test]
    fn ring_atom_primitives_are_respected() {
        assert!(matches(&QueryMol::from_str("[R]").unwrap(), "C1CC1").unwrap());
        assert!(!matches(&QueryMol::from_str("[R]").unwrap(), "CCC").unwrap());
        assert!(matches(&QueryMol::from_str("[R0]").unwrap(), "CCC").unwrap());
        assert!(matches(&QueryMol::from_str("[r3]").unwrap(), "C1CC1").unwrap());
        assert!(matches(&QueryMol::from_str("[x2]").unwrap(), "C1CC1").unwrap());
    }

    #[test]
    fn ring_bond_primitives_are_respected() {
        assert!(matches(&QueryMol::from_str("C@C").unwrap(), "C1CC1").unwrap());
        assert!(!matches(&QueryMol::from_str("C@C").unwrap(), "CC").unwrap());
        assert!(matches(&QueryMol::from_str("C!@C").unwrap(), "CC").unwrap());
        assert!(!matches(&QueryMol::from_str("C!@C").unwrap(), "C1CC1").unwrap());
    }

    #[test]
    fn ordinary_attached_hydrogens_are_not_matchable_as_hydrogen_atoms() {
        assert!(!matches(&QueryMol::from_str("[H]").unwrap(), "[H]Cl").unwrap());
        assert!(!matches(&QueryMol::from_str("[#1]").unwrap(), "[H]Cl").unwrap());
        assert!(!matches(&QueryMol::from_str("[H]Cl").unwrap(), "[H]Cl").unwrap());
        assert!(matches(&QueryMol::from_str("[H][H]").unwrap(), "[H][H]").unwrap());
        assert!(matches(&QueryMol::from_str("[2H]").unwrap(), "[2H]Cl").unwrap());
    }

    #[test]
    fn zero_level_grouping_requires_same_target_component() {
        let query = QueryMol::from_str("(C.C)").unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CC").unwrap());
        let compiled = CompiledQuery::new(query.clone()).unwrap();
        let mapping = [Some(0), None];
        assert_eq!(query.component_groups(), &[Some(0), Some(0)]);
        assert_eq!(target.connected_component(0), Some(0));
        assert_eq!(target.connected_component(1), Some(0));
        assert!(component_constraints_match(1, 1, &query, &target, &mapping));
        assert!(query_matches(&compiled, &target));
        assert!(matches(&query, "CC").unwrap());
        assert!(matches(&QueryMol::from_str("(C.C)").unwrap(), "CC.C").unwrap());
        assert!(!matches(&QueryMol::from_str("(C.C)").unwrap(), "C.C").unwrap());
    }

    #[test]
    fn separate_groups_require_different_target_components() {
        assert!(!matches(&QueryMol::from_str("(C).(C)").unwrap(), "CC").unwrap());
        assert!(matches(&QueryMol::from_str("(C).(C)").unwrap(), "C.C").unwrap());
        assert!(matches(&QueryMol::from_str("(C).(C)").unwrap(), "CC.C").unwrap());
    }

    #[test]
    fn grouped_and_ungrouped_components_interact_correctly() {
        assert!(matches(&QueryMol::from_str("(C).C").unwrap(), "CC").unwrap());
        assert!(matches(&QueryMol::from_str("(C).C").unwrap(), "C.C").unwrap());
        assert!(matches(&QueryMol::from_str("(C).(C).C").unwrap(), "CC.C").unwrap());
        assert!(!matches(&QueryMol::from_str("(C).(C.C)").unwrap(), "CCC").unwrap());
        assert!(matches(&QueryMol::from_str("(C).(C.C)").unwrap(), "CC.C").unwrap());
    }

    #[test]
    fn recursive_queries_are_respected() {
        assert!(matches(&QueryMol::from_str("[$(C)]").unwrap(), "C").unwrap());
        assert!(!matches(&QueryMol::from_str("[$(O)]").unwrap(), "C").unwrap());
        assert!(matches(&QueryMol::from_str("[$(CO)]").unwrap(), "CO").unwrap());
        assert!(!matches(&QueryMol::from_str("[$(CO)]").unwrap(), "CC").unwrap());
        assert!(matches(&QueryMol::from_str("[C&$(*O)]").unwrap(), "CO").unwrap());
        assert!(!matches(&QueryMol::from_str("[C&$(*O)]").unwrap(), "CC").unwrap());
    }

    #[test]
    fn explicit_tetrahedral_chirality_is_respected() {
        assert!(matches(&QueryMol::from_str("F[C@](Cl)Br").unwrap(), "F[C@H](Cl)Br").unwrap());
        assert!(!matches(&QueryMol::from_str("F[C@](Cl)Br").unwrap(), "F[C@@H](Cl)Br").unwrap());
        assert!(matches(
            &QueryMol::from_str("F[C@@](Cl)Br").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
        assert!(!matches(&QueryMol::from_str("F[C@@](Cl)Br").unwrap(), "F[C@H](Cl)Br").unwrap());
        assert!(matches(&QueryMol::from_str("F[C@H](Cl)Br").unwrap(), "F[C@H](Cl)Br").unwrap());
        assert!(!matches(
            &QueryMol::from_str("F[C@H](Cl)Br").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
        assert!(matches(
            &QueryMol::from_str("F[C@@H](Cl)Br").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
        assert!(!matches(
            &QueryMol::from_str("F[C@@H](Cl)Br").unwrap(),
            "F[C@H](Cl)Br"
        )
        .unwrap());
        assert!(matches(
            &QueryMol::from_str("F[C@TH1](Cl)Br").unwrap(),
            "F[C@H](Cl)Br"
        )
        .unwrap());
        assert!(!matches(
            &QueryMol::from_str("F[C@TH1](Cl)Br").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
        assert!(matches(
            &QueryMol::from_str("F[C@TH2](Cl)Br").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
        assert!(!matches(&QueryMol::from_str("F[C@TH2](Cl)Br").unwrap(), "FC(Cl)Br").unwrap());
        assert!(matches(
            &QueryMol::from_str("Br[C@TH1](Cl)F").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
        assert!(!matches(
            &QueryMol::from_str("Br[C@TH1](Cl)F").unwrap(),
            "F[C@H](Cl)Br"
        )
        .unwrap());
        assert!(matches(
            &QueryMol::from_str("[C@TH2](Br)(Cl)F").unwrap(),
            "F[C@H](Cl)Br"
        )
        .unwrap());
        assert!(!matches(
            &QueryMol::from_str("[C@TH2](Br)(Cl)F").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
        assert!(matches(&QueryMol::from_str("Br[C@](Cl)F").unwrap(), "F[C@@H](Cl)Br").unwrap());
        assert!(matches(&QueryMol::from_str("Br[C@@](Cl)F").unwrap(), "F[C@H](Cl)Br").unwrap());
        assert!(matches(
            &QueryMol::from_str("Br[C@H](Cl)F").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
        assert!(!matches(&QueryMol::from_str("Br[C@H](Cl)F").unwrap(), "F[C@H](Cl)Br").unwrap());
        assert!(matches(
            &QueryMol::from_str("[C@H](F)(Cl)Br").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
        assert!(!matches(
            &QueryMol::from_str("[C@H](F)(Cl)Br").unwrap(),
            "F[C@H](Cl)Br"
        )
        .unwrap());
        assert!(matches(
            &QueryMol::from_str("[C@@H](F)(Cl)Br").unwrap(),
            "F[C@H](Cl)Br"
        )
        .unwrap());
        assert!(!matches(
            &QueryMol::from_str("[C@@H](F)(Cl)Br").unwrap(),
            "F[C@@H](Cl)Br"
        )
        .unwrap());
    }

    #[test]
    fn underconstrained_tetrahedral_queries_do_not_overconstrain() {
        assert!(matches(&QueryMol::from_str("[C@](F)Cl").unwrap(), "F[C@H](Cl)Br").unwrap());
        assert!(matches(&QueryMol::from_str("[C@](F)Cl").unwrap(), "F[C@@H](Cl)Br").unwrap());
        assert!(matches(&QueryMol::from_str("[C@@](F)Cl").unwrap(), "F[C@H](Cl)Br").unwrap());
        assert!(matches(&QueryMol::from_str("[C@@](F)Cl").unwrap(), "F[C@@H](Cl)Br").unwrap());
        assert!(matches(&QueryMol::from_str("[C@H](F)Cl").unwrap(), "F[C@H](Cl)Br").unwrap());
        assert!(matches(&QueryMol::from_str("[C@H](F)Cl").unwrap(), "F[C@@H](Cl)Br").unwrap());
        assert!(matches(&QueryMol::from_str("[C@@H](F)Cl").unwrap(), "F[C@H](Cl)Br").unwrap());
        assert!(matches(&QueryMol::from_str("[C@@H](F)Cl").unwrap(), "F[C@@H](Cl)Br").unwrap());
    }

    #[test]
    fn supported_nontetrahedral_query_classes_do_not_constrain_matches() {
        for smarts in [
            "[C@AL1](F)(Cl)Br",
            "[C@AL2](F)(Cl)Br",
            "[C@SP1](F)(Cl)Br",
            "[C@SP2](F)(Cl)Br",
            "[C@SP3](F)(Cl)Br",
            "[C@TB1](F)(Cl)Br",
            "[C@TB20](F)(Cl)Br",
            "[C@OH1](F)(Cl)Br",
            "[C@OH30](F)(Cl)Br",
        ] {
            let query = QueryMol::from_str(smarts).unwrap();
            assert!(matches(&query, "F[C@H](Cl)Br").unwrap(), "{smarts}");
            assert!(matches(&query, "F[C@@H](Cl)Br").unwrap(), "{smarts}");
            assert!(matches(&query, "FC(Cl)Br").unwrap(), "{smarts}");
        }
    }

    #[test]
    fn multidirectional_endpoint_queries_are_respected() {
        assert!(matches(
            &QueryMol::from_str("F/C=C(/Cl)\\Br").unwrap(),
            "F/C=C(/Cl)Br"
        )
        .unwrap());
        assert!(!matches(
            &QueryMol::from_str("F/C=C(/Cl)\\Br").unwrap(),
            "F/C=C(\\Cl)Br"
        )
        .unwrap());
        assert!(matches(&QueryMol::from_str("F/,\\C=C/F").unwrap(), "F/C=C/F").unwrap());
        assert!(matches(&QueryMol::from_str("F/,\\C=C/F").unwrap(), "F\\C=C\\F").unwrap());
        assert!(!matches(&QueryMol::from_str("F/,\\C=C/F").unwrap(), "F/C=C\\F").unwrap());
        assert!(matches(&QueryMol::from_str("F/;\\C=C/F").unwrap(), "F/C=C/F").unwrap());
        assert!(matches(&QueryMol::from_str("F\\,/C=C/F").unwrap(), "F/C=C\\F").unwrap());
        assert!(matches(&QueryMol::from_str("F/&\\C=C/F").unwrap(), "F/C=C/F").unwrap());
        assert!(matches(&QueryMol::from_str("F\\&/C=C/F").unwrap(), "F/C=C\\F").unwrap());
    }

    #[test]
    fn semantic_double_bond_stereo_is_respected() {
        assert!(matches(&QueryMol::from_str("C/C").unwrap(), "CC").unwrap());
        assert!(matches(&QueryMol::from_str("F/C=C").unwrap(), "F/C=C/F").unwrap());
        assert!(matches(&QueryMol::from_str("F/C=C").unwrap(), "F/C=C\\F").unwrap());
        assert!(matches(&QueryMol::from_str("F/C=C").unwrap(), "FC=CF").unwrap());
        assert!(matches(&QueryMol::from_str("C=C/F").unwrap(), "F/C=C/F").unwrap());
        assert!(matches(&QueryMol::from_str("C=C/F").unwrap(), "FC=CF").unwrap());
        assert!(matches(&QueryMol::from_str("F/C=C/F").unwrap(), "F/C=C/F").unwrap());
        assert!(matches(&QueryMol::from_str("F/C=C/F").unwrap(), "F\\C=C\\F").unwrap());
        assert!(!matches(&QueryMol::from_str("F/C=C/F").unwrap(), "F/C=C\\F").unwrap());
        assert!(!matches(&QueryMol::from_str("F/C=C/F").unwrap(), "FC=CF").unwrap());
        assert!(matches(&QueryMol::from_str("F/C=C\\F").unwrap(), "F\\C=C/F").unwrap());
        assert!(matches(&QueryMol::from_str("C/C=C/C").unwrap(), "C/C=C/C").unwrap());
        assert!(!matches(&QueryMol::from_str("C/C=C/C").unwrap(), "C/C=C\\C").unwrap());
        assert!(!matches(&QueryMol::from_str("C/C=C/C").unwrap(), "CC=CC").unwrap());
    }
}
