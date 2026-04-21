#![no_std]

//! SMARTS validation against target molecules.
//!
//! This crate is the matching layer for compiled SMARTS queries.
//! The current implementation supports a broad molecule-SMARTS slice:
//!
//! - a match entrypoint
//! - reusable compiled SMARTS queries
//! - `smiles-parser` backed target preparation
//! - disconnected and recursive SMARTS matching
//! - ring/count/stereo-aware matching against frozen `RDKit` fixtures
//! - a structured error type
//!
//! The parser crate owns syntax and query IR. This crate will own target
//! adaptation, preparation, and matching.

extern crate alloc;
#[cfg(test)]
extern crate std;

/// Matching and validation error types.
pub mod error;
pub mod geometric_target;
/// Placeholder matching entrypoints.
pub mod matching;
/// Prepared target sidecars and cached molecule properties.
pub mod prepared;
/// Conservative screening summaries for many-query workflows.
pub mod screening;
/// Target graph traits and simple placeholder target types.
pub mod target;

pub use error::SmartsMatchError;
pub use matching::{
    match_count, match_count_compiled, match_count_prepared, matches, matches_compiled,
    matches_prepared, substructure_matches, substructure_matches_compiled,
    substructure_matches_prepared, CompiledQuery,
};
pub use prepared::{EdgeProps, NodeProps, PreparedMolecule, PreparedTarget};
pub use screening::{QueryScreen, TargetScreen};
pub use target::{AtomId, AtomLabel, BondLabel, MoleculeTarget, Neighbor};

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use smarts_parser::QueryMol;

    fn assert_sync<T: Sync>() {}

    #[test]
    fn matches_single_atom_query_against_smiles_target() {
        let query = QueryMol::from_str("[O;H1]").unwrap();
        assert!(crate::matches(&query, "CCO").unwrap());
    }

    #[test]
    fn compiled_query_should_be_sync_for_parallel_matching() {
        assert_sync::<crate::CompiledQuery>();
    }
}
