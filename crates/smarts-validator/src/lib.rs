#![no_std]

//! SMARTS validation against target molecules.
//!
//! This crate is the future matching layer for compiled SMARTS queries.
//! The current implementation supports the first real validator slice:
//!
//! - a match entrypoint
//! - `smiles-parser` backed target preparation
//! - single-atom SMARTS evaluation against frozen `RDKit` fixtures
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
pub use matching::matches;
pub use prepared::{EdgeProps, NodeProps, PreparedMolecule, PreparedTarget};
pub use screening::{QueryScreen, TargetScreen};
pub use target::{AtomId, AtomLabel, BondLabel, MoleculeTarget, Neighbor};

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use smarts_parser::QueryMol;

    #[test]
    fn matches_single_atom_query_against_smiles_target() {
        let query = QueryMol::from_str("[O;H1]").unwrap();
        assert!(crate::matches(&query, "CCO").unwrap());
    }
}
