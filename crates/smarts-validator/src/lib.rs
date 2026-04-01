#![no_std]

//! SMARTS validation against target molecules.
//!
//! This crate is the future matching layer for compiled SMARTS queries.
//! For now it only provides a small, thread-safe scaffold:
//!
//! - a match entrypoint
//! - target/preparation placeholders
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
/// Target graph traits and simple placeholder target types.
pub mod target;

pub use error::SmartsMatchError;
pub use matching::matches;
pub use prepared::{EdgeProps, NodeProps, PreparedMolecule, PreparedTarget};
pub use target::{AtomId, AtomLabel, BondLabel, MoleculeTarget, Neighbor};

#[cfg(test)]
mod tests {
    use super::error::SmartsMatchError;
    use super::target::TargetText;

    #[test]
    fn target_text_rejects_empty_input() {
        let err = TargetText::new("").unwrap_err();
        assert_eq!(err, SmartsMatchError::EmptyTarget);
    }

    #[test]
    fn target_text_preserves_input() {
        let target = TargetText::new("CCO").unwrap();
        assert_eq!(target.as_str(), "CCO");
    }
}
