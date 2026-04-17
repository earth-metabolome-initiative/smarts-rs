//! Error types for SMARTS matching and validation.

use thiserror::Error;

/// Errors raised by SMARTS validation.
#[derive(Debug, Error)]
pub enum SmartsMatchError {
    /// The supplied target input was empty.
    #[error("target input is empty")]
    EmptyTarget,
    /// The target string was not valid SMILES.
    #[error("invalid target SMILES at {start}..{end}: {source}")]
    InvalidTargetSmiles {
        /// Underlying SMILES parse error kind.
        #[source]
        source: smiles_parser::SmilesError,
        /// Byte start offset for the error.
        start: usize,
        /// Byte end offset for the error.
        end: usize,
    },
    /// The query uses an atom primitive that is not implemented yet.
    #[error("unsupported SMARTS atom primitive in current validator slice: {primitive}")]
    UnsupportedAtomPrimitive {
        /// Human-readable primitive name.
        primitive: &'static str,
    },
    /// The query uses a bond primitive that is not implemented yet.
    #[error("unsupported SMARTS bond primitive in current validator slice: {primitive}")]
    UnsupportedBondPrimitive {
        /// Human-readable primitive name.
        primitive: &'static str,
    },
}
