//! Error types for SMARTS matching and validation.

use core::fmt;

/// Errors raised by SMARTS validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmartsMatchError {
    /// The supplied target input was empty.
    EmptyTarget,
}

impl fmt::Display for SmartsMatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTarget => f.write_str("target input is empty"),
        }
    }
}

impl core::error::Error for SmartsMatchError {}
