use thiserror::Error;

/// High-level reasons why SMARTS parsing can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SmartsParseErrorKind {
    /// The input string was empty after trimming the accepted outer whitespace.
    #[error("SMARTS input is empty")]
    EmptyInput,
    /// The parser reached the end of the input before completing the pattern.
    #[error("unexpected end of SMARTS input")]
    UnexpectedEndOfInput,
    /// The parser encountered a character that is not valid in the current context.
    #[error("unexpected character `{0}`")]
    UnexpectedCharacter(char),
    /// Two ends of the same ring closure specified incompatible bond operators.
    #[error("conflicting bond specifications for ring closure")]
    ConflictingRingClosureBond,
    /// A ring closure label was opened but never closed.
    #[error("ring closure was opened but not closed")]
    UnclosedRingClosure,
    /// A bracket atom started with `[` but was not terminated by `]`.
    #[error("unterminated bracket atom")]
    UnterminatedBracketAtom,
    /// The pattern uses syntax that is deliberately out of scope for the current parser.
    #[error("unsupported SMARTS feature: {0}")]
    UnsupportedFeature(UnsupportedFeature),
}

/// SMARTS syntax families that are intentionally deferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum UnsupportedFeature {
    /// Unsupported bracket primitives beyond the current subset.
    #[error("atom primitive")]
    AtomPrimitive,
    /// Branch syntax that is not yet handled by the current parser mode.
    #[error("branches")]
    Branch,
    /// Explicit bond syntax that is outside the supported subset.
    #[error("explicit bond operators")]
    ExplicitBond,
    /// Reaction SMARTS using `>`.
    #[error("reaction SMARTS")]
    Reaction,
    /// Recursive SMARTS using `$()` when not supported in the current mode.
    #[error("recursive SMARTS")]
    RecursiveSmarts,
    /// Ring closure syntax that is outside the supported subset.
    #[error("ring closures")]
    RingClosure,
}

/// Structured parse error carrying a machine-readable kind.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{kind}")]
pub struct SmartsParseError {
    kind: SmartsParseErrorKind,
}

impl UnsupportedFeature {
    /// Returns a stable machine-readable error code for one unsupported feature.
    #[inline]
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::AtomPrimitive => "unsupported_atom_primitive",
            Self::Branch => "unsupported_branch",
            Self::ExplicitBond => "unsupported_explicit_bond",
            Self::Reaction => "unsupported_reaction",
            Self::RecursiveSmarts => "unsupported_recursive_smarts",
            Self::RingClosure => "unsupported_ring_closure",
        }
    }
}

impl SmartsParseError {
    /// Creates an error.
    #[inline]
    #[must_use]
    pub const fn new(kind: SmartsParseErrorKind) -> Self {
        Self { kind }
    }

    /// Returns the structured error kind.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> SmartsParseErrorKind {
        self.kind
    }

    /// Returns a stable machine-readable error code.
    #[inline]
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self.kind {
            SmartsParseErrorKind::EmptyInput => "empty_input",
            SmartsParseErrorKind::UnexpectedEndOfInput => "unexpected_end_of_input",
            SmartsParseErrorKind::UnexpectedCharacter(_) => "unexpected_character",
            SmartsParseErrorKind::ConflictingRingClosureBond => "conflicting_ring_closure_bond",
            SmartsParseErrorKind::UnclosedRingClosure => "unclosed_ring_closure",
            SmartsParseErrorKind::UnterminatedBracketAtom => "unterminated_bracket_atom",
            SmartsParseErrorKind::UnsupportedFeature(feature) => feature.code(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn displays_error_kinds_and_codes() {
        let cases = [
            (
                SmartsParseErrorKind::EmptyInput,
                "SMARTS input is empty",
                "empty_input",
            ),
            (
                SmartsParseErrorKind::UnexpectedEndOfInput,
                "unexpected end of SMARTS input",
                "unexpected_end_of_input",
            ),
            (
                SmartsParseErrorKind::UnexpectedCharacter('x'),
                "unexpected character `x`",
                "unexpected_character",
            ),
            (
                SmartsParseErrorKind::ConflictingRingClosureBond,
                "conflicting bond specifications for ring closure",
                "conflicting_ring_closure_bond",
            ),
            (
                SmartsParseErrorKind::UnclosedRingClosure,
                "ring closure was opened but not closed",
                "unclosed_ring_closure",
            ),
            (
                SmartsParseErrorKind::UnterminatedBracketAtom,
                "unterminated bracket atom",
                "unterminated_bracket_atom",
            ),
            (
                SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::AtomPrimitive),
                "unsupported SMARTS feature: atom primitive",
                "unsupported_atom_primitive",
            ),
            (
                SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::Branch),
                "unsupported SMARTS feature: branches",
                "unsupported_branch",
            ),
            (
                SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::ExplicitBond),
                "unsupported SMARTS feature: explicit bond operators",
                "unsupported_explicit_bond",
            ),
            (
                SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::Reaction),
                "unsupported SMARTS feature: reaction SMARTS",
                "unsupported_reaction",
            ),
            (
                SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::RecursiveSmarts),
                "unsupported SMARTS feature: recursive SMARTS",
                "unsupported_recursive_smarts",
            ),
            (
                SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::RingClosure),
                "unsupported SMARTS feature: ring closures",
                "unsupported_ring_closure",
            ),
        ];

        for (kind, display, code) in cases {
            let error = SmartsParseError::new(kind);
            assert_eq!(kind.to_string(), display);
            assert_eq!(error.to_string(), display);
            assert_eq!(error.code(), code);
            assert_eq!(error.kind(), kind);
        }
    }

    #[test]
    fn constructs_error_without_span_metadata() {
        let error = SmartsParseError::new(SmartsParseErrorKind::UnexpectedCharacter(']'));
        assert_eq!(error.kind(), SmartsParseErrorKind::UnexpectedCharacter(']'));
        assert_eq!(error.code(), "unexpected_character");
    }

    #[test]
    fn unsupported_feature_codes_are_individually_covered() {
        assert_eq!(
            SmartsParseError::new(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::AtomPrimitive,
            ))
            .code(),
            "unsupported_atom_primitive"
        );
        assert_eq!(
            SmartsParseError::new(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::Branch,
            ))
            .code(),
            "unsupported_branch"
        );
        assert_eq!(
            SmartsParseError::new(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::ExplicitBond,
            ))
            .code(),
            "unsupported_explicit_bond"
        );
        assert_eq!(
            SmartsParseError::new(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::Reaction,
            ))
            .code(),
            "unsupported_reaction"
        );
        assert_eq!(
            SmartsParseError::new(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::RecursiveSmarts,
            ))
            .code(),
            "unsupported_recursive_smarts"
        );
        assert_eq!(
            SmartsParseError::new(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::RingClosure,
            ))
            .code(),
            "unsupported_ring_closure"
        );
    }
}
