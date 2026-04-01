use crate::span::Span;
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
    /// Atom maps such as `:1`.
    #[error("atom maps")]
    AtomMap,
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

/// Structured parse error carrying a machine-readable kind and optional span.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{kind}")]
pub struct SmartsParseError {
    kind: SmartsParseErrorKind,
    span: Option<Span>,
}

impl SmartsParseError {
    /// Creates an error without span information.
    #[inline]
    #[must_use]
    pub fn new(kind: SmartsParseErrorKind) -> Self {
        Self { kind, span: None }
    }

    /// Creates an error tied to a source span.
    #[inline]
    #[must_use]
    pub fn with_span(kind: SmartsParseErrorKind, span: Span) -> Self {
        Self {
            kind,
            span: Some(span),
        }
    }

    /// Returns the structured error kind.
    #[inline]
    #[must_use]
    pub fn kind(&self) -> SmartsParseErrorKind {
        self.kind
    }

    /// Returns the source span associated with the error, when available.
    #[inline]
    #[must_use]
    pub fn span(&self) -> Option<Span> {
        self.span
    }

    /// Returns a stable machine-readable error code.
    #[inline]
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self.kind {
            SmartsParseErrorKind::EmptyInput => "empty_input",
            SmartsParseErrorKind::UnexpectedEndOfInput => "unexpected_end_of_input",
            SmartsParseErrorKind::UnexpectedCharacter(_) => "unexpected_character",
            SmartsParseErrorKind::ConflictingRingClosureBond => "conflicting_ring_closure_bond",
            SmartsParseErrorKind::UnclosedRingClosure => "unclosed_ring_closure",
            SmartsParseErrorKind::UnterminatedBracketAtom => "unterminated_bracket_atom",
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::AtomMap) => {
                "unsupported_atom_map"
            }
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::AtomPrimitive) => {
                "unsupported_atom_primitive"
            }
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::Branch) => {
                "unsupported_branch"
            }
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::ExplicitBond) => {
                "unsupported_explicit_bond"
            }
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::Reaction) => {
                "unsupported_reaction"
            }
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::RecursiveSmarts) => {
                "unsupported_recursive_smarts"
            }
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::RingClosure) => {
                "unsupported_ring_closure"
            }
        }
    }
}
