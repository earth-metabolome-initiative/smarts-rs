use crate::span::Span;
use core::fmt;

/// High-level reasons why SMARTS parsing can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartsParseErrorKind {
    /// The input string was empty after trimming the accepted outer whitespace.
    EmptyInput,
    /// The parser reached the end of the input before completing the pattern.
    UnexpectedEndOfInput,
    /// The parser encountered a character that is not valid in the current context.
    UnexpectedCharacter(char),
    /// Two ends of the same ring closure specified incompatible bond operators.
    ConflictingRingClosureBond,
    /// A ring closure label was opened but never closed.
    UnclosedRingClosure,
    /// A bracket atom started with `[` but was not terminated by `]`.
    UnterminatedBracketAtom,
    /// The pattern uses syntax that is deliberately out of scope for the current parser.
    UnsupportedFeature(UnsupportedFeature),
}

/// SMARTS syntax families that are intentionally deferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedFeature {
    /// Atom maps such as `:1`.
    AtomMap,
    /// Unsupported bracket primitives beyond the current subset.
    AtomPrimitive,
    /// Branch syntax that is not yet handled by the current parser mode.
    Branch,
    /// Explicit bond syntax that is outside the supported subset.
    ExplicitBond,
    /// Reaction SMARTS using `>`.
    Reaction,
    /// Recursive SMARTS using `$()` when not supported in the current mode.
    RecursiveSmarts,
    /// Ring closure syntax that is outside the supported subset.
    RingClosure,
}

/// Structured parse error carrying a machine-readable kind and optional span.
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl fmt::Display for SmartsParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            SmartsParseErrorKind::EmptyInput => f.write_str("SMARTS input is empty"),
            SmartsParseErrorKind::UnexpectedEndOfInput => {
                f.write_str("unexpected end of SMARTS input")
            }
            SmartsParseErrorKind::UnexpectedCharacter(ch) => {
                write!(f, "unexpected character `{ch}`")
            }
            SmartsParseErrorKind::ConflictingRingClosureBond => {
                f.write_str("conflicting bond specifications for ring closure")
            }
            SmartsParseErrorKind::UnclosedRingClosure => {
                f.write_str("ring closure was opened but not closed")
            }
            SmartsParseErrorKind::UnterminatedBracketAtom => {
                f.write_str("unterminated bracket atom")
            }
            SmartsParseErrorKind::UnsupportedFeature(feature) => {
                write!(f, "unsupported SMARTS feature: {feature}")
            }
        }
    }
}

impl core::error::Error for SmartsParseError {}

impl fmt::Display for UnsupportedFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtomMap => f.write_str("atom maps"),
            Self::AtomPrimitive => f.write_str("atom primitive"),
            Self::Branch => f.write_str("branches"),
            Self::ExplicitBond => f.write_str("explicit bond operators"),
            Self::Reaction => f.write_str("reaction SMARTS"),
            Self::RecursiveSmarts => f.write_str("recursive SMARTS"),
            Self::RingClosure => f.write_str("ring closures"),
        }
    }
}
