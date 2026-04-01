use alloc::{boxed::Box, vec, vec::Vec};
use core::fmt;
use elements_rs::Isotope;

use crate::parse::parse_smarts;
use crate::query::{
    parse_supported_bracket_element, AtomPrimitive, BracketExpr, BracketExprTree, HydrogenKind,
};
use crate::{SmartsParseError, SmartsParseErrorKind, Span};

/// Parses the inside of one bracket atom into its boolean expression tree.
///
/// The input must not include the surrounding `[` and `]` characters.
///
/// # Errors
///
/// Returns [`BracketParseError`] when the bracket contents are empty or contain
/// malformed or unsupported syntax.
pub fn parse_bracket_text(text: &str) -> Result<BracketExpr, BracketParseError> {
    BracketParser::new(text).parse()
}

/// Structured parse error emitted while parsing bracket atom contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BracketParseError {
    kind: BracketParseErrorKind,
    span: Span,
}

/// High-level reasons why bracket atom parsing can fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BracketParseErrorKind {
    /// The bracket contents were empty.
    Empty,
    /// The parser encountered a character that is not valid in the current context.
    UnexpectedCharacter(char),
    /// The bracket contents ended before the parser could finish a construct.
    UnexpectedEnd,
    /// A nested recursive SMARTS payload could not be parsed.
    RecursiveSmarts(SmartsParseErrorKind),
    /// The current parser intentionally does not support the encountered primitive.
    UnsupportedPrimitive,
}

impl BracketParseError {
    fn new(kind: BracketParseErrorKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Returns the structured error kind.
    #[must_use]
    pub fn kind(&self) -> &BracketParseErrorKind {
        &self.kind
    }

    /// Returns the span relative to the bracket atom contents.
    #[must_use]
    pub fn span(&self) -> Span {
        self.span
    }
}

impl fmt::Display for BracketParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            BracketParseErrorKind::Empty => f.write_str("empty bracket atom"),
            BracketParseErrorKind::UnexpectedCharacter(ch) => {
                write!(f, "unexpected character `{ch}` in bracket atom")
            }
            BracketParseErrorKind::UnexpectedEnd => f.write_str("unexpected end of bracket atom"),
            BracketParseErrorKind::RecursiveSmarts(kind) => {
                write!(
                    f,
                    "invalid recursive SMARTS: {}",
                    SmartsParseError::new(kind)
                )
            }
            BracketParseErrorKind::UnsupportedPrimitive => {
                f.write_str("unsupported bracket atom primitive")
            }
        }
    }
}

impl core::error::Error for BracketParseError {}

struct BracketParser<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> BracketParser<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }

    fn parse(mut self) -> Result<BracketExpr, BracketParseError> {
        if self.text.is_empty() {
            return Err(self.error(BracketParseErrorKind::Empty));
        }

        let tree = self.parse_low_and(true)?;
        if !self.is_eof() {
            return Err(self.error_here(BracketParseErrorKind::UnexpectedCharacter(self.peek())));
        }

        Ok(BracketExpr { tree })
    }

    fn parse_low_and(
        &mut self,
        allow_hydrogen_symbol: bool,
    ) -> Result<BracketExprTree, BracketParseError> {
        let mut terms = vec![self.parse_or(allow_hydrogen_symbol)?];
        loop {
            if self.is_eof() || self.peek() != ';' {
                break;
            }
            self.pos += 1;
            terms.push(self.parse_or(false)?);
        }
        Ok(collapse_logic(terms, BracketExprTree::LowAnd))
    }

    fn parse_or(
        &mut self,
        allow_hydrogen_symbol: bool,
    ) -> Result<BracketExprTree, BracketParseError> {
        let mut terms = vec![self.parse_high_and(allow_hydrogen_symbol)?];
        loop {
            if self.is_eof() || self.peek() != ',' {
                break;
            }
            self.pos += 1;
            terms.push(self.parse_high_and(true)?);
        }
        Ok(collapse_logic(terms, BracketExprTree::Or))
    }

    fn parse_high_and(
        &mut self,
        allow_hydrogen_symbol: bool,
    ) -> Result<BracketExprTree, BracketParseError> {
        let mut terms = vec![self.parse_unary(allow_hydrogen_symbol)?];
        loop {
            if self.is_eof() {
                break;
            }
            if self.peek() == '&' {
                self.pos += 1;
                terms.push(self.parse_unary(false)?);
                continue;
            }
            if starts_implicit_and(self.peek()) {
                terms.push(self.parse_unary(false)?);
                continue;
            }
            break;
        }
        Ok(collapse_logic(terms, BracketExprTree::HighAnd))
    }

    fn parse_unary(
        &mut self,
        allow_hydrogen_symbol: bool,
    ) -> Result<BracketExprTree, BracketParseError> {
        if self.is_eof() {
            return Err(self.error(BracketParseErrorKind::UnexpectedEnd));
        }

        if self.peek() == '!' {
            self.pos += 1;
            return Ok(BracketExprTree::Not(Box::new(
                self.parse_unary(allow_hydrogen_symbol)?,
            )));
        }

        self.parse_primitive(allow_hydrogen_symbol)
    }

    fn parse_primitive(
        &mut self,
        allow_hydrogen_symbol: bool,
    ) -> Result<BracketExprTree, BracketParseError> {
        if self.is_eof() {
            return Err(self.error(BracketParseErrorKind::UnexpectedEnd));
        }

        let primitive = match self.peek() {
            '*' => {
                self.pos += 1;
                AtomPrimitive::Wildcard
            }
            '$' if self.remaining().starts_with("$(") => self.parse_recursive_smarts()?,
            '#' => {
                self.pos += 1;
                AtomPrimitive::AtomicNumber(self.parse_required_u16()?)
            }
            'D' => {
                self.pos += 1;
                AtomPrimitive::Degree(self.parse_optional_u8()?)
            }
            'X' => {
                self.pos += 1;
                AtomPrimitive::Connectivity(self.parse_optional_u8()?)
            }
            'v' => {
                self.pos += 1;
                AtomPrimitive::Valence(self.parse_optional_u8()?)
            }
            'A' => {
                self.pos += 1;
                AtomPrimitive::AliphaticAny
            }
            'a' => {
                if let Some((element, aromatic, width)) =
                    parse_supported_bracket_element(self.remaining())
                {
                    self.pos += width;
                    AtomPrimitive::Symbol { element, aromatic }
                } else {
                    self.pos += 1;
                    AtomPrimitive::AromaticAny
                }
            }
            'H' => {
                self.pos += 1;
                if allow_hydrogen_symbol && !self.next_char_is_ascii_digit() {
                    AtomPrimitive::Symbol {
                        element: elements_rs::Element::H,
                        aromatic: false,
                    }
                } else {
                    AtomPrimitive::Hydrogen(HydrogenKind::Total, self.parse_optional_u8()?)
                }
            }
            'h' => {
                self.pos += 1;
                AtomPrimitive::Hydrogen(HydrogenKind::Implicit, self.parse_optional_u8()?)
            }
            'R' => {
                self.pos += 1;
                AtomPrimitive::RingMembership(self.parse_optional_u8()?)
            }
            'r' => {
                self.pos += 1;
                AtomPrimitive::RingSize(self.parse_optional_u8()?)
            }
            'x' => {
                self.pos += 1;
                AtomPrimitive::RingConnectivity(self.parse_optional_u8()?)
            }
            '+' | '-' => self.parse_charge()?,
            _ if self.peek().is_ascii_alphabetic() => {
                if let Some((element, aromatic, width)) =
                    parse_supported_bracket_element(self.remaining())
                {
                    self.pos += width;
                    AtomPrimitive::Symbol { element, aromatic }
                } else {
                    return Err(self.error(BracketParseErrorKind::UnsupportedPrimitive));
                }
            }
            '1'..='9' => {
                let mass = self.parse_required_u16()?;
                let primitive = self.parse_symbol_after_isotope(mass)?;
                return Ok(BracketExprTree::Primitive(primitive));
            }
            other => return Err(self.error_here(BracketParseErrorKind::UnexpectedCharacter(other))),
        };

        Ok(BracketExprTree::Primitive(primitive))
    }

    fn parse_symbol_after_isotope(
        &mut self,
        mass: u16,
    ) -> Result<AtomPrimitive, BracketParseError> {
        if self.is_eof() {
            return Err(self.error(BracketParseErrorKind::UnexpectedEnd));
        }

        let Some((element, aromatic, width)) = parse_supported_bracket_element(self.remaining())
        else {
            return Err(self.error_here(BracketParseErrorKind::UnexpectedCharacter(self.peek())));
        };

        let isotope = Isotope::try_from((element, mass))
            .map_err(|_| self.error(BracketParseErrorKind::UnsupportedPrimitive))?;
        self.pos += width;

        Ok(AtomPrimitive::Isotope { isotope, aromatic })
    }

    fn parse_recursive_smarts(&mut self) -> Result<AtomPrimitive, BracketParseError> {
        self.pos += 2;
        let content_start = self.pos;
        let mut depth = 1usize;

        while !self.is_eof() {
            let ch = self.peek();
            if ch.is_ascii_whitespace() {
                return Err(self.error_here(BracketParseErrorKind::UnexpectedCharacter(ch)));
            }
            self.pos += ch.len_utf8();

            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        let content = &self.text[content_start..self.pos - 1];
                        if content.trim().is_empty() {
                            return Err(self.error(BracketParseErrorKind::UnexpectedEnd));
                        }
                        let nested = parse_smarts(content).map_err(|err| {
                            let relative_span = err.span().unwrap_or(Span::new(0, content.len()));
                            let span = Span::new(
                                content_start + relative_span.start,
                                content_start + relative_span.end,
                            );
                            BracketParseError::new(
                                BracketParseErrorKind::RecursiveSmarts(err.kind()),
                                span,
                            )
                        })?;
                        return Ok(AtomPrimitive::RecursiveQuery(Box::new(nested)));
                    }
                }
                _ => {}
            }
        }

        Err(self.error(BracketParseErrorKind::UnexpectedEnd))
    }

    fn parse_charge(&mut self) -> Result<AtomPrimitive, BracketParseError> {
        let sign = if self.peek() == '+' { 1 } else { -1 };
        self.pos += 1;
        let mut magnitude = 1_u16;
        let mut repeated = 0_u16;

        while !self.is_eof() && self.peek() == if sign > 0 { '+' } else { '-' } {
            repeated += 1;
            self.pos += 1;
        }

        if repeated > 0 {
            magnitude += repeated;
        } else if !self.is_eof() && self.peek().is_ascii_digit() {
            magnitude = self.parse_required_u16()?;
        }

        let magnitude = i8::try_from(magnitude)
            .map_err(|_| self.error(BracketParseErrorKind::UnsupportedPrimitive))?;

        Ok(AtomPrimitive::Charge(sign * magnitude))
    }

    fn parse_optional_u8(&mut self) -> Result<Option<u8>, BracketParseError> {
        if self.is_eof() || !self.peek().is_ascii_digit() {
            return Ok(None);
        }

        let value = self.parse_required_u16()?;
        Ok(Some(u8::try_from(value).map_err(|_| {
            self.error(BracketParseErrorKind::UnsupportedPrimitive)
        })?))
    }

    fn parse_required_u16(&mut self) -> Result<u16, BracketParseError> {
        let start = self.pos;
        while !self.is_eof() && self.peek().is_ascii_digit() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(self.error(BracketParseErrorKind::UnexpectedEnd));
        }
        self.text[start..self.pos]
            .parse::<u16>()
            .map_err(|_| self.error(BracketParseErrorKind::UnsupportedPrimitive))
    }

    fn remaining(&self) -> &str {
        &self.text[self.pos..]
    }

    fn next_char_is_ascii_digit(&self) -> bool {
        self.text
            .as_bytes()
            .get(self.pos)
            .is_some_and(u8::is_ascii_digit)
    }

    fn peek(&self) -> char {
        self.text[self.pos..].chars().next().expect("peek past eof")
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.text.len()
    }

    fn error(&self, kind: BracketParseErrorKind) -> BracketParseError {
        BracketParseError::new(kind, Span::new(self.pos, self.pos))
    }

    fn error_here(&self, kind: BracketParseErrorKind) -> BracketParseError {
        let width = if self.is_eof() {
            0
        } else {
            self.peek().len_utf8()
        };
        BracketParseError::new(kind, Span::new(self.pos, self.pos + width))
    }
}

fn collapse_logic(
    mut terms: Vec<BracketExprTree>,
    constructor: fn(Vec<BracketExprTree>) -> BracketExprTree,
) -> BracketExprTree {
    if terms.len() == 1 {
        terms.pop().expect("single term")
    } else {
        constructor(terms)
    }
}

fn starts_implicit_and(ch: char) -> bool {
    matches!(
        ch,
        '!' | '*'
            | '#'
            | '$'
            | 'A'
            | 'D'
            | 'H'
            | 'R'
            | 'X'
            | 'a'
            | 'b'
            | 'c'
            | 'h'
            | 'n'
            | 'o'
            | 'p'
            | 'r'
            | 's'
            | 'v'
            | 'x'
            | 'B'
            | 'C'
            | 'F'
            | 'I'
            | 'N'
            | 'O'
            | 'P'
            | 'S'
            | '+'
            | '-'
            | '1'..='9'
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::ToString, vec};
    use elements_rs::{Element, Isotope};

    #[test]
    fn parses_isotope_symbol() {
        let expr = parse_bracket_text("12C").unwrap();
        assert_eq!(
            expr.tree,
            BracketExprTree::Primitive(AtomPrimitive::Isotope {
                isotope: Isotope::try_from((Element::C, 12_u16)).unwrap(),
                aromatic: false,
            })
        );
    }

    #[test]
    fn parses_full_element_symbols_in_brackets() {
        let sodium = parse_bracket_text("Na").unwrap();
        let platinum = parse_bracket_text("Pt").unwrap();
        let oganesson = parse_bracket_text("Og").unwrap();

        assert_eq!(
            sodium.tree,
            BracketExprTree::Primitive(AtomPrimitive::Symbol {
                element: Element::Na,
                aromatic: false,
            })
        );
        assert_eq!(
            platinum.tree,
            BracketExprTree::Primitive(AtomPrimitive::Symbol {
                element: Element::Pt,
                aromatic: false,
            })
        );
        assert_eq!(
            oganesson.tree,
            BracketExprTree::Primitive(AtomPrimitive::Symbol {
                element: Element::Og,
                aromatic: false,
            })
        );
    }

    #[test]
    fn parses_atomic_hydrogen_and_isotopes() {
        let hydrogen = parse_bracket_text("H").unwrap();
        let deuterium = parse_bracket_text("2H").unwrap();

        assert_eq!(
            hydrogen.tree,
            BracketExprTree::Primitive(AtomPrimitive::Symbol {
                element: Element::H,
                aromatic: false,
            })
        );
        assert_eq!(
            deuterium.tree,
            BracketExprTree::Primitive(AtomPrimitive::Isotope {
                isotope: Isotope::try_from((Element::H, 2_u16)).unwrap(),
                aromatic: false,
            })
        );
    }

    #[test]
    fn preserves_hydrogen_count_and_atom_disambiguation() {
        let carbon_with_hydrogen = parse_bracket_text("CH").unwrap();
        let carbon_or_hydrogen = parse_bracket_text("C,H").unwrap();

        assert_eq!(
            carbon_with_hydrogen.tree,
            BracketExprTree::HighAnd(vec![
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: false,
                }),
                BracketExprTree::Primitive(AtomPrimitive::Hydrogen(HydrogenKind::Total, None)),
            ])
        );
        assert_eq!(
            carbon_or_hydrogen.tree,
            BracketExprTree::Or(vec![
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: false,
                }),
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::H,
                    aromatic: false,
                }),
            ])
        );
    }

    #[test]
    fn parses_degree_connectivity_and_valence() {
        let degree = parse_bracket_text("D2").unwrap();
        let connectivity = parse_bracket_text("X4").unwrap();
        let valence = parse_bracket_text("v3").unwrap();
        let ring_connectivity = parse_bracket_text("x2").unwrap();

        assert_eq!(
            degree.tree,
            BracketExprTree::Primitive(AtomPrimitive::Degree(Some(2)))
        );
        assert_eq!(
            connectivity.tree,
            BracketExprTree::Primitive(AtomPrimitive::Connectivity(Some(4)))
        );
        assert_eq!(
            valence.tree,
            BracketExprTree::Primitive(AtomPrimitive::Valence(Some(3)))
        );
        assert_eq!(
            ring_connectivity.tree,
            BracketExprTree::Primitive(AtomPrimitive::RingConnectivity(Some(2)))
        );
    }

    #[test]
    fn parses_precedence() {
        let expr = parse_bracket_text("c,n&H1").unwrap();
        assert_eq!(
            expr.tree,
            BracketExprTree::Or(vec![
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: true,
                }),
                BracketExprTree::HighAnd(vec![
                    BracketExprTree::Primitive(AtomPrimitive::Symbol {
                        element: Element::N,
                        aromatic: true,
                    }),
                    BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                        HydrogenKind::Total,
                        Some(1)
                    )),
                ]),
            ])
        );
    }

    #[test]
    fn parses_negation_and_charge() {
        let neg = parse_bracket_text("!#1").unwrap();
        let charge = parse_bracket_text("++").unwrap();

        assert_eq!(
            neg.tree,
            BracketExprTree::Not(Box::new(BracketExprTree::Primitive(
                AtomPrimitive::AtomicNumber(1),
            )))
        );
        assert_eq!(
            charge.tree,
            BracketExprTree::Primitive(AtomPrimitive::Charge(2))
        );
    }

    #[test]
    fn parses_recursive_smarts_primitive() {
        let expr = parse_bracket_text("$(CO)").unwrap();
        let BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) = &expr.tree else {
            panic!("expected nested recursive query, got {:?}", expr.tree);
        };
        assert_eq!(nested.to_string(), "CO");
        assert_eq!(nested.atom_count(), 2);
        assert_eq!(nested.bond_count(), 1);
    }

    #[test]
    fn preserves_nested_parentheses_in_recursive_smarts() {
        let expr = parse_bracket_text("$(C(=O)O)").unwrap();
        let BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) = &expr.tree else {
            panic!("expected nested recursive query, got {:?}", expr.tree);
        };
        assert_eq!(nested.to_string(), "C(=O)O");
        assert_eq!(nested.atom_count(), 3);
        assert_eq!(nested.bond_count(), 2);
    }

    #[test]
    fn rejects_empty_recursive_smarts() {
        let err = parse_bracket_text("$()").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnexpectedEnd);

        let err = parse_bracket_text("$(   )").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnexpectedCharacter(' '));
    }

    #[test]
    fn rejects_whitespace_inside_recursive_smarts() {
        let err = parse_bracket_text("$(C O)").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnexpectedCharacter(' '));

        let err = parse_bracket_text("$(C\tO)").unwrap_err();
        assert_eq!(
            err.kind(),
            &BracketParseErrorKind::UnexpectedCharacter('\t')
        );

        let err = parse_bracket_text("$(C\nO)").unwrap_err();
        assert_eq!(
            err.kind(),
            &BracketParseErrorKind::UnexpectedCharacter('\n')
        );
    }

    #[test]
    fn rejects_whitespace_inside_brackets() {
        let err = parse_bracket_text("C ; H1").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnexpectedCharacter(' '));

        let err = parse_bracket_text("C\t;H1").unwrap_err();
        assert_eq!(
            err.kind(),
            &BracketParseErrorKind::UnexpectedCharacter('\t')
        );

        let err = parse_bracket_text(" $(CO)").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnexpectedCharacter(' '));

        let err = parse_bracket_text("$(CO) ").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnexpectedCharacter(' '));
    }

    #[test]
    fn rejects_overflowing_charge_runs() {
        let input = format!("S{}", "-".repeat(200));
        let err = parse_bracket_text(&input).unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnsupportedPrimitive);
    }
}
