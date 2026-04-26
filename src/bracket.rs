use alloc::{boxed::Box, vec, vec::Vec};
use elements_rs::Isotope;
use smiles_parser::atom::bracketed::chirality::Chirality;
use thiserror::Error;

use crate::parse::parse_smarts;
use crate::query::{
    parse_supported_bracket_element, AtomPrimitive, BracketExpr, BracketExprTree, HydrogenKind,
    NumericQuery, NumericRange,
};
use crate::SmartsParseErrorKind;

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
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{kind}")]
pub struct BracketParseError {
    kind: BracketParseErrorKind,
}

/// High-level reasons why bracket atom parsing can fail.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BracketParseErrorKind {
    /// The bracket contents were empty.
    #[error("empty bracket atom")]
    Empty,
    /// The parser encountered a character that is not valid in the current context.
    #[error("unexpected character `{0}` in bracket atom")]
    UnexpectedCharacter(char),
    /// The bracket contents ended before the parser could finish a construct.
    #[error("unexpected end of bracket atom")]
    UnexpectedEnd,
    /// A nested recursive SMARTS payload could not be parsed.
    #[error("invalid recursive SMARTS: {0}")]
    RecursiveSmarts(SmartsParseErrorKind),
    /// The current parser intentionally does not support the encountered primitive.
    #[error("unsupported bracket atom primitive")]
    UnsupportedPrimitive,
}

impl BracketParseError {
    const fn new(kind: BracketParseErrorKind) -> Self {
        Self { kind }
    }

    /// Returns the structured error kind.
    #[must_use]
    pub const fn kind(&self) -> &BracketParseErrorKind {
        &self.kind
    }
}

struct BracketParser<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> BracketParser<'a> {
    const fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }

    fn parse(mut self) -> Result<BracketExpr, BracketParseError> {
        if self.text.is_empty() {
            return Err(self.error(BracketParseErrorKind::Empty));
        }

        let tree = self.parse_low_and(true)?;
        let atom_map = if !self.is_eof() && self.peek() == ':' {
            self.pos += 1;
            Some(u32::from(self.parse_required_u16()?))
        } else {
            None
        };
        if !self.is_eof() {
            return Err(self.error_here(BracketParseErrorKind::UnexpectedCharacter(self.peek())));
        }

        Ok(BracketExpr { tree, atom_map })
    }

    fn parse_low_and(
        &mut self,
        allow_hydrogen_symbol: bool,
    ) -> Result<BracketExprTree, BracketParseError> {
        let mut terms = vec![self.parse_or(allow_hydrogen_symbol)?];
        while !self.is_eof() && self.peek() == ';' {
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
        while !self.is_eof() && self.peek() == ',' {
            self.pos += 1;
            terms.push(self.parse_high_and(false)?);
        }
        Ok(collapse_logic(terms, BracketExprTree::Or))
    }

    fn parse_high_and(
        &mut self,
        allow_hydrogen_symbol: bool,
    ) -> Result<BracketExprTree, BracketParseError> {
        let mut terms = vec![self.parse_unary(allow_hydrogen_symbol)?];
        while !self.is_eof() {
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
        let mut negated = false;
        let mut saw_negation = false;
        while !self.is_eof() && self.peek() == '!' {
            self.pos += 1;
            negated = !negated;
            saw_negation = true;
        }

        if self.is_eof() {
            return Err(self.error(BracketParseErrorKind::UnexpectedEnd));
        }

        let tree = self.parse_primitive(allow_hydrogen_symbol && !saw_negation)?;
        if negated {
            Ok(BracketExprTree::Not(Box::new(tree)))
        } else {
            Ok(tree)
        }
    }

    #[allow(clippy::too_many_lines)]
    fn parse_primitive(
        &mut self,
        allow_hydrogen_symbol: bool,
    ) -> Result<BracketExprTree, BracketParseError> {
        if self.is_eof() {
            return Err(self.error(BracketParseErrorKind::UnexpectedEnd));
        }

        if let Some((element, aromatic, width)) = parse_supported_bracket_element(self.remaining())
        {
            if width > 1 {
                self.pos += width;
                return Ok(BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element,
                    aromatic,
                }));
            }
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
                AtomPrimitive::Degree(self.parse_optional_numeric_query()?)
            }
            'X' => {
                self.pos += 1;
                AtomPrimitive::Connectivity(self.parse_optional_numeric_query()?)
            }
            'v' => {
                self.pos += 1;
                AtomPrimitive::Valence(self.parse_optional_numeric_query()?)
            }
            'A' => {
                if let Some((element, aromatic, width)) =
                    parse_supported_bracket_element(self.remaining())
                {
                    if width > 1 {
                        self.pos += width;
                        AtomPrimitive::Symbol { element, aromatic }
                    } else {
                        self.pos += 1;
                        AtomPrimitive::AliphaticAny
                    }
                } else {
                    self.pos += 1;
                    AtomPrimitive::AliphaticAny
                }
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
                if let Some((element, aromatic, width)) =
                    parse_supported_bracket_element(self.remaining())
                {
                    if width > 1 {
                        self.pos += width;
                        AtomPrimitive::Symbol { element, aromatic }
                    } else {
                        self.parse_single_h_primitive(allow_hydrogen_symbol)?
                    }
                } else {
                    self.parse_single_h_primitive(allow_hydrogen_symbol)?
                }
            }
            'h' => {
                self.pos += 1;
                AtomPrimitive::Hydrogen(
                    HydrogenKind::Implicit,
                    self.parse_optional_numeric_query()?,
                )
            }
            'R' => {
                if let Some((element, aromatic, width)) =
                    parse_supported_bracket_element(self.remaining())
                {
                    if width > 1 {
                        self.pos += width;
                        AtomPrimitive::Symbol { element, aromatic }
                    } else {
                        self.pos += 1;
                        AtomPrimitive::RingMembership(self.parse_optional_numeric_query()?)
                    }
                } else {
                    self.pos += 1;
                    AtomPrimitive::RingMembership(self.parse_optional_numeric_query()?)
                }
            }
            'r' => {
                self.pos += 1;
                AtomPrimitive::RingSize(self.parse_optional_numeric_query()?)
            }
            'x' => {
                self.pos += 1;
                AtomPrimitive::RingConnectivity(self.parse_optional_numeric_query()?)
            }
            '^' => {
                self.pos += 1;
                AtomPrimitive::Hybridization(self.parse_required_numeric_query()?)
            }
            'z' => {
                self.pos += 1;
                AtomPrimitive::HeteroNeighbor(self.parse_optional_numeric_query()?)
            }
            'Z' => {
                self.pos += 1;
                AtomPrimitive::AliphaticHeteroNeighbor(self.parse_optional_numeric_query()?)
            }
            '@' => AtomPrimitive::Chirality(self.parse_chirality()?),
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
            '0'..='9' => {
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

        if self.peek() == '*' {
            self.pos += 1;
            return Ok(AtomPrimitive::IsotopeWildcard(mass));
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
                            BracketParseError::new(BracketParseErrorKind::RecursiveSmarts(
                                err.kind(),
                            ))
                        })?;
                        return Ok(AtomPrimitive::RecursiveQuery(Box::new(nested)));
                    }
                }
                _ => {}
            }
        }

        Err(self.error(BracketParseErrorKind::UnexpectedEnd))
    }

    fn parse_chirality(&mut self) -> Result<Chirality, BracketParseError> {
        debug_assert_eq!(self.peek(), '@');
        self.pos += 1;

        if !self.is_eof() && self.peek() == '@' {
            self.pos += 1;
            return Ok(Chirality::AtAt);
        }

        if self.remaining().starts_with("TH") {
            self.pos += 2;
            return self.parse_chiral_permutation(Chirality::try_th);
        } else if self.remaining().starts_with("AL") {
            self.pos += 2;
            return self.parse_chiral_permutation(Chirality::try_al);
        } else if self.remaining().starts_with("SP") {
            self.pos += 2;
            return self.parse_chiral_permutation(Chirality::try_sp);
        } else if self.remaining().starts_with("TB") {
            self.pos += 2;
            return self.parse_chiral_permutation(Chirality::try_tb);
        } else if self.remaining().starts_with("OH") {
            self.pos += 2;
            return self.parse_chiral_permutation(Chirality::try_oh);
        }

        Ok(Chirality::At)
    }

    fn parse_chiral_permutation(
        &mut self,
        constructor: fn(u8) -> Result<Chirality, smiles_parser::SmilesError>,
    ) -> Result<Chirality, BracketParseError> {
        let value = self.parse_optional_u8()?.unwrap_or(1);
        constructor(value).map_err(|_| self.error(BracketParseErrorKind::UnsupportedPrimitive))
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

    fn parse_optional_numeric_query(&mut self) -> Result<Option<NumericQuery>, BracketParseError> {
        if self.is_eof() {
            return Ok(None);
        }

        if self.peek().is_ascii_digit() {
            return Ok(Some(NumericQuery::Exact(self.parse_required_u16()?)));
        }

        if self.peek() != '{' {
            return Ok(None);
        }

        self.pos += 1;
        let min = if !self.is_eof() && self.peek().is_ascii_digit() {
            Some(self.parse_required_u16()?)
        } else {
            None
        };

        let max = if !self.is_eof() && self.peek() == '}' {
            self.pos += 1;
            min
        } else {
            if self.is_eof() || self.peek() != '-' {
                return Err(self.error(BracketParseErrorKind::UnsupportedPrimitive));
            }
            self.pos += 1;
            let max = if !self.is_eof() && self.peek().is_ascii_digit() {
                Some(self.parse_required_u16()?)
            } else {
                None
            };
            if self.is_eof() || self.peek() != '}' {
                return Err(self.error(BracketParseErrorKind::UnsupportedPrimitive));
            }
            self.pos += 1;
            max
        };

        if min.is_none() && max.is_none() {
            return Err(self.error(BracketParseErrorKind::UnsupportedPrimitive));
        }
        if let (Some(min), Some(max)) = (min, max) {
            if min > max {
                return Err(self.error(BracketParseErrorKind::UnsupportedPrimitive));
            }
            if min == max {
                return Ok(Some(NumericQuery::Exact(min)));
            }
        }

        Ok(Some(NumericQuery::Range(NumericRange { min, max })))
    }

    fn parse_required_numeric_query(&mut self) -> Result<NumericQuery, BracketParseError> {
        self.parse_optional_numeric_query()?
            .ok_or_else(|| self.error(BracketParseErrorKind::UnexpectedEnd))
    }

    fn parse_single_h_primitive(
        &mut self,
        allow_hydrogen_symbol: bool,
    ) -> Result<AtomPrimitive, BracketParseError> {
        self.pos += 1;
        if allow_hydrogen_symbol && self.h_is_atomic_hydrogen_context() {
            return Ok(AtomPrimitive::Symbol {
                element: elements_rs::Element::H,
                aromatic: false,
            });
        }
        Ok(AtomPrimitive::Hydrogen(
            HydrogenKind::Total,
            self.parse_optional_numeric_query()?,
        ))
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

    fn h_is_atomic_hydrogen_context(&self) -> bool {
        if self.next_char_is_ascii_digit() {
            return false;
        }
        let Some(next) = self.text.as_bytes().get(self.pos).copied() else {
            return true;
        };
        if next != b'+' && next != b'-' {
            return false;
        }

        let mut index = self.pos + 1;
        while let Some(ch) = self.text.as_bytes().get(index).copied() {
            if ch == next {
                index += 1;
                continue;
            }
            if ch.is_ascii_digit() {
                index += 1;
                continue;
            }
            return false;
        }
        true
    }

    fn peek(&self) -> char {
        self.text[self.pos..].chars().next().expect("peek past eof")
    }

    const fn is_eof(&self) -> bool {
        self.pos >= self.text.len()
    }

    #[allow(clippy::unused_self)]
    const fn error(&self, kind: BracketParseErrorKind) -> BracketParseError {
        BracketParseError::new(kind)
    }

    #[allow(clippy::unused_self)]
    const fn error_here(&self, kind: BracketParseErrorKind) -> BracketParseError {
        BracketParseError::new(kind)
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

const fn starts_implicit_and(ch: char) -> bool {
    matches!(
        ch,
        '!' | '*'
            | '#'
            | '$'
            | '@'
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
            | '^'
            | 'z'
            | 'B'
            | 'C'
            | 'F'
            | 'I'
            | 'N'
            | 'O'
            | 'P'
            | 'S'
            | 'Z'
            | '+'
            | '-'
            | '0'..='9'
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
    fn parses_isotope_wildcard() {
        let expr = parse_bracket_text("89*").unwrap();
        let zero = parse_bracket_text("0*").unwrap();
        assert_eq!(
            expr.tree,
            BracketExprTree::Primitive(AtomPrimitive::IsotopeWildcard(89))
        );
        assert_eq!(
            zero.tree,
            BracketExprTree::Primitive(AtomPrimitive::IsotopeWildcard(0))
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
        let hydrogen_or_chlorine = parse_bracket_text("H,Cl").unwrap();
        let double_hydrogen = parse_bracket_text("HH").unwrap();
        let not_hydrogen = parse_bracket_text("!H").unwrap();

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
                BracketExprTree::Primitive(AtomPrimitive::Hydrogen(HydrogenKind::Total, None)),
            ])
        );
        assert_eq!(
            hydrogen_or_chlorine.tree,
            BracketExprTree::Or(vec![
                BracketExprTree::Primitive(AtomPrimitive::Hydrogen(HydrogenKind::Total, None)),
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::Cl,
                    aromatic: false,
                }),
            ])
        );
        assert_eq!(
            double_hydrogen.tree,
            BracketExprTree::HighAnd(vec![
                BracketExprTree::Primitive(AtomPrimitive::Hydrogen(HydrogenKind::Total, None)),
                BracketExprTree::Primitive(AtomPrimitive::Hydrogen(HydrogenKind::Total, None)),
            ])
        );
        assert_eq!(
            not_hydrogen.tree,
            BracketExprTree::Not(Box::new(BracketExprTree::Primitive(
                AtomPrimitive::Hydrogen(HydrogenKind::Total, None),
            )))
        );
    }

    #[test]
    fn parses_degree_connectivity_valence_and_ring_connectivity() {
        let degree = parse_bracket_text("D2").unwrap();
        let connectivity = parse_bracket_text("X4").unwrap();
        let valence = parse_bracket_text("v3").unwrap();
        let ring_connectivity = parse_bracket_text("x2").unwrap();

        assert_eq!(
            degree.tree,
            BracketExprTree::Primitive(AtomPrimitive::Degree(Some(NumericQuery::Exact(2))))
        );
        assert_eq!(
            connectivity.tree,
            BracketExprTree::Primitive(AtomPrimitive::Connectivity(Some(NumericQuery::Exact(4))))
        );
        assert_eq!(
            valence.tree,
            BracketExprTree::Primitive(AtomPrimitive::Valence(Some(NumericQuery::Exact(3))))
        );
        assert_eq!(
            ring_connectivity.tree,
            BracketExprTree::Primitive(AtomPrimitive::RingConnectivity(Some(NumericQuery::Exact(
                2
            ),)))
        );
    }

    #[test]
    fn parses_numeric_ranges() {
        let hydrogen = parse_bracket_text("h{1-}").unwrap();
        let degree = parse_bracket_text("D{2-3}").unwrap();
        let ring_membership = parse_bracket_text("R{1-}").unwrap();
        let ring_connectivity = parse_bracket_text("x{2-}").unwrap();

        assert_eq!(
            hydrogen.tree,
            BracketExprTree::Primitive(AtomPrimitive::Hydrogen(
                HydrogenKind::Implicit,
                Some(NumericQuery::Range(NumericRange {
                    min: Some(1),
                    max: None,
                })),
            ))
        );
        assert_eq!(
            degree.tree,
            BracketExprTree::Primitive(AtomPrimitive::Degree(Some(NumericQuery::Range(
                NumericRange {
                    min: Some(2),
                    max: Some(3),
                },
            ))))
        );
        assert_eq!(
            ring_membership.tree,
            BracketExprTree::Primitive(AtomPrimitive::RingMembership(Some(NumericQuery::Range(
                NumericRange {
                    min: Some(1),
                    max: None,
                }
            ),)))
        );
        assert_eq!(
            ring_connectivity.tree,
            BracketExprTree::Primitive(AtomPrimitive::RingConnectivity(Some(NumericQuery::Range(
                NumericRange {
                    min: Some(2),
                    max: None,
                }
            ),)))
        );
    }

    #[test]
    fn parses_atom_stereo_primitives() {
        let clockwise = parse_bracket_text("C@H").unwrap();
        let counter = parse_bracket_text("C@@H").unwrap();
        let tetrahedral = parse_bracket_text("C@TH1").unwrap();
        let square_planar = parse_bracket_text("C@SP").unwrap();
        let trigonal_bipyramidal = parse_bracket_text("C@TB10").unwrap();
        let octahedral = parse_bracket_text("C@OH30").unwrap();

        assert_eq!(
            clockwise.tree,
            BracketExprTree::HighAnd(vec![
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: false,
                }),
                BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::At)),
                BracketExprTree::Primitive(AtomPrimitive::Hydrogen(HydrogenKind::Total, None)),
            ])
        );
        assert_eq!(
            counter.tree,
            BracketExprTree::HighAnd(vec![
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: false,
                }),
                BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::AtAt)),
                BracketExprTree::Primitive(AtomPrimitive::Hydrogen(HydrogenKind::Total, None)),
            ])
        );
        assert_eq!(
            tetrahedral.tree,
            BracketExprTree::HighAnd(vec![
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: false,
                }),
                BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::TH(1))),
            ])
        );
        assert_eq!(
            square_planar.tree,
            BracketExprTree::HighAnd(vec![
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: false,
                }),
                BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::SP(1))),
            ])
        );
        assert_eq!(
            trigonal_bipyramidal.tree,
            BracketExprTree::HighAnd(vec![
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: false,
                }),
                BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::TB(10))),
            ])
        );
        assert_eq!(
            octahedral.tree,
            BracketExprTree::HighAnd(vec![
                BracketExprTree::Primitive(AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: false,
                }),
                BracketExprTree::Primitive(AtomPrimitive::Chirality(Chirality::OH(30))),
            ])
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
                        Some(NumericQuery::Exact(1)),
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
    fn parses_deep_negation_chains_without_recursing() {
        let even = format!("{}#6", "!".repeat(50_000));
        let odd = format!("{}#6", "!".repeat(50_001));

        assert_eq!(parse_bracket_text(&even).unwrap().to_string(), "#6");
        assert_eq!(parse_bracket_text(&odd).unwrap().to_string(), "!#6");
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
    fn rejects_invalid_atom_stereo_primitives() {
        let err = parse_bracket_text("C@?").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnexpectedCharacter('?'));

        let err = parse_bracket_text("C@TH0").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnsupportedPrimitive);

        let err = parse_bracket_text("C@SP4").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnsupportedPrimitive);

        let err = parse_bracket_text("C@TB21").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnsupportedPrimitive);

        let err = parse_bracket_text("C@OH31").unwrap_err();
        assert_eq!(err.kind(), &BracketParseErrorKind::UnsupportedPrimitive);
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

    #[test]
    fn covers_remaining_bracket_helper_paths() {
        assert_eq!(parse_bracket_text("*").unwrap().to_string(), "*");
        assert_eq!(parse_bracket_text("#6").unwrap().to_string(), "#6");
        assert_eq!(parse_bracket_text("D").unwrap().to_string(), "D");
        assert_eq!(parse_bracket_text("X7").unwrap().to_string(), "X7");
        assert_eq!(parse_bracket_text("v").unwrap().to_string(), "v");
        assert_eq!(parse_bracket_text("A").unwrap().to_string(), "A");
        assert_eq!(parse_bracket_text("a").unwrap().to_string(), "a");
        assert_eq!(parse_bracket_text("H2").unwrap().to_string(), "H2");
        assert_eq!(parse_bracket_text("h3").unwrap().to_string(), "h3");
        assert_eq!(parse_bracket_text("R").unwrap().to_string(), "R");
        assert_eq!(parse_bracket_text("r4").unwrap().to_string(), "r4");
        assert_eq!(parse_bracket_text("x").unwrap().to_string(), "x");
        assert_eq!(parse_bracket_text("^2").unwrap().to_string(), "^2");
        assert_eq!(parse_bracket_text("Na").unwrap().to_string(), "Na");
        assert_eq!(parse_bracket_text("80se").unwrap().to_string(), "80se");
        assert_eq!(parse_bracket_text("@AL1").unwrap().to_string(), "@AL1");
        assert_eq!(parse_bracket_text("+15").unwrap().to_string(), "+15");
        assert_eq!(parse_bracket_text("---").unwrap().to_string(), "-3");
        assert_eq!(parse_bracket_text("C;N").unwrap().to_string(), "C;N");
        assert_eq!(parse_bracket_text("C,N").unwrap().to_string(), "C,N");
        assert_eq!(parse_bracket_text("C&H1").unwrap().to_string(), "C&H1");
        assert_eq!(
            parse_bracket_text("$(C(=O)O)").unwrap().to_string(),
            "$(C(=O)O)"
        );
        assert_eq!(parse_bracket_text("$((C))").unwrap().to_string(), "$((C))");

        assert_eq!(
            parse_bracket_text("!").unwrap_err().kind(),
            &BracketParseErrorKind::UnexpectedEnd
        );
        assert_eq!(
            parse_bracket_text("#").unwrap_err().kind(),
            &BracketParseErrorKind::UnexpectedEnd
        );
        assert_eq!(
            parse_bracket_text("12").unwrap_err().kind(),
            &BracketParseErrorKind::UnexpectedEnd
        );
        assert_eq!(
            parse_bracket_text("12?").unwrap_err().kind(),
            &BracketParseErrorKind::UnexpectedCharacter('?')
        );
        assert_eq!(
            parse_bracket_text("999C").unwrap_err().kind(),
            &BracketParseErrorKind::UnsupportedPrimitive
        );
        assert_eq!(
            parse_bracket_text("$(C").unwrap_err().kind(),
            &BracketParseErrorKind::UnexpectedEnd
        );
        assert_eq!(
            parse_bracket_text("D256").unwrap().tree,
            BracketExprTree::Primitive(AtomPrimitive::Degree(Some(NumericQuery::Exact(256))))
        );
        assert_eq!(
            parse_bracket_text("65536C").unwrap_err().kind(),
            &BracketParseErrorKind::UnsupportedPrimitive
        );

        let parser = BracketParser::new("C");
        assert_eq!(parser.remaining(), "C");
        assert!(!parser.next_char_is_ascii_digit());
        assert!(!parser.is_eof());

        let eof = BracketParser { text: "", pos: 0 };
        assert!(eof.is_eof());
        let err = eof.error_here(BracketParseErrorKind::UnexpectedEnd);
        assert_eq!(err.kind(), &BracketParseErrorKind::UnexpectedEnd);
    }

    #[test]
    fn direct_bracket_parser_methods_cover_internal_branches() {
        let mut parser = BracketParser::new("C;N");
        assert_eq!(parser.parse_low_and(true).unwrap().to_string(), "C;N");

        let mut parser = BracketParser::new("C,N");
        assert_eq!(parser.parse_or(true).unwrap().to_string(), "C,N");

        let mut parser = BracketParser::new("C&H1");
        assert_eq!(parser.parse_high_and(true).unwrap().to_string(), "C&H1");

        let mut parser = BracketParser::new("");
        assert_eq!(
            parser.parse_unary(true).unwrap_err().kind(),
            &BracketParseErrorKind::UnexpectedEnd
        );

        for (text, expected) in [
            ("*", "*"),
            ("#6", "#6"),
            ("D2", "D2"),
            ("X", "X"),
            ("v3", "v3"),
            ("A", "A"),
            ("a", "a"),
            ("H", "#1"),
            ("h", "h"),
            ("R2", "R2"),
            ("r", "r"),
            ("x2", "x2"),
            ("^2", "^2"),
        ] {
            let mut parser = BracketParser::new(text);
            assert_eq!(parser.parse_primitive(true).unwrap().to_string(), expected);
        }

        let mut parser = BracketParser::new("?");
        assert_eq!(
            parser.parse_symbol_after_isotope(12).unwrap_err().kind(),
            &BracketParseErrorKind::UnexpectedCharacter('?')
        );

        let mut parser = BracketParser::new("$(C)");
        assert_eq!(parser.parse_recursive_smarts().unwrap().to_string(), "$(C)");

        let mut parser = BracketParser::new("@AL1");
        assert_eq!(parser.parse_chirality().unwrap().to_string(), "@AL1");

        let mut parser = BracketParser::new("+15");
        assert_eq!(parser.parse_charge().unwrap().to_string(), "+15");

        let mut parser = BracketParser::new("256");
        assert_eq!(
            parser.parse_optional_u8().unwrap_err().kind(),
            &BracketParseErrorKind::UnsupportedPrimitive
        );

        let mut parser = BracketParser::new("x");
        assert_eq!(
            parser.parse_required_u16().unwrap_err().kind(),
            &BracketParseErrorKind::UnexpectedEnd
        );

        let mut parser = BracketParser::new("^");
        assert_eq!(
            parser.parse_required_numeric_query().unwrap_err().kind(),
            &BracketParseErrorKind::UnexpectedEnd
        );
    }

    #[test]
    fn direct_bracket_parser_methods_cover_remaining_numeric_and_symbol_fallbacks() {
        for (text, expected) in [
            ("As", "As"),
            ("as", "as"),
            ("Hg", "Hg"),
            ("Rb", "Rb"),
            ("z", "z"),
            ("Z2", "Z2"),
        ] {
            let mut parser = BracketParser::new(text);
            assert_eq!(parser.parse_primitive(true).unwrap().to_string(), expected);
        }

        let mut parser = BracketParser::new("$((C))");
        assert_eq!(
            parser.parse_recursive_smarts().unwrap().to_string(),
            "$((C))"
        );

        let mut parser = BracketParser::new("{2}");
        assert_eq!(
            parser.parse_optional_numeric_query().unwrap(),
            Some(NumericQuery::Exact(2))
        );

        for text in ["{}", "{2x}", "{2-3x}", "{3-2}"] {
            let mut parser = BracketParser::new(text);
            assert_eq!(
                parser.parse_optional_numeric_query().unwrap_err().kind(),
                &BracketParseErrorKind::UnsupportedPrimitive
            );
        }

        let eof = BracketParser::new("");
        assert!(eof.h_is_atomic_hydrogen_context());

        let charge_run = BracketParser::new("+12");
        assert!(charge_run.h_is_atomic_hydrogen_context());

        let bad_suffix = BracketParser::new("+-");
        assert!(!bad_suffix.h_is_atomic_hydrogen_context());

        let plain_symbol = BracketParser::new("C");
        assert!(!plain_symbol.h_is_atomic_hydrogen_context());
    }
}
