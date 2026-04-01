use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::bracket::{parse_bracket_text, BracketParseError, BracketParseErrorKind};
use crate::error::UnsupportedFeature;
use crate::query::{
    parse_supported_bare_element, AtomExpr, BondExpr, BondExprTree, BondPrimitive, QueryAtom,
    QueryBond, QueryMol,
};
use crate::{SmartsParseError, SmartsParseErrorKind};

/// Parses a SMARTS pattern into a query graph.
///
/// # Errors
///
/// Returns [`SmartsParseError`] when the input is empty or when the pattern
/// contains malformed or unsupported SMARTS syntax.
pub fn parse_smarts(input: &str) -> Result<QueryMol, SmartsParseError> {
    Parser::new(input).parse()
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
    atoms: Vec<QueryAtom>,
    bonds: Vec<QueryBond>,
    current_atom: Option<usize>,
    current_component: usize,
    current_group: Option<usize>,
    next_group_id: usize,
    component_groups: Vec<Option<usize>>,
    open_ring_closures: BTreeMap<u32, PendingRingClosure>,
}

#[derive(Debug, Clone)]
struct PendingRingClosure {
    atom_id: usize,
    explicit_bond: Option<BondExpr>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            atoms: Vec::new(),
            bonds: Vec::new(),
            current_atom: None,
            current_component: 0,
            current_group: None,
            next_group_id: 0,
            component_groups: Vec::new(),
            open_ring_closures: BTreeMap::new(),
        }
    }

    fn parse(mut self) -> Result<QueryMol, SmartsParseError> {
        self.skip_whitespace();
        if self.is_eof() {
            return Err(self.error(SmartsParseErrorKind::EmptyInput));
        }
        self.parse_scope(false)?;

        if self.open_ring_closures.values().next().is_some() {
            return Err(self.error(SmartsParseErrorKind::UnclosedRingClosure));
        }

        Ok(QueryMol::from_parts(
            self.atoms,
            self.bonds,
            self.current_component + 1,
            self.component_groups,
        ))
    }

    fn parse_scope(&mut self, stop_at_group_end: bool) -> Result<(), SmartsParseError> {
        let mut parsed_any = false;
        let mut just_closed_group = false;

        loop {
            self.skip_whitespace();

            if self.is_eof() {
                if stop_at_group_end {
                    return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
                }
                break;
            }

            if stop_at_group_end && self.peek() == ')' {
                if !parsed_any || (self.current_atom.is_none() && !just_closed_group) {
                    return Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter(')')));
                }
                self.pos += 1;
                self.current_atom = None;
                return Ok(());
            }

            if self.current_atom.is_none() {
                if just_closed_group {
                    if self.peek() == '.' {
                        self.parse_component_separator();
                        just_closed_group = false;
                        continue;
                    }
                    return Err(
                        self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek()))
                    );
                }

                if self.peek() == '(' {
                    self.parse_zero_level_group()?;
                    just_closed_group = true;
                } else {
                    self.parse_component_start()?;
                    just_closed_group = false;
                }
            } else {
                self.parse_component_continuation()?;
                just_closed_group = false;
            }

            parsed_any = true;
        }

        if self.current_atom.is_none() && parsed_any && !just_closed_group {
            return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
        }

        Ok(())
    }

    fn parse_component_start(&mut self) -> Result<(), SmartsParseError> {
        match self.peek() {
            '(' | ')' => Err(self.error_here(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::Branch,
            ))),
            '>' => Err(self.error_here(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::Reaction,
            ))),
            '%' => Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter('%'))),
            ch if is_supported_explicit_bond_char(ch) => {
                Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter(ch)))
            }
            ch if ch.is_ascii_digit() || ch == '.' => {
                Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter(ch)))
            }
            _ => self.parse_atom(None),
        }
    }

    fn parse_component_continuation(&mut self) -> Result<(), SmartsParseError> {
        match self.peek() {
            '.' => {
                self.parse_component_separator();
                Ok(())
            }
            '(' => self.parse_branch(),
            '%' => self.parse_ring_closure(None),
            ch if ch.is_ascii_digit() => self.parse_ring_closure(None),
            ch if is_supported_explicit_bond_char(ch) => self.parse_bonded_item(),
            ')' => Err(self.error_here(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::Branch,
            ))),
            '>' => Err(self.error_here(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::Reaction,
            ))),
            _ => self.parse_atom(Some(BondExpr::Elided)),
        }
    }

    fn parse_zero_level_group(&mut self) -> Result<(), SmartsParseError> {
        let outer_group = self.current_group;
        let group_id = outer_group.unwrap_or_else(|| {
            let group_id = self.next_group_id;
            self.next_group_id += 1;
            group_id
        });

        self.pos += 1;
        self.current_group = Some(group_id);
        self.current_atom = None;
        let result = self.parse_scope(true);
        self.current_atom = None;
        self.current_group = outer_group;
        result
    }

    fn parse_branch(&mut self) -> Result<(), SmartsParseError> {
        let branch_root = self
            .current_atom
            .expect("branch encountered without current atom");
        let outer_current = self.current_atom;
        self.pos += 1;

        self.skip_whitespace();
        if self.is_eof() {
            return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
        }

        self.current_atom = Some(branch_root);
        let mut saw_content = false;

        while !self.is_eof() {
            self.skip_whitespace();
            if self.is_eof() {
                return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
            }

            match self.peek() {
                ')' => {
                    if !saw_content {
                        return Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter(')')));
                    }
                    self.pos += 1;
                    self.current_atom = outer_current;
                    return Ok(());
                }
                '.' => return Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter('.'))),
                '(' if !saw_content => {
                    return Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter('(')));
                }
                '(' => {
                    self.parse_branch()?;
                    saw_content = true;
                }
                _ => {
                    self.parse_component_continuation()?;
                    saw_content = true;
                }
            }
        }

        Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput))
    }

    fn parse_component_separator(&mut self) {
        self.pos += 1;
        self.current_component += 1;
        self.current_atom = None;
    }

    fn parse_bonded_item(&mut self) -> Result<(), SmartsParseError> {
        let bond = self.parse_bond_expr()?;

        self.skip_whitespace();
        if self.is_eof() {
            return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
        }

        match self.peek() {
            '%' => self.parse_ring_closure(Some(bond)),
            ch if ch.is_ascii_digit() => self.parse_ring_closure(Some(bond)),
            _ => self.parse_atom(Some(bond)),
        }
    }

    fn parse_bond_expr(&mut self) -> Result<BondExpr, SmartsParseError> {
        Ok(BondExpr::Query(self.parse_bond_low_and()?))
    }

    fn parse_bond_low_and(&mut self) -> Result<BondExprTree, SmartsParseError> {
        let mut terms = vec![self.parse_bond_or()?];
        while !self.is_eof() && self.peek() == ';' {
            self.pos += 1;
            terms.push(self.parse_bond_or()?);
        }
        Ok(collapse_bond_logic(terms, BondExprTree::LowAnd))
    }

    fn parse_bond_or(&mut self) -> Result<BondExprTree, SmartsParseError> {
        let mut terms = vec![self.parse_bond_high_and()?];
        while !self.is_eof() && self.peek() == ',' {
            self.pos += 1;
            terms.push(self.parse_bond_high_and()?);
        }
        Ok(collapse_bond_logic(terms, BondExprTree::Or))
    }

    fn parse_bond_high_and(&mut self) -> Result<BondExprTree, SmartsParseError> {
        let mut terms = vec![self.parse_bond_unary()?];
        while !self.is_eof() {
            if self.peek() == '&' {
                self.pos += 1;
                terms.push(self.parse_bond_unary()?);
                continue;
            }
            if starts_implicit_bond_and(self.peek()) {
                terms.push(self.parse_bond_unary()?);
                continue;
            }
            break;
        }
        Ok(collapse_bond_logic(terms, BondExprTree::HighAnd))
    }

    fn parse_bond_unary(&mut self) -> Result<BondExprTree, SmartsParseError> {
        if self.is_eof() {
            return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
        }

        if self.peek() == '!' {
            self.pos += 1;
            return Ok(BondExprTree::Not(Box::new(self.parse_bond_unary()?)));
        }

        Ok(BondExprTree::Primitive(self.parse_bond_primitive()?))
    }

    fn parse_bond_primitive(&mut self) -> Result<BondPrimitive, SmartsParseError> {
        let primitive = match self.peek() {
            '-' => BondPrimitive::Single,
            '=' => BondPrimitive::Double,
            '#' => BondPrimitive::Triple,
            ':' => BondPrimitive::Aromatic,
            '~' => BondPrimitive::Any,
            '@' => BondPrimitive::Ring,
            '/' => BondPrimitive::Up,
            '\\' => BondPrimitive::Down,
            other => {
                return Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter(other)));
            }
        };
        self.pos += 1;
        Ok(primitive)
    }

    fn parse_ring_closure(
        &mut self,
        explicit_bond: Option<BondExpr>,
    ) -> Result<(), SmartsParseError> {
        let current_atom = self
            .current_atom
            .expect("ring closure without current atom");
        let label = self.parse_ring_label()?;

        let explicit_bond_expr = explicit_bond;

        if let Some(open) = self.open_ring_closures.remove(&label) {
            let bond_expr = resolve_ring_bond(open.explicit_bond, explicit_bond_expr)
                .ok_or_else(|| self.error(SmartsParseErrorKind::ConflictingRingClosureBond))?;

            self.bonds.push(QueryBond {
                id: self.bonds.len(),
                src: open.atom_id,
                dst: current_atom,
                expr: bond_expr,
            });
        } else {
            self.open_ring_closures.insert(
                label,
                PendingRingClosure {
                    atom_id: current_atom,
                    explicit_bond: explicit_bond_expr,
                },
            );
        }

        Ok(())
    }

    fn parse_ring_label(&mut self) -> Result<u32, SmartsParseError> {
        match self.peek() {
            '%' => {
                self.pos += 1;
                if self.is_eof() {
                    return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
                }

                if self.peek() == '(' {
                    self.pos += 1;
                    let digits_start = self.pos;
                    let mut digit_count = 0usize;
                    while !self.is_eof() && self.peek().is_ascii_digit() {
                        if digit_count == 5 {
                            return Err(self.error_here(
                                SmartsParseErrorKind::UnexpectedCharacter(self.peek()),
                            ));
                        }
                        self.pos += 1;
                        digit_count += 1;
                    }

                    if digits_start == self.pos {
                        if self.is_eof() {
                            return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
                        }
                        return Err(
                            self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek()))
                        );
                    }

                    if self.is_eof() {
                        return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
                    }

                    if self.peek() != ')' {
                        return Err(
                            self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek()))
                        );
                    }

                    let label = self.input[digits_start..self.pos]
                        .parse::<u32>()
                        .map_err(|_| self.error(SmartsParseErrorKind::UnexpectedCharacter('%')))?;
                    self.pos += 1;
                    Ok(label)
                } else {
                    if !self.peek().is_ascii_digit() || self.peek() == '0' {
                        return Err(
                            self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek()))
                        );
                    }
                    let tens = self.peek().to_digit(10).expect("ascii digit");
                    self.pos += 1;

                    if self.is_eof() {
                        return Err(self.error(SmartsParseErrorKind::UnexpectedEndOfInput));
                    }

                    if !self.peek().is_ascii_digit() {
                        return Err(
                            self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek()))
                        );
                    }
                    let ones = self.peek().to_digit(10).expect("ascii digit");
                    self.pos += 1;

                    if !self.is_eof() && self.peek().is_ascii_digit() {
                        return Err(
                            self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek()))
                        );
                    }

                    Ok(tens * 10 + ones)
                }
            }
            ch if ch.is_ascii_digit() => {
                self.pos += 1;
                Ok(ch.to_digit(10).expect("ascii digit"))
            }
            other => Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter(other))),
        }
    }

    fn parse_atom(&mut self, incoming_bond: Option<BondExpr>) -> Result<(), SmartsParseError> {
        let expr = match self.peek() {
            '[' => self.parse_bracket_atom()?,
            _ => self.parse_bare_atom()?,
        };

        let atom_id = self.atoms.len();
        let atom = QueryAtom {
            id: atom_id,
            component: self.current_component,
            expr,
        };

        if let Some(prev_id) = self.current_atom {
            let bond_expr = incoming_bond.unwrap_or(BondExpr::Elided);
            self.bonds.push(QueryBond {
                id: self.bonds.len(),
                src: prev_id,
                dst: atom_id,
                expr: bond_expr,
            });
        }

        self.atoms.push(atom);
        self.ensure_component_group_entry();
        self.current_atom = Some(atom_id);
        Ok(())
    }

    fn ensure_component_group_entry(&mut self) {
        if self.component_groups.len() <= self.current_component {
            self.component_groups.push(self.current_group);
        }
    }

    fn parse_bracket_atom(&mut self) -> Result<AtomExpr, SmartsParseError> {
        let start = self.pos;
        self.pos += 1;
        let mut recursive_depth = 0usize;

        while !self.is_eof() {
            if self.input[self.pos..].starts_with("$(") {
                recursive_depth += 1;
                self.pos += 2;
                continue;
            }

            let ch = self.peek();
            if recursive_depth == 0 {
                if ch == ']' {
                    break;
                }
                if ch == '[' {
                    return Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter('[')));
                }
            } else if ch == '(' {
                recursive_depth += 1;
            } else if ch == ')' {
                recursive_depth = recursive_depth.saturating_sub(1);
            }

            self.pos += ch.len_utf8();
        }

        if self.is_eof() {
            return Err(self.error(SmartsParseErrorKind::UnterminatedBracketAtom));
        }

        let end = self.pos;
        let text = &self.input[start + 1..end];
        if find_atom_map(text).is_some() {
            return Err(self.error(SmartsParseErrorKind::UnsupportedFeature(
                UnsupportedFeature::AtomMap,
            )));
        }
        let bracket = parse_bracket_text(text).map_err(|err| self.map_bracket_error(&err))?;
        self.pos += 1;

        Ok(AtomExpr::Bracket(bracket))
    }

    fn parse_bare_atom(&mut self) -> Result<AtomExpr, SmartsParseError> {
        let remaining = &self.input[self.pos..];

        if remaining.starts_with('*') {
            self.pos += 1;
            return Ok(AtomExpr::Wildcard);
        }

        let Some((element, aromatic, width)) = parse_supported_bare_element(remaining) else {
            return Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek())));
        };

        self.pos += width;
        Ok(AtomExpr::Bare { element, aromatic })
    }

    fn skip_whitespace(&mut self) {
        while !self.is_eof() && self.peek().is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> char {
        self.input[self.pos..]
            .chars()
            .next()
            .expect("parser peek past eof")
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn error_here(&self, kind: SmartsParseErrorKind) -> SmartsParseError {
        self.error(kind)
    }

    fn error(&self, kind: SmartsParseErrorKind) -> SmartsParseError {
        SmartsParseError::new(kind)
    }

    fn map_bracket_error(&self, err: &BracketParseError) -> SmartsParseError {
        let kind = match err.kind() {
            BracketParseErrorKind::Empty | BracketParseErrorKind::UnexpectedEnd => {
                SmartsParseErrorKind::UnexpectedEndOfInput
            }
            BracketParseErrorKind::UnexpectedCharacter(ch) => {
                SmartsParseErrorKind::UnexpectedCharacter(*ch)
            }
            BracketParseErrorKind::RecursiveSmarts(kind) => *kind,
            BracketParseErrorKind::UnsupportedPrimitive => {
                SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::AtomPrimitive)
            }
        };

        SmartsParseError::new(kind)
    }
}

fn resolve_ring_bond(left: Option<BondExpr>, right: Option<BondExpr>) -> Option<BondExpr> {
    match (left, right) {
        (Some(lhs), Some(rhs)) if lhs == rhs => Some(lhs),
        (Some(_), Some(_)) => None,
        (Some(lhs), None) => Some(lhs),
        (None, Some(rhs)) => Some(rhs),
        (None, None) => Some(BondExpr::Elided),
    }
}

fn collapse_bond_logic(
    mut terms: Vec<BondExprTree>,
    constructor: fn(Vec<BondExprTree>) -> BondExprTree,
) -> BondExprTree {
    if terms.len() == 1 {
        terms.pop().expect("single term")
    } else {
        constructor(terms)
    }
}

fn find_atom_map(text: &str) -> Option<usize> {
    let mut recursive_depth = 0usize;
    let mut chars = text.char_indices().peekable();

    while let Some((index, ch)) = chars.next() {
        if text[index..].starts_with("$(") {
            recursive_depth += 1;
            chars.next();
            continue;
        }

        if recursive_depth == 0 {
            if ch == ':' && chars.peek().is_some_and(|(_, next)| next.is_ascii_digit()) {
                return Some(index);
            }
            continue;
        }

        match ch {
            '(' => recursive_depth += 1,
            ')' => recursive_depth = recursive_depth.saturating_sub(1),
            _ => {}
        }
    }

    None
}

fn is_supported_explicit_bond_char(ch: char) -> bool {
    matches!(ch, '!' | '-' | '=' | '#' | ':' | '~' | '/' | '\\' | '@')
}

fn starts_implicit_bond_and(ch: char) -> bool {
    matches!(ch, '!' | '-' | '=' | '#' | ':' | '~' | '/' | '\\' | '@')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bracket::parse_bracket_text;
    use alloc::string::ToString;

    #[test]
    fn helper_functions_cover_internal_bond_and_atom_map_logic() {
        let single = BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Single));
        let double = BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Double));

        assert_eq!(resolve_ring_bond(None, None), Some(BondExpr::Elided));
        assert_eq!(
            resolve_ring_bond(Some(single.clone()), None),
            Some(single.clone())
        );
        assert_eq!(
            resolve_ring_bond(None, Some(double.clone())),
            Some(double.clone())
        );
        assert_eq!(
            resolve_ring_bond(Some(single.clone()), Some(single.clone())),
            Some(single.clone())
        );
        assert_eq!(resolve_ring_bond(Some(single), Some(double)), None);

        assert_eq!(
            collapse_bond_logic(
                vec![BondExprTree::Primitive(BondPrimitive::Ring)],
                BondExprTree::Or,
            ),
            BondExprTree::Primitive(BondPrimitive::Ring)
        );
        assert_eq!(
            collapse_bond_logic(
                vec![
                    BondExprTree::Primitive(BondPrimitive::Single),
                    BondExprTree::Primitive(BondPrimitive::Double),
                ],
                BondExprTree::Or,
            ),
            BondExprTree::Or(vec![
                BondExprTree::Primitive(BondPrimitive::Single),
                BondExprTree::Primitive(BondPrimitive::Double),
            ])
        );

        assert_eq!(find_atom_map("C:1"), Some(1));
        assert_eq!(find_atom_map("$(C:1)"), None);
        assert_eq!(find_atom_map("$(C(C):1):2"), Some(9));

        for ch in ['!', '-', '=', '#', ':', '~', '/', '\\', '@'] {
            assert!(is_supported_explicit_bond_char(ch));
            assert!(starts_implicit_bond_and(ch));
        }
        assert!(!is_supported_explicit_bond_char('C'));
        assert!(!starts_implicit_bond_and('C'));
    }

    #[test]
    fn parser_reports_component_and_branch_error_edges() {
        assert_eq!(
            parse_smarts("(").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("()").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter(')')
        );
        assert_eq!(
            parse_smarts("(C)X").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('X')
        );
        assert_eq!(
            parse_smarts(")").unwrap_err().kind(),
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::Branch)
        );
        assert_eq!(
            parse_smarts(">").unwrap_err().kind(),
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::Reaction)
        );
        assert_eq!(
            parse_smarts("%").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('%')
        );
        assert_eq!(
            parse_smarts("-C").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('-')
        );
        assert_eq!(
            parse_smarts("1").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('1')
        );
        assert_eq!(
            parse_smarts(".").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('.')
        );
        assert_eq!(
            parse_smarts("C)").unwrap_err().kind(),
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::Branch)
        );
        assert_eq!(
            parse_smarts("C>").unwrap_err().kind(),
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::Reaction)
        );
        assert_eq!(
            parse_smarts("C(").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("C(C").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("C()").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter(')')
        );
        assert_eq!(
            parse_smarts("C(.)").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('.')
        );
        assert_eq!(
            parse_smarts("C((").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('(')
        );
        assert_eq!(
            parse_smarts("C-").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("C!").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("C[").unwrap_err().kind(),
            SmartsParseErrorKind::UnterminatedBracketAtom
        );
        assert_eq!(
            parse_smarts("[[C]]").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('[')
        );
        assert_eq!(parse_smarts(" * ").unwrap().to_string(), "*");
    }

    #[test]
    fn parser_reports_ring_label_error_edges() {
        assert_eq!(
            parse_smarts("C%").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("C%(").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("C%(1").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("C%(1x").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('x')
        );
        assert_eq!(
            parse_smarts("C%1").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("C%1x").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('x')
        );
        assert_eq!(
            parse_smarts("C%123").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('3')
        );
    }

    #[test]
    fn parser_helper_methods_cover_eof_and_bracket_error_mapping() {
        let parser = Parser::new("");
        let eof_error = parser.error_here(SmartsParseErrorKind::UnexpectedEndOfInput);
        assert_eq!(eof_error.kind(), SmartsParseErrorKind::UnexpectedEndOfInput);

        let mapped_empty = parser.map_bracket_error(&parse_bracket_text("").unwrap_err());
        assert_eq!(
            mapped_empty.kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );

        let mapped_unexpected_end =
            parser.map_bracket_error(&parse_bracket_text("$()").unwrap_err());
        assert_eq!(
            mapped_unexpected_end.kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );

        let mapped_recursive = parser.map_bracket_error(&parse_bracket_text("$(1)").unwrap_err());
        assert_eq!(
            mapped_recursive.kind(),
            SmartsParseErrorKind::UnexpectedCharacter('1')
        );

        let mapped_unsupported = parser.map_bracket_error(&parse_bracket_text("z").unwrap_err());
        assert_eq!(
            mapped_unsupported.kind(),
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::AtomPrimitive)
        );
    }

    #[test]
    fn parser_covers_additional_success_and_error_paths() {
        let grouped = parse_smarts("(C.C)").unwrap();
        assert_eq!(grouped.to_string(), "(C.C)");

        let disjoint = parse_smarts("C.C").unwrap();
        assert_eq!(disjoint.component_count(), 2);

        let adjacent = parse_smarts("CC").unwrap();
        assert_eq!(adjacent.bonds()[0].expr, BondExpr::Elided);

        let nested_branch = parse_smarts("C(C(C))").unwrap();
        assert_eq!(nested_branch.atom_count(), 3);
        assert_eq!(nested_branch.bond_count(), 2);

        let explicit_ring = parse_smarts("C-1CC1").unwrap();
        assert_eq!(explicit_ring.to_string(), "C-1CC-1");

        let bonded_low_and = parse_smarts("C-;:N").unwrap();
        assert_eq!(bonded_low_and.to_string(), "C-;:N");

        let bonded_or = parse_smarts("C-,=N").unwrap();
        assert_eq!(bonded_or.to_string(), "C-,=N");

        let bonded_high_and = parse_smarts("C-&@N").unwrap();
        assert_eq!(bonded_high_and.to_string(), "C-&@N");

        let recursive_nested = parse_smarts("[$((C))]").unwrap();
        assert_eq!(recursive_nested.to_string(), "[$((C))]");

        assert_eq!(
            parse_smarts("C.").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );
        assert_eq!(
            parse_smarts("C%(x)").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('x')
        );
        assert_eq!(
            parse_smarts("C?").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('?')
        );
        assert_eq!(
            parse_smarts("C-?").unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('?')
        );
    }

    #[test]
    fn direct_parser_methods_cover_internal_branches() {
        let mut parser = Parser::new("C");
        parser.parse_scope(false).unwrap();
        assert_eq!(parser.atoms.len(), 1);

        let mut parser = Parser::new(".");
        parser.current_atom = Some(0);
        parser.parse_component_continuation().unwrap();
        assert_eq!(parser.current_component, 1);
        assert_eq!(parser.current_atom, None);

        let mut parser = Parser::new("(C)");
        parser.atoms.push(QueryAtom {
            id: 0,
            component: 0,
            expr: AtomExpr::Bare {
                element: elements_rs::Element::C,
                aromatic: false,
            },
        });
        parser.current_atom = Some(0);
        parser.parse_branch().unwrap();
        assert_eq!(parser.atoms.len(), 2);

        let mut parser = Parser::new("(C(C))");
        parser.atoms.push(QueryAtom {
            id: 0,
            component: 0,
            expr: AtomExpr::Bare {
                element: elements_rs::Element::C,
                aromatic: false,
            },
        });
        parser.current_atom = Some(0);
        parser.parse_branch().unwrap();
        assert_eq!(parser.atoms.len(), 3);

        let mut parser = Parser::new("-1");
        parser.atoms.push(QueryAtom {
            id: 0,
            component: 0,
            expr: AtomExpr::Bare {
                element: elements_rs::Element::C,
                aromatic: false,
            },
        });
        parser.current_atom = Some(0);
        parser.parse_bonded_item().unwrap();
        assert_eq!(parser.open_ring_closures.len(), 1);

        let mut parser = Parser::new("-;:");
        assert_eq!(parser.parse_bond_low_and().unwrap().to_string(), "-;:");

        let mut parser = Parser::new("-,=");
        assert_eq!(parser.parse_bond_or().unwrap().to_string(), "-,=");

        let mut parser = Parser::new("-&@");
        assert_eq!(parser.parse_bond_high_and().unwrap().to_string(), "-&@");

        let mut parser = Parser::new("x");
        assert_eq!(
            parser.parse_ring_label().unwrap_err().kind(),
            SmartsParseErrorKind::UnexpectedCharacter('x')
        );

        let mapped_empty = Parser::new("").map_bracket_error(&parse_bracket_text("").unwrap_err());
        assert_eq!(
            mapped_empty.kind(),
            SmartsParseErrorKind::UnexpectedEndOfInput
        );

        let mapped_unsupported =
            Parser::new("").map_bracket_error(&parse_bracket_text("z").unwrap_err());
        assert_eq!(
            mapped_unsupported.kind(),
            SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::AtomPrimitive)
        );
    }
}
