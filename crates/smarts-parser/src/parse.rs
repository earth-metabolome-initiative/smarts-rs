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
use crate::{SmartsParseError, SmartsParseErrorKind, Span};

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
    span: Span,
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
            return Err(self.error(SmartsParseErrorKind::EmptyInput, Span::new(0, 0)));
        }
        self.parse_scope(false)?;

        if let Some(pending) = self.open_ring_closures.values().next() {
            return Err(self.error(SmartsParseErrorKind::UnclosedRingClosure, pending.span));
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
                    return Err(self.error(
                        SmartsParseErrorKind::UnexpectedEndOfInput,
                        Span::new(self.input.len(), self.input.len()),
                    ));
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
            return Err(self.error(
                SmartsParseErrorKind::UnexpectedEndOfInput,
                Span::new(self.input.len(), self.input.len()),
            ));
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
            return Err(self.error(
                SmartsParseErrorKind::UnexpectedEndOfInput,
                Span::new(self.input.len(), self.input.len()),
            ));
        }

        self.current_atom = Some(branch_root);
        let mut saw_content = false;

        loop {
            self.skip_whitespace();
            if self.is_eof() {
                return Err(self.error(
                    SmartsParseErrorKind::UnexpectedEndOfInput,
                    Span::new(self.input.len(), self.input.len()),
                ));
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
                '.' => {
                    return Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter('.')));
                }
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
    }

    fn parse_component_separator(&mut self) {
        self.pos += 1;
        self.current_component += 1;
        self.current_atom = None;
    }

    fn parse_bonded_item(&mut self) -> Result<(), SmartsParseError> {
        let bond_start = self.pos;
        let bond = self.parse_bond_expr()?;
        let bond_span = Span::new(bond_start, self.pos);

        self.skip_whitespace();
        if self.is_eof() {
            return Err(self.error(
                SmartsParseErrorKind::UnexpectedEndOfInput,
                Span::new(self.input.len(), self.input.len()),
            ));
        }

        match self.peek() {
            '%' => self.parse_ring_closure(Some((bond, bond_span))),
            ch if ch.is_ascii_digit() => self.parse_ring_closure(Some((bond, bond_span))),
            _ => self.parse_atom(Some(bond)),
        }
    }

    fn parse_bond_expr(&mut self) -> Result<BondExpr, SmartsParseError> {
        Ok(BondExpr::Query(self.parse_bond_low_and()?))
    }

    fn parse_bond_low_and(&mut self) -> Result<BondExprTree, SmartsParseError> {
        let mut terms = vec![self.parse_bond_or()?];
        loop {
            if self.is_eof() || self.peek() != ';' {
                break;
            }
            self.pos += 1;
            terms.push(self.parse_bond_or()?);
        }
        Ok(collapse_bond_logic(terms, BondExprTree::LowAnd))
    }

    fn parse_bond_or(&mut self) -> Result<BondExprTree, SmartsParseError> {
        let mut terms = vec![self.parse_bond_high_and()?];
        loop {
            if self.is_eof() || self.peek() != ',' {
                break;
            }
            self.pos += 1;
            terms.push(self.parse_bond_high_and()?);
        }
        Ok(collapse_bond_logic(terms, BondExprTree::Or))
    }

    fn parse_bond_high_and(&mut self) -> Result<BondExprTree, SmartsParseError> {
        let mut terms = vec![self.parse_bond_unary()?];
        loop {
            if self.is_eof() {
                break;
            }
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
            return Err(self.error(
                SmartsParseErrorKind::UnexpectedEndOfInput,
                Span::new(self.input.len(), self.input.len()),
            ));
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
        explicit_bond: Option<(BondExpr, Span)>,
    ) -> Result<(), SmartsParseError> {
        let current_atom = self
            .current_atom
            .expect("ring closure without current atom");
        let (label, label_span) = self.parse_ring_label()?;

        let explicit_bond_expr = explicit_bond.as_ref().map(|(bond, _)| bond.clone());
        let pending_span = explicit_bond.as_ref().map_or(label_span, |(_, span)| {
            Span::new(span.start, label_span.end)
        });

        if let Some(open) = self.open_ring_closures.remove(&label) {
            let bond_expr =
                resolve_ring_bond(open.explicit_bond, explicit_bond_expr).ok_or_else(|| {
                    self.error(
                        SmartsParseErrorKind::ConflictingRingClosureBond,
                        pending_span,
                    )
                })?;

            self.bonds.push(QueryBond {
                id: self.bonds.len(),
                src: open.atom_id,
                dst: current_atom,
                span: Span::new(open.span.start, label_span.end),
                expr: bond_expr,
            });
        } else {
            self.open_ring_closures.insert(
                label,
                PendingRingClosure {
                    atom_id: current_atom,
                    explicit_bond: explicit_bond_expr,
                    span: pending_span,
                },
            );
        }

        Ok(())
    }

    fn parse_ring_label(&mut self) -> Result<(u32, Span), SmartsParseError> {
        let start = self.pos;
        match self.peek() {
            '%' => {
                self.pos += 1;
                if self.is_eof() {
                    return Err(self.error(
                        SmartsParseErrorKind::UnexpectedEndOfInput,
                        Span::new(self.input.len(), self.input.len()),
                    ));
                }

                if self.peek() == '(' {
                    let percent_start = start;
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
                            return Err(self.error(
                                SmartsParseErrorKind::UnexpectedEndOfInput,
                                Span::new(self.input.len(), self.input.len()),
                            ));
                        }
                        return Err(
                            self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek()))
                        );
                    }

                    if self.is_eof() {
                        return Err(self.error(
                            SmartsParseErrorKind::UnexpectedEndOfInput,
                            Span::new(self.input.len(), self.input.len()),
                        ));
                    }

                    if self.peek() != ')' {
                        return Err(
                            self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek()))
                        );
                    }

                    let label =
                        self.input[digits_start..self.pos]
                            .parse::<u32>()
                            .map_err(|_| {
                                self.error(
                                    SmartsParseErrorKind::UnexpectedCharacter('%'),
                                    Span::new(percent_start, self.pos),
                                )
                            })?;
                    self.pos += 1;
                    Ok((label, Span::new(percent_start, self.pos)))
                } else {
                    if !self.peek().is_ascii_digit() || self.peek() == '0' {
                        return Err(
                            self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek()))
                        );
                    }
                    let tens = self.peek().to_digit(10).expect("ascii digit");
                    self.pos += 1;

                    if self.is_eof() {
                        return Err(self.error(
                            SmartsParseErrorKind::UnexpectedEndOfInput,
                            Span::new(self.input.len(), self.input.len()),
                        ));
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

                    Ok((tens * 10 + ones, Span::new(start, self.pos)))
                }
            }
            ch if ch.is_ascii_digit() => {
                self.pos += 1;
                Ok((
                    ch.to_digit(10).expect("ascii digit"),
                    Span::new(start, self.pos),
                ))
            }
            other => Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter(other))),
        }
    }

    fn parse_atom(&mut self, incoming_bond: Option<BondExpr>) -> Result<(), SmartsParseError> {
        let bond_start = self.pos;
        let (expr, span) = match self.peek() {
            '[' => self.parse_bracket_atom()?,
            _ => self.parse_bare_atom()?,
        };

        let atom_id = self.atoms.len();
        let atom = QueryAtom {
            id: atom_id,
            component: self.current_component,
            span,
            expr,
        };

        if let Some(prev_id) = self.current_atom {
            let bond_expr = incoming_bond.unwrap_or(BondExpr::Elided);
            let bond_span = match bond_expr {
                BondExpr::Elided => {
                    let prev_span = self.atoms[prev_id].span;
                    Span::new(prev_span.end, span.start)
                }
                _ => Span::new(bond_start.saturating_sub(1), bond_start),
            };

            self.bonds.push(QueryBond {
                id: self.bonds.len(),
                src: prev_id,
                dst: atom_id,
                span: bond_span,
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

    fn parse_bracket_atom(&mut self) -> Result<(AtomExpr, Span), SmartsParseError> {
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
            return Err(self.error(
                SmartsParseErrorKind::UnterminatedBracketAtom,
                Span::new(start, self.input.len()),
            ));
        }

        let end = self.pos;
        let text = &self.input[start + 1..end];
        if let Some(index) = find_atom_map(text) {
            let span = Span::new(start + 1 + index, start + 1 + index + 1);
            return Err(self.error(
                SmartsParseErrorKind::UnsupportedFeature(UnsupportedFeature::AtomMap),
                span,
            ));
        }
        let bracket =
            parse_bracket_text(text).map_err(|err| self.map_bracket_error(&err, start + 1))?;
        self.pos += 1;

        Ok((AtomExpr::Bracket(bracket), Span::new(start, self.pos)))
    }

    fn parse_bare_atom(&mut self) -> Result<(AtomExpr, Span), SmartsParseError> {
        let start = self.pos;
        let remaining = &self.input[self.pos..];

        if remaining.starts_with('*') {
            self.pos += 1;
            return Ok((AtomExpr::Wildcard, Span::new(start, self.pos)));
        }

        let Some((element, aromatic, width)) = parse_supported_bare_element(remaining) else {
            return Err(self.error_here(SmartsParseErrorKind::UnexpectedCharacter(self.peek())));
        };

        self.pos += width;
        Ok((
            AtomExpr::Bare { element, aromatic },
            Span::new(start, self.pos),
        ))
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
        if self.is_eof() {
            return self.error(kind, Span::new(self.pos, self.pos));
        }

        let width = self.peek().len_utf8();
        self.error(kind, Span::new(self.pos, self.pos + width))
    }

    fn error(&self, kind: SmartsParseErrorKind, span: Span) -> SmartsParseError {
        SmartsParseError::with_span(kind, span)
    }
    fn map_bracket_error(&self, err: &BracketParseError, offset: usize) -> SmartsParseError {
        let span = err.span();
        let absolute_span = Span::new(offset + span.start, offset + span.end);

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

        SmartsParseError::with_span(kind, absolute_span)
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
