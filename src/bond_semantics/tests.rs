use alloc::{format, string::String, vec, vec::Vec};

use smiles_rs::bond::Bond;

use super::{bond_state_bit, spellings, BondSemantics, DISTINGUISHABLE_LABELS};
use crate::{
    parse::parse_bond_expr_text, target::BondLabel, BondExprTree, BondPrimitive,
    SmartsParseErrorKind,
};

const LABELS: usize = DISTINGUISHABLE_LABELS.len();
const TABLE_LEN: usize = 1 << (2 * LABELS);
const STATE_MASK: usize = TABLE_LEN - 1;
const UP: usize = TABLE_LEN;
const STATE_COUNT: usize = 2 * TABLE_LEN;
const CHAIN_MASK: usize = (1 << LABELS) - 1;
const SINGLE_POSITIONS: usize = (1 << LABELS) | 1;

const fn on_chain_and_ring(position: usize) -> usize {
    (1 << position) | (1 << (position + LABELS))
}

fn states_of_spelling_index(index: usize) -> u16 {
    let mut states = 0u16;
    for (ring_offset, ring) in [(0, false), (LABELS, true)] {
        for (position, label) in DISTINGUISHABLE_LABELS.into_iter().enumerate() {
            if index & (1 << (position + ring_offset)) == 0 {
                continue;
            }
            states |= bond_state_bit(label, ring);
            if label == BondLabel::Single {
                states |=
                    bond_state_bit(BondLabel::Up, ring) | bond_state_bit(BondLabel::Down, ring);
            }
        }
    }
    states
}

/// Unary bond terms, as `(state, text)` where a state is a table index plus [`UP`].
fn literals() -> Vec<(usize, &'static str)> {
    let positive = [
        (on_chain_and_ring(0), "-"),
        (on_chain_and_ring(1), "="),
        (on_chain_and_ring(2), "#"),
        (on_chain_and_ring(3), ":"),
        (STATE_MASK, "~"),
        (STATE_MASK & !CHAIN_MASK, "@"),
    ];
    let negated = ["!-", "!=", "!#", "!:", "!~", "!@"];
    let mut literals = positive.to_vec();
    literals.extend(
        positive
            .iter()
            .zip(negated)
            .map(|(&(state, _), text)| (!state & STATE_MASK, text)),
    );
    literals.push((on_chain_and_ring(0) | UP, "/"));
    literals
}

const fn intersect(left: usize, right: usize) -> usize {
    (left & right & STATE_MASK) | ((left | right) & UP)
}

const fn unite(left: usize, right: usize) -> usize {
    left | right
}

#[derive(Clone, Copy)]
enum Step {
    Unreached,
    Literal(usize),
    Lower,
    JoinLiteral { left: usize, literal: usize },
    Join { left: usize, right: usize },
}

struct Level {
    len: Vec<usize>,
    step: Vec<Step>,
}

impl Level {
    fn unreached() -> Self {
        Self {
            len: vec![usize::MAX; STATE_COUNT],
            step: vec![Step::Unreached; STATE_COUNT],
        }
    }

    fn above(lower: &Self) -> Self {
        Self {
            len: lower.len.clone(),
            step: lower
                .len
                .iter()
                .map(|&len| {
                    if len == usize::MAX {
                        Step::Unreached
                    } else {
                        Step::Lower
                    }
                })
                .collect(),
        }
    }

    fn offer(&mut self, state: usize, len: usize, step: Step) -> bool {
        if len < self.len[state] {
            self.len[state] = len;
            self.step[state] = step;
            true
        } else {
            false
        }
    }

    fn reached(&self) -> impl Iterator<Item = usize> + '_ {
        (0..STATE_COUNT).filter(|&state| self.len[state] != usize::MAX)
    }
}

/// Shortest spellings by precedence level: `&` joins unary terms, `,` joins
/// `&` terms and `;` joins `,` terms. Equal lengths keep the first spelling found.
struct ShortestSpellings {
    literals: Vec<(usize, &'static str)>,
    high: Level,
    or: Level,
    low: Level,
}

impl ShortestSpellings {
    fn compute() -> Self {
        let literals = literals();
        let mut high = Level::unreached();
        for (index, &(state, text)) in literals.iter().enumerate() {
            high.offer(state, text.len(), Step::Literal(index));
        }
        let mut changed = true;
        while changed {
            changed = false;
            for left in high.reached().collect::<Vec<_>>() {
                for (literal, &(state, text)) in literals.iter().enumerate() {
                    let len = high.len[left] + 1 + text.len();
                    changed |= high.offer(
                        intersect(left, state),
                        len,
                        Step::JoinLiteral { left, literal },
                    );
                }
            }
        }
        let or = Self::close(&high, unite);
        let low = Self::close(&or, intersect);
        Self {
            literals,
            high,
            or,
            low,
        }
    }

    fn close(lower: &Level, combine: fn(usize, usize) -> usize) -> Level {
        let mut level = Level::above(lower);
        let rights = lower.reached().collect::<Vec<_>>();
        let mut changed = true;
        while changed {
            changed = false;
            for left in level.reached().collect::<Vec<_>>() {
                for &right in &rights {
                    let len = level.len[left] + 1 + lower.len[right];
                    changed |= level.offer(combine(left, right), len, Step::Join { left, right });
                }
            }
        }
        level
    }

    fn spelling(&self, state: usize) -> Option<String> {
        (self.low.len[state] != usize::MAX).then(|| self.spell_low(state))
    }

    fn spell_low(&self, state: usize) -> String {
        match self.low.step[state] {
            Step::Lower => self.spell_or(state),
            Step::Join { left, right } => {
                format!("{};{}", self.spell_low(left), self.spell_or(right))
            }
            _ => unreachable!("low level joins `,` terms"),
        }
    }

    fn spell_or(&self, state: usize) -> String {
        match self.or.step[state] {
            Step::Lower => self.spell_high(state),
            Step::Join { left, right } => {
                format!("{},{}", self.spell_or(left), self.spell_high(right))
            }
            _ => unreachable!("or level joins `&` terms"),
        }
    }

    fn spell_high(&self, state: usize) -> String {
        match self.high.step[state] {
            Step::Literal(literal) => String::from(self.literals[literal].1),
            Step::JoinLiteral { left, literal } => {
                format!("{}&{}", self.spell_high(left), self.literals[literal].1)
            }
            _ => unreachable!("high level joins unary terms"),
        }
    }
}

fn holds_a_single_bond(index: usize) -> bool {
    index & SINGLE_POSITIONS != 0
}

#[test]
fn every_bond_meaning_has_a_spelling_that_means_it() {
    for index in 0..TABLE_LEN {
        let states = states_of_spelling_index(index);
        let plain = parse_bond_expr_text(spellings::PLAIN[index]).unwrap();
        assert_eq!(
            BondSemantics::of_tree(&plain),
            BondSemantics::new(states, None),
            "plain spelling {index}"
        );
        if holds_a_single_bond(index) {
            let up = parse_bond_expr_text(spellings::UP[index]).unwrap();
            assert_eq!(
                BondSemantics::of_tree(&up),
                BondSemantics::new(states, Some(Bond::Up)),
                "directional spelling {index}"
            );
        } else {
            assert_eq!(spellings::UP[index], "", "directional spelling {index}");
        }
    }
}

#[test]
fn bond_spellings_are_the_shortest() {
    let shortest = ShortestSpellings::compute();
    for index in 0..TABLE_LEN {
        assert_eq!(
            spellings::PLAIN[index].len(),
            shortest.low.len[index],
            "plain spelling {index}"
        );
        if holds_a_single_bond(index) {
            assert_eq!(
                spellings::UP[index].len(),
                shortest.low.len[index | UP],
                "directional spelling {index}"
            );
        }
    }
}

#[test]
fn negated_direction_matches_nothing() {
    for source in ["!/", "-&!\\", "/,!/", "!/;-"] {
        let tree = parse_bond_expr_text(source).unwrap();
        assert_eq!(
            BondSemantics::of_tree(&tree),
            BondSemantics::new(0, None),
            "{source}"
        );
    }
}

#[test]
fn quadruple_bond_matches_nothing() {
    let quadruple = BondExprTree::Primitive(BondPrimitive::Bond(Bond::Quadruple));
    assert_eq!(
        BondSemantics::of_tree(&quadruple),
        BondSemantics::new(0, None)
    );
}

#[test]
fn bare_bond_expression_rejects_trailing_input() {
    assert_eq!(
        parse_bond_expr_text("-C").unwrap_err().kind(),
        SmartsParseErrorKind::UnexpectedCharacter('C')
    );
}

#[test]
fn first_direction_is_the_bond_direction() {
    let up_first = BondSemantics::of_tree(&parse_bond_expr_text("/,\\").unwrap());
    let down_first = BondSemantics::of_tree(&parse_bond_expr_text("=,\\&@,/").unwrap());
    assert_eq!(up_first.direction(), Some(Bond::Up));
    assert_eq!(down_first.direction(), Some(Bond::Down));
}

#[test]
fn bond_without_single_states_carries_no_direction() {
    let tree = parse_bond_expr_text("/&=").unwrap();
    assert_eq!(BondSemantics::of_tree(&tree), BondSemantics::new(0, None));
    let tree = parse_bond_expr_text("/&!@,=").unwrap();
    assert_eq!(BondSemantics::of_tree(&tree).direction(), Some(Bond::Up));
}

type LiteralKey = (bool, bool, usize);

fn literal_key(literal: &str) -> LiteralKey {
    let symbol = literal.trim_start_matches('!');
    let position = "/-=#:~@".find(symbol).expect("bond literal");
    (symbol == "@", literal.starts_with('!'), position)
}

/// Orders literals within `&` terms, terms within `,` clauses and clauses within
/// `;`, bond orders before ring membership and plain before negated literals.
fn in_reading_order(spelling: &str) -> String {
    let mut clauses = spelling
        .split(';')
        .map(|clause| {
            let mut terms = clause
                .split(',')
                .map(|term| {
                    let mut literals = term.split('&').collect::<Vec<_>>();
                    literals.sort_by_key(|literal| literal_key(literal));
                    literals
                })
                .collect::<Vec<_>>();
            terms.sort_by_key(|term| {
                term.iter()
                    .map(|literal| literal_key(literal))
                    .collect::<Vec<_>>()
            });
            terms
        })
        .collect::<Vec<_>>();
    clauses.sort_by_key(|clause| {
        clause
            .iter()
            .map(|term| {
                term.iter()
                    .map(|literal| literal_key(literal))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    clauses
        .iter()
        .map(|clause| {
            clause
                .iter()
                .map(|term| term.join("&"))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join(";")
}

#[test]
#[ignore = "rewrites src/bond_semantics/spellings.rs from the shortest-spelling search"]
fn write_bond_spelling_table() {
    use core::fmt::Write;

    let shortest = ShortestSpellings::compute();
    let mut source = String::from(
        "//! Shortest spellings of every bond meaning, indexed by `spelling_index`.\n\
         //!\n\
         //! Written by `bond_semantics::tests::write_bond_spelling_table`.\n\n\
         /// Spellings that carry no direction.\n",
    );
    writeln!(source, "pub(super) static PLAIN: [&str; {TABLE_LEN}] = [").unwrap();
    for index in 0..TABLE_LEN {
        let spelling =
            in_reading_order(&shortest.spelling(index).expect("every state is reachable"));
        writeln!(source, "    {spelling:?},").unwrap();
    }
    source.push_str(
        "];\n\n/// Spellings that carry `/`, empty where the states hold no single bond.\n",
    );
    writeln!(source, "pub(super) static UP: [&str; {TABLE_LEN}] = [").unwrap();
    for index in 0..TABLE_LEN {
        let spelling = if holds_a_single_bond(index) {
            in_reading_order(
                &shortest
                    .spelling(index | UP)
                    .expect("every directional state is reachable"),
            )
        } else {
            String::new()
        };
        writeln!(source, "    {spelling:?},").unwrap();
    }
    source.push_str("];\n");
    std::fs::write(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/bond_semantics/spellings.rs"
        ),
        source,
    )
    .unwrap();
}
