use std::{
    sync::OnceLock,
    time::{Duration, Instant},
};

use smarts_rs::{PreparedTarget, QueryMol, TargetCorpusIndex};
use smiles_parser::Smiles;

pub const MAX_INPUT_LEN: usize = 768;
pub const MAX_QUERY_LEN: usize = 512;
pub const MAX_TARGET_LEN: usize = 192;
pub const MAX_QUERY_ATOMS: usize = 32;
pub const MAX_QUERY_BONDS: usize = 48;
pub const MAX_QUERY_COMPONENTS: usize = 10;
pub const MAX_QUERY_COMPLEXITY: usize = 250_000;
pub const TARGETS_PER_INPUT: usize = 8;
pub const PHASE_SLOW_LIMIT: Duration = Duration::from_secs(30);

const GA_MUTATION_SEEDS: &[&str] = &[
    "*@[B,X{16-}].[!R,b].([!#16]~[!#8].[!107*&r6]-[#6&$([#6,#7])&H]=&@[D3])",
    "*-,:,=C.(*(!=[!-]-&@1)!-[#6]-&@1~[!#6&H]-[!Cl].[!#6&H]-[!Cl]-,/&@[#6&R])",
    "([!!-2,8610*,A,X16&!58236*&@TH1&R15,Z{-16},h16,#50].[!#6;X3][!a,!r{8-}&40476*;X;v&A;R1;Z{7-};$([#6]~[#8]);a&R{14-}&v2][#6].[!$([!#1])])",
    "*!:;~[!^{-16}].[r].[v2:14569]",
    "[!!12C,16O,D,D11,h,v14].([!r6&$([#6,#7]):57773].[R:30837])",
    "*.[!!37585*,!v{10-};#79;#85;v9;x11,o,z11;53903*;A;A;r30;r7,$([#7:43954]),r15]!:&~[!r6:36126].[#7;R:30837]",
];

const GA_ATOM_TERMS: &[&str] = &[
    "*",
    "!*",
    "#6",
    "!#6",
    "#7",
    "!#7",
    "C",
    "c",
    "N",
    "n",
    "A",
    "a",
    "R",
    "!R",
    "r",
    "r6",
    "!r6",
    "H0",
    "!H0",
    "D3",
    "!D3",
    "X3",
    "X16",
    "v2",
    "v14",
    "x2",
    "z",
    "z11",
    "+0",
    "-2",
    "$([#6])",
    "$([#7])",
    "$([!#1])",
    "$([#6,#7])",
    "$([#6]-[#8].[#7])",
    "$([!#6,#7]!-[$([#6,#7])])",
];

const GA_BONDS: &[&str] = &["", "-", "~", ":", "=", "#", "!:", "-&@", "-;@", "-,@", "-&!@"];

const TARGET_SMILES: &[&str] = &[
    "CCO",
    "CCCC",
    "c1ccccc1",
    "C1CCCCC1O.CC",
    "ClCC1=CCCCC1.CC",
    "CC(=O)NC1=CC=C(O)C=C1",
    "COC1=CC=CC(=C1OCC2=NC(=CC=C2)OC)CCN",
    "C1=CC=C(C(=C1)C(CNCCC2=C(C=C(C=C2)Cl)Br)O)N",
    "C1CCN(C1)CCOC2=CC=C(C=C2)CC3=C(SC4=CC=CC=C43)C5=CC=C(C=C5)CCNCC6=CN=CC=C6",
    "C[C@@H]([C@@H](C1=CC=C(C=C1)OCC2=CC=CC=C2)O)NCCC3=CC=C(C=C3)OC",
    "CCN1C(=O)C(=CNC2=CC=C(C=C2)CCN3CCN(CC3)C)SC1=C(C#N)C(=O)OCC",
    "C1CN(C[C@H]1NC(=O)NC23CC4CC(C2)CC(C4)C3)CCC5=CC=C(C=C5)F",
    "CC1=C(SC=C1)C(=O)N(C)CC2CCCN(C2)CCC3=CC=C(C=C3)F",
    "CC1=CC(=CC(=C1)CSC2=C(C=CC(=C2)Br)CCN)C",
    "C1=CC(=C(C(=C1)F)OCC2=CC(=C(C=C2)F)C#N)CCN",
    "CN=C(NCC1CCN(C1)CCC2=CC=CC=C2)NCC3(CCCC3)CCOC",
    "CC(C)S(=O)(=O)CCOC1=C(C=C(C=C1Br)CCN)OC",
    "C1=CC=C(C=C1)SCCOC2=C(C=C(C=C2Br)CCN)Br",
    "C1=CC(=CC(=C1)C2=CN=C(S2)C3=CC(=CC=C3)C4=NC=C(S4)C5=CC=CC(=C5)CCN)CCN",
    "CC.CC.C1=CC=C(C=C1)CCN",
    "O=C(O)c1ccccc1",
    "N#CC1=CC=CC=C1Cl",
    "CC(C)(C)OC(=O)N1CCC(CC1)CO",
    "CCOC(=O)C1=CC=CC=C1N",
];

pub struct TargetCase {
    pub smiles: &'static str,
    pub prepared: PreparedTarget,
}

pub struct TargetFixture {
    pub targets: Vec<TargetCase>,
    pub index: TargetCorpusIndex,
}

static TARGET_FIXTURE: OnceLock<TargetFixture> = OnceLock::new();

pub fn target_fixture() -> &'static TargetFixture {
    TARGET_FIXTURE.get_or_init(|| {
        let mut targets = Vec::with_capacity(TARGET_SMILES.len());
        let mut prepared_targets = Vec::with_capacity(TARGET_SMILES.len());
        for &smiles in TARGET_SMILES {
            let parsed = smiles
                .parse::<Smiles>()
                .expect("embedded timeout-fuzz target SMILES must parse");
            let prepared = PreparedTarget::new(parsed);
            prepared_targets.push(PreparedTarget::new(
                smiles
                    .parse::<Smiles>()
                    .expect("embedded timeout-fuzz target SMILES must parse"),
            ));
            targets.push(TargetCase { smiles, prepared });
        }
        let index = TargetCorpusIndex::new(&prepared_targets);
        TargetFixture { targets, index }
    })
}

pub fn split_query_and_target(input: &str) -> (&str, Option<&str>) {
    let split_at = input
        .char_indices()
        .find_map(|(index, ch)| (ch == '\n' || ch == '\0').then_some(index));
    let Some(split_at) = split_at else {
        return (input, None);
    };

    let query = &input[..split_at];
    let rest = &input[split_at + 1..];
    let target = rest
        .split(['\n', '\0'])
        .next()
        .unwrap_or_default()
        .trim_matches('\r');
    let target = (!target.is_empty() && target.len() <= MAX_TARGET_LEN).then_some(target);
    (query, target)
}

pub fn query_is_too_large(query_text: &str, query: &QueryMol) -> bool {
    query_text.len() > MAX_QUERY_LEN
        || query.atom_count() > MAX_QUERY_ATOMS
        || query.bond_count() > MAX_QUERY_BONDS
        || query.component_count() > MAX_QUERY_COMPONENTS
        || query.complexity() > MAX_QUERY_COMPLEXITY
}

pub fn run_phase_with_slow_guard<R>(
    phase: &str,
    input: &str,
    query: &QueryMol,
    run: impl FnOnce() -> R,
) -> R {
    let started = Instant::now();
    let result = run();
    let elapsed = started.elapsed();
    assert!(
        elapsed <= PHASE_SLOW_LIMIT,
        "slow matching-timeout fuzz phase `{phase}` after {:?} on input `{input}` (atoms={}, bonds={}, components={}, complexity={})",
        elapsed,
        query.atom_count(),
        query.bond_count(),
        query.component_count(),
        query.complexity()
    );
    result
}

pub fn selected_target_ids(data: &[u8], target_count: usize) -> [usize; TARGETS_PER_INPUT] {
    let seed = data.iter().fold(0usize, |acc, byte| {
        acc.wrapping_mul(167).wrapping_add(usize::from(*byte))
    });
    let mut ids = [0usize; TARGETS_PER_INPUT];
    for (offset, id) in ids.iter_mut().enumerate() {
        let byte = data.get(offset).copied().unwrap_or(0);
        *id = seed
            .wrapping_add(offset.wrapping_mul(17))
            .wrapping_add(usize::from(byte))
            % target_count;
    }
    ids
}

struct ByteCursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn next(&mut self) -> u8 {
        if self.data.is_empty() {
            return 0;
        }
        let byte = self.data[self.offset % self.data.len()];
        self.offset += 1;
        byte
    }

    fn choose<'b>(&mut self, choices: &'b [&str]) -> &'b str {
        choices[usize::from(self.next()) % choices.len()]
    }
}

fn ga_like_atom(cursor: &mut ByteCursor<'_>) -> String {
    let term_count = 1 + usize::from(cursor.next() % 6);
    let mut atom = String::from("[");
    for term_index in 0..term_count {
        if term_index > 0 {
            atom.push_str(match cursor.next() % 3 {
                0 => "&",
                1 => ",",
                _ => ";",
            });
        }
        if cursor.next().is_multiple_of(17) {
            atom.push('!');
        }
        atom.push_str(cursor.choose(GA_ATOM_TERMS));
    }
    if cursor.next().is_multiple_of(11) {
        atom.push(':');
        atom.push_str(&(usize::from(cursor.next()) * 257 % 65_535).to_string());
    }
    atom.push(']');
    atom
}

fn ga_like_branch(cursor: &mut ByteCursor<'_>) -> String {
    let mut branch = String::from("(");
    branch.push_str(cursor.choose(GA_BONDS));
    branch.push_str(&ga_like_atom(cursor));
    if cursor.next().is_multiple_of(3) {
        branch.push_str(cursor.choose(GA_BONDS));
        branch.push_str(&ga_like_atom(cursor));
    }
    branch.push(')');
    branch
}

fn append_ga_like_component(cursor: &mut ByteCursor<'_>, query: &mut String) {
    query.push('.');
    query.push_str(&ga_like_atom(cursor));
    if cursor.next().is_multiple_of(2) {
        query.push_str(cursor.choose(GA_BONDS));
        query.push_str(&ga_like_atom(cursor));
    }
    if cursor.next().is_multiple_of(2) {
        query.push_str(&ga_like_branch(cursor));
    }
}

fn synthesize_ga_like_query(cursor: &mut ByteCursor<'_>) -> String {
    let mut query = match cursor.next() % 4 {
        0 => {
            let mut query = cursor.choose(GA_MUTATION_SEEDS).to_owned();
            append_ga_like_component(cursor, &mut query);
            query
        }
        1 => {
            let mut query = ga_like_atom(cursor);
            query.push_str(&ga_like_branch(cursor));
            append_ga_like_component(cursor, &mut query);
            query
        }
        2 => {
            let mut query = String::from("*");
            query.push_str(cursor.choose(GA_BONDS));
            query.push_str(&ga_like_atom(cursor));
            query.push_str(&ga_like_branch(cursor));
            append_ga_like_component(cursor, &mut query);
            query
        }
        _ => {
            let mut query = ga_like_atom(cursor);
            query.push_str(cursor.choose(GA_BONDS));
            query.push_str(&ga_like_atom(cursor));
            query.push('.');
            query.push_str(&ga_like_atom(cursor));
            query
        }
    };

    for _ in 0..usize::from(cursor.next() % 4) {
        if query.len() > MAX_QUERY_LEN.saturating_sub(96) {
            break;
        }
        append_ga_like_component(cursor, &mut query);
    }
    query.truncate(MAX_QUERY_LEN);
    query
}

pub fn for_each_fuzz_input(data: &[u8], mut evaluate: impl FnMut(&str, &[u8])) {
    if data.is_empty() {
        return;
    }

    if data.len() <= MAX_INPUT_LEN {
        let input = String::from_utf8_lossy(data);
        evaluate(input.as_ref(), data);
    }

    let mut cursor = ByteCursor::new(data);
    let projected = synthesize_ga_like_query(&mut cursor);
    evaluate(&projected, data);
}
