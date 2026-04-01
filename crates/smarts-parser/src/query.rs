use crate::Span;
use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt;
use elements_rs::{Element, ElementVariant, Isotope, MassNumber};

/// Dense atom identifier inside one parsed SMARTS query.
pub type AtomId = usize;
/// Dense bond identifier inside one parsed SMARTS query.
pub type BondId = usize;
/// Dense component identifier inside one parsed SMARTS query.
pub type ComponentId = usize;
/// Dense zero-level component-group identifier inside one parsed SMARTS query.
pub type ComponentGroupId = usize;

/// One parsed atom expression in the SMARTS query graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomExpr {
    /// Unqualified wildcard atom `*`.
    Wildcard,
    /// Bare atom syntax such as `C` or `c`.
    Bare {
        /// Chemical element identity.
        element: Element,
        /// Whether the element was written using aromatic lowercase SMARTS syntax.
        aromatic: bool,
    },
    /// Bracket atom syntax such as `[C;H1]`.
    Bracket(BracketExpr),
}

/// Parsed contents of one bracket atom without the surrounding `[` and `]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BracketExpr {
    /// Root of the boolean expression tree for the bracket atom.
    pub tree: BracketExprTree,
}

/// Boolean expression tree used inside bracket atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BracketExprTree {
    /// Leaf predicate.
    Primitive(AtomPrimitive),
    /// Unary negation.
    Not(Box<BracketExprTree>),
    /// High-precedence conjunction using `&` or implicit juxtaposition.
    HighAnd(Vec<BracketExprTree>),
    /// Disjunction using `,`.
    Or(Vec<BracketExprTree>),
    /// Low-precedence conjunction using `;`.
    LowAnd(Vec<BracketExprTree>),
}

/// One atomic predicate used inside a bracket atom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomPrimitive {
    /// Wildcard atom `*`.
    Wildcard,
    /// Aliphatic-any atom `A`.
    AliphaticAny,
    /// Aromatic-any atom `a`.
    AromaticAny,
    /// Element symbol predicate such as `C`, `n`, or `se`.
    Symbol {
        /// Chemical element identity.
        element: Element,
        /// Whether the predicate was written using aromatic lowercase SMARTS syntax.
        aromatic: bool,
    },
    /// Exact isotope plus element predicate such as `12C`.
    Isotope {
        /// Exact isotope identity.
        isotope: Isotope,
        /// Whether the isotope predicate was written using aromatic lowercase SMARTS syntax.
        aromatic: bool,
    },
    /// Atomic number predicate such as `#6`.
    AtomicNumber(u16),
    /// Degree predicate such as `D2`.
    Degree(Option<u8>),
    /// Connectivity predicate such as `X3`.
    Connectivity(Option<u8>),
    /// Valence predicate such as `v4`.
    Valence(Option<u8>),
    /// Recursive SMARTS predicate such as `$(CO)`.
    RecursiveQuery(Box<QueryMol>),
    /// Hydrogen-count predicate such as `H1` or `h1`.
    Hydrogen(HydrogenKind, Option<u8>),
    /// Ring-membership predicate such as `R` or `R2`.
    RingMembership(Option<u8>),
    /// Ring-size predicate such as `r5`.
    RingSize(Option<u8>),
    /// Ring-connectivity predicate such as `x2`.
    RingConnectivity(Option<u8>),
    /// Formal charge predicate such as `+`, `-`, or `+2`.
    Charge(i8),
}

/// Kind of hydrogen-count predicate used in SMARTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HydrogenKind {
    /// Total hydrogen count `H`.
    Total,
    /// Implicit hydrogen count `h`.
    Implicit,
}

/// Parsed bond expression between two query atoms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum BondExpr {
    /// Omitted bond operator.
    Elided,
    /// Parsed explicit bond query.
    Query(BondExprTree),
}

/// Boolean expression tree used inside bond expressions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum BondExprTree {
    /// Leaf predicate.
    Primitive(BondPrimitive),
    /// Unary negation.
    Not(Box<BondExprTree>),
    /// High-precedence conjunction using `&` or implicit juxtaposition.
    HighAnd(Vec<BondExprTree>),
    /// Disjunction using `,`.
    Or(Vec<BondExprTree>),
    /// Low-precedence conjunction using `;`.
    LowAnd(Vec<BondExprTree>),
}

/// One primitive bond predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BondPrimitive {
    /// Single bond `-`.
    Single,
    /// Double bond `=`.
    Double,
    /// Triple bond `#`.
    Triple,
    /// Aromatic bond `:`.
    Aromatic,
    /// Any bond `~`.
    Any,
    /// Directional up bond `/`.
    Up,
    /// Directional down bond `\`.
    Down,
    /// Ring bond `@`.
    Ring,
}

/// One parsed query atom in the compiled SMARTS graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryAtom {
    /// Dense atom identifier.
    pub id: AtomId,
    /// Connected-component identifier.
    pub component: ComponentId,
    /// Source span covering the atom token.
    pub span: Span,
    /// Parsed atom expression.
    pub expr: AtomExpr,
}

/// One parsed query bond in the compiled SMARTS graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryBond {
    /// Dense bond identifier.
    pub id: BondId,
    /// Source atom identifier.
    pub src: AtomId,
    /// Destination atom identifier.
    pub dst: AtomId,
    /// Source span covering the bond token or inferred adjacency span.
    pub span: Span,
    /// Parsed bond expression.
    pub expr: BondExpr,
}

/// Parsed SMARTS query graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryMol {
    atoms: Vec<QueryAtom>,
    bonds: Vec<QueryBond>,
    component_count: usize,
    component_groups: Vec<Option<ComponentGroupId>>,
}

impl QueryMol {
    #[inline]
    pub(crate) fn from_parts(
        atoms: Vec<QueryAtom>,
        bonds: Vec<QueryBond>,
        component_count: usize,
        component_groups: Vec<Option<ComponentGroupId>>,
    ) -> Self {
        Self {
            atoms,
            bonds,
            component_count,
            component_groups,
        }
    }

    /// Returns the parsed atoms in dense identifier order.
    #[inline]
    #[must_use]
    pub fn atoms(&self) -> &[QueryAtom] {
        &self.atoms
    }

    /// Returns the parsed bonds in dense identifier order.
    #[inline]
    #[must_use]
    pub fn bonds(&self) -> &[QueryBond] {
        &self.bonds
    }

    /// Returns the number of atoms in the query graph.
    #[inline]
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Returns the number of bonds in the query graph.
    #[inline]
    #[must_use]
    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// Returns the number of disconnected components in the query graph.
    #[inline]
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.component_count
    }

    /// Returns the zero-level component group for one disconnected component.
    ///
    /// Components without zero-level grouping return `None`.
    #[inline]
    #[must_use]
    pub fn component_group(&self, component_id: ComponentId) -> Option<ComponentGroupId> {
        self.component_groups.get(component_id).copied().flatten()
    }

    /// Returns the dense per-component zero-level grouping table.
    #[inline]
    #[must_use]
    pub fn component_groups(&self) -> &[Option<ComponentGroupId>] {
        &self.component_groups
    }

    /// Returns whether the query contains no atoms.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

impl fmt::Display for QueryMol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        QueryMolWriter::new(self).write(f)
    }
}

impl fmt::Display for AtomExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wildcard => f.write_str("*"),
            Self::Bare { element, aromatic } => write_smarts_element(f, *element, *aromatic),
            Self::Bracket(expr) => write!(f, "[{expr}]"),
        }
    }
}

impl fmt::Display for BracketExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.tree.fmt(f)
    }
}

impl fmt::Display for BracketExprTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primitive(primitive) => primitive.fmt(f),
            Self::Not(inner) => write!(f, "!{inner}"),
            Self::HighAnd(items) => write_joined(f, items, "&"),
            Self::Or(items) => write_joined(f, items, ","),
            Self::LowAnd(items) => write_joined(f, items, ";"),
        }
    }
}

impl fmt::Display for AtomPrimitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wildcard => f.write_str("*"),
            Self::AliphaticAny => f.write_str("A"),
            Self::AromaticAny => f.write_str("a"),
            Self::Symbol { element, aromatic } => write_smarts_element(f, *element, *aromatic),
            Self::Isotope { isotope, aromatic } => write_smarts_isotope(f, *isotope, *aromatic),
            Self::AtomicNumber(atomic_number) => write!(f, "#{atomic_number}"),
            Self::Degree(None) => f.write_str("D"),
            Self::Degree(Some(value)) => write!(f, "D{value}"),
            Self::Connectivity(None) => f.write_str("X"),
            Self::Connectivity(Some(value)) => write!(f, "X{value}"),
            Self::Valence(None) => f.write_str("v"),
            Self::Valence(Some(value)) => write!(f, "v{value}"),
            Self::RecursiveQuery(query) => write!(f, "$({query})"),
            Self::Hydrogen(HydrogenKind::Total, None) => f.write_str("H"),
            Self::Hydrogen(HydrogenKind::Total, Some(value)) => write!(f, "H{value}"),
            Self::Hydrogen(HydrogenKind::Implicit, None) => f.write_str("h"),
            Self::Hydrogen(HydrogenKind::Implicit, Some(value)) => write!(f, "h{value}"),
            Self::RingMembership(None) => f.write_str("R"),
            Self::RingMembership(Some(value)) => write!(f, "R{value}"),
            Self::RingSize(None) => f.write_str("r"),
            Self::RingSize(Some(value)) => write!(f, "r{value}"),
            Self::RingConnectivity(None) => f.write_str("x"),
            Self::RingConnectivity(Some(value)) => write!(f, "x{value}"),
            Self::Charge(charge) => write_charge(f, *charge),
        }
    }
}

impl fmt::Display for BondExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Elided => Ok(()),
            Self::Query(tree) => tree.fmt(f),
        }
    }
}

impl fmt::Display for BondExprTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primitive(primitive) => primitive.fmt(f),
            Self::Not(inner) => write!(f, "!{inner}"),
            Self::HighAnd(items) => write_joined(f, items, "&"),
            Self::Or(items) => write_joined(f, items, ","),
            Self::LowAnd(items) => write_joined(f, items, ";"),
        }
    }
}

impl fmt::Display for BondPrimitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Single => "-",
            Self::Double => "=",
            Self::Triple => "#",
            Self::Aromatic => ":",
            Self::Any => "~",
            Self::Up => "/",
            Self::Down => "\\",
            Self::Ring => "@",
        })
    }
}

fn write_joined(
    f: &mut fmt::Formatter<'_>,
    items: &[impl fmt::Display],
    separator: &str,
) -> fmt::Result {
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            f.write_str(separator)?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

fn write_smarts_element(
    f: &mut fmt::Formatter<'_>,
    element: Element,
    aromatic: bool,
) -> fmt::Result {
    if aromatic {
        let symbol: &str = element.as_ref();
        for ch in symbol.chars() {
            write!(f, "{}", ch.to_ascii_lowercase())?;
        }
        Ok(())
    } else {
        write!(f, "{element}")
    }
}

fn write_smarts_isotope(
    f: &mut fmt::Formatter<'_>,
    isotope: Isotope,
    aromatic: bool,
) -> fmt::Result {
    write!(f, "{}", isotope.mass_number())?;
    write_smarts_element(f, isotope.element(), aromatic)
}

fn write_charge(f: &mut fmt::Formatter<'_>, charge: i8) -> fmt::Result {
    if charge == 0 {
        return f.write_str("+0");
    }

    let sign = if charge > 0 { '+' } else { '-' };
    let magnitude = charge.unsigned_abs();
    if magnitude == 1 {
        write!(f, "{sign}")
    } else {
        write!(f, "{sign}{magnitude}")
    }
}

#[derive(Debug, Clone, Copy)]
struct IncidentBond {
    bond_id: BondId,
    other: AtomId,
}

#[derive(Debug, Clone)]
struct RingToken {
    label: u32,
    expr: BondExpr,
}

#[derive(Debug, Clone)]
struct TopLevelEntry {
    components: Vec<AtomId>,
    grouped: bool,
}

struct QueryMolWriter<'a> {
    mol: &'a QueryMol,
    parent_bond_by_atom: Vec<Option<BondId>>,
    children_by_atom: Vec<Vec<AtomId>>,
    ring_tokens_by_atom: Vec<Vec<RingToken>>,
    top_level_entries: Vec<TopLevelEntry>,
}

impl<'a> QueryMolWriter<'a> {
    fn new(mol: &'a QueryMol) -> Self {
        let mut adjacency = vec![Vec::new(); mol.atom_count()];
        for bond in &mol.bonds {
            adjacency[bond.src].push(IncidentBond {
                bond_id: bond.id,
                other: bond.dst,
            });
            adjacency[bond.dst].push(IncidentBond {
                bond_id: bond.id,
                other: bond.src,
            });
        }

        let mut parent_bond_by_atom = vec![None; mol.atom_count()];
        let mut component_roots = vec![None; mol.component_count()];

        for atom in &mol.atoms {
            let atom_id = atom.id;
            let mut parent = None;
            let mut parent_end = 0usize;

            for incident in &adjacency[atom_id] {
                if incident.other >= atom_id {
                    continue;
                }
                let bond = &mol.bonds[incident.bond_id];
                if bond.span.end <= atom.span.start && bond.span.end >= parent_end {
                    parent = Some(incident.bond_id);
                    parent_end = bond.span.end;
                }
            }

            parent_bond_by_atom[atom_id] = parent;
            if parent.is_none() {
                component_roots[atom.component] = Some(atom_id);
            }
        }

        let component_roots = component_roots
            .into_iter()
            .map(|root| root.expect("each component must have a root atom"))
            .collect::<Vec<_>>();

        let mut children_by_atom = vec![Vec::new(); mol.atom_count()];
        for (atom_id, parent_bond) in parent_bond_by_atom.iter().copied().enumerate() {
            if let Some(parent_bond) = parent_bond {
                let bond = &mol.bonds[parent_bond];
                let parent = if bond.src == atom_id {
                    bond.dst
                } else {
                    bond.src
                };
                children_by_atom[parent].push(atom_id);
            }
        }
        for children in &mut children_by_atom {
            children.sort_unstable();
        }

        let mut ring_tokens_by_atom = vec![Vec::new(); mol.atom_count()];
        let mut ring_bond_ids = mol
            .bonds
            .iter()
            .filter(|bond| {
                !(parent_bond_by_atom[bond.src] == Some(bond.id)
                    || parent_bond_by_atom[bond.dst] == Some(bond.id))
            })
            .map(|bond| bond.id)
            .collect::<Vec<_>>();
        ring_bond_ids.sort_by_key(|&bond_id| {
            let bond = &mol.bonds[bond_id];
            (
                bond.src.min(bond.dst),
                bond.src.max(bond.dst),
                bond_expr_rank(&bond.expr),
                bond.id,
            )
        });

        for (ring_label, bond_id) in (1_u32..).zip(ring_bond_ids) {
            let bond = &mol.bonds[bond_id];
            let first = bond.src.min(bond.dst);
            let second = bond.src.max(bond.dst);
            let token = RingToken {
                label: ring_label,
                expr: bond.expr.clone(),
            };
            ring_tokens_by_atom[first].push(token.clone());
            ring_tokens_by_atom[second].push(token);
        }
        for tokens in &mut ring_tokens_by_atom {
            tokens.sort_by_key(|token| token.label);
        }

        let mut top_level_entries = Vec::new();
        let mut component_id = 0usize;
        while component_id < component_roots.len() {
            if let Some(group_id) = mol.component_group(component_id) {
                let mut components = vec![component_roots[component_id]];
                component_id += 1;
                while component_id < component_roots.len()
                    && mol.component_group(component_id) == Some(group_id)
                {
                    components.push(component_roots[component_id]);
                    component_id += 1;
                }
                top_level_entries.push(TopLevelEntry {
                    components,
                    grouped: true,
                });
            } else {
                top_level_entries.push(TopLevelEntry {
                    components: vec![component_roots[component_id]],
                    grouped: false,
                });
                component_id += 1;
            }
        }

        Self {
            mol,
            parent_bond_by_atom,
            children_by_atom,
            ring_tokens_by_atom,
            top_level_entries,
        }
    }

    fn write(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, entry) in self.top_level_entries.iter().enumerate() {
            if index > 0 {
                f.write_str(".")?;
            }
            if entry.grouped {
                f.write_str("(")?;
            }
            for (component_index, root) in entry.components.iter().enumerate() {
                if component_index > 0 {
                    f.write_str(".")?;
                }
                self.write_atom_with_subtree(*root, f)?;
            }
            if entry.grouped {
                f.write_str(")")?;
            }
        }
        Ok(())
    }

    fn write_atom_with_subtree(&self, atom_id: AtomId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mol.atoms[atom_id].expr)?;
        self.write_ring_tokens(atom_id, f)?;

        let children = &self.children_by_atom[atom_id];
        if children.is_empty() {
            return Ok(());
        }

        for child in &children[..children.len() - 1] {
            f.write_str("(")?;
            self.write_bond_to_child(*child, f)?;
            self.write_atom_with_subtree(*child, f)?;
            f.write_str(")")?;
        }

        let main_child = *children.last().expect("children non-empty");
        self.write_bond_to_child(main_child, f)?;
        self.write_atom_with_subtree(main_child, f)
    }

    fn write_bond_to_child(&self, child_id: AtomId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bond_id = self.parent_bond_by_atom[child_id].expect("child must have a parent bond");
        let bond = &self.mol.bonds[bond_id];
        write!(f, "{}", bond.expr)
    }

    fn write_ring_tokens(&self, atom_id: AtomId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for token in &self.ring_tokens_by_atom[atom_id] {
            write!(f, "{}", token.expr)?;
            write_ring_label(f, token.label)?;
        }
        Ok(())
    }
}

fn write_ring_label(f: &mut fmt::Formatter<'_>, label: u32) -> fmt::Result {
    if label < 10 {
        write!(f, "{label}")
    } else if label < 100 {
        write!(f, "%{label:02}")
    } else {
        write!(f, "%({label})")
    }
}

fn bond_expr_rank(expr: &BondExpr) -> (u8, String) {
    let rank = match expr {
        BondExpr::Elided => 0,
        BondExpr::Query(BondExprTree::Primitive(_)) => 1,
        BondExpr::Query(BondExprTree::Not(_)) => 2,
        BondExpr::Query(BondExprTree::HighAnd(_)) => 3,
        BondExpr::Query(BondExprTree::Or(_)) => 4,
        BondExpr::Query(BondExprTree::LowAnd(_)) => 5,
    };
    (rank, expr.to_string())
}

pub(crate) fn parse_supported_bare_element(input: &str) -> Option<(Element, bool, usize)> {
    if input.starts_with("Cl") {
        Some((Element::Cl, false, 2))
    } else if input.starts_with("Br") {
        Some((Element::Br, false, 2))
    } else {
        parse_aromatic_bare_element(input).or_else(|| match input.as_bytes().first().copied()? {
            b'B' => Some((Element::B, false, 1)),
            b'C' => Some((Element::C, false, 1)),
            b'N' => Some((Element::N, false, 1)),
            b'O' => Some((Element::O, false, 1)),
            b'P' => Some((Element::P, false, 1)),
            b'S' => Some((Element::S, false, 1)),
            b'F' => Some((Element::F, false, 1)),
            b'I' => Some((Element::I, false, 1)),
            _ => None,
        })
    }
}

pub(crate) fn parse_supported_bracket_element(input: &str) -> Option<(Element, bool, usize)> {
    parse_aromatic_bracket_element(input)
        .or_else(|| parse_element_symbol(input).map(|(element, width)| (element, false, width)))
}

fn parse_aromatic_bracket_element(input: &str) -> Option<(Element, bool, usize)> {
    if input.starts_with("as") {
        Some((Element::As, true, 2))
    } else if input.starts_with("se") {
        Some((Element::Se, true, 2))
    } else {
        match input.as_bytes().first().copied()? {
            b'b' => Some((Element::B, true, 1)),
            b'c' => Some((Element::C, true, 1)),
            b'n' => Some((Element::N, true, 1)),
            b'o' => Some((Element::O, true, 1)),
            b'p' => Some((Element::P, true, 1)),
            b's' => Some((Element::S, true, 1)),
            _ => None,
        }
    }
}

fn parse_aromatic_bare_element(input: &str) -> Option<(Element, bool, usize)> {
    match input.as_bytes().first().copied()? {
        b'b' => Some((Element::B, true, 1)),
        b'c' => Some((Element::C, true, 1)),
        b'n' => Some((Element::N, true, 1)),
        b'o' => Some((Element::O, true, 1)),
        b'p' => Some((Element::P, true, 1)),
        b's' => Some((Element::S, true, 1)),
        _ => None,
    }
}

fn parse_element_symbol(input: &str) -> Option<(Element, usize)> {
    let bytes = input.as_bytes();
    let first = *bytes.first()?;
    if !first.is_ascii_uppercase() {
        return None;
    }

    if let Some(&second) = bytes.get(1) {
        if second.is_ascii_lowercase() {
            let symbol = &input[..2];
            if let Ok(element) = Element::try_from(symbol) {
                return Some((element, 2));
            }
        }
    }

    let symbol = &input[..1];
    Element::try_from(symbol).ok().map(|element| (element, 1))
}
