use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt;
use elements_rs::{Element, ElementVariant, Isotope, MassNumber};
use smiles_parser::{atom::bracketed::chirality::Chirality, bond::Bond};

/// Dense atom identifier inside one parsed SMARTS query.
pub type AtomId = usize;
/// Dense bond identifier inside one parsed SMARTS query.
pub type BondId = usize;
/// Dense component identifier inside one parsed SMARTS query.
pub type ComponentId = usize;
/// Dense zero-level component-group identifier inside one parsed SMARTS query.
pub type ComponentGroupId = usize;

/// One numeric SMARTS count constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericQuery {
    /// Exact numeric value such as `D2`.
    Exact(u16),
    /// Inclusive range value such as `D{2-3}` or `h{1-}`.
    Range(NumericRange),
}

/// Inclusive numeric SMARTS range constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericRange {
    /// Inclusive lower bound, if present.
    pub min: Option<u16>,
    /// Inclusive upper bound, if present.
    pub max: Option<u16>,
}

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
    /// Optional atom-map number written at the end of the bracket atom, such as `:1`.
    pub atom_map: Option<u32>,
}

/// Boolean expression tree used inside bracket atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BracketExprTree {
    /// Leaf predicate.
    Primitive(AtomPrimitive),
    /// Unary negation.
    Not(Box<Self>),
    /// High-precedence conjunction using `&` or implicit juxtaposition.
    HighAnd(Vec<Self>),
    /// Disjunction using `,`.
    Or(Vec<Self>),
    /// Low-precedence conjunction using `;`.
    LowAnd(Vec<Self>),
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
    /// Isotope mass number paired with wildcard atom syntax such as `89*` or `0*`.
    IsotopeWildcard(u16),
    /// Atomic number predicate such as `#6`.
    AtomicNumber(u16),
    /// Degree predicate such as `D2` or `D{2-3}`.
    Degree(Option<NumericQuery>),
    /// Connectivity predicate such as `X3` or `X{2-4}`.
    Connectivity(Option<NumericQuery>),
    /// Valence predicate such as `v4` or `v{3-}`.
    Valence(Option<NumericQuery>),
    /// Recursive SMARTS predicate such as `$(CO)`.
    RecursiveQuery(Box<QueryMol>),
    /// Hydrogen-count predicate such as `H1`, `h1`, or `h{1-}`.
    Hydrogen(HydrogenKind, Option<NumericQuery>),
    /// Ring-membership predicate such as `R` or `R2`.
    RingMembership(Option<NumericQuery>),
    /// Ring-size predicate such as `r5` or `r{5-6}`.
    RingSize(Option<NumericQuery>),
    /// Ring-connectivity predicate such as `x2` or `x{2-}`.
    RingConnectivity(Option<NumericQuery>),
    /// Atom chirality predicate such as `@`, `@@`, or `@TH1`.
    Chirality(Chirality),
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
    Not(Box<Self>),
    /// High-precedence conjunction using `&` or implicit juxtaposition.
    HighAnd(Vec<Self>),
    /// Disjunction using `,`.
    Or(Vec<Self>),
    /// Low-precedence conjunction using `;`.
    LowAnd(Vec<Self>),
}

/// One primitive bond predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondPrimitive {
    /// Concrete directional or order-specific bond kinds reused from `smiles-parser`.
    Bond(Bond),
    /// Any bond `~`.
    Any,
    /// Ring bond `@`.
    Ring,
}

impl PartialOrd for BondPrimitive {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BondPrimitive {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        bond_primitive_order_key(*self).cmp(&bond_primitive_order_key(*other))
    }
}

/// One parsed query atom in the compiled SMARTS graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryAtom {
    /// Dense atom identifier.
    pub id: AtomId,
    /// Connected-component identifier.
    pub component: ComponentId,
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
    /// Builds a parsed SMARTS query from already-lowered graph parts.
    #[inline]
    #[must_use]
    pub const fn from_parts(
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
    pub const fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Returns the number of bonds in the query graph.
    #[inline]
    #[must_use]
    pub const fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// Returns the number of disconnected components in the query graph.
    #[inline]
    #[must_use]
    pub const fn component_count(&self) -> usize {
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
    pub const fn is_empty(&self) -> bool {
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
        self.tree.fmt(f)?;
        if let Some(atom_map) = self.atom_map {
            write!(f, ":{atom_map}")?;
        }
        Ok(())
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
            Self::IsotopeWildcard(mass_number) => write!(f, "{mass_number}*"),
            Self::AtomicNumber(atomic_number) => write!(f, "#{atomic_number}"),
            Self::Degree(query) => write_numeric_query_suffix(f, "D", *query),
            Self::Connectivity(query) => write_numeric_query_suffix(f, "X", *query),
            Self::Valence(query) => write_numeric_query_suffix(f, "v", *query),
            Self::RecursiveQuery(query) => write!(f, "$({query})"),
            Self::Hydrogen(HydrogenKind::Total, query) => {
                write_numeric_query_suffix(f, "H", *query)
            }
            Self::Hydrogen(HydrogenKind::Implicit, query) => {
                write_numeric_query_suffix(f, "h", *query)
            }
            Self::RingMembership(query) => write_numeric_query_suffix(f, "R", *query),
            Self::RingSize(query) => write_numeric_query_suffix(f, "r", *query),
            Self::RingConnectivity(query) => write_numeric_query_suffix(f, "x", *query),
            Self::Chirality(chirality) => chirality.fmt(f),
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
        match self {
            Self::Bond(bond) => bond.fmt(f),
            Self::Any => f.write_str("~"),
            Self::Ring => f.write_str("@"),
        }
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

fn write_numeric_query_suffix(
    f: &mut fmt::Formatter<'_>,
    prefix: &str,
    query: Option<NumericQuery>,
) -> fmt::Result {
    f.write_str(prefix)?;
    match query {
        None => Ok(()),
        Some(NumericQuery::Exact(value)) => write!(f, "{value}"),
        Some(NumericQuery::Range(range)) => write_numeric_range(f, range),
    }
}

fn write_numeric_range(f: &mut fmt::Formatter<'_>, range: NumericRange) -> fmt::Result {
    f.write_str("{")?;
    if let Some(min) = range.min {
        write!(f, "{min}")?;
    }
    if range.min != range.max || range.min.is_none() || range.max.is_none() {
        f.write_str("-")?;
        if let Some(max) = range.max {
            write!(f, "{max}")?;
        }
    }
    f.write_str("}")
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

const fn bond_primitive_order_key(primitive: BondPrimitive) -> u8 {
    match primitive {
        BondPrimitive::Bond(Bond::Single) => 0,
        BondPrimitive::Bond(Bond::Double) => 1,
        BondPrimitive::Bond(Bond::Triple) => 2,
        BondPrimitive::Bond(Bond::Quadruple) => 3,
        BondPrimitive::Bond(Bond::Aromatic) => 4,
        BondPrimitive::Bond(Bond::Up) => 5,
        BondPrimitive::Bond(Bond::Down) => 6,
        BondPrimitive::Any => 7,
        BondPrimitive::Ring => 8,
    }
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
        let mut parent_bond_by_atom = vec![None; mol.atom_count()];
        for bond in &mol.bonds {
            if bond.dst < parent_bond_by_atom.len() && parent_bond_by_atom[bond.dst].is_none() {
                parent_bond_by_atom[bond.dst] = Some(bond.id);
            }
        }

        let mut component_roots = vec![None; mol.component_count()];
        for atom in &mol.atoms {
            if parent_bond_by_atom[atom.id].is_none() && component_roots[atom.component].is_none() {
                component_roots[atom.component] = Some(atom.id);
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
                children_by_atom[bond.src].push(atom_id);
            }
        }
        for children in &mut children_by_atom {
            children.sort_unstable();
        }

        let mut is_parent_bond = vec![false; mol.bond_count()];
        for parent_bond in parent_bond_by_atom.iter().flatten().copied() {
            is_parent_bond[parent_bond] = true;
        }

        let mut ring_tokens_by_atom = vec![Vec::new(); mol.atom_count()];
        let mut ring_bond_ids = mol
            .bonds
            .iter()
            .filter(|bond| !is_parent_bond[bond.id])
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

#[allow(clippy::redundant_pub_crate)]
pub(crate) fn parse_supported_bare_element(input: &str) -> Option<(Element, bool, usize)> {
    if let Some((element, width)) = parse_rdkit_bare_multiletter_element(input) {
        Some((element, false, width))
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

#[allow(clippy::redundant_pub_crate)]
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

fn parse_rdkit_bare_multiletter_element(input: &str) -> Option<(Element, usize)> {
    const BARE_TWO_LETTER_ELEMENTS: [(&str, Element); 21] = [
        ("Cl", Element::Cl),
        ("Br", Element::Br),
        ("Na", Element::Na),
        ("Ca", Element::Ca),
        ("Sc", Element::Sc),
        ("Co", Element::Co),
        ("As", Element::As),
        ("Nb", Element::Nb),
        ("In", Element::In),
        ("Sn", Element::Sn),
        ("Sb", Element::Sb),
        ("Cs", Element::Cs),
        ("Ba", Element::Ba),
        ("Os", Element::Os),
        ("Pb", Element::Pb),
        ("Po", Element::Po),
        ("Ac", Element::Ac),
        ("Pa", Element::Pa),
        ("Np", Element::Np),
        ("No", Element::No),
        ("Cn", Element::Cn),
    ];

    for (symbol, element) in BARE_TWO_LETTER_ELEMENTS {
        if input.starts_with(symbol) {
            return Some((element, 2));
        }
    }

    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn atom(
        id: AtomId,
        component: ComponentId,
        _start: usize,
        _end: usize,
        expr: AtomExpr,
    ) -> QueryAtom {
        QueryAtom {
            id,
            component,
            expr,
        }
    }

    fn bond(
        id: BondId,
        src: AtomId,
        dst: AtomId,
        _start: usize,
        _end: usize,
        expr: BondExpr,
    ) -> QueryBond {
        QueryBond { id, src, dst, expr }
    }

    #[test]
    fn query_accessors_handle_empty_and_out_of_range_component_groups() {
        let query = QueryMol::from_parts(Vec::new(), Vec::new(), 0, Vec::new());
        assert!(query.is_empty());
        assert_eq!(query.atom_count(), 0);
        assert_eq!(query.bond_count(), 0);
        assert_eq!(query.component_count(), 0);
        assert_eq!(query.component_group(0), None);
        assert_eq!(query.component_groups(), &[]);
        assert_eq!(query.to_string(), "");
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn displays_all_atom_and_bond_primitive_variants() {
        let aromatic_isotope = Isotope::try_from((Element::Se, 80_u16)).unwrap();
        let recursive = QueryMol::from_parts(
            vec![atom(
                0,
                0,
                0,
                1,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            )],
            Vec::new(),
            1,
            vec![None],
        );

        let atom_cases = [
            (AtomExpr::Wildcard.to_string(), "*"),
            (
                AtomExpr::Bare {
                    element: Element::Cl,
                    aromatic: false,
                }
                .to_string(),
                "Cl",
            ),
            (
                AtomExpr::Bracket(BracketExpr {
                    tree: BracketExprTree::Primitive(AtomPrimitive::Wildcard),
                    atom_map: None,
                })
                .to_string(),
                "[*]",
            ),
            (AtomPrimitive::AliphaticAny.to_string(), "A"),
            (AtomPrimitive::AromaticAny.to_string(), "a"),
            (
                AtomPrimitive::Symbol {
                    element: Element::As,
                    aromatic: true,
                }
                .to_string(),
                "as",
            ),
            (
                AtomPrimitive::Isotope {
                    isotope: aromatic_isotope,
                    aromatic: true,
                }
                .to_string(),
                "80se",
            ),
            (AtomPrimitive::IsotopeWildcard(89).to_string(), "89*"),
            (AtomPrimitive::AtomicNumber(6).to_string(), "#6"),
            (AtomPrimitive::Degree(None).to_string(), "D"),
            (AtomPrimitive::Connectivity(None).to_string(), "X"),
            (AtomPrimitive::Valence(None).to_string(), "v"),
            (
                AtomPrimitive::RecursiveQuery(Box::new(recursive)).to_string(),
                "$(C)",
            ),
            (
                AtomPrimitive::Hydrogen(HydrogenKind::Implicit, None).to_string(),
                "h",
            ),
            (AtomPrimitive::RingMembership(None).to_string(), "R"),
            (AtomPrimitive::RingSize(None).to_string(), "r"),
            (AtomPrimitive::RingConnectivity(None).to_string(), "x"),
            (AtomPrimitive::Charge(0).to_string(), "+0"),
            (AtomPrimitive::Charge(-1).to_string(), "-"),
            (
                AtomPrimitive::Chirality(Chirality::AL(1)).to_string(),
                "@AL1",
            ),
            (Chirality::At.to_string(), "@"),
            (Chirality::AtAt.to_string(), "@@"),
            (Chirality::SP(1).to_string(), "@SP1"),
            (Chirality::TB(1).to_string(), "@TB1"),
            (Chirality::OH(1).to_string(), "@OH1"),
            (
                BracketExprTree::Not(Box::new(BracketExprTree::Primitive(
                    AtomPrimitive::AtomicNumber(1),
                )))
                .to_string(),
                "!#1",
            ),
            (
                BracketExprTree::HighAnd(vec![
                    BracketExprTree::Primitive(AtomPrimitive::AliphaticAny),
                    BracketExprTree::Primitive(AtomPrimitive::Degree(Some(NumericQuery::Exact(4)))),
                ])
                .to_string(),
                "A&D4",
            ),
            (
                BracketExprTree::Or(vec![
                    BracketExprTree::Primitive(AtomPrimitive::AromaticAny),
                    BracketExprTree::Primitive(AtomPrimitive::RingMembership(Some(
                        NumericQuery::Exact(2),
                    ))),
                ])
                .to_string(),
                "a,R2",
            ),
            (
                BracketExprTree::LowAnd(vec![
                    BracketExprTree::Primitive(AtomPrimitive::Valence(Some(NumericQuery::Exact(
                        3,
                    )))),
                    BracketExprTree::Primitive(AtomPrimitive::Charge(-2)),
                ])
                .to_string(),
                "v3;-2",
            ),
            (BondExpr::Elided.to_string(), ""),
            (
                BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Any)).to_string(),
                "~",
            ),
            (
                BondExprTree::Not(Box::new(BondExprTree::Primitive(BondPrimitive::Ring)))
                    .to_string(),
                "!@",
            ),
            (
                BondExprTree::HighAnd(vec![
                    BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
                    BondExprTree::Primitive(BondPrimitive::Ring),
                ])
                .to_string(),
                "-&@",
            ),
            (
                BondExprTree::Or(vec![
                    BondExprTree::Primitive(BondPrimitive::Bond(Bond::Up)),
                    BondExprTree::Primitive(BondPrimitive::Bond(Bond::Down)),
                ])
                .to_string(),
                "/,\\",
            ),
            (
                BondExprTree::LowAnd(vec![
                    BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)),
                    BondExprTree::Primitive(BondPrimitive::Bond(Bond::Aromatic)),
                ])
                .to_string(),
                "=;:",
            ),
        ];

        for (actual, expected) in atom_cases {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn helper_functions_cover_element_and_label_formatting() {
        assert_eq!(
            parse_supported_bare_element("Cl"),
            Some((Element::Cl, false, 2))
        );
        assert_eq!(
            parse_supported_bare_element("Br"),
            Some((Element::Br, false, 2))
        );
        assert_eq!(
            parse_supported_bare_element("c"),
            Some((Element::C, true, 1))
        );
        assert_eq!(parse_supported_bare_element("Z"), None);

        assert_eq!(
            parse_supported_bracket_element("as"),
            Some((Element::As, true, 2))
        );
        assert_eq!(
            parse_supported_bracket_element("se"),
            Some((Element::Se, true, 2))
        );
        assert_eq!(
            parse_supported_bracket_element("Na"),
            Some((Element::Na, false, 2))
        );
        assert_eq!(
            parse_supported_bracket_element("Pt"),
            Some((Element::Pt, false, 2))
        );
        assert_eq!(parse_supported_bracket_element("?"), None);

        assert_eq!(
            parse_aromatic_bare_element("n"),
            Some((Element::N, true, 1))
        );
        assert_eq!(parse_aromatic_bare_element("Na"), None);
        assert_eq!(
            parse_aromatic_bracket_element("b"),
            Some((Element::B, true, 1))
        );
        assert_eq!(parse_aromatic_bracket_element("te"), None);
        assert_eq!(parse_element_symbol("Xe"), Some((Element::Xe, 2)));
        assert_eq!(parse_element_symbol("C"), Some((Element::C, 1)));
        assert_eq!(parse_element_symbol("cl"), None);

        assert_eq!(bond_expr_rank(&BondExpr::Elided), (0, String::new()));
        assert_eq!(
            bond_expr_rank(&BondExpr::Query(BondExprTree::Primitive(
                BondPrimitive::Bond(Bond::Single)
            )))
            .0,
            1,
        );
        assert_eq!(
            bond_expr_rank(&BondExpr::Query(BondExprTree::Not(Box::new(
                BondExprTree::Primitive(BondPrimitive::Ring),
            ))))
            .0,
            2,
        );
        assert_eq!(
            bond_expr_rank(&BondExpr::Query(BondExprTree::HighAnd(vec![
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
                BondExprTree::Primitive(BondPrimitive::Ring),
            ])))
            .0,
            3,
        );
        assert_eq!(
            bond_expr_rank(&BondExpr::Query(BondExprTree::Or(vec![
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)),
            ])))
            .0,
            4,
        );
        assert_eq!(
            bond_expr_rank(&BondExpr::Query(BondExprTree::LowAnd(vec![
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single)),
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)),
            ])))
            .0,
            5,
        );
    }

    #[test]
    fn ring_label_writer_formats_all_ranges() {
        struct RingLabel(u32);
        impl fmt::Display for RingLabel {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write_ring_label(f, self.0)
            }
        }

        assert_eq!(RingLabel(9).to_string(), "9");
        assert_eq!(RingLabel(10).to_string(), "%10");
        assert_eq!(RingLabel(100).to_string(), "%(100)");
    }

    #[test]
    fn writer_handles_custom_orientation_groups_and_ring_labels() {
        let atoms = vec![
            atom(
                0,
                0,
                0,
                1,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            ),
            atom(
                1,
                0,
                2,
                3,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            ),
            atom(
                2,
                0,
                4,
                5,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            ),
            atom(
                3,
                1,
                6,
                7,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            ),
            atom(
                4,
                2,
                8,
                9,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            ),
        ];
        let bonds = vec![
            bond(0, 1, 0, 1, 2, BondExpr::Elided),
            bond(
                1,
                2,
                1,
                3,
                4,
                BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double))),
            ),
            bond(
                2,
                2,
                0,
                5,
                6,
                BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Ring)),
            ),
        ];
        let query = QueryMol::from_parts(atoms, bonds, 3, vec![Some(0), Some(0), None]);
        assert_eq!(query.component_group(0), Some(0));
        assert_eq!(query.component_group(2), None);
        assert_eq!(query.to_string(), "(C@1=CC@1.C).C");
    }

    #[test]
    fn covers_remaining_display_and_helper_variants() {
        assert_eq!(
            AtomPrimitive::Connectivity(Some(NumericQuery::Exact(7))).to_string(),
            "X7"
        );
        assert_eq!(
            AtomPrimitive::Hydrogen(HydrogenKind::Total, Some(NumericQuery::Exact(2))).to_string(),
            "H2"
        );
        assert_eq!(
            AtomPrimitive::Hydrogen(HydrogenKind::Implicit, Some(NumericQuery::Exact(3)))
                .to_string(),
            "h3"
        );
        assert_eq!(
            AtomPrimitive::RingSize(Some(NumericQuery::Exact(4))).to_string(),
            "r4"
        );
        assert_eq!(
            AtomPrimitive::RingConnectivity(Some(NumericQuery::Exact(2))).to_string(),
            "x2"
        );
        assert_eq!(
            AtomPrimitive::Hydrogen(
                HydrogenKind::Implicit,
                Some(NumericQuery::Range(NumericRange {
                    min: Some(1),
                    max: None,
                })),
            )
            .to_string(),
            "h{1-}"
        );
        assert_eq!(
            AtomPrimitive::Chirality(Chirality::TH(1)).to_string(),
            "@TH1",
        );
        assert_eq!(BondPrimitive::Bond(Bond::Triple).to_string(), "#");

        assert_eq!(
            parse_supported_bare_element("F"),
            Some((Element::F, false, 1))
        );
        assert_eq!(
            parse_supported_bare_element("I"),
            Some((Element::I, false, 1))
        );
        assert_eq!(
            parse_aromatic_bracket_element("o"),
            Some((Element::O, true, 1))
        );
        assert_eq!(
            parse_aromatic_bracket_element("p"),
            Some((Element::P, true, 1))
        );
        assert_eq!(
            parse_aromatic_bracket_element("s"),
            Some((Element::S, true, 1))
        );
        assert_eq!(
            parse_aromatic_bare_element("b"),
            Some((Element::B, true, 1))
        );
        assert_eq!(
            parse_aromatic_bare_element("o"),
            Some((Element::O, true, 1))
        );
        assert_eq!(
            parse_aromatic_bare_element("p"),
            Some((Element::P, true, 1))
        );
        assert_eq!(
            parse_aromatic_bare_element("s"),
            Some((Element::S, true, 1))
        );
    }

    #[test]
    fn writer_sorts_multiple_ring_bonds_by_rank_and_label() {
        let atoms = vec![
            atom(
                0,
                0,
                0,
                1,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            ),
            atom(
                1,
                0,
                2,
                3,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            ),
            atom(
                2,
                0,
                4,
                5,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            ),
            atom(
                3,
                0,
                6,
                7,
                AtomExpr::Bare {
                    element: Element::C,
                    aromatic: false,
                },
            ),
        ];
        let bonds = vec![
            bond(0, 0, 1, 0, 1, BondExpr::Elided),
            bond(1, 1, 2, 2, 3, BondExpr::Elided),
            bond(2, 2, 3, 4, 5, BondExpr::Elided),
            bond(
                3,
                0,
                2,
                6,
                7,
                BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Single))),
            ),
            bond(
                4,
                1,
                3,
                8,
                9,
                BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double))),
            ),
        ];
        let query = QueryMol::from_parts(atoms, bonds, 1, vec![None]);
        assert_eq!(query.to_string(), "C-1C=2C-1C=2");
    }
}
