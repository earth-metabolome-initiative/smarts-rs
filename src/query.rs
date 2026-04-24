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
    /// Hybridization predicate such as `^2`.
    Hybridization(NumericQuery),
    /// Hetero-neighbor count predicate such as `z2` or `z{1-}`.
    HeteroNeighbor(Option<NumericQuery>),
    /// Aliphatic-hetero-neighbor count predicate such as `Z2` or `Z{1-}`.
    AliphaticHeteroNeighbor(Option<NumericQuery>),
    /// Atom chirality predicate such as `@`, `@@`, or `@TH1`.
    Chirality(Chirality),
    /// Formal charge predicate such as `+`, `-`, or `+2`.
    Charge(i8),
}

impl AtomPrimitive {
    /// Returns the nested recursive SMARTS query, if this primitive stores one.
    #[inline]
    #[must_use]
    pub fn recursive_query(&self) -> Option<&QueryMol> {
        match self {
            Self::RecursiveQuery(query) => Some(query),
            _ => None,
        }
    }

    /// Returns the nested recursive SMARTS query mutably, if this primitive stores one.
    #[inline]
    #[must_use]
    pub fn recursive_query_mut(&mut self) -> Option<&mut QueryMol> {
        match self {
            Self::RecursiveQuery(query) => Some(query),
            _ => None,
        }
    }

    /// Replaces this primitive with one recursive SMARTS query.
    #[inline]
    pub fn set_recursive_query(&mut self, query: QueryMol) {
        *self = Self::RecursiveQuery(Box::new(query));
    }
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
    neighbors_by_atom: Vec<Vec<AtomId>>,
    incident_bonds_by_atom: Vec<Vec<BondId>>,
    atoms_by_component: Vec<Vec<AtomId>>,
    bonds_by_component: Vec<Vec<BondId>>,
}

type TopologyIndexes = (
    Vec<Vec<AtomId>>,
    Vec<Vec<BondId>>,
    Vec<Vec<AtomId>>,
    Vec<Vec<BondId>>,
);

impl QueryMol {
    /// Builds a parsed SMARTS query from already-lowered graph parts.
    #[inline]
    #[must_use]
    pub fn from_parts(
        atoms: Vec<QueryAtom>,
        bonds: Vec<QueryBond>,
        component_count: usize,
        component_groups: Vec<Option<ComponentGroupId>>,
    ) -> Self {
        let (neighbors_by_atom, incident_bonds_by_atom, atoms_by_component, bonds_by_component) =
            build_topology_indexes(&atoms, &bonds, component_count);
        Self {
            atoms,
            bonds,
            component_count,
            component_groups,
            neighbors_by_atom,
            incident_bonds_by_atom,
            atoms_by_component,
            bonds_by_component,
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

    /// Returns one atom by dense identifier.
    #[inline]
    #[must_use]
    pub fn atom(&self, atom_id: AtomId) -> Option<&QueryAtom> {
        self.atoms.get(atom_id)
    }

    /// Returns one bond by dense identifier.
    #[inline]
    #[must_use]
    pub fn bond(&self, bond_id: BondId) -> Option<&QueryBond> {
        self.bonds.get(bond_id)
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

    /// Returns the neighboring atom ids for one atom.
    #[inline]
    #[must_use]
    pub fn neighbors(&self, atom_id: AtomId) -> &[AtomId] {
        self.neighbors_by_atom
            .get(atom_id)
            .map_or(&[], Vec::as_slice)
    }

    /// Returns the incident bond ids for one atom.
    #[inline]
    #[must_use]
    pub fn incident_bonds(&self, atom_id: AtomId) -> &[BondId] {
        self.incident_bonds_by_atom
            .get(atom_id)
            .map_or(&[], Vec::as_slice)
    }

    /// Returns the graph degree for one atom.
    #[inline]
    #[must_use]
    pub fn degree(&self, atom_id: AtomId) -> usize {
        self.neighbors(atom_id).len()
    }

    /// Returns the dense bond id between two atoms, if present.
    #[must_use]
    pub fn bond_between(&self, a: AtomId, b: AtomId) -> Option<BondId> {
        self.incident_bonds(a).iter().copied().find(|&bond_id| {
            let Some(bond) = self.bonds.get(bond_id) else {
                return false;
            };
            (bond.src == a && bond.dst == b) || (bond.src == b && bond.dst == a)
        })
    }

    /// Returns the atom ids for one connected component.
    #[inline]
    #[must_use]
    pub fn component_atoms(&self, component_id: ComponentId) -> &[AtomId] {
        self.atoms_by_component
            .get(component_id)
            .map_or(&[], Vec::as_slice)
    }

    /// Returns the bond ids for one connected component.
    #[inline]
    #[must_use]
    pub fn component_bonds(&self, component_id: ComponentId) -> &[BondId] {
        self.bonds_by_component
            .get(component_id)
            .map_or(&[], Vec::as_slice)
    }

    #[must_use]
    pub(crate) fn cloned_parts(
        &self,
    ) -> (
        Vec<QueryAtom>,
        Vec<QueryBond>,
        usize,
        Vec<Option<ComponentGroupId>>,
    ) {
        (
            self.atoms.clone(),
            self.bonds.clone(),
            self.component_count,
            self.component_groups.clone(),
        )
    }

    /// Returns a target-independent estimate of exact matching cost.
    ///
    /// Larger values indicate SMARTS that are expected to require more matcher
    /// work: broad atom predicates, disconnected singleton components,
    /// negation, disjunction, recursive SMARTS, wildcard bonds, cycles, and
    /// branched query topology all increase the score. The value is a heuristic
    /// for scheduling or penalizing generated SMARTS, not a semantic property
    /// and not a replacement for benchmarking against a concrete target corpus.
    #[inline]
    #[must_use]
    pub fn complexity(&self) -> usize {
        query_complexity(self)
    }
}

fn query_complexity(query: &QueryMol) -> usize {
    let atom_score = query.atoms.iter().fold(0usize, |score, atom| {
        score.saturating_add(atom_expr_complexity(&atom.expr))
    });
    let bond_score = query.bonds.iter().fold(0usize, |score, bond| {
        score.saturating_add(bond_expr_complexity(&bond.expr))
    });

    1usize
        .saturating_add(query.atom_count().saturating_mul(4))
        .saturating_add(query.bond_count().saturating_mul(8))
        .saturating_add(atom_score)
        .saturating_add(bond_score)
        .saturating_add(query_topology_complexity(query))
}

fn query_topology_complexity(query: &QueryMol) -> usize {
    let mut score = 0usize;

    if query.component_count > 1 {
        score = score.saturating_add(square(query.component_count - 1).saturating_mul(12));
    }
    if query.bond_count() == 0 && query.atom_count() > 1 {
        score = score.saturating_add(square(query.atom_count()).saturating_mul(16));
    }

    let grouped_components = query
        .component_groups
        .iter()
        .filter(|group| group.is_some())
        .count();
    score = score
        .saturating_add(grouped_components.saturating_mul(8))
        .saturating_add(component_group_complexity(query));

    let cycle_rank = query
        .bond_count()
        .saturating_add(query.component_count)
        .saturating_sub(query.atom_count());
    score = score.saturating_add(
        cycle_rank
            .saturating_mul(16)
            .saturating_add(square(cycle_rank).saturating_mul(16)),
    );

    query
        .atoms
        .iter()
        .fold(score, |score, atom| {
            let degree = query.degree(atom.id);
            let atom_score = match degree {
                0 if query.atom_count() > 1 => 10,
                0 | 1 => 0,
                2 => 2,
                degree => square(degree - 2).saturating_mul(10),
            };
            score.saturating_add(atom_score)
        })
        .saturating_add(disconnected_singleton_complexity(query))
        .saturating_add(component_interaction_complexity(query))
}

const fn square(value: usize) -> usize {
    value.saturating_mul(value)
}

fn disconnected_singleton_complexity(query: &QueryMol) -> usize {
    if query.component_count <= 1 {
        return 0;
    }

    let mut singleton_count = 0usize;
    let mut singleton_expr_score = 0usize;
    for component_id in 0..query.component_count {
        let atoms = query.component_atoms(component_id);
        if atoms.len() != 1 || !query.component_bonds(component_id).is_empty() {
            continue;
        }

        singleton_count = singleton_count.saturating_add(1);
        let atom_id = atoms[0];
        singleton_expr_score =
            singleton_expr_score.saturating_add(atom_expr_complexity(&query.atoms[atom_id].expr));
    }

    if singleton_count < 2 {
        return 0;
    }

    let base = square(singleton_count).saturating_mul(24).saturating_add(
        singleton_expr_score
            .saturating_mul(singleton_count)
            .saturating_mul(2),
    );
    if query.bond_count() == 0 {
        base.saturating_mul(2)
    } else {
        base
    }
}

fn component_interaction_complexity(query: &QueryMol) -> usize {
    if query.component_count <= 2 || query.component_groups.iter().any(Option::is_some) {
        return 0;
    }

    let component_scores = (0..query.component_count)
        .map(|component_id| component_search_space_score(query, component_id))
        .collect::<Vec<_>>();
    let mut score = 0usize;
    for left in 0..component_scores.len() {
        for right in left + 1..component_scores.len() {
            score = score
                .saturating_add(component_scores[left].saturating_mul(component_scores[right]) / 7);
        }
    }

    score
}

fn component_search_space_score(query: &QueryMol, component_id: ComponentId) -> usize {
    let atoms = query.component_atoms(component_id);
    let bonds = query.component_bonds(component_id);
    let mut broad_atoms = 0usize;
    let atom_score = atoms.iter().fold(0usize, |score, &atom_id| {
        let atom_complexity = atom_expr_complexity(&query.atoms[atom_id].expr);
        if atom_complexity >= 8 {
            broad_atoms = broad_atoms.saturating_add(1);
        }
        score.saturating_add(atom_complexity)
    });
    let mut broad_bonds = 0usize;
    let bond_score = bonds.iter().fold(0usize, |score, &bond_id| {
        let bond_complexity = bond_expr_complexity(&query.bonds[bond_id].expr);
        if bond_complexity >= 12 {
            broad_bonds = broad_bonds.saturating_add(1);
        }
        score.saturating_add(bond_complexity)
    });

    let mut score = atom_score
        .saturating_add(bond_score)
        .saturating_add(atoms.len().saturating_mul(2))
        .saturating_add(bonds.len().saturating_mul(3));

    if atoms.len() == 1 && bonds.is_empty() && broad_atoms == 1 {
        score = score
            .saturating_add(64)
            .saturating_add(atom_score.saturating_mul(3));
    }
    score
        .saturating_add(square(broad_atoms).saturating_mul(16))
        .saturating_add(square(broad_bonds).saturating_mul(12))
        .max(1)
}

fn component_group_complexity(query: &QueryMol) -> usize {
    let mut score = 0usize;
    for component_id in 0..query.component_count {
        let Some(group_id) = query.component_group(component_id) else {
            continue;
        };
        if query
            .component_groups
            .iter()
            .take(component_id)
            .any(|group| *group == Some(group_id))
        {
            continue;
        }

        let mut group_size = 0usize;
        let mut singleton_count = 0usize;
        let mut group_expr_score = 0usize;
        let mut singleton_expr_score = 0usize;
        for grouped_component_id in 0..query.component_count {
            if query.component_group(grouped_component_id) != Some(group_id) {
                continue;
            }

            group_size = group_size.saturating_add(1);
            let atoms = query.component_atoms(grouped_component_id);
            for &atom_id in atoms {
                group_expr_score = group_expr_score
                    .saturating_add(atom_expr_complexity(&query.atoms[atom_id].expr));
            }
            if atoms.len() == 1 && query.component_bonds(grouped_component_id).is_empty() {
                singleton_count = singleton_count.saturating_add(1);
                singleton_expr_score = singleton_expr_score
                    .saturating_add(atom_expr_complexity(&query.atoms[atoms[0]].expr));
            }
        }

        if group_size > 1 {
            score = score
                .saturating_add(square(group_size).saturating_mul(64))
                .saturating_add(group_expr_score.saturating_mul(group_size));
        }
        if singleton_count > 0 {
            score = score
                .saturating_add(square(singleton_count).saturating_mul(128))
                .saturating_add(singleton_expr_score.saturating_mul(4));
        }
    }

    score
}

fn atom_expr_complexity(expr: &AtomExpr) -> usize {
    match expr {
        AtomExpr::Wildcard => 16,
        AtomExpr::Bare { aromatic, .. } => {
            if *aromatic {
                3
            } else {
                2
            }
        }
        AtomExpr::Bracket(expr) => {
            bracket_tree_complexity(&expr.tree).saturating_add(usize::from(expr.atom_map.is_some()))
        }
    }
}

fn bracket_tree_complexity(tree: &BracketExprTree) -> usize {
    match tree {
        BracketExprTree::Primitive(primitive) => atom_primitive_complexity(primitive),
        BracketExprTree::Not(inner) => match inner.as_ref() {
            BracketExprTree::Not(grandchild) => bracket_tree_complexity(grandchild),
            _ => 12usize.saturating_add(bracket_tree_complexity(inner).saturating_mul(2)),
        },
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => {
            items.iter().fold(items.len(), |score, item| {
                score.saturating_add(bracket_tree_complexity(item))
            })
        }
        BracketExprTree::Or(items) => items.iter().fold(
            8usize.saturating_add(square(items.len()).saturating_mul(4)),
            |score, item| score.saturating_add(bracket_tree_complexity(item)),
        ),
    }
}

fn atom_primitive_complexity(primitive: &AtomPrimitive) -> usize {
    match primitive {
        AtomPrimitive::Wildcard => 16,
        AtomPrimitive::AliphaticAny
        | AtomPrimitive::AromaticAny
        | AtomPrimitive::IsotopeWildcard(_)
        | AtomPrimitive::Chirality(_) => 10,
        AtomPrimitive::Symbol { aromatic, .. } => {
            if *aromatic {
                3
            } else {
                2
            }
        }
        AtomPrimitive::Isotope { .. } | AtomPrimitive::Charge(_) => 4,
        AtomPrimitive::AtomicNumber(_) => 3,
        AtomPrimitive::Degree(query)
        | AtomPrimitive::Connectivity(query)
        | AtomPrimitive::Valence(query)
        | AtomPrimitive::Hydrogen(_, query)
        | AtomPrimitive::RingMembership(query)
        | AtomPrimitive::RingSize(query)
        | AtomPrimitive::RingConnectivity(query)
        | AtomPrimitive::HeteroNeighbor(query)
        | AtomPrimitive::AliphaticHeteroNeighbor(query) => {
            6usize.saturating_add(optional_numeric_query_complexity(*query))
        }
        AtomPrimitive::Hybridization(query) => {
            6usize.saturating_add(numeric_query_complexity(*query))
        }
        AtomPrimitive::RecursiveQuery(query) => {
            32usize.saturating_add(query.complexity().saturating_mul(3))
        }
    }
}

fn optional_numeric_query_complexity(query: Option<NumericQuery>) -> usize {
    query.map_or(2, numeric_query_complexity)
}

const fn numeric_query_complexity(query: NumericQuery) -> usize {
    match query {
        NumericQuery::Exact(_) => 2,
        NumericQuery::Range(range) => 8usize
            .saturating_add(if range.min.is_none() { 4 } else { 0 })
            .saturating_add(if range.max.is_none() { 4 } else { 0 }),
    }
}

fn bond_expr_complexity(expr: &BondExpr) -> usize {
    match expr {
        BondExpr::Elided => 1,
        BondExpr::Query(tree) => bond_tree_complexity(tree),
    }
}

fn bond_tree_complexity(tree: &BondExprTree) -> usize {
    match tree {
        BondExprTree::Primitive(primitive) => bond_primitive_complexity(*primitive),
        BondExprTree::Not(inner) => match inner.as_ref() {
            BondExprTree::Not(grandchild) => bond_tree_complexity(grandchild),
            _ => 12usize.saturating_add(bond_tree_complexity(inner).saturating_mul(2)),
        },
        BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => {
            items.iter().fold(items.len(), |score, item| {
                score.saturating_add(bond_tree_complexity(item))
            })
        }
        BondExprTree::Or(items) => items.iter().fold(
            6usize.saturating_add(square(items.len()).saturating_mul(3)),
            |score, item| score.saturating_add(bond_tree_complexity(item)),
        ),
    }
}

const fn bond_primitive_complexity(primitive: BondPrimitive) -> usize {
    match primitive {
        BondPrimitive::Any => 14,
        BondPrimitive::Ring => 8,
        BondPrimitive::Bond(Bond::Up | Bond::Down) => 4,
        BondPrimitive::Bond(Bond::Single | Bond::Double | Bond::Triple | Bond::Aromatic) => 2,
        BondPrimitive::Bond(Bond::Quadruple) => 3,
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
            Self::Hybridization(query) => write_required_numeric_query_suffix(f, "^", *query),
            Self::HeteroNeighbor(query) => write_numeric_query_suffix(f, "z", *query),
            Self::AliphaticHeteroNeighbor(query) => write_numeric_query_suffix(f, "Z", *query),
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

fn write_required_numeric_query_suffix(
    f: &mut fmt::Formatter<'_>,
    prefix: &str,
    query: NumericQuery,
) -> fmt::Result {
    f.write_str(prefix)?;
    match query {
        NumericQuery::Exact(value) => write!(f, "{value}"),
        NumericQuery::Range(range) => write_numeric_range(f, range),
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

fn build_topology_indexes(
    atoms: &[QueryAtom],
    bonds: &[QueryBond],
    component_count: usize,
) -> TopologyIndexes {
    let mut neighbors_by_atom = vec![Vec::new(); atoms.len()];
    let mut incident_bonds_by_atom = vec![Vec::new(); atoms.len()];
    let mut atoms_by_component = vec![Vec::new(); component_count];
    let mut bonds_by_component = vec![Vec::new(); component_count];

    for atom in atoms {
        if atom.component < atoms_by_component.len() {
            atoms_by_component[atom.component].push(atom.id);
        }
    }

    for bond in bonds {
        if let Some(incident) = incident_bonds_by_atom.get_mut(bond.src) {
            incident.push(bond.id);
        }
        if bond.dst != bond.src {
            if let Some(incident) = incident_bonds_by_atom.get_mut(bond.dst) {
                incident.push(bond.id);
            }
        }

        if let Some(neighbors) = neighbors_by_atom.get_mut(bond.src) {
            neighbors.push(bond.dst);
        }
        if bond.dst != bond.src {
            if let Some(neighbors) = neighbors_by_atom.get_mut(bond.dst) {
                neighbors.push(bond.src);
            }
        }

        let Some(src_atom) = atoms.get(bond.src) else {
            continue;
        };
        let Some(dst_atom) = atoms.get(bond.dst) else {
            continue;
        };
        if src_atom.component == dst_atom.component && src_atom.component < bonds_by_component.len()
        {
            bonds_by_component[src_atom.component].push(bond.id);
        }
    }

    for neighbors in &mut neighbors_by_atom {
        neighbors.sort_unstable();
    }
    for incident in &mut incident_bonds_by_atom {
        incident.sort_unstable();
    }
    for atom_ids in &mut atoms_by_component {
        atom_ids.sort_unstable();
    }
    for bond_ids in &mut bonds_by_component {
        bond_ids.sort_unstable();
    }

    (
        neighbors_by_atom,
        incident_bonds_by_atom,
        atoms_by_component,
        bonds_by_component,
    )
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
        let (parent_bond_by_atom, component_roots) = build_spanning_forest(mol);
        let component_roots = component_roots
            .into_iter()
            .map(|root| root.expect("each component must have a root atom"))
            .collect::<Vec<_>>();

        let mut children_by_atom = vec![Vec::new(); mol.atom_count()];
        for (atom_id, parent_bond) in parent_bond_by_atom.iter().copied().enumerate() {
            if let Some(parent_bond) = parent_bond {
                let bond = &mol.bonds[parent_bond];
                let parent_atom = if bond.src == atom_id {
                    bond.dst
                } else {
                    bond.src
                };
                children_by_atom[parent_atom].push(atom_id);
            }
        }
        for children in &mut children_by_atom {
            children.sort_unstable();
        }

        let mut is_parent_bond = vec![false; mol.bond_count()];
        for parent_bond in parent_bond_by_atom.iter().flatten().copied() {
            is_parent_bond[parent_bond] = true;
        }

        let ring_tokens_by_atom = build_ring_tokens_by_atom(mol, &is_parent_bond);
        let top_level_entries = build_top_level_entries(mol, &component_roots);

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
        if self.requires_parenthesized_main_child(atom_id, main_child) {
            f.write_str("(")?;
            self.write_bond_to_child(main_child, f)?;
            self.write_atom_with_subtree(main_child, f)?;
            f.write_str(")")?;
            return Ok(());
        }

        self.write_bond_to_child(main_child, f)?;
        self.write_atom_with_subtree(main_child, f)
    }

    fn write_bond_to_child(&self, child_id: AtomId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bond_id = self.parent_bond_by_atom[child_id].expect("child must have a parent bond");
        let bond = &self.mol.bonds[bond_id];
        write!(f, "{}", bond.expr)
    }

    fn requires_parenthesized_main_child(&self, atom_id: AtomId, child_id: AtomId) -> bool {
        let bond_id = self.parent_bond_by_atom[child_id].expect("child must have a parent bond");
        let bond = &self.mol.bonds[bond_id];
        if !matches!(bond.expr, BondExpr::Elided) {
            return false;
        }

        let parent_text = self.mol.atoms[atom_id].expr.to_string();
        let child_text = self.mol.atoms[child_id].expr.to_string();
        let parent_width = parent_text.len();
        let combined = parent_text + &child_text;

        parse_supported_bare_element(&combined).is_some_and(|(_, _, width)| width > parent_width)
    }

    fn write_ring_tokens(&self, atom_id: AtomId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for token in &self.ring_tokens_by_atom[atom_id] {
            write!(f, "{}", token.expr)?;
            write_ring_label(f, token.label)?;
        }
        Ok(())
    }
}

fn build_spanning_forest(mol: &QueryMol) -> (Vec<Option<BondId>>, Vec<Option<AtomId>>) {
    let mut component_roots = vec![None; mol.component_count()];
    let mut parent_bond_by_atom = vec![None; mol.atom_count()];
    let mut bond_ids_by_atom = vec![Vec::new(); mol.atom_count()];
    for bond in &mol.bonds {
        if bond.src < bond_ids_by_atom.len() {
            bond_ids_by_atom[bond.src].push(bond.id);
        }
        if bond.dst < bond_ids_by_atom.len() && bond.dst != bond.src {
            bond_ids_by_atom[bond.dst].push(bond.id);
        }
    }
    for (atom_id, bond_ids) in bond_ids_by_atom.iter_mut().enumerate() {
        bond_ids.sort_by_key(|&bond_id| {
            let bond = &mol.bonds[bond_id];
            (
                if bond.src == atom_id {
                    bond.dst
                } else {
                    bond.src
                },
                bond_expr_rank(&bond.expr),
                bond.id,
            )
        });
    }

    let mut visited = vec![false; mol.atom_count()];
    for atom in &mol.atoms {
        if visited[atom.id] {
            continue;
        }

        if component_roots[atom.component].is_none() {
            component_roots[atom.component] = Some(atom.id);
        }

        visited[atom.id] = true;
        let mut stack = vec![atom.id];
        while let Some(current_atom) = stack.pop() {
            for &bond_id in bond_ids_by_atom[current_atom].iter().rev() {
                let bond = &mol.bonds[bond_id];
                let next_atom = if bond.src == current_atom {
                    bond.dst
                } else {
                    bond.src
                };

                if next_atom >= visited.len() || visited[next_atom] {
                    continue;
                }

                visited[next_atom] = true;
                parent_bond_by_atom[next_atom] = Some(bond_id);
                stack.push(next_atom);
            }
        }
    }

    (parent_bond_by_atom, component_roots)
}

fn build_ring_tokens_by_atom(mol: &QueryMol, is_parent_bond: &[bool]) -> Vec<Vec<RingToken>> {
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

    ring_tokens_by_atom
}

fn build_top_level_entries(mol: &QueryMol, component_roots: &[AtomId]) -> Vec<TopLevelEntry> {
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
    top_level_entries
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
            (
                AtomPrimitive::Hybridization(NumericQuery::Exact(2)).to_string(),
                "^2",
            ),
            (AtomPrimitive::HeteroNeighbor(None).to_string(), "z"),
            (
                AtomPrimitive::AliphaticHeteroNeighbor(None).to_string(),
                "Z",
            ),
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
        assert_eq!(query.to_string(), "(C(C=1)@C=1.C).C");
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
            AtomPrimitive::Hybridization(NumericQuery::Exact(3)).to_string(),
            "^3"
        );
        assert_eq!(
            AtomPrimitive::HeteroNeighbor(Some(NumericQuery::Exact(2))).to_string(),
            "z2"
        );
        assert_eq!(
            AtomPrimitive::AliphaticHeteroNeighbor(Some(NumericQuery::Exact(2))).to_string(),
            "Z2"
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
        assert_eq!(query.to_string(), "C(C1=C2)-C12");
    }
}
