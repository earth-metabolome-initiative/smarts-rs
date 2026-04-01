# SMARTS Implementation Study

## Bottom line

SMARTS is not just "SMILES with a few extra tokens".

The syntax is manageable, but a practical implementation becomes complex because SMARTS mixes:

- graph syntax from SMILES
- atom and bond predicate logic
- recursive subqueries
- ring and aromaticity semantics
- stereochemistry
- disconnected-component semantics
- reaction-specific matching rules

If the goal is a useful Rust implementation, the right starting point is a scoped subset of **molecule SMARTS**, not full Daylight + RDKit/CXSMARTS compatibility on day one.

## What the reference language contains

Daylight's SMARTS reference already includes the features that make implementation non-trivial:

- atom primitives like `D`, `H`, `h`, `R`, `r`, `v`, `X`, `x`, charge, atomic number, chirality, isotope/mass
- bond primitives like implicit single-or-aromatic, `-`, `=`, `#`, `:`, `~`, `@`, directional `/` and `\`
- logical operators with precedence: `!`, `&`, `,`, `;`
- recursive SMARTS: `$(SMARTS)`
- component-level grouping with zero-level parentheses: `(SMARTS).(SMARTS)`
- reaction SMARTS with `>` separators and atom-map semantics

The Daylight theory manual explicitly notes that reaction atom maps are global-scope constraints and are more expensive than normal incremental SMARTS matching. That is a good signal for project scope: **reaction SMARTS is a separate difficulty class**.

## What existing implementations actually do

### 1. RDKit

RDKit is the clearest example of a modern, production-quality SMARTS stack.

- Parsing is split into a lexer and grammar:
  - `Code/GraphMol/SmilesParse/smarts.ll`
  - `Code/GraphMol/SmilesParse/smarts.yy`
- The parser does not produce a "simple AST". It directly builds query atoms, query bonds, recursive query objects, ring closures, branch state, and chiral/query metadata.
- There is also non-trivial parse/post-parse work in `SmilesParseOps`:
  - close rings
  - set unspecified bond types
  - validate chirality
  - parse CXSMARTS/CXSMILES extensions

RDKit also documents that it supports most Daylight SMARTS, but not all of it. The explicitly missing pieces include:

- non-tetrahedral chiral classes
- `@?`
- explicit atomic masses
- component-level grouping requiring matches in different components, for example `(C).(C)`

At the same time, RDKit adds extensions:

- hybridization queries like `^2`
- range queries like `D{2-4}`
- quadruple bonds `$`
- dative bonds `<-` and `->`
- CXSMARTS/CXSMILES features
- reaction SMARTS

Important implementation signal: the RDKit book says recursive queries are not inherently thread-safe during matching unless guarded. That means even mature implementations end up with matcher-state complications.

### 2. Open Babel

Open Babel takes a different approach:

- parsing is largely hand-written in `src/parsmart.cpp`
- the internal representation is custom:
  - `AtomExpr`
  - `BondExpr`
  - `Pattern`
  - `AtomSpec`
  - `BondSpec`
- matching is custom backtracking code, not just a generic graph-isomorphism library call

The code shows the following design choices:

- recursive SMARTS is stored as nested `Pattern` objects
- explicit hydrogen handling may require matching on a hydrogen-added copy of the molecule
- chirality is filtered after the main structural match
- recursive SMARTS results are cached during evaluation
- there are separate paths for exhaustive matching vs fast single-match search

This is useful because it shows the real implementation burden: even with a hand-written parser, most of the work sits in query evaluation and search.

### 3. CDK

CDK shows a third style:

- the parser is hand-written in Java (`Smarts.java`)
- SMARTS compiles into expression trees (`Expr`) attached to query atoms and bonds
- `SmartsPattern` wraps the compiled query and plugs into a general substructure matcher

Two things matter here:

1. CDK exposes multiple SMARTS "flavors", including Daylight, CACTVS, MOE, OEChem, and CDK-specific behavior. This is a strong sign that "SMARTS" is really a family of dialects.
2. CDK has an explicit target-preparation step before matching:
   - ring perception
   - aromaticity perception

That separation is worth copying in Rust. It keeps parser logic independent from target-molecule perception.

## Complexity breakdown

### Syntax complexity: moderate

A parser for core molecule SMARTS is not the hardest part.

You need:

- SMILES-like graph parsing
- branch handling
- ring closures
- atom expressions with precedence
- bond expressions with precedence
- recursive SMARTS nesting

This is substantial, but tractable with:

- a hand-written recursive-descent parser plus Pratt/parser-precedence logic, or
- a parser generator

### Semantic complexity: high

This is where the project becomes real.

You need correct semantics for:

- aromatic vs aliphatic atoms/bonds
- implicit vs explicit H counts
- degree, valence, connectivity, ring membership, ring bond count
- bond defaults, especially implicit single-or-aromatic semantics
- chirality and directional bonds
- recursive SMARTS evaluation

These depend on the target molecule model, not just the query string.

### Matching complexity: high

A usable SMARTS engine needs more than parsing:

- candidate filtering
- backtracking subgraph search
- query-atom/query-bond predicate evaluation
- recursive SMARTS execution
- connected vs disconnected matching rules
- unique-match handling

This is where implementation quality will dominate runtime.

### Dialect complexity: very high

If you aim for "whatever users expect from SMARTS in RDKit/Open Babel/CDK", complexity jumps fast.

The biggest scope multipliers are:

- reaction SMARTS
- atom maps
- component-level grouping
- non-Daylight extensions
- CXSMARTS
- non-tetrahedral stereochemistry

## Practical scope recommendation for Rust

### Good first target

Implement **core molecule SMARTS**, not reaction SMARTS.

Recommended phase 1:

- atom primitives: `*`, `a`, `A`, `D`, `H`, `h`, `R`, `r`, `v`, `X`, `x`, charge, atomic number, isotope
- bond primitives: implicit, `-`, `=`, `#`, `:`, `~`, `@`
- logical operators: `!`, `&`, `,`, `;`
- branches and ring closures
- recursive SMARTS
- basic tetrahedral chirality only if your molecule model already supports it cleanly

Recommended deferrals:

- reaction SMARTS
- atom-map semantics
- component-level grouping across disconnected components
- CXSMARTS
- non-tetrahedral stereo
- toolkit-specific extensions like `^n`, ranges, dative bonds

### Suggested architecture

Use a three-layer design.

1. Parser
- Input: SMARTS string
- Output: query graph + atom/bond predicate trees

2. Target preparation
- ring membership
- aromaticity
- valence/connectivity/hydrogen-derived properties

3. Matcher
- candidate generation
- backtracking subgraph search
- predicate evaluation
- recursive query execution

This is closer to CDK/RDKit than to a "single parser function", and that is a good thing.

### Suggested internal representation

For Rust, a good representation would be:

- `QueryMol`
  - atoms: `Vec<QueryAtom>`
  - bonds: `Vec<QueryBond>`
- `QueryAtom`
  - base atom metadata
  - predicate tree: `AtomExpr`
- `QueryBond`
  - predicate tree: `BondExpr`
- `AtomExpr` / `BondExpr`
  - `True`
  - primitive predicate
  - `Not`
  - `And`
  - `Or`
  - for atoms: `Recursive(Box<QueryMol>)`

That maps well onto both the Daylight semantics and what RDKit/Open Babel/CDK already converged on.

## Testing strategy

Do not invent semantics from scratch.

Use differential testing against existing toolkits:

- Daylight examples/tutorial cases for reference behavior
- RDKit for a modern open-source baseline
- Open Babel and/or CDK to expose dialect mismatches

Build a corpus in layers:

1. parser acceptance/rejection tests
2. query-graph normalization tests
3. per-primitive semantic tests
4. substructure-match goldens
5. recursive SMARTS edge cases
6. aromaticity-sensitive cases
7. stereo-sensitive cases

## Recommended project framing

If you say "implement SMARTS in Rust", that sounds like one feature.
In practice, it is closer to building a small query language compiler plus a chemistry-aware subgraph engine.

Reasonable framing:

- **small project**: parser + query IR for a documented subset
- **medium project**: useful molecule SMARTS matcher with recursive SMARTS and ring/aromaticity support
- **large project**: broad compatibility with RDKit/Open Babel/CDK dialects
- **very large project**: reaction SMARTS and CXSMARTS compatibility

## Recommendation

The best next step is to target:

- **Daylight-like molecule SMARTS subset**
- **no reaction SMARTS**
- **no CXSMARTS**
- **no dialect extensions in v1 unless they are cheap and clearly isolated**

That is realistic and still valuable.

## Sources

- Daylight theory manual, SMARTS section:
  - https://daylight.com/dayhtml/doc/theory/theory.smarts.html
- Daylight SMARTS tutorial:
  - https://www.daylight.com/dayhtml_tutorials/languages/smarts/index.html
- RDKit book, SMARTS support/extensions and reaction SMARTS:
  - https://www.rdkit.org/docs/RDKit_Book.html
- RDKit SMARTS lexer:
  - https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smarts.ll
- RDKit SMARTS grammar:
  - https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smarts.yy
- RDKit parse operations:
  - https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/SmilesParseOps.h
- RDKit SMARTS tests:
  - https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smarts_catch_tests.cpp
- Open Babel SMARTS parser:
  - https://github.com/openbabel/openbabel/blob/master/src/parsmart.cpp
- Open Babel SMARTS parser header/internal structures:
  - https://github.com/openbabel/openbabel/blob/master/include/openbabel/parsmart.h
- CDK SMARTS parser/generator:
  - https://github.com/cdk/cdk/blob/main/tool/smarts/src/main/java/org/openscience/cdk/smarts/Smarts.java
- CDK SMARTS pattern wrapper:
  - https://github.com/cdk/cdk/blob/main/tool/smarts/src/main/java/org/openscience/cdk/smarts/SmartsPattern.java
- CDK CXSMARTS parser:
  - https://github.com/cdk/cdk/blob/main/tool/smarts/src/main/java/org/openscience/cdk/smarts/CxSmartsParser.java
- Rust Open Babel bindings:
  - https://docs.rs/openbabel/latest/openbabel/smartspattern/index.html
