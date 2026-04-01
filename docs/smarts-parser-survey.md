# SMARTS Parser Survey

## Purpose

This note looks specifically at how major toolkits implement SMARTS parsing and where they have historically run into trouble.

The focus is narrower than the earlier implementation study:

- parser architecture
- parser-adjacent normalization
- parser-specific or parser-triggered bug clusters
- thread-safety implications
- lessons for a Rust `smarts-parser`

The three main reference implementations here are:

- RDKit
- Open Babel
- CDK

## Bottom line

The surface SMARTS grammar is not the main reason these implementations are large.

The real size comes from the fact that parsing is rarely "just parsing". In all three libraries, the parser is coupled to one or more of these:

- query-graph construction
- hydrogen normalization
- recursive SMARTS bookkeeping
- ring-closure bookkeeping
- stereo interpretation
- aromaticity and ring-preparation requirements
- dialect selection
- error-reporting and thread-safety constraints

That coupling is where complexity and bugs accumulate.

The most important design lesson for Rust is:

- keep the parser responsible for syntax, spans, and immutable query IR
- keep target-molecule preparation and match-time caches out of the parser
- do not hide semantic rewrites behind parse-time convenience flags unless they are clearly isolated

## RDKit

### Relevant files

- `Code/GraphMol/SmilesParse/smarts.ll`
- `Code/GraphMol/SmilesParse/smarts.yy`
- `Code/GraphMol/SmilesParse/SmilesParse.h`
- `Code/GraphMol/SmilesParse/SmilesParse.cpp`

### Architecture

RDKit uses a classic lexer/parser-generator stack:

- `smarts.ll` is the lexer
- `smarts.yy` is the Bison grammar
- the semantic actions in the grammar build the query molecule directly
- `SmilesParse.cpp` wraps the generated parser and performs additional preprocessing and postprocessing

This is not an AST-first design.

The grammar actions allocate and connect query atoms and query bonds while parsing. They also handle:

- branches
- ring closures
- bond defaults
- recursive SMARTS
- chiral markers
- atom-expression and bond-expression trees

So the parser is already acting as:

- a syntax recognizer
- a graph builder
- a semantic normalizer

### What is parser-specific in RDKit

#### 1. Stateful lexer

The lexer in `smarts.ll` uses explicit modes such as:

- `IN_ATOM_STATE`
- `IN_BRANCH_STATE`
- `IN_RECURSION_STATE`

That is a signal that token meaning is context-sensitive enough that a flat tokenizer is not sufficient.

One especially telling comment is the workaround around `]` handling for recursive SMARTS. The lexer includes a note that the solution used for bracket handling inside recursion may not work in all cases. That is exactly the kind of localized fix that appears when parser state and SMARTS nesting rules interact in awkward ways.

#### 2. Grammar actions build chemistry-aware queries directly

In `smarts.yy`, reductions do not just build syntax nodes. They:

- add atoms to an `RWMol`
- add bonds to the current query graph
- manage bookmarks for ring closures
- attach atom and bond query predicates
- track active atoms across branch nesting

This makes parsing efficient, but it also means parse-time bugs and query-construction bugs are tightly coupled.

#### 3. Recursive SMARTS preprocessing

`SmilesParse.cpp` contains `labelRecursivePatterns()`, which rewrites recursive SMARTS before the parser consumes them.

That is important. It means the parser entrypoint already assumes that nested SMARTS needs an extra normalization pass. This is not "pure grammar parsing"; it is grammar plus preprocessing to make recursive queries manageable.

#### 4. Parse-time behavior is controlled by semantic flags

`SmartsParserParams` includes options such as:

- `allowCXSMILES`
- `strictCXSMILES`
- `parseName`
- `mergeHs`
- `skipCleanup`
- `debugParse`

This is another sign that the parse entrypoint is doing more than syntax.

The most consequential flag here is `mergeHs`, because it changes the resulting query semantics after parse.

### Where RDKit has found problems

#### 1. Hydrogen handling is a persistent bug source

The grammar itself contains comments calling hydrogen handling "ugliness". That is not cosmetic commentary. It reflects a long-running source of ambiguity:

- `[H]` can mean an actual hydrogen atom in some contexts
- `H` inside bracket expressions can also mean hydrogen-count predicates
- atom lists containing hydrogen interact badly with normalization
- recursive SMARTS can complicate hydrogen merging

RDKit has a visible bug trail around `mergeQueryHs()` and hydrogen-in-list behavior:

- issue #544: recursive SMARTS and `mergeQueryHs()`
- issue #557: existing H query interaction
- issue #558: OR-query interaction
- issue #8071: SMARTS atom-list parsing problems
- issue #8072: hydrogen parsing in atom lists
- issue #8073: hydrogen in atom lists with `mergeQueryHs()`
- issue #8362: invalid SMARTS parsing with hydrogens in OR clauses

The common pattern is simple: a parse result that looks syntactically fine becomes unstable once the toolkit applies post-parse hydrogen normalization.

#### 2. Recursive SMARTS is not "just nesting"

RDKit has both:

- recursive SMARTS relabeling before parse
- explicit lexer state for recursion

The grammar comments also note possible leakage concerns around recursive-query handling.

That tells you recursive SMARTS is a structural feature with real implementation weight, not a small grammar extension.

#### 3. Thread safety is affected by parser-global state

`MolFromSmarts()` is documented as generally safe in multithreaded contexts except when `debugParse` differs across threads. The reason is that parser debugging uses a global `yysmarts_debug`.

That is a narrow example, but it matters. It shows how parser-generator-era global state can leak into the public thread-safety story.

RDKit also has a separate history of thread-safety concerns in substructure matching, especially from Python-facing use, which reinforces the same design lesson: immutable query objects and per-call runtime state are safer than shared mutable machinery.

#### 4. Parser settings expand into API complexity

Issue #4905 asked for parser-setting control on `ReactionFromSmarts()`. That is the normal consequence of a parser entrypoint growing more semantic knobs over time: eventually higher-level APIs need those knobs too.

That is an argument for keeping the Rust parser surface narrow and explicit.

### RDKit lessons

- A generated grammar is viable.
- Stateful lexing is unavoidable for full SMARTS.
- Building query graphs directly during parse is workable, but it couples syntax and semantics tightly.
- Hydrogen normalization is one of the highest-risk areas.
- Recursive SMARTS deserves a first-class representation, not an afterthought.
- Global parser flags are a thread-safety footgun.

## Open Babel

### Relevant files

- `include/openbabel/parsmart.h`
- `src/parsmart.cpp`

### Architecture

Open Babel uses a hand-written parser rather than a lexer/parser-generator pair.

The parser is built around:

- a character cursor
- recursive parsing functions
- manual operator-precedence handling
- a custom internal query representation

Key internal types include:

- `Pattern`
- `AtomExpr`
- `BondExpr`
- `AtomSpec`
- `BondSpec`
- `ParseState`

This implementation is older in style than RDKit's parser, but it is very revealing because the control flow is explicit.

### What is parser-specific in Open Babel

#### 1. Operator precedence is implemented manually

`ParseAtomExpr(int level)` and `ParseBondExpr(int level)` handle precedence by recursion over explicit levels:

- low-precedence `;`
- disjunction `,`
- high-precedence `&` and implicit conjunction
- unary negation and primitives

This is effectively a hand-written Pratt or precedence parser, except encoded manually.

It is compact, but it also means expression bugs are easy to localize directly to parser logic.

#### 2. Ring closures are handled with explicit parser state

`ParseState` keeps ring closure information in fixed arrays, including fields such as:

- `closord[100]`
- `closure[100]`

That reflects a very direct implementation strategy: parse first, keep explicit mutable closure state, validate later.

There is also `EquivalentBondExpr()`, used to decide whether two sides of a ring closure are consistent. The implementation explicitly distinguishes patterns that should be rejected, such as mismatched ring bond specifications.

#### 3. `[H]` is a parser-level special case

The parser treats bracket hydrogen specially. It also tracks whether the pattern contains explicit hydrogen through `pat->hasExplicitH`.

That parser result later changes matcher behavior: if the pattern has explicit hydrogen, Open Babel can construct a hydrogen-added copy of the target molecule before matching.

This is a good example of parser output carrying semantic triggers that force target mutation or target adaptation later.

#### 4. Recursive SMARTS is embedded as nested patterns

Recursive SMARTS does not become a neutral syntax node. It becomes nested `Pattern` data that is evaluated later with cache support.

That makes the recursive feature operationally expensive and structurally important.

### Where Open Babel has found problems

#### 1. Thread safety was a real historical issue

Issue #1524 discusses thread-safe SMARTS matching and points to older shared-state behavior. The header also makes a point of describing the newer API as "more thread safe".

That wording matters. It usually means the original design had parser or matcher state stored in places that were too shared or too mutable.

#### 2. Stereo support has been a recurring weak point

Open Babel has a long issue trail around SMARTS stereo:

- issue #553: chirality SMARTS problems
- issue #554: cis/trans SMARTS problems
- issue #578: mixed meanings of bond direction flags
- issue #919: valid cis/trans SMARTS rejected
- issue #1479: incomplete or incorrect cis/trans SMARTS support

One parser-adjacent detail is especially revealing: `ParseBondPrimitive()` maps `/` and `\` into a generic single-bond expression. That means directional syntax is not preserved as a rich bond-query primitive at parse time. If later stereo logic needs more structure, it must recover that meaning from reduced information or side channels.

That is a strong warning for a Rust parser: do not erase stereo-relevant syntax too early.

#### 3. Unsupported SMARTS features cause ambiguity unless rejected clearly

Issue #1403 covers SMARTS component-level grouping, which Open Babel does not support. The lesson is not just "some features are missing"; it is that unsupported features need an explicit parse or validation policy.

If a feature is outside scope, one of these must happen:

- reject it cleanly during parse
- retain it in the IR but mark it unsupported during validation

Silent approximation is the worst option.

#### 4. Valid patterns have historically been rejected or crashed older code paths

Issues such as #971 and #443 show two common failure modes:

- valid SMARTS rejected because the parser or its assumptions were too narrow
- invalid or unsupported SMARTS leading to bad runtime behavior instead of structured errors

That is the downside of a hand-written pointer parser with tight coupling to mutable internal data structures: error recovery and diagnostics are harder to keep disciplined.

### Open Babel lessons

- A hand-written parser is entirely viable.
- Manual precedence logic is manageable for SMARTS.
- Fixed parser-side closure state is simple, but brittle.
- Special-casing `[H]` early has large downstream semantic consequences.
- Stereo syntax should remain explicit in the query IR.
- Unsupported features must be rejected deliberately.
- Thread-safe wrappers are harder than starting from immutable query data.

## CDK

### Relevant files

- `tool/smarts/src/main/java/org/openscience/cdk/smarts/Smarts.java`
- `tool/smarts/src/main/java/org/openscience/cdk/smarts/SmartsPattern.java`
- `base/isomorphism/src/main/java/org/openscience/cdk/isomorphism/DfPattern.java`
- `base/isomorphism/src/main/java/org/openscience/cdk/isomorphism/Pattern.java`

### Architecture

CDK uses a hand-written parser like Open Babel, but the surrounding architecture is cleaner and more modular.

The main split is:

- `Smarts.java` parses SMARTS into a query graph with expression objects
- `SmartsPattern` prepares targets and runs matching
- `DfPattern` and related classes implement the graph-matching engine

This is the clearest example of separating:

- query parsing
- target preparation
- matching

### What is parser-specific in CDK

#### 1. The parser supports multiple SMARTS flavors

`Smarts.java` includes flavor flags such as:

- `FLAVOR_DAYLIGHT`
- `FLAVOR_CACTVS`
- `FLAVOR_MOE`
- `FLAVOR_OECHEM`
- `FLAVOR_CDK`
- `FLAVOR_CDK_LEGACY`

This is useful, but it is also a warning. Supporting multiple dialects pushes complexity into the parser because syntax and semantics are no longer fixed.

#### 2. The parser exposes richer syntax than strict Daylight SMARTS

CDK includes parsing support for features like:

- ranges such as `{2-4}`
- inequality-like forms such as `>`, `<`

That expands expressiveness, but it also increases the surface that later preparation and matching code must understand.

#### 3. Error handling had to evolve for thread safety

`Smarts.java` still contains legacy static error reporting through `lastError`, but also provides `parseToResult(...)` as the thread-safe path returning a `SmartsResult`.

This is an extremely clear lesson for Rust. Even if the parse itself is structurally thread-safe, any shared mutable error sink can damage the API story.

### Where CDK has found problems

#### 1. Ring semantics are a real correctness boundary

Issue #926 deals with bond expressions involving `@`, `!@`, and related combinations.

Issue #1271 covers incorrect ring-membership detection for patterns like `[r5]`.

These are not superficial matcher bugs. They show that certain SMARTS primitives depend on well-defined ring perception and on precise interpretation of ring-related bond predicates.

#### 2. Parser and target-preparation logic must stay synchronized

`SmartsPattern` computes which preparation steps are required by the query and then applies operations such as:

- ring marking
- Daylight aromaticity

That is a strong design, but it creates a maintenance obligation: if the parser learns a new query expression, the preparation logic must also know whether that expression requires rings, aromaticity, stereo prep, or something else.

The code even includes a defensive failure mode for unknown expression types, which is a sign the maintainers know this boundary is fragile.

#### 3. Flavor support creates semantic drift pressure

Once one parser supports multiple SMARTS dialects, the problem is no longer just syntax. The parser and matcher become a negotiation layer across several incompatible expectations.

That grows the implementation even when each individual feature seems small.

### CDK lessons

- Clean separation of parser, prep, and matcher is worth copying.
- A thread-safe parse-result API is preferable to global error state.
- Supporting multiple flavors increases parser complexity fast.
- Ring and aromaticity semantics should be treated as explicit query requirements.
- Query-expression space and preparation logic must evolve together.

## Recurring problem clusters across toolkits

The same kinds of pain appear in all three libraries.

### 1. Hydrogen semantics

This is the single most obvious recurring trouble spot.

The hard cases are:

- `[H]` as atomic hydrogen versus hydrogen-count semantics
- hydrogen inside atom lists
- hydrogen combined with disjunctions
- hydrogen merged or normalized after parse
- recursive SMARTS that contain hydrogen-sensitive subqueries

If the Rust parser tries to "simplify" these too early, it will likely recreate the same bugs.

### 2. Recursive SMARTS

`$(...)` is structurally expensive.

It introduces:

- nested query graphs
- recursive parse structure
- matcher re-entry
- caching requirements
- more complicated diagnostics

All three toolkits treat recursive SMARTS as a first-class complication.

### 3. Ring closure and ring-property handling

Ring digits are syntactically simple, but semantically they trigger:

- deferred bond construction
- bond-spec compatibility checks
- ring membership constraints
- ring-size interpretation
- aromaticity interactions

The parser cannot solve all of that, but it must preserve enough information for later stages to solve it correctly.

### 4. Stereo and directional bonds

Directional bonds and stereochemical markers are easy to tokenize and hard to preserve correctly.

The consistent lesson from Open Babel and RDKit issue history is that flattening or overloading these signals too early leads to long-lived bugs.

### 5. Shared mutable state

The problem is not only matching caches.

Shared mutable state shows up in:

- parser debug flags
- static error sinks
- recursive-query bookkeeping
- query objects that accumulate runtime state

If the goal is a thread-safe Rust implementation, this must be designed out early.

### 6. Dialect creep

Every implementation eventually accumulates extra SMARTS-like features.

That creates pressure to expand:

- lexer/token set
- parser grammar
- query IR
- target preparation
- matcher semantics
- public parser parameters

This is why a narrow v1 scope matters.

## What this implies for a Rust `smarts-parser`

### 1. The parser should produce immutable query IR, not mutable runtime objects

The parser should return a stable data structure that is safe to share across threads.

That IR should preserve:

- atom-expression structure
- bond-expression structure
- branch structure as graph topology
- ring closure intent
- recursive SMARTS subqueries
- source spans

It should not contain:

- target-specific caches
- mutable match-time fields
- global parser configuration state

### 2. Keep parse-time rewrites to a minimum

If the parser silently rewrites the user's SMARTS into a "simpler" query, those rewrites become a bug magnet.

The safer rule is:

- parse faithfully
- lower into a clear IR
- perform any later normalization in explicitly named passes

Hydrogen rewriting is the clearest example of why this matters.

### 3. Treat recursive SMARTS as an explicit IR node

Do not encode recursive SMARTS as an opaque string after parse.

Instead, compile it into:

- a nested `QueryMol`, or
- an interned query identifier into a query arena

That makes both diagnostics and later matcher design cleaner.

### 4. Preserve stereo-relevant tokens explicitly

Do not reduce `/`, `\`, `@`, `@@`, and related syntax into generic bond or atom categories too early.

Even if v1 validation does not fully support all stereo forms, the parser should preserve the information faithfully so later phases do not need a parser rewrite.

### 5. Reject unsupported features clearly

If v1 excludes:

- reaction SMARTS
- component-level grouping
- CXSMARTS
- dialect-specific extensions

the parser should report that clearly and precisely, ideally with spans.

That is better than partial acceptance with ambiguous semantics.

### 6. Keep parser errors local and structured

Do not use shared global error state.

Return structured diagnostics that include:

- message
- span
- expected-versus-found detail where possible
- unsupported-feature classification when relevant

### 7. Separate parser from target preparation

CDK's split is the right model here.

The parser should not be responsible for:

- aromaticity perception
- ring perception on targets
- explicit-hydrogen expansion of target molecules
- recursive-match caching

Those belong in `smarts-validator`.

## Concrete recommendations for phase 1

### Parser scope

Implement first:

- SMILES-like atom and bond graph syntax
- bracket atom expressions
- bond expressions
- precedence for `!`, `&`, `,`, `;`
- branches
- ring closures
- recursive SMARTS
- spans and diagnostics

Defer initially:

- reaction SMARTS
- zero-level component grouping
- CXSMARTS
- flavor switches
- range and inequality extensions

### Internal model

Prefer these layers:

1. tokens
2. parse result with spans
3. immutable query IR

If a CST is added, it should serve diagnostics and testing, not become the primary runtime representation.

### Testing priorities

The test suite should concentrate early on the places existing toolkits have struggled:

- bracket hydrogen forms
- hydrogen in atom lists
- recursive SMARTS nesting
- ring closure bond consistency
- directional bond syntax
- unsupported feature rejection
- span accuracy for nested expressions

## Sources

### RDKit

- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smarts.ll`
- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smarts.yy`
- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/SmilesParse.h`
- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/SmilesParse.cpp`
- `https://github.com/rdkit/rdkit/issues/544`
- `https://github.com/rdkit/rdkit/issues/557`
- `https://github.com/rdkit/rdkit/issues/558`
- `https://github.com/rdkit/rdkit/issues/629`
- `https://github.com/rdkit/rdkit/issues/1489`
- `https://github.com/rdkit/rdkit/issues/4905`
- `https://github.com/rdkit/rdkit/issues/8071`
- `https://github.com/rdkit/rdkit/issues/8072`
- `https://github.com/rdkit/rdkit/issues/8073`
- `https://github.com/rdkit/rdkit/issues/8362`

### Open Babel

- `https://github.com/openbabel/openbabel/blob/master/include/openbabel/parsmart.h`
- `https://github.com/openbabel/openbabel/blob/master/src/parsmart.cpp`
- `https://github.com/openbabel/openbabel/issues/443`
- `https://github.com/openbabel/openbabel/issues/553`
- `https://github.com/openbabel/openbabel/issues/554`
- `https://github.com/openbabel/openbabel/issues/578`
- `https://github.com/openbabel/openbabel/issues/919`
- `https://github.com/openbabel/openbabel/issues/971`
- `https://github.com/openbabel/openbabel/issues/1403`
- `https://github.com/openbabel/openbabel/issues/1479`
- `https://github.com/openbabel/openbabel/issues/1524`

### CDK

- `https://github.com/cdk/cdk/blob/main/tool/smarts/src/main/java/org/openscience/cdk/smarts/Smarts.java`
- `https://github.com/cdk/cdk/blob/main/tool/smarts/src/main/java/org/openscience/cdk/smarts/SmartsPattern.java`
- `https://github.com/cdk/cdk/blob/main/base/isomorphism/src/main/java/org/openscience/cdk/isomorphism/DfPattern.java`
- `https://github.com/cdk/cdk/blob/main/base/isomorphism/src/main/java/org/openscience/cdk/isomorphism/Pattern.java`
- `https://github.com/cdk/cdk/issues/926`
- `https://github.com/cdk/cdk/issues/1271`
