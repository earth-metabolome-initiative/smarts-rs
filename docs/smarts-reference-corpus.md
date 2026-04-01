# SMARTS Reference Corpus

## Purpose

This note collects public SMARTS sources that are useful for test-driven development.

The goal is not to find one perfect dataset.
The goal is to build a layered reference corpus with:

- parse-valid cases
- parse-invalid cases
- match gold sets
- difficult semantic cases
- known cross-tool divergences

For a SMARTS implementation, this is more important than a large random corpus.
The hard bugs are not volume bugs. They are edge-case bugs.

## Short answer on PubChem

Yes, PubChem has usable SMARTS material, but not the thing we most want.

What PubChem clearly provides:

- PubChem structure search accepts SMARTS input
- the official PubChem Substructure Fingerprint specification includes SMARTS-pattern sections

What PubChem does **not** appear to provide:

- a public SMARTS parser conformance suite
- a public matcher gold-set for difficult SMARTS
- a curated corpus focused on recursive SMARTS, hydrogen edge cases, stereo, or component grouping

So PubChem is useful as a **positive pattern source**, but not sufficient as the main TDD reference collection.

## Best public sources

### 1. RDKit test suite

Primary file:

- `Code/GraphMol/SmilesParse/smatest.cpp`

What it gives:

- valid parse cases
- invalid parse cases
- roundtrip cases
- serialization stability cases
- many target-based matching cases
- regression cases from real issues

Why it matters:

- it contains many of the difficult parser-adjacent features in one place
- it is maintained by a toolkit with broad SMARTS coverage
- it includes cases that only show up after years of production use

High-value properties:

- recursive SMARTS
- hydrogen-sensitive queries
- ring queries
- directional bond stereo
- chirality
- high ring-closure numbers
- atom maps
- unsupported versus extended syntax boundaries

Weakness:

- it is embedded in C++ tests, not published as a clean flat dataset
- it includes RDKit extensions, so not every valid RDKit case should be treated as core SMARTS

### 2. Open Babel test corpus

Primary files:

- `test/files/validsmarts.txt`
- `test/files/invalid-smarts.txt`
- `test/files/smartstest.txt`
- `test/files/smartsresults.txt`
- `test/smartsparse.cpp`
- `test/smartstest.cpp`

What it gives:

- a flat valid-parse corpus
- a flat invalid-parse corpus
- a flat SMARTS pattern list for matching
- a gold-set of match results against a fixed molecule panel

Why it matters:

- this is the cleanest machine-readable starter corpus among the toolkits examined
- it is immediately usable for TDD
- it already separates parse-validity from match behavior

High-value properties:

- recursive SMARTS
- ring predicates
- aromatic and aliphatic distinctions
- charge cases
- explicit hydrogen forms
- invalid syntax cases

Weakness:

- it reflects Open Babel's historical behavior, including some dialect choices
- stereo and some advanced semantics are not as strong as RDKit or CDK

### 3. CDK parser and matcher tests

Primary files:

- `tool/smarts/src/test/java/org/openscience/cdk/smarts/ParserTest.java`
- `tool/smarts/src/test/java/org/openscience/cdk/smarts/SmartsPatternTest.java`

What it gives:

- a large parse-validity corpus
- explicit parser failure cases
- explicit matcher behavior for difficult semantics
- component grouping tests
- isotope tests
- recursive stereo tests
- reaction and atom-map tests
- ring-size regression tests

Why it matters:

- CDK has some of the clearest difficult semantic tests
- it cleanly exposes cases that depend on target preparation and not just syntax

High-value properties:

- components
- isotopes
- recursive SMARTS
- stereo inside recursion
- ring-size checks like `[r5]` and `[r6]`
- reaction SMARTS and atom maps
- non-tetrahedral stereo in later-stage tests

Weakness:

- the data is embedded in Java tests
- part of it is beyond a sensible v1 scope

### 4. PubChem fingerprint definitions

Primary file:

- `https://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt`

What it gives:

- a large official set of real-world positive patterns
- fixed pattern definitions tied to PubChem fingerprint keys

The most useful sections are:

- section 6: simple SMARTS patterns, bits `460` to `712`
- section 7: complex SMARTS patterns, bits `713` to `880`

That is:

- 253 simple SMARTS-pattern bits
- 168 complex SMARTS-pattern bits
- 421 pattern-based bits total across sections 6 and 7

Why it matters:

- it is an official public source
- it gives many real substructure patterns that should parse cleanly
- it is a good supplement for positive parse coverage

Weakness:

- it is not a parser test suite
- it does not provide invalid cases
- it does not provide match gold sets for hard SMARTS semantics
- it is weak on recursive SMARTS, hydrogen ambiguity, stereo edge cases, and components

### 5. Daylight theory/tutorial examples

Primary sources:

- Daylight SMARTS theory manual
- Daylight SMARTS tutorial

What they give:

- canonical language examples
- operator-precedence examples
- component-grouping examples
- reaction SMARTS examples

Why they matter:

- they are the language reference point
- they are good for spec-facing unit tests

Weakness:

- small corpus
- not machine-oriented
- not a regression suite

## Recommended TDD corpus layout

The corpus should not be one undifferentiated list.

Build these layers separately:

### Layer A: parse-valid core corpus

Use:

- Open Babel `validsmarts.txt`
- Daylight examples
- selected PubChem section 6/7 patterns

Purpose:

- parser acceptance
- AST/IR structure assertions
- roundtrip tests where applicable

### Layer B: parse-invalid corpus

Use:

- Open Babel `invalid-smarts.txt`
- selected bad-reaction and bad-prefix cases from CDK
- selected malformed bracket/ring cases from RDKit tests

Purpose:

- diagnostics
- recovery behavior
- unsupported-feature reporting

### Layer C: difficult semantic corpus

Use:

- selected RDKit `smatest.cpp` cases
- selected CDK `SmartsPatternTest.java` cases

Purpose:

- recursive SMARTS
- hydrogen semantics
- ring semantics
- stereo
- components
- atom maps

### Layer D: real-world positive corpus

Use:

- PubChem fingerprint patterns

Purpose:

- broad positive parsing
- realistic aromatic/ring fragments
- practical substructure idioms

### Layer E: known-divergence corpus

Use:

- toolkit issue reproducers

Purpose:

- document intentional or temporary compatibility gaps
- avoid silently changing behavior later

This layer should be marked clearly as:

- `expected_rdkit`
- `expected_openbabel`
- `expected_cdk`
- `our_v1_policy`

## What to record for each test case

Each corpus entry should carry metadata.

Minimum schema:

- `id`
- `source_toolkit`
- `source_file_or_url`
- `smarts`
- `kind`
  - `parse_valid`
  - `parse_invalid`
  - `match`
  - `regression`
- `dialect`
  - `daylight`
  - `rdkit_extension`
  - `openbabel_extension`
  - `cdk_extension`
  - `pubchem_pattern`
- `tags`
- `notes`

For match cases also record:

- `target_smiles`
- `expected_match`
- `expected_count` when known
- `expected_mapping` when known

Recommended tags:

- `recursive`
- `hydrogen`
- `atom_list`
- `ring_closure`
- `ring_predicate`
- `aromatic`
- `charge`
- `stereo_directional`
- `stereo_tetrahedral`
- `component_grouping`
- `reaction`
- `atom_map`
- `extension`
- `unsupported_v1`

## Difficult SMARTS worth carrying immediately

These are high-value cases to seed first because they exercise the same areas that repeatedly break in existing toolkits.

| SMARTS | Source | Key properties | Why it matters |
| --- | --- | --- | --- |
| `[$(CO)]CO` | RDKit `smatest.cpp` | `recursive` | Minimal recursive SMARTS that is still readable |
| `[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]` | RDKit `smatest.cpp` | `recursive`, `ring_predicate`, `negation` | Dense query logic with low-level precedence pressure |
| `[C;!$(C-[OH])]=O` | RDKit `smatest.cpp` | `recursive`, `low_and` | Common functional-group style query |
| `Cl/C=C\\Cl` | RDKit `smatest.cpp` | `stereo_directional` | Smallest useful directional-bond regression |
| `[2H,13C]` | RDKit `smatest.cpp` | `isotope`, `atom_list` | Isotope parsing inside a list |
| `C%(1000)CC%(1000)` | RDKit `smatest.cpp` | `ring_closure`, `extension` | High ring-closure index handling |
| `[H]` | Open Babel `validsmarts.txt`, CDK `ParserTest.java` | `hydrogen` | Must be in the first parser pack |
| `[H+]` | Open Babel `validsmarts.txt`, CDK `ParserTest.java` | `hydrogen`, `charge` | Hydrogen plus charge parsing |
| `[nH1]` | Open Babel `validsmarts.txt` | `hydrogen`, `aromatic` | Aromatic hydrogen case |
| `[c,n&H1]` | Open Babel `validsmarts.txt` | `atom_list`, `hydrogen`, `precedence` | List plus high-precedence `&` |
| `C=1CCCCC#1` | Open Babel `invalid-smarts.txt` | `ring_closure`, `bond_consistency` | Invalid ring closure with conflicting bond types |
| `C-1CCCCC:1` | Open Babel `invalid-smarts.txt` | `ring_closure`, `bond_consistency` | Another invalid closure form worth keeping |
| `[C;;C]` | Open Babel `invalid-smarts.txt` | `precedence`, `syntax_error` | Simple malformed boolean syntax |
| `(O).(O)` | CDK `SmartsPatternTest.java` | `component_grouping` | Important out-of-scope or later-scope boundary |
| `[12*]` | CDK `SmartsPatternTest.java` | `isotope` | Explicit isotope wildcard |
| `[$(*/C=C/*)]` | CDK `SmartsPatternTest.java` | `recursive`, `stereo_directional` | Recursive SMARTS with double-bond geometry |
| `[$(C(/*)=C/*)]` | CDK `SmartsPatternTest.java` | `recursive`, `stereo_directional` | Complementary cis/trans recursive case |
| `[$([C@](C)(CC)(N)O)]` | CDK `SmartsPatternTest.java` | `recursive`, `stereo_tetrahedral` | Recursive query carrying tetrahedral stereo |
| `[N&r5]` | CDK `SmartsPatternTest.java` | `ring_predicate` | Regression for ring-size semantics |
| `[N&r6]` | CDK `SmartsPatternTest.java` | `ring_predicate` | Pair with the previous case |
| `Cc1ccc(Cl)cc1` | PubChem section 7 | `aromatic`, `ring_closure`, `real_world` | Good real positive pattern |
| `CC1CCC(Br)CC1` | PubChem section 7 | `ring_closure`, `real_world` | Saturated ring pattern from a public corpus |
| `C:C-C=C` | PubChem section 6 | `aromatic`, `bond_order`, `real_world` | Small realistic mixed aromatic/aliphatic bond pattern |
| `[#1]-N-N-[#1]` | PubChem section 6 | `hydrogen`, `real_world` | Explicit hydrogen in a public pattern corpus |

## Immediate recommendation

For v1, build the corpus in this order:

1. Open Babel valid and invalid flat files
2. a curated difficult subset from RDKit
3. a curated difficult subset from CDK
4. PubChem section 6 and 7 as extra positive patterns

That gives you:

- easy parser smoke coverage
- hard parser and semantic edge cases
- at least one large public positive corpus

It also avoids a common mistake:

- starting from PubChem alone

PubChem is useful, but by itself it will miss the hardest parser and semantic bugs.

## Sources

### PubChem

- `https://pubchem.ncbi.nlm.nih.gov/search/help_search.html`
- `https://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt`

### RDKit

- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smatest.cpp`

### Open Babel

- `https://github.com/openbabel/openbabel/blob/master/test/files/validsmarts.txt`
- `https://github.com/openbabel/openbabel/blob/master/test/files/invalid-smarts.txt`
- `https://github.com/openbabel/openbabel/blob/master/test/files/smartstest.txt`
- `https://github.com/openbabel/openbabel/blob/master/test/files/smartsresults.txt`
- `https://github.com/openbabel/openbabel/blob/master/test/smartsparse.cpp`
- `https://github.com/openbabel/openbabel/blob/master/test/smartstest.cpp`

### CDK

- `https://github.com/cdk/cdk/blob/main/tool/smarts/src/test/java/org/openscience/cdk/smarts/ParserTest.java`
- `https://github.com/cdk/cdk/blob/main/tool/smarts/src/test/java/org/openscience/cdk/smarts/SmartsPatternTest.java`

### Daylight

- `https://daylight.com/dayhtml/doc/theory/theory.smarts.html`
- `https://www.daylight.com/dayhtml_tutorials/languages/smarts/index.html`
