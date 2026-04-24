//! Generate a richer SMARTS benchmark corpus from downstream example SMILES.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write,
    fs,
    path::PathBuf,
    str::FromStr,
};

use smiles_parser::{bond::Bond, Smiles};

const INPUT_TSV: &str = "corpus/benchmark/smarts-evolution-example-smiles-v0.tsv";
const OUTPUT_SMARTS: &str = "corpus/benchmark/smarts-evolution-complex-queries-v0.smarts";
const LARGE_OUTPUT_SMARTS: &str =
    "corpus/benchmark/smarts-evolution-complex-queries-large-v0.smarts";
const QUERIES_PER_DATASET: usize = 40;
const LARGE_QUERIES_PER_DATASET: usize = 200;
const MAX_PATH_ATOMS: usize = 4;

#[derive(Clone, Debug)]
struct CandidateStats {
    smarts: String,
    support: usize,
    metrics: ComplexityMetrics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ComplexityMetrics {
    bracket_atoms: usize,
    branches: usize,
    multiple_bonds: usize,
    aromatic_atoms: usize,
    ring_atoms: usize,
    hydrogen_terms: usize,
    length: usize,
}

fn main() {
    let datasets = load_dataset_smiles();
    let candidates_by_dataset = datasets
        .iter()
        .map(|(dataset, smiles_set)| (dataset.clone(), mine_dataset_candidates(smiles_set)))
        .collect::<BTreeMap<_, _>>();

    write_query_fixture(
        &datasets,
        &candidates_by_dataset,
        QUERIES_PER_DATASET,
        OUTPUT_SMARTS,
    );
    write_query_fixture(
        &datasets,
        &candidates_by_dataset,
        LARGE_QUERIES_PER_DATASET,
        LARGE_OUTPUT_SMARTS,
    );
}

fn write_query_fixture(
    datasets: &BTreeMap<String, Vec<Smiles>>,
    candidates_by_dataset: &BTreeMap<String, Vec<CandidateStats>>,
    queries_per_dataset: usize,
    output_smarts: &str,
) {
    let mut selected = Vec::new();

    for dataset in datasets.keys() {
        let candidates = &candidates_by_dataset[dataset];
        let chosen = choose_dataset_queries(candidates, queries_per_dataset);
        assert!(
            chosen.len() >= queries_per_dataset,
            "dataset {dataset} produced only {} complex candidates",
            chosen.len()
        );
        selected.extend(
            chosen
                .into_iter()
                .map(|candidate| (dataset.clone(), candidate)),
        );
    }

    selected.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| compare_candidates(&left.1, &right.1))
    });

    let output_path = repo_root().join(output_smarts);
    let mut rendered = String::new();
    for (_dataset, candidate) in &selected {
        rendered.push_str(&candidate.smarts);
        rendered.push('\n');
    }
    fs::write(&output_path, rendered)
        .unwrap_or_else(|_| panic!("failed to write {}", output_path.display()));

    eprintln!(
        "wrote {} queries across {} datasets to {}",
        selected.len(),
        datasets.len(),
        output_path.display()
    );
    for (dataset, count) in selected
        .iter()
        .fold(BTreeMap::new(), |mut acc, (dataset, _)| {
            *acc.entry(dataset.clone()).or_insert(0usize) += 1;
            acc
        })
    {
        eprintln!("{dataset}\t{count}");
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_dataset_smiles() -> BTreeMap<String, Vec<Smiles>> {
    let path = repo_root().join(INPUT_TSV);
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("failed to read {}", path.display()));
    let mut per_dataset = BTreeMap::<String, Vec<Smiles>>::new();

    for (line_idx, line) in raw.lines().enumerate() {
        if line_idx == 0 {
            assert_eq!(line, "dataset\tlabel\tsmiles", "fixture header drifted");
            continue;
        }
        let mut fields = line.splitn(3, '\t');
        let dataset = fields
            .next()
            .unwrap_or_else(|| panic!("missing dataset field at line {}", line_idx + 1));
        let _label = fields
            .next()
            .unwrap_or_else(|| panic!("missing label field at line {}", line_idx + 1));
        let smiles = fields
            .next()
            .unwrap_or_else(|| panic!("missing SMILES field at line {}", line_idx + 1));
        per_dataset.entry(dataset.to_string()).or_default().push(
            Smiles::from_str(smiles).unwrap_or_else(|_| {
                panic!(
                    "invalid SMILES in fixture at line {}: {smiles}",
                    line_idx + 1
                )
            }),
        );
    }

    per_dataset
}

fn mine_dataset_candidates(smiles_set: &[Smiles]) -> Vec<CandidateStats> {
    let mut support = HashMap::<String, usize>::new();
    for smiles in smiles_set {
        for candidate in molecule_candidates(smiles, MAX_PATH_ATOMS) {
            *support.entry(candidate).or_default() += 1;
        }
    }

    let mut candidates = support
        .into_iter()
        .map(|(smarts, support)| CandidateStats {
            metrics: compute_complexity_metrics(&smarts),
            smarts,
            support,
        })
        .filter(is_complex_candidate)
        .collect::<Vec<_>>();
    candidates.sort_by(compare_candidates);
    candidates
}

fn choose_dataset_queries(
    candidates: &[CandidateStats],
    target_count: usize,
) -> Vec<CandidateStats> {
    let mut selected = Vec::new();
    let mut seen = HashSet::new();

    for candidate in candidates {
        if selected.len() == target_count {
            break;
        }
        if seen.insert(candidate.smarts.clone()) {
            selected.push(candidate.clone());
        }
    }

    selected
}

fn compare_candidates(left: &CandidateStats, right: &CandidateStats) -> Ordering {
    right
        .metrics
        .cmp(&left.metrics)
        .then_with(|| right.support.cmp(&left.support))
        .then_with(|| left.smarts.cmp(&right.smarts))
}

impl Ord for ComplexityMetrics {
    fn cmp(&self, other: &Self) -> Ordering {
        self.branches
            .cmp(&other.branches)
            .then(self.multiple_bonds.cmp(&other.multiple_bonds))
            .then(self.aromatic_atoms.cmp(&other.aromatic_atoms))
            .then(self.ring_atoms.cmp(&other.ring_atoms))
            .then(self.hydrogen_terms.cmp(&other.hydrogen_terms))
            .then(self.bracket_atoms.cmp(&other.bracket_atoms))
            .then(self.length.cmp(&other.length))
    }
}

impl PartialOrd for ComplexityMetrics {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

const fn is_complex_candidate(candidate: &CandidateStats) -> bool {
    candidate.support >= 4
        && candidate.metrics.bracket_atoms >= 3
        && candidate.metrics.length >= 18
        && (candidate.metrics.branches > 0
            || candidate.metrics.multiple_bonds > 0
            || candidate.metrics.aromatic_atoms > 0
            || candidate.metrics.ring_atoms > 0
            || candidate.metrics.hydrogen_terms > 0)
}

fn compute_complexity_metrics(smarts: &str) -> ComplexityMetrics {
    ComplexityMetrics {
        bracket_atoms: smarts.matches('[').count(),
        branches: smarts.matches('(').count(),
        multiple_bonds: smarts
            .chars()
            .filter(|ch| matches!(ch, '=' | '#' | ':' | '$'))
            .count(),
        aromatic_atoms: smarts.matches(";a").count(),
        ring_atoms: smarts.matches(";R").count(),
        hydrogen_terms: smarts.matches(";H").count(),
        length: smarts.len(),
    }
}

fn molecule_candidates(smiles: &Smiles, max_path_atoms: usize) -> HashSet<String> {
    let ring_membership = smiles.ring_membership();
    let mut candidates = HashSet::new();
    let mut visited = vec![false; smiles.nodes().len()];
    let mut atom_ids = Vec::new();
    let mut bond_tokens = Vec::new();

    for atom_id in 0..smiles.nodes().len() {
        collect_paths(
            smiles,
            &ring_membership,
            atom_id,
            max_path_atoms,
            &mut visited,
            &mut atom_ids,
            &mut bond_tokens,
            &mut candidates,
        );
        collect_branch_pairs(smiles, &ring_membership, atom_id, &mut candidates);
        collect_carboxylate_tail_candidates(
            smiles,
            &ring_membership,
            atom_id,
            max_path_atoms.max(2),
            &mut candidates,
        );
    }

    candidates
}

#[allow(clippy::too_many_arguments)]
fn collect_paths(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    atom_id: usize,
    max_path_atoms: usize,
    visited: &mut [bool],
    atom_ids: &mut Vec<usize>,
    bond_tokens: &mut Vec<&'static str>,
    candidates: &mut HashSet<String>,
) {
    visited[atom_id] = true;
    atom_ids.push(atom_id);
    candidates.insert(canonical_path_smarts(
        smiles,
        ring_membership,
        atom_ids,
        bond_tokens,
    ));

    if atom_ids.len() < max_path_atoms {
        for edge in smiles.edges_for_node(atom_id) {
            let next = if edge.0 == atom_id { edge.1 } else { edge.0 };
            if visited[next] {
                continue;
            }
            bond_tokens.push(bond_smarts(edge.2));
            collect_paths(
                smiles,
                ring_membership,
                next,
                max_path_atoms,
                visited,
                atom_ids,
                bond_tokens,
                candidates,
            );
            bond_tokens.pop();
        }
    }

    atom_ids.pop();
    visited[atom_id] = false;
}

fn collect_branch_pairs(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    center: usize,
    candidates: &mut HashSet<String>,
) {
    let neighbors = smiles.edges_for_node(center).collect::<Vec<_>>();
    for left_index in 0..neighbors.len() {
        for right_index in (left_index + 1)..neighbors.len() {
            let left = neighbors[left_index];
            let right = neighbors[right_index];
            let left_id = if left.0 == center { left.1 } else { left.0 };
            let right_id = if right.0 == center { right.1 } else { right.0 };
            if left_id == right_id {
                continue;
            }
            let forward = branch_pair_smarts(
                smiles,
                ring_membership,
                center,
                left_id,
                bond_smarts(left.2),
                right_id,
                bond_smarts(right.2),
            );
            let reverse = branch_pair_smarts(
                smiles,
                ring_membership,
                center,
                right_id,
                bond_smarts(right.2),
                left_id,
                bond_smarts(left.2),
            );
            candidates.insert(forward.min(reverse));
        }
    }
}

fn collect_carboxylate_tail_candidates(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    center: usize,
    max_tail_atoms: usize,
    candidates: &mut HashSet<String>,
) {
    if smiles
        .node_by_id(center)
        .and_then(smiles_parser::atom::Atom::element)
        .map(u8::from)
        != Some(6)
    {
        return;
    }

    let mut double_oxygen_neighbors = Vec::new();
    let mut single_oxygen_neighbors = Vec::new();
    let mut tail_neighbors = Vec::new();

    for edge in smiles.edges_for_node(center) {
        let neighbor = if edge.0 == center { edge.1 } else { edge.0 };
        let atomic_number = smiles
            .node_by_id(neighbor)
            .and_then(smiles_parser::atom::Atom::element)
            .map_or(0, u8::from);
        match (edge.2, atomic_number) {
            (Bond::Double, 8) => double_oxygen_neighbors.push(neighbor),
            (bond, 8) if is_single_like_bond(bond) => {
                single_oxygen_neighbors.push((neighbor, bond_smarts(bond)));
            }
            (bond, 6) if is_single_like_bond(bond) => {
                tail_neighbors.push((neighbor, bond_smarts(bond)));
            }
            _ => {}
        }
    }

    if double_oxygen_neighbors.is_empty()
        || single_oxygen_neighbors.is_empty()
        || tail_neighbors.is_empty()
    {
        return;
    }

    let double_oxygen = double_oxygen_neighbors[0];
    let mut visited = vec![false; smiles.nodes().len()];
    visited[center] = true;

    for (tail_start, bond_to_center) in tail_neighbors {
        let mut tail_atoms = Vec::new();
        let mut tail_bonds = Vec::new();
        collect_carboxylate_tail_paths(
            smiles,
            ring_membership,
            tail_start,
            max_tail_atoms,
            &mut visited,
            &mut tail_atoms,
            &mut tail_bonds,
            center,
            bond_to_center,
            double_oxygen,
            &single_oxygen_neighbors,
            candidates,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_carboxylate_tail_paths(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    atom_id: usize,
    max_tail_atoms: usize,
    visited: &mut [bool],
    tail_atoms: &mut Vec<usize>,
    tail_bonds: &mut Vec<&'static str>,
    center: usize,
    bond_to_center: &'static str,
    double_oxygen: usize,
    single_oxygen_neighbors: &[(usize, &'static str)],
    candidates: &mut HashSet<String>,
) {
    let atomic_number = smiles
        .node_by_id(atom_id)
        .and_then(smiles_parser::atom::Atom::element)
        .map_or(0, u8::from);
    if atomic_number != 6 {
        return;
    }

    visited[atom_id] = true;
    tail_atoms.push(atom_id);

    for &(single_oxygen, oxygen_bond) in single_oxygen_neighbors {
        candidates.insert(carboxylate_tail_smarts(
            smiles,
            ring_membership,
            tail_atoms,
            tail_bonds,
            center,
            bond_to_center,
            double_oxygen,
            single_oxygen,
            oxygen_bond,
        ));
    }

    if tail_atoms.len() < max_tail_atoms {
        for edge in smiles.edges_for_node(atom_id) {
            let next = if edge.0 == atom_id { edge.1 } else { edge.0 };
            if visited[next] {
                continue;
            }
            tail_bonds.push(bond_smarts(edge.2));
            collect_carboxylate_tail_paths(
                smiles,
                ring_membership,
                next,
                max_tail_atoms,
                visited,
                tail_atoms,
                tail_bonds,
                center,
                bond_to_center,
                double_oxygen,
                single_oxygen_neighbors,
                candidates,
            );
            tail_bonds.pop();
        }
    }

    tail_atoms.pop();
    visited[atom_id] = false;
}

#[allow(clippy::too_many_arguments)]
fn carboxylate_tail_smarts(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    tail_atoms: &[usize],
    tail_bonds: &[&str],
    center: usize,
    bond_to_center: &str,
    double_oxygen: usize,
    single_oxygen: usize,
    oxygen_bond: &str,
) -> String {
    let mut atoms = tail_atoms.to_vec();
    let mut bonds = tail_bonds.to_vec();
    atoms.reverse();
    bonds.reverse();

    let mut smarts = path_smarts(smiles, ring_membership, &atoms, &bonds);
    smarts.push_str(bond_to_center);
    smarts.push_str(&atom_smarts(smiles, ring_membership, center));
    smarts.push('(');
    smarts.push('=');
    smarts.push_str(&atom_smarts(smiles, ring_membership, double_oxygen));
    smarts.push(')');
    smarts.push_str(oxygen_bond);
    smarts.push_str(&atom_smarts(smiles, ring_membership, single_oxygen));
    smarts
}

const fn is_single_like_bond(bond: Bond) -> bool {
    matches!(bond, Bond::Single | Bond::Up | Bond::Down)
}

fn branch_pair_smarts(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    center: usize,
    branch_atom: usize,
    branch_bond: &str,
    tail_atom: usize,
    tail_bond: &str,
) -> String {
    format!(
        "{}({}{}){}{}",
        atom_smarts(smiles, ring_membership, center),
        branch_bond,
        atom_smarts(smiles, ring_membership, branch_atom),
        tail_bond,
        atom_smarts(smiles, ring_membership, tail_atom)
    )
}

fn canonical_path_smarts(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    atom_ids: &[usize],
    bond_tokens: &[&str],
) -> String {
    let forward = path_smarts(smiles, ring_membership, atom_ids, bond_tokens);
    if atom_ids.len() <= 1 {
        return forward;
    }

    let mut reverse_atoms = atom_ids.to_vec();
    reverse_atoms.reverse();
    let mut reverse_bonds = bond_tokens.to_vec();
    reverse_bonds.reverse();
    let reverse = path_smarts(smiles, ring_membership, &reverse_atoms, &reverse_bonds);
    forward.min(reverse)
}

fn path_smarts(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    atom_ids: &[usize],
    bond_tokens: &[&str],
) -> String {
    let mut smarts = atom_smarts(smiles, ring_membership, atom_ids[0]);
    for (index, &atom_id) in atom_ids.iter().enumerate().skip(1) {
        smarts.push_str(bond_tokens[index - 1]);
        smarts.push_str(&atom_smarts(smiles, ring_membership, atom_id));
    }
    smarts
}

fn atom_smarts(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    atom_id: usize,
) -> String {
    let atom = smiles.node_by_id(atom_id).unwrap();
    let atomic_number = atom.element().map_or(0, u8::from);
    let degree = smiles.edges_for_node(atom_id).count();
    let total_hydrogens = atom.hydrogen_count() + smiles.implicit_hydrogen_count(atom_id);
    let is_hetero = !matches!(atomic_number, 0 | 1 | 6);
    let mut smarts = format!("[#{atomic_number}");
    if atom.aromatic() {
        smarts.push_str(";a");
    } else if ring_membership.contains_atom(atom_id) {
        smarts.push_str(";R");
    }
    if degree > 0 {
        let _ = write!(smarts, ";D{degree}");
    }
    if total_hydrogens > 0 && (is_hetero || degree <= 1) {
        let _ = write!(smarts, ";H{total_hydrogens}");
    }
    smarts.push(']');
    smarts
}

const fn bond_smarts(bond: Bond) -> &'static str {
    match bond {
        Bond::Single | Bond::Up | Bond::Down => "-",
        Bond::Double => "=",
        Bond::Triple => "#",
        Bond::Quadruple => "$",
        Bond::Aromatic => ":",
    }
}
