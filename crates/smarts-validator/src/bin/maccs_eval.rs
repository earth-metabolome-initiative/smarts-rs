//! Stream MACCS evaluation for SMILES lines using the local `smarts-rs` matcher.

use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use smarts_parser::QueryMol;
use smarts_validator::{match_count_compiled, matches_compiled, CompiledQuery, PreparedTarget};
use smiles_parser::Smiles;

#[derive(Debug, Deserialize)]
struct CatalogEntry {
    key_id: u16,
    smarts: Option<String>,
    count_threshold: u8,
    special_case: Option<String>,
}

#[derive(Debug)]
enum CompiledEntry {
    Undefined,
    AromaticRingCountGreaterThanOne,
    FragmentCountGreaterThanOne,
    Query {
        compiled: Box<CompiledQuery>,
        count_threshold: u8,
    },
}

#[derive(Debug, Serialize)]
struct EvalOutput {
    ok: bool,
    bits: Vec<u16>,
    error: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let catalog_path = parse_args(env::args().skip(1))?;
    let entries = load_catalog(&catalog_path)?;

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    for line in BufReader::new(stdin.lock()).lines() {
        let smiles = line?;
        if smiles.trim().is_empty() {
            continue;
        }

        let output = evaluate_smiles(smiles.trim(), &entries);
        serde_json::to_writer(&mut writer, &output)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }

    writer.flush()?;
    Ok(())
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<PathBuf, String> {
    match (args.next().as_deref(), args.next(), args.next()) {
        (Some("--catalog"), Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err("usage: maccs_eval --catalog <catalog.json>".to_string()),
    }
}

fn load_catalog(path: &Path) -> Result<Vec<(u16, CompiledEntry)>, Box<dyn std::error::Error>> {
    let catalog_text = fs::read_to_string(path)?;
    let entries: Vec<CatalogEntry> = serde_json::from_str(&catalog_text)?;
    let mut compiled_entries = Vec::with_capacity(entries.len());

    for entry in entries {
        let compiled = match (entry.smarts.as_deref(), entry.special_case.as_deref()) {
            (None, Some("undefined")) => CompiledEntry::Undefined,
            (None, Some("aromatic_ring_count_gt_1")) => {
                CompiledEntry::AromaticRingCountGreaterThanOne
            }
            (None, Some("fragment_count_gt_1")) => CompiledEntry::FragmentCountGreaterThanOne,
            (Some(smarts), None) => {
                let query = smarts.parse::<QueryMol>()?;
                let compiled = CompiledQuery::new(query)?;
                CompiledEntry::Query {
                    compiled: Box::new(compiled),
                    count_threshold: entry.count_threshold,
                }
            }
            _ => {
                return Err(format!(
                    "unsupported MACCS catalog entry shape for key {}",
                    entry.key_id
                )
                .into());
            }
        };
        compiled_entries.push((entry.key_id, compiled));
    }

    Ok(compiled_entries)
}

fn evaluate_smiles(smiles: &str, entries: &[(u16, CompiledEntry)]) -> EvalOutput {
    let parsed = match smiles.parse::<Smiles>() {
        Ok(parsed) => parsed,
        Err(error) => {
            return EvalOutput {
                ok: false,
                bits: Vec::new(),
                error: Some(error.to_string()),
            };
        }
    };
    let prepared = PreparedTarget::new(parsed);
    let mut bits = Vec::new();

    for (key_id, entry) in entries {
        let is_on = match entry {
            CompiledEntry::Undefined => false,
            CompiledEntry::AromaticRingCountGreaterThanOne => {
                has_multiple_aromatic_rings(&prepared)
            }
            CompiledEntry::FragmentCountGreaterThanOne => has_multiple_fragments(&prepared),
            CompiledEntry::Query {
                compiled,
                count_threshold,
            } => {
                if *count_threshold == 0 {
                    matches_compiled(compiled, &prepared)
                } else {
                    match_count_compiled(compiled, &prepared) > usize::from(*count_threshold)
                }
            }
        };
        if is_on {
            bits.push(*key_id);
        }
    }

    EvalOutput {
        ok: true,
        bits,
        error: None,
    }
}

fn has_multiple_aromatic_rings(target: &PreparedTarget) -> bool {
    let mut aromatic_ring_count = 0_u8;

    for cycle in target.target().symm_sssr_result().cycles() {
        if cycle.len() < 2 {
            continue;
        }

        let mut all_aromatic = true;
        for edge in cycle.windows(2) {
            if !target.is_aromatic_bond(edge[0], edge[1]) {
                all_aromatic = false;
                break;
            }
        }

        if all_aromatic && !target.is_aromatic_bond(cycle[cycle.len() - 1], cycle[0]) {
            all_aromatic = false;
        }

        if all_aromatic {
            aromatic_ring_count = aromatic_ring_count.saturating_add(1);
            if aromatic_ring_count > 1 {
                return true;
            }
        }
    }

    false
}

fn has_multiple_fragments(target: &PreparedTarget) -> bool {
    let Some(first_component) = target.connected_component(0) else {
        return false;
    };

    (1..target.atom_count()).any(|atom_id| {
        target
            .connected_component(atom_id)
            .is_some_and(|component| component != first_component)
    })
}
