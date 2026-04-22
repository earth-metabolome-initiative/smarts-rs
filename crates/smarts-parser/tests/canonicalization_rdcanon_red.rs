//! Red tests for known `RDCanon` literature-group canonicalization gaps.

use serde::Deserialize;
use smarts_parser::QueryMol;
use std::str::FromStr;

#[derive(Debug, Deserialize)]
struct CanonGroup {
    id: String,
    source: String,
    members: Vec<String>,
}

#[test]
fn rdcanon_stereo_isomer_family_1_should_converge() {
    assert_group_converges(&load_group("rdcanon-stereo-isomer-family-1"));
}

#[test]
fn rdcanon_stereo_isomer_family_2_should_converge() {
    assert_group_converges(&load_group("rdcanon-stereo-isomer-family-2"));
}

#[test]
fn rdcanon_recursive_charge_bundle_should_converge() {
    assert_group_converges(&load_group("rdcanon-recursive-charge-bundle"));
}

fn assert_group_converges(group: &CanonGroup) {
    assert!(
        group.members.len() >= 2,
        "group {} from {} must contain at least two members",
        group.id,
        group.source
    );

    let canonical_strings = group
        .members
        .iter()
        .map(|smarts| {
            QueryMol::from_str(smarts)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to parse group {} member from {}: {}\nerror: {}",
                        group.id, group.source, smarts, error
                    )
                })
                .to_canonical_smarts()
        })
        .collect::<Vec<_>>();

    let expected = &canonical_strings[0];
    for (index, canonical) in canonical_strings.iter().enumerate().skip(1) {
        assert_eq!(
            expected, canonical,
            "group {} from {} diverged at member {}",
            group.id, group.source, index
        );
    }
}

fn load_group(group_id: &str) -> CanonGroup {
    let groups: Vec<CanonGroup> = load_json("rdcanon-groups-v0.json");
    groups
        .into_iter()
        .find(|group| group.id == group_id)
        .unwrap_or_else(|| panic!("missing RDCanon group fixture: {group_id}"))
}

fn load_json<T>(file_name: &str) -> T
where
    T: for<'de> Deserialize<'de>,
{
    let path = format!(
        "{}/../../corpus/canonicalization/{}",
        env!("CARGO_MANIFEST_DIR"),
        file_name
    );
    let raw =
        std::fs::read_to_string(path).expect("failed to read canonicalization corpus fixture");
    serde_json::from_str(&raw).expect("failed to parse canonicalization corpus fixture JSON")
}
