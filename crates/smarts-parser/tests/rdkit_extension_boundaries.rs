//! Explicit parser boundary tests for `SMARTS` forms that should stay unsupported.

use smarts_parser::parse_smarts;

#[test]
fn documented_rdkit_boundary_forms_remain_unsupported() {
    let smarts = "[C@?H](F)Cl";
    assert!(
        parse_smarts(smarts).is_err(),
        "unexpectedly accepted unsupported SMARTS boundary case {smarts}"
    );
}
