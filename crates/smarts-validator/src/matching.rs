use crate::error::SmartsMatchError;
use crate::prepared::PreparedTarget;
use crate::target::TargetText;

/// Match a compiled SMARTS query against a target string.
///
/// This is a placeholder implementation. It validates the target is non-empty
/// and returns `Ok(false)` until the real matcher lands.
///
/// # Errors
///
/// Returns [`SmartsMatchError::EmptyTarget`] when `target` is empty.
pub fn matches(_query: &smarts_parser::QueryMol, target: &str) -> Result<bool, SmartsMatchError> {
    let target = TargetText::new(target)?;
    let _prepared = PreparedTarget::new(target);

    Ok(false)
}
