use alloc::{vec, vec::Vec};

type ConjunctionItems<T, K> = for<'a> fn(&'a T) -> Option<(&'a [T], K)>;
type DisjunctionItems<T> = for<'a> fn(&'a T) -> Option<&'a [T]>;

pub(super) struct CoveredNegatedDisjunctionOps<T, K> {
    pub(super) consensus_items: fn(&T) -> (Vec<T>, K),
    pub(super) sort_and_dedup: fn(&mut Vec<T>),
    pub(super) trees_are_complements: fn(&T, &T) -> bool,
    pub(super) items_without_index: fn(&[T], usize) -> Vec<T>,
    pub(super) can_render_and_as_high: fn(&[T]) -> bool,
    pub(super) simplify_and: fn(Vec<T>, K) -> T,
    pub(super) high_kind: K,
    pub(super) low_kind: K,
}

pub(super) struct AbsorbedDisjunctionConjunctionOps<T, K> {
    pub(super) conjunction_items: ConjunctionItems<T, K>,
    pub(super) disjunction_items: DisjunctionItems<T>,
    pub(super) tree_implies: fn(&T, &T) -> bool,
    pub(super) simplify_or: fn(Vec<T>) -> T,
    pub(super) simplify_and: fn(Vec<T>, K) -> T,
}

pub(super) fn remove_absorbed_disjunction_terms<T>(
    items: &mut Vec<T>,
    item_absorbs: fn(&T, &T) -> bool,
) {
    let mut index = 0usize;
    while index < items.len() {
        let is_absorbed = items
            .iter()
            .enumerate()
            .any(|(other_index, other)| other_index != index && item_absorbs(other, &items[index]));
        if is_absorbed {
            items.remove(index);
        } else {
            index += 1;
        }
    }
}

pub(super) fn collapse_disjunction_consensus_terms<T>(
    items: &mut Vec<T>,
    disjunction_consensus: fn(&T, &T) -> Option<T>,
) -> bool {
    for left_index in 0..items.len() {
        for right_index in (left_index + 1)..items.len() {
            let Some(consensus) = disjunction_consensus(&items[left_index], &items[right_index])
            else {
                continue;
            };
            items.remove(right_index);
            items.remove(left_index);
            items.push(consensus);
            return true;
        }
    }

    false
}

pub(super) fn relax_negated_conjunction_terms<T>(
    items: &mut [T],
    negated_inner: for<'a> fn(&'a T) -> Option<&'a T>,
    remove_term_from_conjunction: fn(&T, &T) -> Option<T>,
) -> bool {
    for negated_index in 0..items.len() {
        let Some(general) = negated_inner(&items[negated_index]) else {
            continue;
        };
        let mut candidate_index = 0;
        while candidate_index < items.len() {
            if candidate_index == negated_index {
                candidate_index += 1;
                continue;
            }
            let Some(relaxed) = remove_term_from_conjunction(&items[candidate_index], general)
            else {
                candidate_index += 1;
                continue;
            };
            items[candidate_index] = relaxed;
            return true;
        }
    }

    false
}

pub(super) fn relax_complemented_disjunction_terms<T>(
    items: &mut [T],
    remove_negated_term_from_conjunction: fn(&T, &T) -> Option<T>,
) -> bool {
    for base_index in 0..items.len() {
        for candidate_index in 0..items.len() {
            if base_index == candidate_index {
                continue;
            }
            let Some(relaxed) =
                remove_negated_term_from_conjunction(&items[candidate_index], &items[base_index])
            else {
                continue;
            };
            items[candidate_index] = relaxed;
            return true;
        }
    }
    false
}

pub(super) fn remove_redundant_disjunction_consensus_terms<T>(
    items: &mut Vec<T>,
    disjunction_consensus_term_is_redundant: fn(&T, &T, &T) -> bool,
) -> bool {
    for candidate_index in 0..items.len() {
        for left_index in 0..items.len() {
            if left_index == candidate_index {
                continue;
            }
            for right_index in (left_index + 1)..items.len() {
                if right_index == candidate_index
                    || !disjunction_consensus_term_is_redundant(
                        &items[left_index],
                        &items[right_index],
                        &items[candidate_index],
                    )
                {
                    continue;
                }

                items.remove(candidate_index);
                return true;
            }
        }
    }

    false
}

pub(super) fn relax_absorbed_disjunction_conjunction_alternatives<T, K>(
    items: &mut Vec<T>,
    ops: &AbsorbedDisjunctionConjunctionOps<T, K>,
) -> bool
where
    T: Clone,
    K: Copy,
{
    for base_index in 0..items.len() {
        for candidate_index in 0..items.len() {
            if base_index == candidate_index {
                continue;
            }
            let Some((candidate_items, kind)) = (ops.conjunction_items)(&items[candidate_index])
            else {
                continue;
            };

            for disjunction_index in 0..candidate_items.len() {
                let Some(alternatives) =
                    (ops.disjunction_items)(&candidate_items[disjunction_index])
                else {
                    continue;
                };
                let retained = alternatives
                    .iter()
                    .filter(|alternative| !(ops.tree_implies)(alternative, &items[base_index]))
                    .cloned()
                    .collect::<Vec<_>>();

                if retained.len() == alternatives.len() {
                    continue;
                }
                if retained.is_empty() {
                    items.remove(candidate_index);
                    return true;
                }

                let mut replacement = candidate_items.to_vec();
                replacement[disjunction_index] = (ops.simplify_or)(retained);
                items[candidate_index] = (ops.simplify_and)(replacement, kind);
                return true;
            }
        }
    }

    false
}

pub(super) fn relax_covered_negated_disjunction_terms<T, K>(
    items: &mut Vec<T>,
    ops: &CoveredNegatedDisjunctionOps<T, K>,
) -> bool
where
    T: PartialEq,
    K: Copy,
{
    for base_index in 0..items.len() {
        let (mut base_terms, _) = (ops.consensus_items)(&items[base_index]);
        (ops.sort_and_dedup)(&mut base_terms);
        if base_terms.len() < 2 {
            continue;
        }

        let mut coverages: Vec<(Vec<T>, Vec<usize>, Vec<usize>)> = Vec::new();
        for (alternative_index, alternative) in items.iter().enumerate() {
            if alternative_index == base_index {
                continue;
            }

            let (alternative_terms, _) = (ops.consensus_items)(alternative);
            for (base_term_index, base_term) in base_terms.iter().enumerate() {
                let Some(complement_index) = alternative_terms
                    .iter()
                    .position(|term| (ops.trees_are_complements)(term, base_term))
                else {
                    continue;
                };
                let mut residual = (ops.items_without_index)(&alternative_terms, complement_index);
                (ops.sort_and_dedup)(&mut residual);

                if let Some((_, covered_terms, alternative_indices)) = coverages
                    .iter_mut()
                    .find(|(known_residual, _, _)| known_residual.as_slice() == residual.as_slice())
                {
                    if !covered_terms.contains(&base_term_index) {
                        covered_terms.push(base_term_index);
                    }
                    if !alternative_indices.contains(&alternative_index) {
                        alternative_indices.push(alternative_index);
                    }
                } else {
                    coverages.push((residual, vec![base_term_index], vec![alternative_index]));
                }
            }
        }

        let Some((residual, _, alternative_indices)) =
            coverages.into_iter().find(|(_, covered_terms, _)| {
                (0..base_terms.len())
                    .all(|base_term_index| covered_terms.contains(&base_term_index))
            })
        else {
            continue;
        };

        super::remove_indices_descending(items, &alternative_indices);
        let kind = if (ops.can_render_and_as_high)(&residual) {
            ops.high_kind
        } else {
            ops.low_kind
        };
        items.push((ops.simplify_and)(residual, kind));
        return true;
    }

    false
}
