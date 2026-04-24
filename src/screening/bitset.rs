use alloc::{boxed::Box, vec, vec::Vec};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CountBitsetIndex {
    thresholds: Box<[usize]>,
    bitsets: Box<[Box<[u64]>]>,
    populations: Box<[usize]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RequiredCountFilter<'a> {
    pub(super) population: usize,
    pub(super) source: &'a [u64],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CachedFeatureMask {
    pub(super) population: usize,
    pub(super) words: Box<[u64]>,
}

impl CountBitsetIndex {
    pub(super) fn new(counts: &[usize]) -> Self {
        let mut thresholds = counts
            .iter()
            .copied()
            .filter(|&count| count > 0)
            .collect::<Vec<_>>();
        thresholds.sort_unstable();
        thresholds.dedup();

        let word_count = bitset_word_count(counts.len());
        let bitsets = thresholds
            .iter()
            .map(|&threshold| {
                let mut words = vec![0u64; word_count];
                for (target_id, &count) in counts.iter().enumerate() {
                    if count >= threshold {
                        set_bit(&mut words, target_id);
                    }
                }
                words.into_boxed_slice()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let populations = bitsets
            .iter()
            .map(|words| words.iter().map(|word| word.count_ones() as usize).sum())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            thresholds: thresholds.into_boxed_slice(),
            bitsets,
            populations,
        }
    }

    fn threshold_index_for_at_least(&self, required: usize) -> Option<usize> {
        let index = self
            .thresholds
            .partition_point(|&threshold| threshold < required);
        self.bitsets.get(index)?;
        Some(index)
    }

    pub(super) fn filter_for_at_least(&self, required: usize) -> Option<RequiredCountFilter<'_>> {
        let index = self.threshold_index_for_at_least(required)?;
        Some(RequiredCountFilter {
            population: self.populations.get(index).copied()?,
            source: self.bitsets.get(index)?.as_ref(),
        })
    }
}

pub(super) const fn bitset_word_count(target_count: usize) -> usize {
    target_count.div_ceil(u64::BITS as usize)
}

pub(super) fn set_bit(words: &mut [u64], target_id: usize) {
    let word = target_id / u64::BITS as usize;
    let bit = target_id % u64::BITS as usize;
    words[word] |= 1u64 << bit;
}

pub(super) fn ensure_zeroed_words(words: &mut Vec<u64>, word_count: usize) {
    if words.len() == word_count {
        words.fill(0);
    } else {
        words.clear();
        words.resize(word_count, 0);
    }
}

pub(super) fn intersect_source(
    candidate_mask: &mut [u64],
    has_active_candidate: &mut bool,
    source: &[u64],
) -> bool {
    if *has_active_candidate {
        for (candidate_word, source_word) in candidate_mask.iter_mut().zip(source) {
            *candidate_word &= source_word;
        }
    } else {
        candidate_mask.copy_from_slice(source);
        *has_active_candidate = true;
    }
    candidate_mask.iter().any(|&word| word != 0)
}

pub(super) fn intersect_source_with_population(
    candidate_mask: &mut [u64],
    has_active_candidate: &mut bool,
    source: &[u64],
    source_population: usize,
) -> Option<usize> {
    if *has_active_candidate {
        let mut population = 0usize;
        for (candidate_word, &source_word) in candidate_mask.iter_mut().zip(source) {
            *candidate_word &= source_word;
            population += candidate_word.count_ones() as usize;
        }
        (population != 0).then_some(population)
    } else {
        candidate_mask.copy_from_slice(source);
        *has_active_candidate = true;
        (source_population != 0).then_some(source_population)
    }
}

pub(super) fn for_each_set_bit<F>(candidate_mask: &[u64], target_count: usize, mut f: F)
where
    F: FnMut(usize),
{
    for (word_index, &word) in candidate_mask.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let target_id = word_index * u64::BITS as usize + bit;
            if target_id < target_count {
                f(target_id);
            }
            remaining &= remaining - 1;
        }
    }
}
