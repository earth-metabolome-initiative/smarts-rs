use elements_rs::Element;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct BondCountScreen {
    pub(super) single: usize,
    pub(super) double: usize,
    pub(super) triple: usize,
    pub(super) aromatic: usize,
    pub(super) ring: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct AtomFeature {
    pub(super) element: Option<Element>,
    pub(super) aromatic: Option<bool>,
    pub(super) requires_ring: bool,
    pub(super) degree: Option<u16>,
    pub(super) total_hydrogens: Option<u16>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct EdgeBondFeature {
    pub(super) kind: Option<RequiredBondKind>,
    pub(super) requires_ring: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct EdgeFeature {
    pub(super) left: AtomFeature,
    pub(super) bond: EdgeBondFeature,
    pub(super) right: AtomFeature,
}

impl EdgeFeature {
    #[must_use]
    pub(super) fn new(left: AtomFeature, bond: EdgeBondFeature, right: AtomFeature) -> Self {
        if right < left {
            Self {
                left: right,
                bond,
                right: left,
            }
        } else {
            Self { left, bond, right }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Path3Feature {
    pub(super) left: AtomFeature,
    pub(super) left_bond: EdgeBondFeature,
    pub(super) center: AtomFeature,
    pub(super) right_bond: EdgeBondFeature,
    pub(super) right: AtomFeature,
}

impl Path3Feature {
    #[must_use]
    pub(super) fn new(
        left: AtomFeature,
        left_bond: EdgeBondFeature,
        center: AtomFeature,
        right_bond: EdgeBondFeature,
        right: AtomFeature,
    ) -> Self {
        let forward = Self {
            left,
            left_bond,
            center,
            right_bond,
            right,
        };
        let reverse = Self {
            left: right,
            left_bond: right_bond,
            center,
            right_bond: left_bond,
            right: left,
        };
        if reverse < forward {
            reverse
        } else {
            forward
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Path4Feature {
    pub(super) left: AtomFeature,
    pub(super) left_bond: EdgeBondFeature,
    pub(super) left_middle: AtomFeature,
    pub(super) center_bond: EdgeBondFeature,
    pub(super) right_middle: AtomFeature,
    pub(super) right_bond: EdgeBondFeature,
    pub(super) right: AtomFeature,
}

impl Path4Feature {
    #[must_use]
    pub(super) fn new(
        left: AtomFeature,
        left_bond: EdgeBondFeature,
        left_middle: AtomFeature,
        center_bond: EdgeBondFeature,
        right_middle: AtomFeature,
        right_bond: EdgeBondFeature,
        right: AtomFeature,
    ) -> Self {
        let forward = Self {
            left,
            left_bond,
            left_middle,
            center_bond,
            right_middle,
            right_bond,
            right,
        };
        let reverse = Self {
            left: right,
            left_bond: right_bond,
            left_middle: right_middle,
            center_bond,
            right_middle: left_middle,
            right_bond: left_bond,
            right: left,
        };
        if reverse < forward {
            reverse
        } else {
            forward
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Star3Arm {
    pub(super) bond: EdgeBondFeature,
    pub(super) atom: AtomFeature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Star3Feature {
    pub(super) center: AtomFeature,
    pub(super) arms: [Star3Arm; 3],
}

impl Star3Feature {
    #[must_use]
    pub(super) fn new(center: AtomFeature, mut arms: [Star3Arm; 3]) -> Self {
        arms.sort_unstable();
        Self { center, arms }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RequiredBondKind {
    Single,
    Double,
    Triple,
    Aromatic,
}
