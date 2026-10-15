"""Fixed source-checked comparisons, not a general experimental-design inference.

Membership is explicit and checked against the installed OSDR cache on every run.
Source checking is a software/data review, not an independent biological validation.
See docs/reviewed-comparisons.md for the source review and remaining limitations.
"""
from dataclasses import dataclass

import pandas as pd

from .cohorts import Cohort, DEFAULT_FACETS, cohort_metadata


@dataclass(frozen=True)
class Comparison:
    key: str
    study: str
    tissue: str
    strain: str
    flight: tuple[str, ...]
    ground: tuple[str, ...]
    durations: tuple[str, ...]
    diet: str
    note: str

    @property
    def label(self):
        return f"{self.study} · {self.tissue}"

    @property
    def source_url(self):
        return f"https://osdr.nasa.gov/bio/repo/data/studies/{self.study}"


def _rr1_members(study, tissue_code, arm, animals):
    return tuple(f"{study}|Mmus_C57-6J_{tissue_code}_{arm}_Rep{i}_M{animal}"
                 for i, animal in enumerate(animals, 1))


COMPARISONS = (
    Comparison("osd-100-eye", "OSD-100", "left eye", "C57BL/6J",
               _rr1_members("OSD-100", "EYE", "FLT", range(23, 29)),
               _rr1_members("OSD-100", "EYE", "GC", range(33, 39)),
               ("37 day",), "Unknown",
               "RR-1, 6 flight and 6 ground mice. Diet is not recorded in the local "
               "sample metadata. Ground and flight are separate animals, not paired replicates."),
    Comparison("osd-104-soleus", "OSD-104", "Soleus-both sides", "C57BL/6J",
               _rr1_members("OSD-104", "SLS", "FLT", range(23, 29)),
               _rr1_members("OSD-104", "SLS", "GC", range(33, 39)),
               ("37 day",), "Nutrient Upgraded Rodent Food Bar (NuRFB)",
               "RR-1, 6 flight and 6 ground mice. Left and right soleus were combined "
               "within each mouse; the two muscles are not independent replicates."),
    Comparison("osd-194-retina", "OSD-194", "Left retina", "BALB/c",
               tuple(f"OSD-194|Mmus_BAL-TAL_LRTN_FLT_Rep{i}_F{animal}"
                     for i, animal in enumerate(range(6, 11), 1)),
               tuple(f"OSD-194|Mmus_BAL-TAL_LRTN_GC_Rep{i}_G{animal}"
                     for i, animal in enumerate((6, 8, 9), 1)),
               ("41 day", "42 day"), "Nutrient Upgraded Rodent Food Bar (NuRFB)",
               "5 flight and 3 ground mice, left retina only. Duration spans 41–42 days "
               "in both groups. Basal controls and right retinas are excluded. "
               "Small, unequal groups limit interpretation."),
)


def get_comparison(key: str) -> Comparison:
    for preset in COMPARISONS:
        if preset.key == key:
            return preset
    raise ValueError("This comparison is not source-checked. Select a listed comparison.")


def resolve_comparison(key: str, metadata: pd.DataFrame | None = None):
    """Refuse drift or missing identities rather than silently changing a group."""
    preset = get_comparison(key)
    meta = cohort_metadata() if metadata is None else metadata
    if "sample_key" not in meta or meta.sample_key.duplicated().any():
        raise ValueError("OSDR sample identities are missing or duplicate.")
    indexed = meta.set_index("sample_key")
    result = []
    for arm, members in (("Space Flight", preset.flight), ("Ground Control", preset.ground)):
        missing = set(members) - set(indexed.index)
        if missing:
            raise ValueError(f"{len(missing)} source-checked comparison samples are missing from the cache.")
        rows = indexed.loc[list(members)]
        expected = dict(study=preset.study, tissue=preset.tissue, strain=preset.strain,
                        spaceflight=arm, sex="Female", diet=preset.diet,
                        habitat="Rodent Habitat", genotype="Unknown")
        for field, value in expected.items():
            if field not in rows or not rows[field].eq(value).all():
                raise ValueError(f"The source-checked {field} no longer matches the installed metadata.")
        if "duration" not in rows or not rows.duration.isin(preset.durations).all():
            raise ValueError("The source-checked duration no longer matches the installed metadata.")
        values = dict(study=preset.study, tissue=preset.tissue, spaceflight=arm)
        result.append(Cohort(f"REVIEWED|{key}|{arm}", preset.study, DEFAULT_FACETS,
                             values, members))
    return tuple(result)
