"""The single seam between Bridge Manifold and Bridge RNA.

Bridge Manifold reuses Bridge RNA's model, preprocessing, and digest gate rather
than reimplementing them, because a subtle preprocessing drift produces
embeddings that look valid but are scientifically wrong. All imports from the
sibling repository are funnelled through this one module so the coupling is
visible and testable in isolation.

The two scripts that need these symbols (``precompute/embed_osdr.py`` and the
LFS preflight) import from here; the serving app never touches torch.
"""

from __future__ import annotations

import sys

from .paths import BRIDGE_RNA_ROOT

# Make the Bridge RNA repo importable. It is prepended so its module names win
# over any same-named local file, matching how the demo scripts run in-tree.
_bridge_rna_str = str(BRIDGE_RNA_ROOT)
if _bridge_rna_str not in sys.path:
    sys.path.insert(0, _bridge_rna_str)


def load_bridge_rna_symbols():
    """Import and return the reusable Bridge RNA interfaces.

    Imported lazily inside a function so that merely importing this module does
    not pull in torch (a multi-second import) for callers that only need the
    path shim. Returns a dict of the verified reusable symbols catalogued in
    REFERENCE.md section 6.
    """
    from generate_archs4_embeddings import (  # noqa: E402
        CANONICAL_GENES_SHA256,
        ExpressionPerformer,
        _strip_module_prefix,
        canonical_gene_order_digest,
    )
    from demo_osdr_top5 import (  # noqa: E402
        build_mouse_to_human_maps,
        normalize_counts_to_tpm_single,
    )

    return {
        "ExpressionPerformer": ExpressionPerformer,
        "CANONICAL_GENES_SHA256": CANONICAL_GENES_SHA256,
        "canonical_gene_order_digest": canonical_gene_order_digest,
        "_strip_module_prefix": _strip_module_prefix,
        "build_mouse_to_human_maps": build_mouse_to_human_maps,
        "normalize_counts_to_tpm_single": normalize_counts_to_tpm_single,
    }
