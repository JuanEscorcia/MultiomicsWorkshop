"""Small, transparent helpers used across the multi-omics workshop notebooks.

Kept intentionally minimal. No classes, no hidden state.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


# ---------- IO ----------

def load_table(path: Path, sep: str | None = None, index_col=None) -> pd.DataFrame:
    """Read a tsv/csv (optionally gzipped). Auto-detects separator from extension if not given."""
    path = Path(path)
    if sep is None:
        sep = "," if ".csv" in path.suffixes else "\t"
    return pd.read_csv(path, sep=sep, index_col=index_col)


def load_optional_table(path: Path, **kwargs) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        warnings.warn(f"Optional table missing: {path.name}. Skipping.")
        return None
    try:
        return load_table(path, **kwargs)
    except Exception as e:
        warnings.warn(f"Could not load {path.name}: {e}")
        return None


# ---------- orientation / alignment ----------

def to_features_by_samples(table: pd.DataFrame, sample_ids) -> pd.DataFrame:
    """Return a numeric DataFrame oriented as features x samples, using sample_ids to detect orientation.

    Drops obvious non-numeric annotation columns (e.g. 'accession_number', 'species_name', 'gene_name', 'm/z').
    """
    sample_set = set(map(str, sample_ids))
    ann_cols = [c for c in table.columns if c in {
        "accession_number", "species_name", "gene_name", "Unnamed: 0",
        "Ion_index", "m/z", "Ion_mz", "feature",
    }]
    work = table.copy()
    # If there is exactly one annotation-like column, use it as index.
    feature_index = None
    for cand in ["accession_number", "species_name", "Ion_index", "m/z", "Unnamed: 0", "feature"]:
        if cand in work.columns:
            feature_index = cand
            break
    if feature_index is not None:
        work = work.set_index(feature_index)
        # Drop remaining annotation cols
        drop = [c for c in ann_cols if c in work.columns]
        if drop:
            work = work.drop(columns=drop)
    cols_in_samples = [c for c in work.columns if str(c) in sample_set]
    rows_in_samples = [r for r in work.index if str(r) in sample_set]
    if len(cols_in_samples) >= len(rows_in_samples):
        return work[cols_in_samples].apply(pd.to_numeric, errors="coerce")
    return work.loc[rows_in_samples].T.apply(pd.to_numeric, errors="coerce")


def align_metadata(table: pd.DataFrame, metadata: pd.DataFrame, sample_col: str = "sample"):
    """Align a features x samples table with a metadata frame on sample_col. Returns (table, meta) sharing samples.

    Metadata rows are deduplicated on sample_col (the same physical sample may be
    listed once per omic with different `identifier` values).
    """
    meta_dedup = metadata.drop_duplicates(subset=[sample_col])
    shared = [s for s in table.columns if s in set(meta_dedup[sample_col])]
    table = table[shared]
    meta = meta_dedup.set_index(sample_col).loc[shared].reset_index()
    return table, meta


# ---------- transforms ----------

def filter_features(table: pd.DataFrame, min_prevalence: float = 0.1, min_abundance: float = 0.0) -> pd.DataFrame:
    prev = (table > min_abundance).mean(axis=1)
    return table.loc[prev >= min_prevalence]


def log_transform(table: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    return np.log10(table.astype(float) + pseudocount)


def clr_transform(table: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    x = table.astype(float) + pseudocount
    log_x = np.log(x)
    return log_x.subtract(log_x.mean(axis=0), axis=1)


def zscore_features(table: pd.DataFrame) -> pd.DataFrame:
    mu = table.mean(axis=1)
    sd = table.std(axis=1).replace(0, np.nan)
    z = table.subtract(mu, axis=0).divide(sd, axis=0)
    return z.fillna(0.0)


# ---------- distances ----------

def compute_distance_matrix(table: pd.DataFrame, metric: str = "braycurtis") -> pd.DataFrame:
    """table is features x samples. Returns sample x sample distance DataFrame."""
    X = table.T.values
    d = squareform(pdist(X, metric=metric))
    return pd.DataFrame(d, index=table.columns, columns=table.columns)


def mantel_test(d1: pd.DataFrame, d2: pd.DataFrame, n_perm: int = 999, random_state: int = 0):
    """Spearman correlation between upper triangles of two distance matrices. Returns (r, p)."""
    common = [s for s in d1.index if s in d2.index]
    a = d1.loc[common, common].values
    b = d2.loc[common, common].values
    iu = np.triu_indices_from(a, k=1)
    r, _ = spearmanr(a[iu], b[iu])
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(random_state)
        idx = np.arange(len(common))
        count = 0
        for _ in range(n_perm):
            p = rng.permutation(idx)
            bp = b[np.ix_(p, p)]
            rp, _ = spearmanr(a[iu], bp[iu])
            if abs(rp) >= abs(r):
                count += 1
        pval = (count + 1) / (n_perm + 1)
    else:
        pval = np.nan
    return float(r), float(pval), len(common)


# ---------- QC ----------

def quick_pca(table: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    X = table.T.fillna(0.0).values
    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n_components)]
    out = pd.DataFrame(pcs, index=table.columns, columns=cols)
    out.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    return out


# ---------- figures ----------

def save_figure(fig, outpath: Path, dpi: int = 300):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")


# ---------- color table ----------

def load_color_table(path: Path) -> pd.DataFrame:
    """Color tables are comma-separated with columns: accession_number, species, color."""
    df = pd.read_csv(path, sep=",")
    df.columns = [c.strip() for c in df.columns]
    return df


def species_color_map(color_df: pd.DataFrame, key: str = "species") -> dict:
    return dict(zip(color_df[key], color_df["color"]))


def load_palette(path: Path) -> pd.DataFrame:
    """Load `Data/Metadata/colour_scheme.tsv.gz` (or rel_abund variant).

    Columns: ID, phylum_gtdb, species_gtdb, phylum_colour, colour.
    `ID` is the NT accession code used everywhere else in the workshop.
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df


def palette_by_id(palette_df: pd.DataFrame, column: str = "colour") -> dict:
    """Return {NT_code: hex_colour} for either the species- or phylum-level palette."""
    return dict(zip(palette_df["ID"].astype(str), palette_df[column].astype(str)))


# ---------- drug palette (used across notebooks) ----------

DRUG_COLOURS = {
    "control":        "#7f7f7f",
    "chlorpromazine": "#d95f02",
    "metformin":      "#1b9e77",
    "niclosamide":    "#7570b3",
}


# ---------- ANCOM parser ----------

# Drug short codes used in the ANCOM tables map to the canonical drug names.
ANCOM_DRUG_MAP = {"Chlor": "chlorpromazine", "Metfo": "metformin", "Niclo": "niclosamide"}

# Run labels in ANCOM column suffixes: "A", "B" are per-replicate; "" (empty) means runs combined.
ANCOM_RUN_MAP = {"A": "runA", "B": "runB", "": "combined"}


def load_ancom_fc_table(path: Path, drugs=("Niclo", "Metfo", "Chlor")) -> pd.DataFrame:
    """Parse a Wuyts-style `ancom_FC_<omic>_time_conditions.txt.gz` into a long-form table.

    The source layout is 9 horizontally concatenated blocks (drugs x runs). Each block has
    13 columns: taxa_id, W, detected_0.9, detected_0.8, detected_0.7, detected_0.6,
    followed by per-timepoint log fold-change columns. For metaG only 5 timepoints exist;
    for 16S/metaT/metaP all 7 timepoints (T0, T15, T1h, T30, T3h, T48h, T96h) exist.

    Returns columns: taxa_id, drug, drug_full, run, timepoint, W, det_0.9, det_0.8, det_0.7, det_0.6, fc.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None, skiprows=1, na_values=["NA", '"NA"'])
    # First column is row index from R ("1","2",...). Drop it.
    raw = raw.iloc[:, 1:].reset_index(drop=True)
    # Strip quotes from any string cells.
    for c in raw.columns:
        if raw[c].dtype == object:
            raw[c] = raw[c].astype(str).str.strip('"')
    # Determine the per-block width from total columns and number of blocks (drugs x 3 runs).
    n_blocks = len(drugs) * 3  # A, B, combined
    block_w = raw.shape[1] // n_blocks
    if raw.shape[1] % n_blocks != 0:
        raise ValueError(f"Cannot evenly split {raw.shape[1]} columns into {n_blocks} blocks (got width {block_w}).")
    n_time = block_w - 6  # 1 taxa_id + 1 W + 4 detected + n_time FC
    # Time labels: 7-point schedule for everything except metaG (5-point).
    time_schedule_7 = ["T0", "T15", "T1h", "T30", "T3h", "T48h", "T96h"]
    time_schedule_5 = ["T0", "T1h", "T3h", "T48h", "T96h"]
    if n_time == 7:
        time_labels = time_schedule_7
    elif n_time == 5:
        time_labels = time_schedule_5
    else:
        time_labels = [f"T{i}" for i in range(n_time)]
    # Minutes for the time labels.
    time_min = {"T0": 0, "T15": 15, "T30": 30, "T1h": 60, "T3h": 180, "T48h": 2880, "T96h": 5760}
    blocks = []
    block_idx = 0
    for drug in drugs:
        for run_code in ("A", "B", ""):
            sl = raw.iloc[:, block_idx * block_w:(block_idx + 1) * block_w].copy()
            sl.columns = ["taxa_id", "W", "det_0.9", "det_0.8", "det_0.7", "det_0.6", *time_labels]
            # Reshape to long: one row per (taxa, time).
            long = sl.melt(
                id_vars=["taxa_id", "W", "det_0.9", "det_0.8", "det_0.7", "det_0.6"],
                value_vars=time_labels, var_name="time_label", value_name="fc",
            )
            long["drug"] = drug
            long["drug_full"] = ANCOM_DRUG_MAP.get(drug, drug)
            long["run"] = ANCOM_RUN_MAP[run_code]
            long["timepoint"] = long["time_label"].map(time_min)
            blocks.append(long)
            block_idx += 1
    out = pd.concat(blocks, ignore_index=True)
    # Cast types.
    for c in ("W", "fc", "timepoint"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("det_0.9", "det_0.8", "det_0.7", "det_0.6"):
        out[c] = out[c].map({"TRUE": True, "FALSE": False, True: True, False: False})
    return out


def ancom_significance_stars(w_col: pd.Series, d09=None, d08=None, d07=None) -> pd.Series:
    """Return *,**,*** based on ANCOM detected_0.7/0.8/0.9 booleans. Highest threshold wins."""
    s = pd.Series([""] * len(w_col), index=w_col.index, dtype=object)
    if d07 is not None:
        s = s.where(~d07.fillna(False), "*")
    if d08 is not None:
        s = s.where(~d08.fillna(False), "**")
    if d09 is not None:
        s = s.where(~d09.fillna(False), "***")
    return s

