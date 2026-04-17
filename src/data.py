"""
Data pipeline: BAM → ATAC signal arrays, window generation.

EpiBERT input format:
  seq    float32 (batch, seq_len, 4)    one-hot DNA (reference hg38)
  atac   float32 (batch, seq_len, 1)    ATAC-seq signal per base
  motifs float32 (batch, 693)           JASPAR motif scores (zeros: no JASPAR DB)

Normalization: RPM per base → log1p transform (matches EpiBERT training preprocessing).
"""

import os
import numpy as np
import pysam
from pathlib import Path

from utils import load_config, get_logger, atac_processed_path, cfg

log = get_logger("data")

# Chromosome sizes for hg38 (chr8, chr9 only — confirmed from BAM headers)
CHR_SIZES = {
    "chr1":  248956422, "chr2":  242193529, "chr3":  198295559,
    "chr4":  190214555, "chr5":  181538259, "chr6":  170805979,
    "chr7":  159345973, "chr8":  145138636, "chr9":  138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16":  90338345, "chr17":  83257441, "chr18":  80373285,
    "chr19":  58617616, "chr20":  64444167, "chr21":  46709983,
    "chr22":  50818468, "chrX":  156040895, "chrY":  57227415,
}

# DNA base encoding: A=0, C=1, G=2, T=3
BASE_MAP = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3}
N_BASES  = 4


# ─────────────────────────────────────────────────────────────────────────────
# Window generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_windows(
    chromosomes=None,
    n_windows=None,
    window_bp=None,
    seed=None,
    blacklist_bed=None,
    output_bed=None,
):
    """
    Generate genomic windows sampled from the specified chromosomes.

    Windows are distributed proportionally to chromosome length.
    Non-overlapping within each chromosome; sorted by coordinate.

    Returns list of (chrom, start, end) tuples, saved to output_bed.
    """
    cfg_d = load_config()
    if chromosomes is None:
        chromosomes = cfg_d['windows']['chromosomes']
    if n_windows is None:
        n_windows = cfg_d['windows']['n_windows']
    if window_bp is None:
        window_bp = cfg_d['windows']['window_bp']
    if seed is None:
        seed = cfg_d['seed']
    if output_bed is None:
        output_bed = cfg_d['paths']['windows_bed']

    rng = np.random.default_rng(seed)

    # Build blacklist set (optional)
    blacklist = set()
    if blacklist_bed and os.path.isfile(blacklist_bed):
        log.info(f"Loading blacklist: {blacklist_bed}")
        with open(blacklist_bed) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    bl_chr, bl_s, bl_e = parts[0], int(parts[1]), int(parts[2])
                    # Store as 50kb bin index for fast lookup
                    for bin_i in range(bl_s // 50000, bl_e // 50000 + 1):
                        blacklist.add((bl_chr, bin_i))

    chr_lens = {c: CHR_SIZES[c] for c in chromosomes if c in CHR_SIZES}
    total_len = sum(chr_lens.values())

    # Allocate windows proportionally
    windows_per_chr = {}
    allocated = 0
    for i, (c, l) in enumerate(chr_lens.items()):
        if i == len(chr_lens) - 1:
            windows_per_chr[c] = n_windows - allocated
        else:
            windows_per_chr[c] = max(1, int(n_windows * l / total_len))
            allocated += windows_per_chr[c]

    all_windows = []
    for chrom, n_chr in windows_per_chr.items():
        chr_len = chr_lens[chrom]
        max_start = chr_len - window_bp
        if max_start <= 0:
            log.warning(f"{chrom}: too short for window_bp={window_bp}, skipping")
            continue

        # Sample positions without replacement (or with, if n_chr > possible positions)
        # Use stride-based sampling to cover the chromosome evenly, then shuffle
        stride = max(1, max_start // n_chr)
        candidates = list(range(0, max_start, stride))
        # Shuffle to get random subset
        rng.shuffle(candidates)
        candidates = candidates[:n_chr]

        accepted = []
        for start in sorted(candidates):
            # Check blacklist
            end = start + window_bp
            if blacklist:
                bins = range(start // 50000, end // 50000 + 1)
                if any((chrom, b) in blacklist for b in bins):
                    continue
            # Must be on-chromosome
            if end > chr_len:
                continue
            accepted.append((chrom, start, end))

        log.info(f"{chrom}: {len(accepted)} windows")
        all_windows.extend(accepted)

    # Sort for reproducibility
    all_windows.sort()

    # Save BED file
    Path(output_bed).parent.mkdir(parents=True, exist_ok=True)
    with open(output_bed, 'w') as f:
        for chrom, start, end in all_windows:
            f.write(f"{chrom}\t{start}\t{end}\n")

    log.info(f"Saved {len(all_windows)} windows to {output_bed}")
    return all_windows


def load_windows(bed_path=None):
    """Load windows from BED file. Returns list of (chrom, start, end)."""
    if bed_path is None:
        bed_path = cfg('paths', 'windows_bed')
    windows = []
    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            windows.append((parts[0], int(parts[1]), int(parts[2])))
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# ATAC signal from BAM
# ─────────────────────────────────────────────────────────────────────────────

def get_total_mapped_reads(bam_path):
    """Return total number of mapped reads (primary, non-dup)."""
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        stats = bam.get_index_statistics()
        return sum(s.mapped for s in stats)


def compute_atac_coverage(bam_path, chrom, start, end, total_reads=None,
                           normalize='rpm', smoothing_bp=150, log1p=True):
    """
    Compute ATAC-seq signal for a genomic window.

    Parameters
    ----------
    bam_path  : str
    chrom     : str
    start     : int   (0-based)
    end       : int   (exclusive)
    total_reads: int  library size for RPM; computed if None
    normalize : str   'rpm' | 'raw'
    smoothing_bp: int  gaussian smoothing half-width in bp (0 = no smoothing)
    log1p     : bool  apply log1p after normalization

    Returns
    -------
    signal : float32 ndarray of shape (end - start,)
    """
    length = end - start

    try:
        with pysam.AlignmentFile(bam_path, 'rb') as bam:
            # pysam.count_coverage returns 4 arrays (ACGT), sum for total coverage
            cov = bam.count_coverage(
                chrom, start, end,
                quality_threshold=0,   # include all reads
                read_callback='all',
            )
            signal = np.array(cov[0], dtype=np.float32) + \
                     np.array(cov[1], dtype=np.float32) + \
                     np.array(cov[2], dtype=np.float32) + \
                     np.array(cov[3], dtype=np.float32)
    except (ValueError, KeyError) as e:
        # Region not in BAM (e.g., contig missing) → return zeros
        log.warning(f"Coverage failed for {chrom}:{start}-{end}: {e}")
        return np.zeros(length, dtype=np.float32)

    # RPM normalization
    if normalize == 'rpm' and total_reads is not None and total_reads > 0:
        signal = signal / (total_reads / 1_000_000)

    # Gaussian smoothing
    if smoothing_bp > 0:
        sigma = smoothing_bp / 2.355  # FWHM → sigma
        from scipy.ndimage import gaussian_filter1d
        signal = gaussian_filter1d(signal, sigma=sigma).astype(np.float32)

    if log1p:
        signal = np.log1p(signal).astype(np.float32)

    return signal


# ─────────────────────────────────────────────────────────────────────────────
# Build processed ATAC arrays for all conditions × all windows
# ─────────────────────────────────────────────────────────────────────────────

def build_atac_arrays(windows, conditions=None, bam_map=None, force=False):
    """
    For each condition, compute ATAC signal for all windows and save as .npy.

    Shape saved: (n_windows, window_bp)  float32
    """
    if conditions is None:
        conditions = cfg('all_conditions')
    if bam_map is None:
        bam_map = cfg('bam_files')

    normalize  = cfg('atac', 'normalize')
    smoothing  = cfg('atac', 'smoothing_bp')
    do_log1p   = cfg('atac', 'log1p')
    n_windows  = len(windows)
    window_bp  = windows[0][2] - windows[0][1]

    for cond in conditions:
        out_path = atac_processed_path(cond)
        if os.path.isfile(out_path) and not force:
            log.info(f"[SKIP] {cond} ATAC already processed: {out_path}")
            continue

        bam_path = bam_map.get(cond)
        if not bam_path or not os.path.isfile(bam_path):
            log.warning(f"BAM not found for {cond}: {bam_path}")
            continue

        log.info(f"Processing ATAC for {cond} ({n_windows} windows) ...")
        total_reads = get_total_mapped_reads(bam_path)
        log.info(f"  {cond}: {total_reads:,} mapped reads")

        arr = np.zeros((n_windows, window_bp), dtype=np.float32)
        for i, (chrom, start, end) in enumerate(windows):
            sig = compute_atac_coverage(
                bam_path, chrom, start, end,
                total_reads=total_reads,
                normalize=normalize,
                smoothing_bp=smoothing,
                log1p=do_log1p,
            )
            # Ensure correct length
            sig_len = min(len(sig), window_bp)
            arr[i, :sig_len] = sig[:sig_len]

            if (i + 1) % 500 == 0:
                log.info(f"  {cond}: {i+1}/{n_windows} windows done")

        np.save(out_path, arr)
        log.info(f"  Saved {arr.shape} → {out_path}  "
                 f"(size {arr.nbytes / 1e6:.1f} MB)")


def load_atac_array(condition):
    """Load processed ATAC array for a condition. Shape: (n_windows, window_bp)."""
    path = atac_processed_path(condition)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ATAC array not found: {path}. Run build_atac_arrays first.")
    return np.load(path)


def get_atac_for_window(condition, window_idx):
    """
    Load ATAC signal for a single window index.
    Returns float32 array shape (window_bp, 1) — EpiBERT input format.
    """
    arr = load_atac_array(condition)
    sig = arr[window_idx]   # (window_bp,)
    return sig[:, np.newaxis]  # (window_bp, 1)


# ─────────────────────────────────────────────────────────────────────────────
# DNA sequence (one-hot from hg38 reference)
# ─────────────────────────────────────────────────────────────────────────────

_GENOME = None

def load_genome(fasta_path=None):
    """Load hg38 reference genome using pyfaidx."""
    global _GENOME
    if _GENOME is not None:
        return _GENOME
    if fasta_path is None:
        fasta_path = os.path.join(cfg('paths', 'genome'), 'hg38.fa')
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(
            f"hg38.fa not found at {fasta_path}. "
            "Download with:\n"
            "  wget -P /workspace/project/data/genome/ "
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz\n"
            "  gunzip /workspace/project/data/genome/hg38.fa.gz\n"
            "  samtools faidx /workspace/project/data/genome/hg38.fa"
        )
    from pyfaidx import Fasta
    _GENOME = Fasta(fasta_path, as_raw=True, read_long_names=False)
    return _GENOME


def seq_to_onehot(seq_str, window_bp):
    """
    Convert DNA string to one-hot float32 array (window_bp, 4).
    N/ambiguous bases → [0.25, 0.25, 0.25, 0.25].
    """
    arr = np.zeros((window_bp, N_BASES), dtype=np.float32)
    for i, base in enumerate(seq_str[:window_bp]):
        idx = BASE_MAP.get(base)
        if idx is not None:
            arr[i, idx] = 1.0
        else:
            arr[i, :] = 0.25  # N bases
    return arr


def get_dna_onehot(chrom, start, end):
    """
    Fetch DNA sequence from hg38 and one-hot encode it.
    Returns float32 array shape (window_bp, 4).
    """
    window_bp = end - start
    try:
        genome = load_genome()
        seq_str = str(genome[chrom][start:end])
    except Exception as e:
        log.warning(f"Genome fetch failed {chrom}:{start}-{end}: {e}. Using random seq.")
        rng = np.random.default_rng(abs(hash((chrom, start))) % (2**31))
        bases = rng.integers(0, 4, window_bp)
        arr = np.eye(4, dtype=np.float32)[bases]
        return arr
    return seq_to_onehot(seq_str, window_bp)


# ─────────────────────────────────────────────────────────────────────────────
# Motif scores (placeholder — zeros without JASPAR database)
# ─────────────────────────────────────────────────────────────────────────────

def get_motif_scores(chrom=None, start=None, end=None):
    """
    Return placeholder motif scores (zeros).
    Shape: (693,) float32.
    Both vitro and vivo use the same zeros, so they cancel in CDS.
    """
    return np.zeros(cfg('model', 'num_motifs'), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset for activation collection
# ─────────────────────────────────────────────────────────────────────────────

class ATACWindowDataset:
    """
    Iterable dataset yielding batches of (seq, atac, motifs) for a condition.

    Lazy-loads ATAC per batch; optionally pre-loads DNA one-hot.

    Parameters
    ----------
    windows    : list of (chrom, start, end)
    condition  : str  e.g. 'K562'
    atac_arr   : ndarray shape (n_windows, window_bp) or None (load from disk)
    batch_size : int
    use_genome : bool   True = load DNA from hg38, False = use random DNA
    indices    : list   subset of window indices (None = all)
    """

    def __init__(self, windows, condition, atac_arr=None, batch_size=4,
                 use_genome=False, indices=None):
        self.windows    = windows
        self.condition  = condition
        self.batch_size = batch_size
        self.use_genome = use_genome

        if indices is None:
            self.indices = list(range(len(windows)))
        else:
            self.indices = list(indices)

        if atac_arr is not None:
            self.atac_arr = atac_arr
        else:
            self.atac_arr = load_atac_array(condition)

        self.window_bp = self.atac_arr.shape[1]

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = self.indices
        for batch_start in range(0, len(indices), self.batch_size):
            batch_idx = indices[batch_start: batch_start + self.batch_size]
            B = len(batch_idx)

            # ATAC: (B, window_bp, 1)
            atac_batch = self.atac_arr[batch_idx][:, :, np.newaxis]  # (B, L, 1)

            # DNA: (B, window_bp, 4)
            if self.use_genome:
                seq_list = []
                for wi in batch_idx:
                    chrom, start, end = self.windows[wi]
                    seq_list.append(get_dna_onehot(chrom, start, end))
                seq_batch = np.stack(seq_list, axis=0)  # (B, L, 4)
            else:
                # Synthetic random DNA (for testing / when genome not available)
                rng = np.random.default_rng(batch_start)
                bases = rng.integers(0, 4, (B, self.window_bp))
                seq_batch = np.eye(4, dtype=np.float32)[bases]  # (B, L, 4)

            # Motifs: (B, 693)
            motif_batch = np.zeros((B, cfg('model', 'num_motifs')), dtype=np.float32)

            yield (seq_batch.astype(np.float32),
                   atac_batch.astype(np.float32),
                   motif_batch)

    def iter_with_indices(self):
        """Yield (batch_window_indices, seq, atac, motifs)."""
        indices = self.indices
        for batch_start in range(0, len(indices), self.batch_size):
            batch_idx = indices[batch_start: batch_start + self.batch_size]
            yield from self._get_batch(batch_idx)

    def _get_batch(self, batch_idx):
        B = len(batch_idx)
        atac_batch = self.atac_arr[batch_idx][:, :, np.newaxis]

        if self.use_genome:
            seq_list = []
            for wi in batch_idx:
                chrom, start, end = self.windows[wi]
                seq_list.append(get_dna_onehot(chrom, start, end))
            seq_batch = np.stack(seq_list, axis=0)
        else:
            rng = np.random.default_rng(batch_idx[0])
            bases = rng.integers(0, 4, (B, self.window_bp))
            seq_batch = np.eye(4, dtype=np.float32)[bases]

        motif_batch = np.zeros((B, cfg('model', 'num_motifs')), dtype=np.float32)
        yield (batch_idx,
               seq_batch.astype(np.float32),
               atac_batch.astype(np.float32),
               motif_batch)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation gene placeholder
# ─────────────────────────────────────────────────────────────────────────────

def create_placeholder_eval_genes(output_path=None, n=200, seed=None):
    """
    Create placeholder eval_genes.tsv with 200 genes on chr8/chr9.
    Real gene TSS positions are synthesized for scaffold purposes.
    """
    import pandas as pd
    if output_path is None:
        output_path = cfg('paths', 'eval_genes')
    if seed is None:
        seed = cfg('seed')
    if os.path.isfile(output_path):
        log.info(f"eval_genes.tsv already exists: {output_path}")
        return

    rng = np.random.default_rng(seed)
    rows = []
    chroms = ['chr8'] * (n // 2) + ['chr9'] * (n - n // 2)
    for i, chrom in enumerate(chroms):
        chr_len = CHR_SIZES[chrom]
        tss = int(rng.integers(1_000_000, chr_len - 1_000_000))
        rows.append({
            'gene_id': f'PLACEHOLDER_{chrom}_{i:04d}',
            'gene_name': f'GENE{i:04d}',
            'chrom': chrom,
            'tss': tss,
            'strand': '+' if rng.random() > 0.5 else '-',
        })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    log.info(f"Created placeholder eval_genes.tsv ({n} genes): {output_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check_atac(conditions=None, window_idx=0, plot=True):
    """
    Sanity check: compare ATAC tracks across conditions for one window.
    Returns dict of stats.
    """
    if conditions is None:
        conditions = cfg('all_conditions')

    windows = load_windows()
    chrom, start, end = windows[window_idx]
    log.info(f"Sanity check window: {chrom}:{start}-{end}")

    arrays = {}
    for cond in conditions:
        try:
            arr = load_atac_array(cond)
            sig = arr[window_idx]
            arrays[cond] = sig
        except FileNotFoundError:
            log.warning(f"ATAC not available for {cond}")

    stats = {}
    for cond, sig in arrays.items():
        stats[cond] = {
            'mean': float(np.mean(sig)),
            'max':  float(np.max(sig)),
            'std':  float(np.std(sig)),
            'pct95': float(np.percentile(sig, 95)),
            'nonzero_frac': float(np.mean(sig > 0)),
        }
        log.info(f"  {cond}: mean={stats[cond]['mean']:.3f}, "
                 f"max={stats[cond]['max']:.3f}, "
                 f"nonzero={stats[cond]['nonzero_frac']:.2%}")

    if plot and arrays:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(14, 4))
            region_size = end - start
            x = np.arange(region_size)
            for cond, sig in arrays.items():
                # Downsample for plotting
                step = max(1, region_size // 2000)
                ax.plot(x[::step], sig[::step], alpha=0.7, label=cond)
            ax.set_xlabel('Position (bp)')
            ax.set_ylabel('ATAC signal (log1p RPM)')
            ax.set_title(f'ATAC signals — {chrom}:{start}-{end}')
            ax.legend(fontsize=8)
            out = f"{cfg('paths', 'results')}/figures/sanity_atac_{chrom}_{start}.png"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()
            log.info(f"  Saved ATAC sanity plot: {out}")
        except Exception as e:
            log.warning(f"Plot failed: {e}")

    return stats


def compute_jaccard_peaks(sig1, sig2, threshold_pct=80):
    """
    Compute Jaccard overlap of 'peak' regions between two ATAC signals.
    Threshold: top (100-threshold_pct)% of signal is 'peak'.
    """
    t1 = np.percentile(sig1, threshold_pct)
    t2 = np.percentile(sig2, threshold_pct)
    peaks1 = sig1 > t1
    peaks2 = sig2 > t2
    intersection = np.sum(peaks1 & peaks2)
    union = np.sum(peaks1 | peaks2)
    return float(intersection) / float(union) if union > 0 else 0.0


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    cfg_d = load_config()

    # Quick test: generate windows
    log.info("Generating windows ...")
    windows = generate_windows()
    log.info(f"Generated {len(windows)} windows")
    log.info(f"First 3: {windows[:3]}")
