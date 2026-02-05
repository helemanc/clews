"""
Compute CLEWS distance matrix for retrieval evaluation.

Usage:
    python compute_distance_matrix.py \
        checkpoint_dir=/path/to/checkpoints \
        dataset=SHS100K \
        output_dir=/path/to/save/distance_matrix \
        path_meta=/path/to/datasets/shs/ \
        path_audio=/path/to/audio/ \
        partition=test \
        ngpus=4

The script will:
1. Load CLEWS model from checkpoint directory
2. Extract embeddings for all songs in the partition
3. Compute pairwise distances between queries and candidates
4. Save distance matrix with metadata to pickle file
5. Compute and print retrieval metrics (MAP, MR1, ARP)
"""

import hashlib
import importlib
import math
import os
import pickle
import sys

import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import OmegaConf

from lib import dataset, eval
from lib import tensor_ops as tops
from utils import print_utils, pytorch_utils

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

args = OmegaConf.from_cli()

# Required arguments
assert "checkpoint_dir" in args, "Must provide checkpoint_dir=/path/to/checkpoints"
assert "dataset" in args, "Must provide dataset=SHS100K or dataset=DiscogsVI"
assert "output_dir" in args, "Must provide output_dir=/path/to/save"
assert "path_meta" in args, "Must provide path_meta=/path/to/metadata"
assert "path_audio" in args, "Must provide path_audio=/path/to/audio"

# Determine checkpoint paths based on dataset
dataset_lower = args.dataset.lower()
if dataset_lower in ["shs", "shs100k"]:
    checkpoint_name = "shs-clews"
    dataset_short = "shs"
elif dataset_lower in ["dvi", "discogs-vi", "discogsvi"]:
    checkpoint_name = "dvi-clews"
    dataset_short = "dvi"
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}. Use shs or dvi")

checkpoint_path = os.path.join(
    args.checkpoint_dir, checkpoint_name, "checkpoint_best.ckpt"
)
config_path = os.path.join(args.checkpoint_dir, checkpoint_name)

# Basic settings
if "ngpus" not in args:
    args.ngpus = 1
if "nnodes" not in args:
    args.nnodes = 1
args.precision = "32"  # CLEWS uses fp32

if "partition" not in args:
    args.partition = "test"
if "limit_num" not in args:
    args.limit_num = None

# Extraction parameters
if "maxlen" not in args:  # maximum audio length in seconds
    args.maxlen = 10 * 60
if "redux" not in args:  # distance reduction strategy
    args.redux = None
if "qslen" not in args:  # query shingle length
    args.qslen = None
if "qshop" not in args:  # query shingle hop (default = every 5 sec)
    args.qshop = 5
if "cslen" not in args:  # candidate shingle length
    args.cslen = None
if "cshop" not in args:  # candidate shingle hop (default = every 5 sec)
    args.cshop = 5

# ============================================================================
# INITIALIZE PYTORCH/FABRIC
# ============================================================================

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)

fabric = Fabric(
    accelerator="cuda",
    devices=args.ngpus,
    num_nodes=args.nnodes,
    strategy=DDPStrategy(broadcast_buffers=False),
    precision=args.precision,
)
fabric.launch()

# Seed for reproducibility
fabric.barrier()
fabric.seed_everything(44 + fabric.global_rank, workers=True)

# Utilities
myprint = lambda s, end="\n": print_utils.myprint(
    s, end=end, doit=fabric.is_global_zero
)
myprogbar = lambda it, desc=None, leave=False: print_utils.myprogbar(
    it, desc=desc, leave=leave, doit=fabric.is_global_zero
)
timer = print_utils.Timer()
fabric.barrier()

# ============================================================================
# LOAD MODEL
# ============================================================================

myprint("=" * 100)
myprint(f"CLEWS Distance Matrix Computation - {args.dataset}")
myprint("=" * 100)
myprint(OmegaConf.to_yaml(args))
myprint("=" * 100)

myprint("Load model configuration...")
conf = OmegaConf.load(os.path.join(config_path, "configuration.yaml"))

myprint("Initialize model...")
module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = module.Model(conf.model, sr=conf.data.samplerate)
model = fabric.setup(model)

myprint(f"Load checkpoint: {checkpoint_path}")
state = pytorch_utils.get_state(model, None, None, conf, None, None, None)
fabric.load(checkpoint_path, state)
model, _, _, conf, epoch, _, best = pytorch_utils.set_state(state)
myprint(f"  ({epoch} epochs; best was {best:.3f})")
model.eval()

# Override paths with command-line arguments
conf.path.audio = args.path_audio
conf.path.meta = args.path_meta
conf.data.path = conf.path

# ============================================================================
# SETUP DATASET
# ============================================================================

myprint("Setup dataset...")
dset = dataset.Dataset(
    conf.data,
    args.partition,
    augment=False,
    fullsongs=True,
    verbose=fabric.is_global_zero,
)
dloader = torch.utils.data.DataLoader(
    dset,
    batch_size=1,
    shuffle=False,
    num_workers=0,  # 0 is needed when working with cache paths
    drop_last=False,
    pin_memory=False,
)
dloader = fabric.setup_dataloaders(dloader)

# Build mapping: raw sequential id -> deterministic id (same as WEALY)
# This ensures version IDs match between CLEWS and WEALY distance matrices
id_to_det_id = {}
for version_key, vinfo in dset.info.items():
    raw_id = vinfo["id"]
    clique_str, version_str = version_key.split("-", 1)
    combined = f"{clique_str}-{version_str}"
    hash_bytes = hashlib.md5(combined.encode("utf-8")).digest()
    det_id = int.from_bytes(hash_bytes[:4], byteorder="big") & 0x7FFFFFFF
    id_to_det_id[raw_id] = det_id
myprint(f"Built deterministic ID mapping for {len(id_to_det_id)} versions")

# ============================================================================
# EXTRACTION FUNCTION
# ============================================================================


@torch.inference_mode()
def extract_embeddings(shingle_len, shingle_hop, dset, desc="Embed", eps=1e-6):
    """Extract CLEWS embeddings with shingles."""

    # Check shingle args
    shinglen, shinghop = model.get_shingle_params()
    if shingle_len is not None:
        shinglen = shingle_len
    if shingle_hop is not None:
        shinghop = shingle_hop
    mxlen = int(args.maxlen * model.sr)
    numshingles = int((mxlen - int(shinglen * model.sr)) / int(shinghop * model.sr))

    # Extract embeddings
    all_c = []
    all_i = []
    all_z = []
    all_m = []

    for batch_idx, batch in enumerate(myprogbar(dloader, desc=desc, leave=True)):
        # Get info & audio
        c, i, x = batch[:3]
        if x.size(1) > mxlen:
            x = x[:, :mxlen]

        # Get embedding (B=1, S, C)
        z = model(
            x,
            shingle_len=int(x.size(1) / model.sr) if shinglen <= 0 else shinglen,
            shingle_hop=int(0.99 * x.size(1) / model.sr) if shinghop <= 0 else shinghop,
        )

        # Make embedding shingles same size
        z = tops.force_length(
            z,
            1 if shinglen <= 0 else numshingles,
            dim=1,
            pad_mode="zeros",
            cut_mode="start",
        )
        m = z.abs().max(-1)[0] < eps

        # Append
        all_c.append(c)
        all_i.append(i)
        all_z.append(z)
        all_m.append(m)

        # Limit number of queries/candidates?
        if args.limit_num is not None and len(all_z) >= args.limit_num / args.ngpus:
            myprint("")
            myprint("  [Max num reached]")
            break

    # Concat single-song batches
    all_c = torch.cat(all_c, dim=0)
    all_i = torch.cat(all_i, dim=0)
    all_z = torch.cat(all_z, dim=0)
    all_m = torch.cat(all_m, dim=0)

    return all_c, all_i, all_z, all_m


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

with torch.inference_mode():
    # Extract query embeddings
    query_c, query_i, query_z, query_m = extract_embeddings(
        args.qslen, args.qshop, dset, desc="Query emb"
    )
    query_c = query_c.int()
    query_i = query_i.int()
    query_z = query_z.half()

    # Extract candidate embeddings (reuse if same params)
    if args.cslen == args.qslen and args.cshop == args.qshop:
        myprint("Cand emb: (copy)")
        cand_c, cand_i, cand_z, cand_m = (
            query_c.clone(),
            query_i.clone(),
            query_z.clone(),
            query_m.clone(),
        )
    else:
        cand_c, cand_i, cand_z, cand_m = extract_embeddings(
            args.cslen, args.cshop, dset, desc="Cand emb"
        )
        cand_c = cand_c.int()
        cand_i = cand_i.int()
        cand_z = cand_z.half()

    # Collect candidates from all GPUs + collapse to batch dim
    fabric.barrier()
    cand_c = fabric.all_gather(cand_c)
    cand_i = fabric.all_gather(cand_i)
    cand_z = fabric.all_gather(cand_z)
    cand_m = fabric.all_gather(cand_m)
    cand_c = torch.cat(torch.unbind(cand_c, dim=0), dim=0)
    cand_i = torch.cat(torch.unbind(cand_i, dim=0), dim=0)
    cand_z = torch.cat(torch.unbind(cand_z, dim=0), dim=0)
    cand_m = torch.cat(torch.unbind(cand_m, dim=0), dim=0)

    # Collect queries from all GPUs for distance matrix computation
    fabric.barrier()
    all_query_c = fabric.all_gather(query_c)
    all_query_i = fabric.all_gather(query_i)
    all_query_z = fabric.all_gather(query_z)
    all_query_m = fabric.all_gather(query_m)
    all_query_c = torch.cat(torch.unbind(all_query_c, dim=0), dim=0)
    all_query_i = torch.cat(torch.unbind(all_query_i, dim=0), dim=0)
    all_query_z = torch.cat(torch.unbind(all_query_z, dim=0), dim=0)
    all_query_m = torch.cat(torch.unbind(all_query_m, dim=0), dim=0)

    # Evaluate and collect distances
    aps = []
    r1s = []
    rpcs = []
    all_distances = []

    for n in myprogbar(range(len(query_z)), desc="Retrieve", leave=True):
        # Compute distances for distance matrix
        # Replicate exact distance computation from eval.compute
        if 2**15 >= len(cand_i):
            dist = model.distances(
                query_z[n : n + 1].float(),
                cand_z.float(),
                qmask=query_m[n : n + 1] if query_m is not None else None,
                cmask=cand_m,
                redux_strategy=args.redux,
            ).squeeze(0)
        else:
            dist = []
            for mstart in range(0, len(cand_i), 2**15):
                mend = min(mstart + 2**15, len(cand_i))
                ddd = model.distances(
                    query_z[n : n + 1].float(),
                    cand_z[mstart:mend].float(),
                    qmask=query_m[n : n + 1] if query_m is not None else None,
                    cmask=cand_m[mstart:mend] if cand_m is not None else None,
                    redux_strategy=args.redux,
                ).squeeze(0)
                dist.append(ddd)
            dist = torch.cat(dist, dim=-1)

        # Store raw distances
        all_distances.append(dist.cpu())

        # Regular evaluation
        ap, r1, rpc = eval.compute(
            model,
            query_c[n : n + 1],
            query_i[n : n + 1],
            query_z[n : n + 1],
            cand_c,
            cand_i,
            cand_z,
            queries_m=query_m[n : n + 1],
            candidates_m=cand_m,
            redux_strategy=args.redux,
            batch_size_candidates=2**15,
        )
        aps.append(ap)
        r1s.append(r1)
        rpcs.append(rpc)

    aps = torch.stack(aps)
    r1s = torch.stack(r1s)
    rpcs = torch.stack(rpcs)

    # Collect measures from all GPUs + collapse to batch dim
    fabric.barrier()
    aps = fabric.all_gather(aps)
    r1s = fabric.all_gather(r1s)
    rpcs = fabric.all_gather(rpcs)
    aps = torch.cat(torch.unbind(aps, dim=0), dim=0)
    r1s = torch.cat(torch.unbind(r1s, dim=0), dim=0)
    rpcs = torch.cat(torch.unbind(rpcs, dim=0), dim=0)

    # Compute and save distance matrix (only on global rank 0)
    fabric.barrier()

    # Collect distances from all GPUs
    if len(all_distances) > 0:
        # Stack distances from this GPU
        local_distances = torch.stack(all_distances)  # (local_queries, n_candidates)

        # Gather from all GPUs
        all_gpu_distances = fabric.all_gather(
            local_distances
        )  # (n_gpus, local_queries, n_candidates)

        # Concatenate along query dimension
        distance_matrix = torch.cat(
            torch.unbind(all_gpu_distances, dim=0), dim=0
        )  # (total_queries, n_candidates)
    else:
        distance_matrix = torch.zeros(len(all_query_c), len(cand_c))

    # Only save on rank 0
    if fabric.is_global_zero:
        myprint("Preparing distance matrix for saving...")

        # Create reference data with exact correspondence to distance matrix indices
        # Use deterministic IDs (same as WEALY) so fusion alignment works
        query_references = []
        for matrix_row_idx in range(len(all_query_c)):
            raw_id = all_query_i[matrix_row_idx].item()
            query_references.append(
                {
                    "clique": all_query_c[matrix_row_idx].item(),
                    "version": id_to_det_id[raw_id],
                    "matrix_row": matrix_row_idx,
                    "original_index": matrix_row_idx,
                }
            )

        candidate_references = []
        for matrix_col_idx in range(len(cand_c)):
            raw_id = cand_i[matrix_col_idx].item()
            candidate_references.append(
                {
                    "clique": cand_c[matrix_col_idx].item(),
                    "version": id_to_det_id[raw_id],
                    "matrix_col": matrix_col_idx,
                    "original_index": matrix_col_idx,
                }
            )

        # Verification - ensure correspondence is correct
        myprint("Verifying distance matrix correspondence...")
        assert len(query_references) == distance_matrix.shape[0], (
            f"Query mismatch: {len(query_references)} vs {distance_matrix.shape[0]}"
        )
        assert len(candidate_references) == distance_matrix.shape[1], (
            f"Candidate mismatch: {len(candidate_references)} vs {distance_matrix.shape[1]}"
        )

        myprint("✓ Distance matrix correspondence verified!")

        # Prepare data to save
        distance_data = {
            "distance_matrix": distance_matrix.cpu().numpy(),
            "query_references": query_references,
            "candidate_references": candidate_references,
            "metadata": {
                "n_queries": len(all_query_c),
                "n_candidates": len(cand_c),
                "dataset": args.dataset,
                "checkpoint": checkpoint_path,
                "partition": args.partition,
                "qslen": args.qslen,
                "qshop": args.qshop,
                "cslen": args.cslen,
                "cshop": args.cshop,
                "maxlen": args.maxlen,
                "redux": args.redux,
                "verification_note": "matrix_row/matrix_col fields ensure exact correspondence",
            },
        }

        # Construct output filename
        output_filename = f"{dataset_short}_clews_distance_matrix_{args.partition}.pkl"
        output_path = os.path.join(args.output_dir, output_filename)

        # Save distance matrix and references
        myprint(f"Saving distance matrix to: {output_path}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(distance_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        myprint(f"Distance matrix shape: {distance_matrix.shape}")
        myprint(
            f"Saved {len(query_references)} query references and {len(candidate_references)} candidate references"
        )
        myprint("Matrix[i,j] = distance from query_refs[i] to candidate_refs[j]")

# ============================================================================
# PRINT RESULTS
# ============================================================================

logdict_mean = {
    "MAP": aps.mean(),
    "MR1": r1s.mean(),
    "ARP": rpcs.mean(),
}
logdict_ci = {
    "MAP": 1.96 * aps.std() / math.sqrt(len(aps)),
    "MR1": 1.96 * r1s.std() / math.sqrt(len(r1s)),
    "ARP": 1.96 * rpcs.std() / math.sqrt(len(rpcs)),
}
myprint("=" * 100)
myprint("Result:")
myprint("  Avg --> " + print_utils.report(logdict_mean, clean_line=False))
myprint("  c.i. -> " + print_utils.report(logdict_ci, clean_line=False))
myprint("=" * 100)
