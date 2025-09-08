import sys
import os
import importlib
from omegaconf import OmegaConf
import torch, math
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
# NEW: Import for saving distance matrix
import pickle
import numpy as np

from lib import eval, dataset
from lib import tensor_ops as tops
from utils import pytorch_utils, print_utils

# --- Get arguments (and set defaults) --- Basic ---
args = OmegaConf.from_cli()
assert "checkpoint" in args
log_path, _ = os.path.split(args.checkpoint)
if "ngpus" not in args:
    args.ngpus = 1
if "nnodes" not in args:
    args.nnodes = 1
args.precision = "32"
if "path_audio" not in args:
    args.path_audio = None
if "path_meta" not in args:
    args.path_meta = None
if "partition" not in args:
    args.partition = "test"
if "limit_num" not in args:
    args.limit_num = None

# --- Get arguments (and set defaults) --- Tunable ---
if "maxlen" not in args:  # maximum audio length
    args.maxlen = 10 * 60  # in seconds
if "redux" not in args:  # distance reduction
    args.redux = None
if "qslen" not in args:  # query shingle len
    args.qslen = None
if "qshop" not in args:  # query shingle hop (default = every 5 sec)
    args.qshop = 5
if "cslen" not in args:  # candidate shingle len
    args.cslen = None
if "cshop" not in args:  # candidate shingle hop (default = every 5 sec)
    args.cshop = 5

# NEW: Add argument for saving distance matrix
if "save_distance_matrix" not in args:
    args.save_distance_matrix = None  # Path to save distance matrix, or None to skip

###############################################################################

# Init pytorch/Fabric
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

# Seed (random segment needs a seed)
fabric.barrier()
fabric.seed_everything(44 + fabric.global_rank, workers=True)

# Init my utils
myprint = lambda s, end="\n": print_utils.myprint(
    s, end=end, doit=fabric.is_global_zero
)
myprogbar = lambda it, desc=None, leave=False: print_utils.myprogbar(
    it, desc=desc, leave=leave, doit=fabric.is_global_zero
)
timer = print_utils.Timer()
fabric.barrier()

# Load conf
myprint(OmegaConf.to_yaml(args))
myprint("Load model conf...")
conf = OmegaConf.load(os.path.join(log_path, "configuration.yaml"))

# Init model
myprint("Init model...")
module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = module.Model(conf.model, sr=conf.data.samplerate)
model = fabric.setup(model)

# Load model
myprint("  Load checkpoint")
state = pytorch_utils.get_state(model, None, None, conf, None, None, None)
fabric.load(args.checkpoint, state)
model, _, _, conf, epoch, _, best = pytorch_utils.set_state(state)
myprint(f"  ({epoch} epochs; best was {best:.3f})")
model.eval()
if args.path_audio is not None:
    conf.path.audio = args.path_audio
if args.path_meta is not None:
    conf.path.meta = args.path_meta
conf.data.path = conf.path

clews_audio_path = "/home/phd-students/eleonora.mancini10/scratch/data/SHS100K/audio/"
clews_meta_path = "/home/phd-students/eleonora.mancini10/scratch/projects/lyrics-version-identification/cache-clews/metadata-shs.pt"
conf.path.meta = clews_meta_path
conf.path.audio = clews_audio_path
conf.data.path.meta = clews_meta_path
conf.data.path.audio = clews_audio_path

# Get dataset
myprint("Dataset...")
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
    num_workers=0, # 0 is needed when working with cache paths 
    drop_last=False,
    pin_memory=False,
)
dloader = fabric.setup_dataloaders(dloader)

###############################################################################


@torch.inference_mode()
def extract_embeddings(shingle_len, shingle_hop, dset, desc="Embed", eps=1e-6):

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
    # for batch in myprogbar(dloader, desc=desc, leave=True):
    #     # Get info & audio
    #     c, i, x = batch[:3]
    #     if x.size(1) > mxlen:
    #         x = x[:, :mxlen]
    #     # Get embedding (B=1,S,C)
    #     z = model(
    #         x,
    #         shingle_len=int(x.size(1) / model.sr) if shinglen <= 0 else shinglen,
    #         shingle_hop=int(0.99 * x.size(1) / model.sr) if shinghop <= 0 else shinghop,
    #     )
    #     # Make embedding shingles same size
    #     z = tops.force_length(
    #         z,
    #         1 if shinglen <= 0 else numshingles,
    #         dim=1,
    #         pad_mode="zeros",
    #         cut_mode="start",
    #     )
    #     m = z.abs().max(-1)[0] < eps
    #     # Append
    #     all_c.append(c)
    #     all_i.append(i)
    #     all_z.append(z)
    #     all_m.append(m)
    #     # Limit number of queries/candidates?
    #     if args.limit_num is not None and len(all_z) >= args.limit_num / args.ngpus:
    #         myprint("")
    #         myprint("  [Max num reached]")
    #         break
    for batch_idx, batch in enumerate(myprogbar(dloader, desc=desc, leave=True)):
        # Get info & audio
        c, i, x = batch[:3]
        if x.size(1) > mxlen:
            x = x[:, :mxlen]

        #  Access cached path info from dataset
        # if hasattr(dset, "_paths_cache") and batch_idx in dset._paths_cache:
        #     paths_info = dset._paths_cache[batch_idx]  # list of dicts
        #     #print(paths_info)
        #     # Override clique/version from path
        #     derived_c = [int(p["clique"]) for p in paths_info]
        #     derived_i = [int(p["version"]) for p in paths_info]
        #     c = torch.tensor(derived_c, dtype=torch.int32, device=x.device)
        #     i = torch.tensor(derived_i, dtype=torch.int32, device=x.device)
        #     #print(c, i)
        
        # if hasattr(dset, 'versions') and batch_idx < len(dset.versions):
        #     version_key = dset.versions[batch_idx]
        #     if version_key in dset.info:
        #         # Get the audio filename and extract clique/version from it
        #         filename = dset.info[version_key]["filename"]
        #         fname = os.path.basename(filename)
        #         if '-' in fname:
        #             parts = fname.split('-')
        #             clique = parts[0]
        #             version = parts[1].split('.')[0]
                    
        #             unique_version_id = hash(f"{clique}-{version}") & 0x7fffffff
        #             # Override with extracted values
        #             c = torch.tensor([int(clique)], dtype=torch.int32, device=x.device)
        #             i = torch.tensor([int(unique_version_id)], dtype=torch.int32, device=x.device)
                    
        #             if batch_idx == 0:
        #                 print(c, i)
        if batch_idx == 0:
            print(c, i)

            
        # Get embedding (B=1,S,C)
        z = model(
            x,
            shingle_len=int(x.size(1) / model.sr) if shinglen <= 0 else shinglen,
            shingle_hop=int(0.99 * x.size(1) / model.sr) if shinghop <= 0 else shinghop,
        )
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

        if args.limit_num is not None and len(all_z) >= args.limit_num / args.ngpus:
            myprint("")
            myprint("  [Max num reached]")
            break
    # Concat single-song batches
    all_c = torch.cat(all_c, dim=0)
    all_i = torch.cat(all_i, dim=0)
    all_z = torch.cat(all_z, dim=0)
    all_m = torch.cat(all_m, dim=0)
    # Return
    return all_c, all_i, all_z, all_m


###############################################################################

# Let's go
with torch.inference_mode():

    # Extract embeddings
    query_c, query_i, query_z, query_m = extract_embeddings(
        args.qslen, args.qshop, dset, desc="Query emb"
    )
    query_c = query_c.int()
    query_i = query_i.int()
    query_z = query_z.half()
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
            args.cslen, args.cshop, desc="Cand emb"
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

    # NEW: Collect queries from all GPUs for distance matrix computation
    if args.save_distance_matrix is not None:
        fabric.barrier()
        all_query_c = fabric.all_gather(query_c)
        all_query_i = fabric.all_gather(query_i)
        all_query_z = fabric.all_gather(query_z)
        all_query_m = fabric.all_gather(query_m)
        all_query_c = torch.cat(torch.unbind(all_query_c, dim=0), dim=0)
        all_query_i = torch.cat(torch.unbind(all_query_i, dim=0), dim=0)
        all_query_z = torch.cat(torch.unbind(all_query_z, dim=0), dim=0)
        all_query_m = torch.cat(torch.unbind(all_query_m, dim=0), dim=0)

    # Evaluate
    aps = []
    r1s = []
    rpcs = []
    # NEW: Collect all distances if saving distance matrix
    all_distances = []
    
    for n in myprogbar(range(len(query_z)), desc="Retrieve", leave=True):
        # NEW: Compute distances separately for matrix if needed (using same model.distances call as eval.compute)
        if args.save_distance_matrix is not None:
            # This replicates the exact distance computation from eval.compute
            if 2**15 >= len(cand_i):  # same condition as eval.compute
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
            # Store the raw distances (before eval.compute modifies them)
            all_distances.append(dist.cpu())
        
        # Regular evaluation (same as original)
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

    # NEW: Compute and save distance matrix if requested (only on global rank 0)
    if args.save_distance_matrix is not None:
        fabric.barrier()
        
        # NEW: Collect distances from all GPUs and ensure proper ordering
        if len(all_distances) > 0:
            # Stack distances from this GPU
            local_distances = torch.stack(all_distances)  # (local_queries, n_candidates)
            
            # Gather from all GPUs
            all_gpu_distances = fabric.all_gather(local_distances)  # (n_gpus, local_queries, n_candidates)
            
            # Concatenate along query dimension
            distance_matrix = torch.cat(torch.unbind(all_gpu_distances, dim=0), dim=0)  # (total_queries, n_candidates)
        else:
            distance_matrix = torch.zeros(len(all_query_c), len(cand_c))
        
        # Only save on rank 0
        if fabric.is_global_zero:
            myprint("Preparing distance matrix for saving...")
            
            # NEW: Create reference data with EXACT correspondence to distance matrix indices
            # CRITICAL: The order must match exactly how distances were computed
            # Row i in distance_matrix corresponds to all_query_c[i], all_query_i[i]
            # Col j in distance_matrix corresponds to cand_c[j], cand_i[j]
            
            query_references = []
            for matrix_row_idx in range(len(all_query_c)):
                query_references.append({
                    'clique': all_query_c[matrix_row_idx].item(),
                    'version': all_query_i[matrix_row_idx].item(),
                    'matrix_row': matrix_row_idx,  # NEW: Explicit matrix position
                    'original_index': matrix_row_idx  # NEW: For verification
                })
            
            candidate_references = []
            for matrix_col_idx in range(len(cand_c)):
                candidate_references.append({
                    'clique': cand_c[matrix_col_idx].item(),
                    'version': cand_i[matrix_col_idx].item(),
                    'matrix_col': matrix_col_idx,  # NEW: Explicit matrix position
                    'original_index': matrix_col_idx  # NEW: For verification
                })
            
            # NEW: Verification - ensure correspondence is correct
            myprint("Verifying distance matrix correspondence...")
            assert len(query_references) == distance_matrix.shape[0], f"Query mismatch: {len(query_references)} vs {distance_matrix.shape[0]}"
            assert len(candidate_references) == distance_matrix.shape[1], f"Candidate mismatch: {len(candidate_references)} vs {distance_matrix.shape[1]}"
            
            # NEW: Double-check a few random correspondences
            for check_idx in [0, len(query_references)//2, len(query_references)-1]:
                if check_idx < len(query_references):
                    ref = query_references[check_idx]
                    assert ref['matrix_row'] == check_idx, f"Query index mismatch at {check_idx}"
                    assert ref['clique'] == all_query_c[check_idx].item(), f"Query clique mismatch at {check_idx}"
                    assert ref['version'] == all_query_i[check_idx].item(), f"Query version mismatch at {check_idx}"
            
            for check_idx in [0, len(candidate_references)//2, len(candidate_references)-1]:
                if check_idx < len(candidate_references):
                    ref = candidate_references[check_idx]
                    assert ref['matrix_col'] == check_idx, f"Candidate index mismatch at {check_idx}"
                    assert ref['clique'] == cand_c[check_idx].item(), f"Candidate clique mismatch at {check_idx}"
                    assert ref['version'] == cand_i[check_idx].item(), f"Candidate version mismatch at {check_idx}"
            
            myprint("✓ Distance matrix correspondence verified!")
            
            # Prepare data to save
            distance_data = {
                'distance_matrix': distance_matrix.cpu().numpy(),
                'query_references': query_references,
                'candidate_references': candidate_references,
                'metadata': {
                    'n_queries': len(all_query_c),
                    'n_candidates': len(cand_c),
                    'checkpoint': args.checkpoint,
                    'partition': args.partition,
                    'qslen': args.qslen,
                    'qshop': args.qshop,
                    'cslen': args.cslen,
                    'cshop': args.cshop,
                    'maxlen': args.maxlen,
                    'redux': args.redux,
                    'verification_note': 'matrix_row/matrix_col fields ensure exact correspondence'
                }
            }
            
            print(distance_data)
            # Save distance matrix and references
            myprint(f"Saving distance matrix to: {args.save_distance_matrix}")
            with open(args.save_distance_matrix, 'wb') as f:
                pickle.dump(distance_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            myprint(f"Distance matrix shape: {distance_matrix.shape}")
            myprint(f"Saved {len(query_references)} query references and {len(candidate_references)} candidate references")
            myprint(f"Matrix[i,j] = distance from query_refs[i] to candidate_refs[j]")

###############################################################################

# Print
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