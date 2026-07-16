#!/usr/bin/env python3
# coding=utf-8

"""Generate ARCHS4 sample embeddings from sharded parquet files.

This script streams all samples from ARCHS4 shards, computes one embedding per sample
using a trained ExpressionPerformer checkpoint, and stores:

1) Embeddings in a float16 memmap matrix for efficient FAISS ingestion.
2) Per-sample metadata (global index, shard location, accession, species) in parquet.

Designed for >1M samples by avoiding full dataset materialization in RAM.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryExpressionEmbedding(nn.Module):
	def __init__(self, dim: int, base: float = 100.0, mask_token_id: float = -10.0):
		super().__init__()
		self.dim = int(dim)
		self.mask_token_id = mask_token_id
		inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer("inv_freq", inv_freq)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_mask_idx = (x == self.mask_token_id).nonzero(as_tuple=False)
		freqs = torch.einsum("bi,j->bij", x, self.inv_freq)
		emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
		if len(x_mask_idx) > 0:
			emb[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = 0
		return emb


class FlashTransformerLayer(nn.Module):
	def __init__(self, hidden_dim: int, ffn_dim: int, n_heads: int):
		super().__init__()
		if hidden_dim % n_heads != 0:
			raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
		self.hidden_dim = int(hidden_dim)
		self.n_heads = int(n_heads)
		self.head_dim = self.hidden_dim // self.n_heads

		self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

		self.norm1 = nn.LayerNorm(self.hidden_dim)
		self.norm2 = nn.LayerNorm(self.hidden_dim)
		self.ffn = nn.Sequential(
			nn.Linear(self.hidden_dim, int(ffn_dim)),
			nn.GELU(),
			nn.Linear(int(ffn_dim), self.hidden_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		h = self.norm1(x)
		bsz, seqlen, _ = h.shape
		q = self.q_proj(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
		k = self.k_proj(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
		v = self.v_proj(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
		attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
		attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_dim)
		x = x + self.out_proj(attn)
		x = x + self.ffn(self.norm2(x))
		return x


class ExpressionPerformer(nn.Module):
	def __init__(
		self,
		num_genes: int,
		hidden_dim: int,
		n_heads: int,
		n_layers: int,
		ffn_dim: int,
		ree_base: float,
		mask_token_id: float,
		feature_type: str,
		compute_type: str,
		include_species_embedding: bool,
		num_species: int,
	):
		super().__init__()
		self.num_genes = int(num_genes)
		self.hidden_dim = int(hidden_dim)
		self.include_species_embedding = bool(include_species_embedding)
		self.feature_type = str(feature_type).strip().lower()
		self.use_flash = self.feature_type == "flash"

		self.gene_embedding = nn.Embedding(self.num_genes, self.hidden_dim)
		self.ree = RotaryExpressionEmbedding(self.hidden_dim, base=ree_base, mask_token_id=mask_token_id)

		if self.include_species_embedding:
			self.species_embedding = nn.Embedding(int(num_species), self.hidden_dim)

		if self.use_flash:
			self.layers = nn.ModuleList([
				FlashTransformerLayer(self.hidden_dim, int(ffn_dim), int(n_heads))
				for _ in range(int(n_layers))
			])
		else:
			# Import SLiM layer lazily so flash-only inference does not require
			# SLiM-specific dependency modules at import time.
			from slim_performer_model import SLiMPerformerLayer

			self.layers = nn.ModuleList([
				SLiMPerformerLayer(
					self.hidden_dim,
					int(ffn_dim),
					int(n_heads),
					feature_type,
					compute_type,
					on_gptln=True,
				)
				for _ in range(int(n_layers))
			])

		self.output_map = nn.Linear(self.hidden_dim, 1)

	def _encode_hidden(self, x: torch.Tensor, species_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
		bsz, genes = x.shape
		device = x.device
		gene_ids = torch.arange(genes, device=device)
		gene_emb = self.gene_embedding(gene_ids)
		ree_emb = self.ree(x)
		h = gene_emb.unsqueeze(0) + ree_emb

		if self.include_species_embedding:
			if species_ids is None:
				species_ids = torch.zeros(bsz, dtype=torch.long, device=device)
			species_emb = self.species_embedding(species_ids.long())
			h = h + species_emb.unsqueeze(1)

		if self.use_flash:
			for layer in self.layers:
				h = layer(h)
		else:
			for layer in self.layers:
				rfs = layer.attention.sample_rfs(device)
				h = layer.full_forward(h, rfs)

		return h

	def forward(self, x: torch.Tensor, species_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
		h = self._encode_hidden(x, species_ids)
		out = self.output_map(h).squeeze(-1)
		return out

	@torch.no_grad()
	def encode(self, x: torch.Tensor, species_ids: Optional[torch.Tensor] = None, normalize: bool = False) -> torch.Tensor:
		h = self._encode_hidden(x, species_ids)
		sample_emb = h.mean(dim=1)
		if normalize:
			sample_emb = F.normalize(sample_emb, dim=-1)
		return sample_emb


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate ARCHS4 sample embeddings at scale.")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path("./data/archs4/train_orthologs_sharded"),
		help="ARCHS4 sharded dataset directory containing batch_files/.",
	)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		required=True,
		help="Checkpoint path (best_model.pt/latest.pt/epoch_XX.pt).",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("./prepared_data/archs4_sample_embeddings"),
		help="Output directory for memmap embeddings and metadata.",
	)
	parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
	parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
	parser.add_argument(
		"--l2-normalize",
		action="store_true",
		help="If set, L2-normalize sample embeddings before writing outputs.",
	)
	parser.add_argument(
		"--normalization",
		choices=["checkpoint", "tpm", "log1p_tpm"],
		default="checkpoint",
		help="Input normalization before encoding.",
	)
	parser.add_argument(
		"--metadata-flush-size",
		type=int,
		default=100000,
		help="Rows per metadata parquet write flush.",
	)
	parser.add_argument(
		"--heartbeat-seconds",
		type=int,
		default=30,
		help="Print progress heartbeat every N seconds (0 disables periodic heartbeat).",
	)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument(
		"--max-shards",
		type=int,
		default=0,
		help="If > 0, only process the first N shards (quick test mode).",
	)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=0,
		help="If > 0, stop after writing this many samples (quick test mode).",
	)
	parser.add_argument("--overwrite", action="store_true")
	return parser.parse_args()


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	if any(k.startswith("module.") for k in state_dict):
		return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
	return state_dict


def _load_species_map(data_dir: Path) -> Dict[str, int]:
	# Prefer metadata.csv for complete species labels.
	meta_csv = data_dir / "metadata.csv"
	samples_json = data_dir / "samples.json"
	sample_to_species: Dict[str, str] = {}

	if meta_csv.exists():
		df = pd.read_csv(meta_csv, usecols=["geo_accession", "species"])
		for _, row in df.iterrows():
			sid = str(row["geo_accession"])
			sp = str(row["species"]).strip().lower()
			if sid and sp:
				sample_to_species[sid] = sp
	elif samples_json.exists():
		with open(samples_json, "r") as f:
			samples = json.load(f)
		if isinstance(samples, list):
			for s in samples:
				if isinstance(s, dict) and "id" in s and "species" in s:
					sample_to_species[str(s["id"])] = str(s["species"]).strip().lower()

	# Canonical id mapping for optional species embedding.
	species_to_id: Dict[str, int] = {}
	for sp in ["human", "mouse"]:
		if any(v.startswith(sp) for v in sample_to_species.values()):
			species_to_id[sp] = len(species_to_id)
	for sp in sorted(set(sample_to_species.values())):
		key = "human" if sp.startswith("human") else "mouse" if sp.startswith("mouse") else sp
		if key not in species_to_id:
			species_to_id[key] = len(species_to_id)

	# Re-map sample values to canonical ids.
	sample_to_species_id: Dict[str, int] = {}
	for sid, sp in sample_to_species.items():
		key = "human" if sp.startswith("human") else "mouse" if sp.startswith("mouse") else sp
		sample_to_species_id[sid] = species_to_id.get(key, 0)
	return sample_to_species_id


def _resolve_norm(arg_norm: str, ckpt_cfg: Dict[str, object]) -> str:
	if arg_norm == "checkpoint":
		return str(ckpt_cfg.get("normalization", "log1p_tpm"))
	return arg_norm


def _estimate_total_rows(batch_files: List[Path]) -> Tuple[int, List[int]]:
	row_counts: List[int] = []
	total = 0
	for p in batch_files:
		n = int(pq.ParquetFile(str(p)).metadata.num_rows)
		row_counts.append(n)
		total += n
	return total, row_counts


def _write_metadata_chunk(writer: pq.ParquetWriter, chunk: Dict[str, List[object]]) -> None:
	table = pa.table(
		{
			"global_index": pa.array(chunk["global_index"], type=pa.int64()),
			"shard_idx": pa.array(chunk["shard_idx"], type=pa.int32()),
			"shard_file": pa.array(chunk["shard_file"], type=pa.string()),
			"row_in_shard": pa.array(chunk["row_in_shard"], type=pa.int32()),
			"geo_accession": pa.array(chunk["geo_accession"], type=pa.string()),
			"species_id": pa.array(chunk["species_id"], type=pa.int16()),
		}
	)
	writer.write_table(table)


def _format_seconds(seconds: float) -> str:
	seconds = max(0, int(seconds))
	hours = seconds // 3600
	minutes = (seconds % 3600) // 60
	secs = seconds % 60
	return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
	args = parse_args()
	data_dir = args.data_dir.resolve()
	batch_dir = data_dir / "batch_files"
	if not batch_dir.exists():
		batch_dir = data_dir

	batch_files = sorted(batch_dir.glob("*.parquet"))
	if not batch_files:
		raise FileNotFoundError(f"No parquet files found in {batch_dir}")
	if args.max_shards > 0:
		batch_files = batch_files[: args.max_shards]

	ckpt_path = args.checkpoint.resolve()
	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

	out_dir = args.output_dir.resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	emb_path = out_dir / f"sample_embeddings.{args.dtype}.mmap"
	meta_path = out_dir / "sample_locations.parquet"
	manifest_path = out_dir / "embedding_manifest.json"

	if not args.overwrite:
		for p in [emb_path, meta_path, manifest_path]:
			if p.exists():
				raise FileExistsError(f"Output exists ({p}). Use --overwrite to replace.")

	print(f"[INFO] Loading checkpoint: {ckpt_path}", flush=True)
	ckpt = torch.load(ckpt_path, map_location="cpu")
	cfg = dict(ckpt.get("config", {}))

	first_pf = pq.ParquetFile(str(batch_files[0]))
	cols = first_pf.schema_arrow.names
	gene_cols = [c for c in cols if c not in ("geo_accession", "__index_level_0__", "gene_symbol")]
	num_genes = len(gene_cols)
	if num_genes <= 0:
		raise RuntimeError("Could not infer gene columns from parquet schema.")

	sample_to_species_id = _load_species_map(data_dir)
	num_species = max(1, max(sample_to_species_id.values(), default=0) + 1)

	model = ExpressionPerformer(
		num_genes=num_genes,
		hidden_dim=int(cfg.get("hidden_dim", 512)),
		n_heads=int(cfg.get("num_heads", 8)),
		n_layers=int(cfg.get("num_layers", 4)),
		ffn_dim=int(cfg.get("ffn_dim", int(cfg.get("hidden_dim", 512)) * 4)),
		ree_base=float(cfg.get("ree_base", 100.0)),
		mask_token_id=float(cfg.get("mask_token", -10.0)),
		feature_type=str(cfg.get("feature_type", "sqr")),
		compute_type=str(cfg.get("compute_type", "iter")),
		include_species_embedding=bool(cfg.get("include_species_embedding", False)),
		num_species=num_species,
	)

	state = _strip_module_prefix(ckpt["model_state_dict"])
	missing, unexpected = model.load_state_dict(state, strict=False)
	if missing:
		print(f"[WARN] Missing model keys: {len(missing)}", flush=True)
	if unexpected:
		print(f"[WARN] Unexpected model keys: {len(unexpected)}", flush=True)

	device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
	model.to(device)
	model.eval()

	emb_dim = int(cfg.get("hidden_dim", 512))
	total_rows_full, row_counts = _estimate_total_rows(batch_files)
	target_rows = total_rows_full if args.max_samples <= 0 else min(total_rows_full, args.max_samples)
	print(f"[INFO] Total samples available: {total_rows_full:,} across {len(batch_files)} shards", flush=True)
	print(f"[INFO] Target samples to write: {target_rows:,}", flush=True)
	print(f"[INFO] Embedding dim: {emb_dim}", flush=True)

	np_dtype = np.float16 if args.dtype == "float16" else np.float32
	emb_mmap = np.memmap(emb_path, dtype=np_dtype, mode="w+", shape=(target_rows, emb_dim))

	meta_schema = pa.schema(
		[
			("global_index", pa.int64()),
			("shard_idx", pa.int32()),
			("shard_file", pa.string()),
			("row_in_shard", pa.int32()),
			("geo_accession", pa.string()),
			("species_id", pa.int16()),
		]
	)
	meta_writer = pq.ParquetWriter(meta_path, meta_schema)

	norm_mode = _resolve_norm(args.normalization, cfg)
	print(f"[INFO] Input normalization: {norm_mode}", flush=True)
	print(f"[INFO] L2 normalize embeddings: {args.l2_normalize}", flush=True)

	t0 = time.time()
	last_heartbeat = t0
	global_idx = 0
	meta_chunk = {
		"global_index": [],
		"shard_idx": [],
		"shard_file": [],
		"row_in_shard": [],
		"geo_accession": [],
		"species_id": [],
	}

	for shard_idx, shard_path in enumerate(batch_files):
		if global_idx >= target_rows:
			break

		pf = pq.ParquetFile(str(shard_path))
		shard_rows = row_counts[shard_idx]
		shard_name = shard_path.name

		shard_written = 0
		for rg in range(pf.metadata.num_row_groups):
			if global_idx >= target_rows:
				break

			table = pf.read_row_group(rg, columns=gene_cols + ["geo_accession"], use_threads=True)
			acc = np.asarray(table.column("geo_accession").to_pylist(), dtype=object)
			x = np.column_stack(
				[table.column(c).combine_chunks().to_numpy(zero_copy_only=False) for c in gene_cols]
			).astype(np.float32, copy=False)

			if norm_mode == "log1p_tpm":
				x = np.log1p(np.maximum(x, 0.0)).astype(np.float32, copy=False)

			species_ids_np = np.asarray([sample_to_species_id.get(str(a), 0) for a in acc], dtype=np.int64)

			row_base = shard_written
			n_rows = x.shape[0]
			for start in range(0, n_rows, args.batch_size):
				if global_idx >= target_rows:
					break

				end = min(start + args.batch_size, n_rows)
				remaining = target_rows - global_idx
				if end - start > remaining:
					end = start + remaining
				xb = torch.from_numpy(x[start:end]).to(device)
				sb = torch.from_numpy(species_ids_np[start:end]).to(device)

				with torch.no_grad(), torch.amp.autocast(
					device_type="cuda",
					dtype=torch.bfloat16,
					enabled=(device.type == "cuda"),
				):
					emb = model.encode(xb, sb, normalize=args.l2_normalize)
				emb_np = emb.detach().float().cpu().numpy().astype(np_dtype, copy=False)

				out_start = global_idx
				out_end = global_idx + emb_np.shape[0]
				emb_mmap[out_start:out_end] = emb_np

				for i in range(emb_np.shape[0]):
					row_in_shard = row_base + start + i
					geo = str(acc[start + i])
					meta_chunk["global_index"].append(global_idx + i)
					meta_chunk["shard_idx"].append(shard_idx)
					meta_chunk["shard_file"].append(shard_name)
					meta_chunk["row_in_shard"].append(int(row_in_shard))
					meta_chunk["geo_accession"].append(geo)
					meta_chunk["species_id"].append(int(species_ids_np[start + i]))

				global_idx = out_end

				now = time.time()
				if args.heartbeat_seconds > 0 and (now - last_heartbeat) >= args.heartbeat_seconds:
					elapsed = now - t0
					rate = global_idx / max(elapsed, 1e-9)
					remaining = target_rows - global_idx
					eta_s = remaining / rate if rate > 1e-9 else float("inf")
					pct = (100.0 * global_idx / target_rows) if target_rows > 0 else 100.0
					eta_str = _format_seconds(eta_s) if np.isfinite(eta_s) else "inf"
					print(
						f"[HEARTBEAT] {global_idx:,}/{target_rows:,} ({pct:.2f}%) | "
						f"{rate:.1f} samples/s | elapsed {_format_seconds(elapsed)} | ETA {eta_str}",
						flush=True,
					)
					last_heartbeat = now

				if len(meta_chunk["global_index"]) >= args.metadata_flush_size:
					_write_metadata_chunk(meta_writer, meta_chunk)
					for k in meta_chunk:
						meta_chunk[k].clear()

			shard_written += n_rows

		if global_idx < target_rows and shard_written != shard_rows:
			raise RuntimeError(
				f"Shard row mismatch for {shard_name}: expected {shard_rows}, wrote {shard_written}"
			)

		elapsed = time.time() - t0
		rate = global_idx / max(elapsed, 1e-9)
		print(
			f"[INFO] Shard {shard_idx+1}/{len(batch_files)} done | "
			f"samples written: {global_idx:,}/{target_rows:,} | {rate:.1f} samples/s",
			flush=True,
		)

	if meta_chunk["global_index"]:
		_write_metadata_chunk(meta_writer, meta_chunk)
	meta_writer.close()
	emb_mmap.flush()

	if global_idx != target_rows:
		raise RuntimeError(f"Expected {target_rows} embeddings, wrote {global_idx}.")

	manifest = {
		"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
		"checkpoint": str(ckpt_path),
		"data_dir": str(data_dir),
		"batch_dir": str(batch_dir),
		"total_samples": int(target_rows),
		"total_samples_available": int(total_rows_full),
		"embedding_dim": int(emb_dim),
		"embedding_dtype": str(args.dtype),
		"l2_normalize": bool(args.l2_normalize),
		"normalization": norm_mode,
		"feature_type": str(cfg.get("feature_type", "sqr")),
		"compute_type": str(cfg.get("compute_type", "iter")),
		"max_shards": int(args.max_shards),
		"max_samples": int(args.max_samples),
		"memmap_path": str(emb_path),
		"metadata_path": str(meta_path),
		"output_dir": str(out_dir),
	}
	with open(manifest_path, "w") as f:
		json.dump(manifest, f, indent=2)

	total_s = time.time() - t0
	print(f"[DONE] Wrote {target_rows:,} embeddings in {total_s/60:.1f} min", flush=True)
	print(f"[DONE] Embeddings: {emb_path}", flush=True)
	print(f"[DONE] Metadata:   {meta_path}", flush=True)
	print(f"[DONE] Manifest:   {manifest_path}", flush=True)


if __name__ == "__main__":
	main()

