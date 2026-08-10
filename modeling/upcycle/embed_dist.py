from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM


LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Visualize pairwise cosine similarities for a list of .npy embeddings or "
			"for matching 2D weight tensors inside a Hugging Face checkpoint."
		)
	)
	source_group = parser.add_mutually_exclusive_group(required=True)
	source_group.add_argument(
		"--embedding-files",
		nargs="+",
		help="One or more .npy embedding files. Each file is treated as one embedding.",
	)
	source_group.add_argument(
		"--model-path",
		help="Local checkpoint directory/file or Hugging Face model ID to inspect.",
	)
	parser.add_argument(
		"--block-search-key",
		help="Substring used to select model tensors in --model-path mode.",
	)
	parser.add_argument(
		"--load-with-automodel",
		action="store_true",
		help=(
			"Load --model-path through transformers.AutoModelForCausalLM.from_pretrained() "
			"and inspect its state_dict instead of downloading checkpoint shards directly."
		),
	)
	parser.add_argument(
		"--trust-remote-code",
		action="store_true",
		help="Pass trust_remote_code=True when using AutoModelForCausalLM loading.",
	)
	parser.add_argument(
		"--output-dir",
		required=True,
		help="Directory where all heatmap images will be written.",
	)
	parser.add_argument(
		"--figsize",
		type=float,
		nargs=2,
		default=(8.0, 6.0),
		metavar=("WIDTH", "HEIGHT"),
		help="Matplotlib figure size in inches.",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=180,
		help="PNG DPI for saved heatmaps.",
	)
	parser.add_argument(
		"--max-embeddings",
		type=int,
		default=10,
		help="Maximum number of embeddings to compare pairwise.",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="Logging verbosity.",
	)
	args = parser.parse_args()

	if args.model_path and not args.block_search_key:
		parser.error("--block-search-key is required when using --model-path.")
	if args.max_embeddings < 2:
		parser.error("--max-embeddings must be at least 2.")
	return args


def configure_logging(level: str) -> None:
	logging.basicConfig(level=getattr(logging, level), format="%(levelname)s: %(message)s")


def sanitize_name(value: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "tensor"


def to_vector(array: np.ndarray, source_name: str) -> np.ndarray:
	if array.size == 0:
		raise ValueError(f"{source_name} is empty")
	vector = np.asarray(array, dtype=np.float32).reshape(-1)
	if not np.any(vector):
		LOG.warning("%s is all zeros; cosine similarities may be undefined but will be stabilized.", source_name)
	return vector


def tensor_to_float32_numpy(tensor: torch.Tensor) -> np.ndarray:
	return tensor.detach().to(device="cpu", dtype=torch.float32).numpy()


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
	matrix = np.asarray(vectors, dtype=np.float32)
	if matrix.ndim != 2:
		raise ValueError(f"Expected a 2D array of row vectors, got shape {matrix.shape}")
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	norms = np.clip(norms, 1e-12, None)
	normalized = matrix / norms
	return normalized @ normalized.T


def save_heatmap(similarity: np.ndarray, labels: list[str], title: str, output_path: Path, figsize: tuple[float, float], dpi: int) -> None:
	fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
	image = ax.imshow(similarity, cmap="viridis", vmin=-1.0, vmax=1.0)
	ax.set_title(title)
	ax.set_xticks(range(len(labels)))
	ax.set_xticklabels(labels, rotation=45, ha="right")
	ax.set_yticks(range(len(labels)))
	ax.set_yticklabels(labels)

	for row in range(similarity.shape[0]):
		for col in range(similarity.shape[1]):
			ax.text(col, row, f"{similarity[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)

	fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi)
	plt.close(fig)
	LOG.info("Wrote %s", output_path)


def load_embedding_file_vectors(paths: list[str]) -> tuple[np.ndarray, list[str]]:
	vectors: list[np.ndarray] = []
	labels: list[str] = []
	reference_shape: tuple[int, ...] | None = None

	for idx, path_str in enumerate(paths):
		path = Path(path_str)
		array = np.load(path)
		vector = to_vector(array, str(path))
		if reference_shape is None:
			reference_shape = vector.shape
		elif vector.shape != reference_shape:
			raise ValueError(
				f"Embedding {idx} from {path} has shape {vector.shape}, expected {reference_shape}"
			)
		vectors.append(vector)
		labels.append(f"embedding {idx}")

	if len(vectors) < 2:
		raise ValueError("Need at least two embedding files to compute pairwise cosine similarity.")
	return np.stack(vectors, axis=0), labels


def resolve_model_source(model_path: str) -> Path:
	candidate = Path(model_path)
	if candidate.exists():
		return candidate.resolve()

	LOG.info("Downloading model snapshot for %s", model_path)
	snapshot_path = snapshot_download(
		repo_id=model_path,
		allow_patterns=[
			"*.safetensors",
			"*.bin",
			"*.pt",
			"*.json",
		],
	)
	return Path(snapshot_path)


def iter_safetensor_matches(file_path: Path, block_search_key: str) -> Iterator[tuple[str, torch.Tensor]]:
	with safe_open(str(file_path), framework="pt", device="cpu") as handle:
		for key in handle.keys():
			if block_search_key in key:
				yield key, handle.get_tensor(key)


def iter_torch_state_dict_matches(file_path: Path, block_search_key: str) -> Iterator[tuple[str, torch.Tensor]]:
	state_dict = torch.load(file_path, map_location="cpu")
	if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
		state_dict = state_dict["state_dict"]
	if not isinstance(state_dict, dict):
		raise ValueError(f"Unsupported checkpoint format in {file_path}")
	for key, value in state_dict.items():
		if block_search_key in key and isinstance(value, torch.Tensor):
			yield key, value.detach().cpu()


def iter_index_matches(index_path: Path, block_search_key: str) -> Iterator[tuple[str, Path, str]]:
	with index_path.open("r") as handle:
		index_data = json.load(handle)
	weight_map = index_data.get("weight_map")
	if not isinstance(weight_map, dict):
		raise ValueError(f"Index file {index_path} does not contain a weight_map")

	for tensor_name, shard_name in weight_map.items():
		if block_search_key in tensor_name:
			yield tensor_name, index_path.parent / shard_name, shard_name


def collect_automodel_matches(
	model_path: str,
	block_search_key: str,
	trust_remote_code: bool,
) -> list[tuple[str, np.ndarray]]:
	LOG.info("Loading %s via AutoModelForCausalLM", model_path)
	model = AutoModelForCausalLM.from_pretrained(
		model_path,
		trust_remote_code=trust_remote_code,
		device_map="cpu",
	)
	try:
		matches: list[tuple[str, np.ndarray]] = []
		for tensor_name, tensor in model.state_dict().items():
			if block_search_key not in tensor_name:
				continue
			if tensor.ndim != 2:
				LOG.warning("Skipping %s with shape %s; expected a 2D weight matrix.", tensor_name, tuple(tensor.shape))
				continue
			zero_row_mask = torch.count_nonzero(tensor, dim=1) == 0
			if bool(zero_row_mask.any()):
				LOG.warning(
					"Tensor %s contains %d all-zero rows.",
					tensor_name,
					int(zero_row_mask.sum().item()),
				)
			matches.append((tensor_name, tensor_to_float32_numpy(tensor)))
		return matches
	finally:
		del model


def collect_matching_weight_matrices(
	model_path: str,
	block_search_key: str,
	load_with_automodel: bool = False,
	trust_remote_code: bool = False,
) -> list[tuple[str, np.ndarray]]:
	if load_with_automodel:
		return collect_automodel_matches(model_path, block_search_key, trust_remote_code)

	source = resolve_model_source(model_path)
	matches: list[tuple[str, np.ndarray]] = []

	if source.is_file():
		files = [source]
	else:
		safetensor_index = source / "model.safetensors.index.json"
		pytorch_index = source / "pytorch_model.bin.index.json"
		if safetensor_index.exists():
			shard_to_keys: dict[Path, list[str]] = {}
			for tensor_name, shard_path, _ in iter_index_matches(safetensor_index, block_search_key):
				shard_to_keys.setdefault(shard_path, []).append(tensor_name)
			for shard_path, tensor_names in shard_to_keys.items():
				with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
					for tensor_name in tensor_names:
						tensor = handle.get_tensor(tensor_name)
						if tensor.ndim != 2:
							LOG.warning("Skipping %s with shape %s; expected a 2D weight matrix.", tensor_name, tuple(tensor.shape))
							continue
						matches.append((tensor_name, tensor_to_float32_numpy(tensor)))
			return matches
		if pytorch_index.exists():
			shard_to_keys = {}
			for tensor_name, shard_path, _ in iter_index_matches(pytorch_index, block_search_key):
				shard_to_keys.setdefault(shard_path, []).append(tensor_name)
			for shard_path, tensor_names in shard_to_keys.items():
				state_dict = torch.load(shard_path, map_location="cpu")
				if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
					state_dict = state_dict["state_dict"]
				for tensor_name in tensor_names:
					tensor = state_dict.get(tensor_name)
					if not isinstance(tensor, torch.Tensor):
						LOG.warning("Skipping %s from %s because it is not a tensor.", tensor_name, shard_path)
						continue
					if tensor.ndim != 2:
						LOG.warning("Skipping %s with shape %s; expected a 2D weight matrix.", tensor_name, tuple(tensor.shape))
						continue
					matches.append((tensor_name, tensor_to_float32_numpy(tensor)))
			return matches

		files = sorted(source.glob("*.safetensors")) + sorted(source.glob("*.bin")) + sorted(source.glob("*.pt"))

	for file_path in files:
		if file_path.suffix == ".safetensors":
			iterator = iter_safetensor_matches(file_path, block_search_key)
		elif file_path.suffix in {".bin", ".pt"}:
			iterator = iter_torch_state_dict_matches(file_path, block_search_key)
		else:
			continue

		for tensor_name, tensor in iterator:
			if tensor.ndim != 2:
				LOG.warning("Skipping %s with shape %s; expected a 2D weight matrix.", tensor_name, tuple(tensor.shape))
				continue
			matches.append((tensor_name, tensor_to_float32_numpy(tensor)))

	return matches


def render_embedding_file_heatmap(
	embedding_files: list[str],
	output_dir: Path,
	figsize: tuple[float, float],
	dpi: int,
	max_embeddings: int,
) -> None:
	vectors, labels = load_embedding_file_vectors(embedding_files[:max_embeddings])
	similarity = cosine_similarity_matrix(vectors)
	save_heatmap(
		similarity=similarity,
		labels=labels,
		title="Pairwise cosine similarity",
		output_path=output_dir / "embedding_pairwise_cossim.png",
		figsize=figsize,
		dpi=dpi,
	)


def render_model_weight_heatmaps(
	model_path: str,
	block_search_key: str,
	output_dir: Path,
	figsize: tuple[float, float],
	dpi: int,
	max_embeddings: int,
	load_with_automodel: bool,
	trust_remote_code: bool,
) -> None:
	matches = collect_matching_weight_matrices(
		model_path,
		block_search_key,
		load_with_automodel=load_with_automodel,
		trust_remote_code=trust_remote_code,
	)
	if not matches:
		raise ValueError(
			f"No 2D tensors matching block search key '{block_search_key}' were found in {model_path}"
		)

	for tensor_name, matrix in matches:
		matrix = matrix[:max_embeddings]
		if matrix.shape[0] < 2:
			LOG.warning("Skipping %s because it has fewer than 2 embeddings after truncation.", tensor_name)
			continue
		labels = [f"embedding {idx}" for idx in range(matrix.shape[0])]
		similarity = cosine_similarity_matrix(matrix)
		output_path = output_dir / f"{sanitize_name(tensor_name)}_pairwise_cossim.png"
		save_heatmap(
			similarity=similarity,
			labels=labels,
			title=tensor_name,
			output_path=output_path,
			figsize=figsize,
			dpi=dpi,
		)


def main() -> None:
	args = parse_args()
	configure_logging(args.log_level)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	if args.embedding_files:
		render_embedding_file_heatmap(
			args.embedding_files,
			output_dir,
			tuple(args.figsize),
			args.dpi,
			args.max_embeddings,
		)
		return

	render_model_weight_heatmaps(
		model_path=args.model_path,
		block_search_key=args.block_search_key,
		output_dir=output_dir,
		figsize=tuple(args.figsize),
		dpi=args.dpi,
		max_embeddings=args.max_embeddings,
		load_with_automodel=args.load_with_automodel,
		trust_remote_code=args.trust_remote_code,
	)


if __name__ == "__main__":
	main()
