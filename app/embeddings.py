import logging

from app.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE, EMBEDDING_MODEL

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = str(exc)
else:
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = None


logger = logging.getLogger(__name__)
SUPPORTED_EMBEDDING_DEVICES = {"auto", "cpu", "cuda", "cuda:0", "cuda:1"}


def embedding_diagnostics() -> dict:
    diagnostics = _torch_diagnostics()
    diagnostics["sentence_transformers_error"] = SENTENCE_TRANSFORMERS_IMPORT_ERROR
    diagnostics["selected_device"] = select_embedding_device(
        diagnostics=diagnostics,
        log_details=False,
    )
    return diagnostics


def select_embedding_device(
    diagnostics: dict | None = None,
    log_details: bool = True
) -> str:
    diagnostics = diagnostics or _torch_diagnostics()
    configured_device = EMBEDDING_DEVICE.strip().lower()

    if configured_device not in SUPPORTED_EMBEDDING_DEVICES:
        logger.warning(
            "[EMBEDDING] Unsupported EMBEDDING_DEVICE=%s; falling back to auto",
            EMBEDDING_DEVICE,
        )
        configured_device = "auto"

    if configured_device == "cpu":
        selected_device = "cpu"
    elif configured_device == "auto":
        selected_device = "cuda" if diagnostics["cuda_available"] else "cpu"
    elif not diagnostics["cuda_available"]:
        logger.warning(
            "[EMBEDDING] EMBEDDING_DEVICE=%s requested but CUDA is not available; falling back to cpu",
            EMBEDDING_DEVICE,
        )
        selected_device = "cpu"
    elif not _cuda_device_exists(configured_device, diagnostics["device_count"]):
        logger.warning(
            "[EMBEDDING] EMBEDDING_DEVICE=%s requested but only %s CUDA device(s) were found; falling back to cpu",
            EMBEDDING_DEVICE,
            diagnostics["device_count"],
        )
        selected_device = "cpu"
    else:
        selected_device = configured_device

    if log_details:
        _log_embedding_diagnostics(diagnostics, selected_device)

    return selected_device


def _torch_diagnostics() -> dict:
    diagnostics = {
        "torch_version": None,
        "cuda_available": False,
        "torch_cuda_version": None,
        "device_count": 0,
        "devices": [],
        "torch_error": None,
    }

    try:
        import torch
    except Exception as exc:
        diagnostics["torch_error"] = str(exc)
        return diagnostics

    diagnostics["torch_version"] = getattr(torch, "__version__", None)
    diagnostics["torch_cuda_version"] = getattr(torch.version, "cuda", None)

    try:
        diagnostics["cuda_available"] = bool(torch.cuda.is_available())
        diagnostics["device_count"] = torch.cuda.device_count()
    except Exception as exc:
        diagnostics["torch_error"] = str(exc)
        return diagnostics

    for index in range(diagnostics["device_count"]):
        try:
            name = torch.cuda.get_device_name(index)
        except Exception as exc:
            name = f"Unavailable CUDA device {index}: {exc}"

        diagnostics["devices"].append({
            "index": index,
            "name": name,
        })

    return diagnostics


def _cuda_device_exists(device: str, device_count: int) -> bool:
    if device == "cuda":
        return device_count > 0

    try:
        requested_index = int(device.split(":", 1)[1])
    except (IndexError, ValueError):
        return False

    return 0 <= requested_index < device_count


def _selected_gpu_name(selected_device: str, diagnostics: dict) -> str | None:
    if not selected_device.startswith("cuda"):
        return None

    device_index = 0
    if ":" in selected_device:
        device_index = int(selected_device.split(":", 1)[1])

    for device in diagnostics["devices"]:
        if device["index"] == device_index:
            return device["name"]

    return None


def _log_embedding_diagnostics(diagnostics: dict, selected_device: str) -> None:
    logger.info("[EMBEDDING] PyTorch version: %s", diagnostics["torch_version"])
    logger.info(
        "[EMBEDDING] CUDA available: %s",
        str(diagnostics["cuda_available"]).lower(),
    )
    logger.info("[EMBEDDING] CUDA version: %s", diagnostics["torch_cuda_version"])
    logger.info("[EMBEDDING] Selected device: %s", selected_device)

    gpu_name = _selected_gpu_name(selected_device, diagnostics)
    if gpu_name:
        logger.info("[EMBEDDING] GPU: %s", gpu_name)

    if diagnostics["torch_error"]:
        logger.warning("[EMBEDDING] Torch diagnostics warning: %s", diagnostics["torch_error"])
    if diagnostics.get("sentence_transformers_error"):
        logger.warning(
            "[EMBEDDING] sentence-transformers import warning: %s",
            diagnostics["sentence_transformers_error"],
        )


class EmbeddingService:
    def __init__(self):
        self.model = None
        self.device = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    f"sentence-transformers could not be imported: {SENTENCE_TRANSFORMERS_IMPORT_ERROR}"
                )

            self.device = select_embedding_device()
            logger.info(
                "[EMBEDDING] Loading embedding model: %s on %s",
                EMBEDDING_MODEL,
                self.device,
            )
            self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
            logger.info(
                "[EMBEDDING] Embedding model loaded: %s on %s",
                EMBEDDING_MODEL,
                self.device,
            )

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
        ).tolist()
