"""Verify downloaded tokenizer variants through the real offline HF loader."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from srgc_research.dispatch.llama31 import model

HAS_TOKENIZER = all(
    importlib.util.find_spec(name) is not None
    for name in ("tokenizers", "transformers")
)
pytestmark = pytest.mark.skipif(
    not HAS_TOKENIZER, reason="requires the separate HF CPU test environment"
)

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    " + message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


@pytest.fixture(scope="module")
def reference_snapshot(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("llama-tokenizer")
    vocab = {f"token{i}": i for i in range(128256)}
    for token, token_id in model.LLAMA_SPECIAL_TOKENS.items():
        del vocab[f"token{token_id}"]
        vocab[token] = token_id
    raw = Tokenizer(WordLevel(vocab, unk_token="token0"))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        unk_token="token0",
        additional_special_tokens=list(model.LLAMA_SPECIAL_TOKENS),
        chat_template=CHAT_TEMPLATE,
    )
    tokenizer.save_pretrained(path)
    (path / "config.json").write_text(
        json.dumps({"model_type": "llama", "vocab_size": 128256})
    )
    (path / "model-00001-of-00004.safetensors").write_bytes(b"fixture weights")
    expected = {}
    for file in path.iterdir():
        data = file.read_bytes()
        expected[file.name] = (
            len(data),
            hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if file.suffix == ".json"
            else None,
        )
    # save_pretrained puts the chat template in its own file in Transformers 4.57.
    # The Hub pin's table only contains its required JSON and weight files.
    expected.pop("chat_template.jinja", None)
    return path, expected


@pytest.fixture
def snapshot(tmp_path, reference_snapshot):
    reference, expected = reference_snapshot
    folder = tmp_path / "models/Llama-3.1-8B-Instruct"
    shutil.copytree(reference, folder)
    environment = {"GROUP_VOLUME": str(tmp_path), "OM_WORK": str(tmp_path / "work")}
    with patch.object(model, "OFFICIAL_FILES", expected):
        yield folder, environment


def rewrite_config(path, transform):
    file = path / "tokenizer_config.json"
    content = json.loads(file.read_text())
    transform(content)
    # A real save/re-serialization: different bytes/size and a changed
    # name_or_path field, but the exact same tokenizer and chat rendering.
    content["name_or_path"] = "/downloaded/local/Llama-3.1-8B-Instruct"
    file.write_text(json.dumps(content, sort_keys=True) + "\n")


@pytest.mark.parametrize("inline_template", [False, True])
def test_different_size_metadata_loads_offline_and_is_bound_to_run(
    snapshot, inline_template
):
    from transformers import AutoTokenizer

    folder, environment = snapshot
    original_size = (folder / "tokenizer_config.json").stat().st_size
    if inline_template:
        (folder / "chat_template.jinja").unlink()
    rewrite_config(
        folder,
        lambda config: config.update(
            {"chat_template": CHAT_TEMPLATE} if inline_template else {}
        ),
    )
    assert (folder / "tokenizer_config.json").stat().st_size != original_size
    before = {file.name: file.read_bytes() for file in folder.iterdir()}
    # Both the validator and the later preparation/rank use the real HF loader.
    # Disable the Hub transport: there must be no hidden network fallback.
    with patch(
        "huggingface_hub.file_download.http_get",
        side_effect=AssertionError("network access forbidden"),
    ):
        path, identity = model.resolve_snapshot(environment)
        assert path == folder
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
        assert tokenizer.encode("<|eot_id|>", add_special_tokens=False) == [128009]
        assert model.resolve_snapshot(environment) == (path, identity)
    assert {file.name: file.read_bytes() for file in folder.iterdir()} == before
    marker = next(Path(environment["OM_WORK"]).rglob("verified-models/*.json"))
    record = json.loads(marker.read_text())
    assert record["files"]["tokenizer_config.json"]["sha256"] == hashlib.sha256(
        (folder / "tokenizer_config.json").read_bytes()
    ).hexdigest()
    assert record["snapshot_sha256"] == identity


def test_even_valid_metadata_changes_cannot_replace_existing_snapshot(snapshot):
    folder, environment = snapshot
    _, identity = model.resolve_snapshot(environment)
    marker = next(Path(environment["OM_WORK"]).rglob("verified-models/*.json"))
    before = marker.read_bytes()
    rewrite_config(folder, lambda config: None)
    with pytest.raises(ValueError, match="changed since the first verification"):
        model.resolve_snapshot(environment)
    assert marker.read_bytes() == before
    assert json.loads(before)["snapshot_sha256"] == identity


@pytest.mark.parametrize(
    "change",
    ["eos", "bos", "decoder", "class", "template", "vocab", "bad-json", "list"],
)
def test_invalid_metadata_is_rejected_before_weight_reads(snapshot, change):
    folder, environment = snapshot

    def corrupt(config):
        if change == "eos":
            config["eos_token"] = "<|end_of_text|>"
        elif change == "bos":
            config["bos_token"] = "<|eot_id|>"
        elif change == "decoder":
            config["added_tokens_decoder"]["128009"]["content"] = "new-token"
        elif change == "class":
            config["tokenizer_class"] = "BertTokenizerFast"
        elif change == "vocab":
            config["added_tokens_decoder"]["128256"] = {
                "content": "new-token",
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "single_word": False,
                "special": True,
            }

    rewrite_config(folder, corrupt)
    if change == "template":
        (folder / "chat_template.jinja").write_text("{{ messages[0]['content'] }}")
    elif change == "bad-json":
        (folder / "tokenizer_config.json").write_text("{")
    elif change == "list":
        (folder / "tokenizer_config.json").write_text("[]")
    original_open = Path.open

    def open_without_weights(path, *args, **kwargs):
        if path.suffix == ".safetensors":
            pytest.fail("invalid metadata should be rejected before hashing weights")
        return original_open(path, *args, **kwargs)

    with (
        patch.object(Path, "open", open_without_weights),
        pytest.raises(ValueError, match="Invalid local Llama tokenizer metadata"),
    ):
        model.resolve_snapshot(environment)
    assert not list(Path(environment["OM_WORK"]).rglob("verified-models/*.json"))


@pytest.mark.parametrize("file", ["tokenizer.json", "config.json"])
def test_other_pinned_files_still_require_exact_content(snapshot, file):
    folder, environment = snapshot
    content = (folder / file).read_bytes()
    # Keep the byte count identical so the pinned content check must catch it.
    changed = (
        content.replace(b"128256", b"128255")
        if file == "config.json"
        else content[:-1] + b" "
    )
    assert changed != content and len(changed) == len(content)
    (folder / file).write_bytes(changed)
    with pytest.raises(ValueError, match="differs from pinned snapshot"):
        model.resolve_snapshot(environment)


def test_named_chat_templates_are_included_in_snapshot_identity(snapshot):
    folder, environment = snapshot
    templates = folder / "chat_templates"
    templates.mkdir()
    (templates / "tools.jinja").write_text(CHAT_TEMPLATE)
    rewrite_config(folder, lambda config: None)
    model.resolve_snapshot(environment)
    marker = next(Path(environment["OM_WORK"]).rglob("verified-models/*.json"))
    before = marker.read_bytes()
    assert "chat_templates/tools.jinja" in json.loads(before)["files"]
    (templates / "tools.jinja").write_text(CHAT_TEMPLATE + " ")
    with pytest.raises(ValueError, match="changed since the first verification"):
        model.resolve_snapshot(environment)
    assert marker.read_bytes() == before
