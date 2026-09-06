"""Typed payload entries preserve KPAK v1 and legacy HSACO defaults."""

import pytest

from rocm_kpack.compression import NoOpCompressor, ZstdCompressor
from rocm_kpack.kpack import PackedKernelArchive


@pytest.mark.parametrize("compressor_type", [NoOpCompressor, ZstdCompressor])
@pytest.mark.parametrize(
    "payload_type,target",
    [
        ("hsaco", "amd:hip:gfx1201"),
        ("cubin", "nvidia:cuda:sm_120"),
        ("ptx", "nvidia:cuda:sm_120"),
        ("spirv", "intel:level-zero:xe2-b70"),
    ],
)
def test_payload_type_roundtrip(tmp_path, compressor_type, payload_type, target):
    archive = PackedKernelArchive(
        "typed", "multi-vendor", [target], compressor=compressor_type()
    )
    payload = b"opaque device payload" * 13
    prepared = archive.prepare_kernel(
        "module", target, payload, payload_type=payload_type
    )
    assert prepared.payload_type == payload_type
    archive.add_kernel(prepared)
    archive.finalize_archive()
    output = tmp_path / "typed.kpack"
    archive.write(output)
    loaded = PackedKernelArchive.read(output)
    assert loaded.toc["module"][target]["type"] == payload_type
    assert loaded.get_kernel("module", target) == payload
    assert loaded.get_kernel("module", "gfx1201") is None


def test_legacy_default_stays_hsaco(tmp_path):
    archive = PackedKernelArchive("legacy", "gfx1201", ["gfx1201"])
    # Preserve positional metadata and the existing default payload type.
    prepared = archive.prepare_kernel("module", "gfx1201", b"legacy", {"legacy": True})
    assert prepared.payload_type == "hsaco"
    archive.add_kernel(prepared)
    archive.finalize_archive()
    output = tmp_path / "legacy.kpack"
    archive.write(output)
    loaded = PackedKernelArchive.read(output)
    assert loaded.toc["module"]["gfx1201"]["type"] == "hsaco"
    assert loaded.toc["module"]["gfx1201"]["metadata"] == {"legacy": True}
    assert loaded.get_kernel("module", "gfx1201") == b"legacy"


def test_unknown_payload_type_rejected():
    archive = PackedKernelArchive("typed", "multi-vendor", [])
    with pytest.raises(ValueError, match="Unsupported payload type"):
        archive.prepare_kernel(
            "module", "nvidia:cuda:sm_120", b"payload", payload_type="unknown"
        )
