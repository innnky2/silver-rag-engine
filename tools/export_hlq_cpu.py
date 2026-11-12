#!/usr/bin/env python3
"""export_hlq_cpu.py — PROD exporter: real 1-bit plane packing (no manifest-only).

Reads real HLQ quant output and writes real packed planes + scales + manifest.
No synthetic data, no RNG, no downloads. scales.json is REQUIRED (no .npy
fallback: numpy layouts varied across checkpoints and silently packing the
wrong layout would fake the HLQ number).

Expected --resume_quant layout (from HLQ-GPTQ / block_ap / e2e_qp):
  <dir>/
    scales.json        # {"q": 2, "group_size": 128, "tensors": {"name": {"s": [...], "z": ...}}}
    <tensor>.bin      # raw float32 bit-values per plane (q * ngroups floats in [0,1])
  scales.json "s" is ONE scale per plane (global-scale prototype path in
  hlq::bitserial_gemv_tmac). Per-group scales g=128 (ELUTQ Tables 7-8/12-14)
  need quantizer-side per-group output + hlq::bitserial_gemv_tmac_pergroup;
  this exporter does NOT invent per-group scales by tiling globals.
  Q4 tensors that are already binary are rejected (packs from float
  bit-values, not from GGUF).

Output (<save_path>/):
  hlq_cpu_manifest.json
  planes.bin            # q bit-planes concatenated, each (ngroups+7)//8 bytes,
                        # 1 bit per group LSB-first (plane j at byte offset
                        # j*plane_bytes). Consumed by hlq::bitserial_gemv_tmac_packed.
  scales.f32            # q float32 scales per tensor, concatenated (global-scale
                        # prototype; per-group output is future work, see above)
  zhat.f32              # per-tensor zhat (single float32, D.2)
  manifest["tensors"]   # per tensor: {name, q, ngroups, plane_bytes}

The manifest records the exact on-disk layout so the C++ side never guesses:
bytes_per_plane = (ngroups + 7) // 8. A read-back popcount check verifies
planes.bin bit counts against the packed inputs before exit.

Bit order is the PROD contract consumed by hlq::bitserial_gemv_tmac*().
"""
from __future__ import annotations
import argparse
import json
import os
import struct
import sys


def _load_scales(d: str):
    p = os.path.join(d, "scales.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _pack_bits(bitvals, ngroups: int) -> bytes:
    out = bytearray((ngroups + 7) // 8)
    for g, b in enumerate(bitvals):
        if b:
            out[g // 8] |= (1 << (g % 8))
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack HLQ checkpoint to HLQ-CPU planes (real).")
    ap.add_argument("--resume_quant", required=True, help="HLQ quant output dir.")
    ap.add_argument("--wbits", type=int, default=2, choices=[2, 3])
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--save_path", required=True, help="HLQ-CPU output dir.")
    args = ap.parse_args()

    meta = _load_scales(args.resume_quant)
    if meta is None:
        print("REFUSED: %s/scales.json not found (no real scales to pack; "
              "manifest-only output is deleted synthetic behavior)." % args.resume_quant,
              file=sys.stderr)
        return 2
    try:
        q = int(meta.get("q", args.wbits))
        tensors = meta["tensors"]
    except Exception as e:
        print("REFUSED: bad scales.json: %s" % e, file=sys.stderr)
        return 2
    if q != args.wbits:
        print("REFUSED: scales.json q=%d != --wbits %d" % (q, args.wbits), file=sys.stderr)
        return 2

    os.makedirs(args.save_path, exist_ok=True)
    planes_path = os.path.join(args.save_path, "planes.bin")
    scales_path = os.path.join(args.save_path, "scales.f32")
    zhat_path = os.path.join(args.save_path, "zhat.f32")
    with open(planes_path, "wb") as fp, open(scales_path, "wb") as fs:
        zhats = {}
        tmeta = []
        for name in sorted(tensors):
            t = tensors[name]
            try:
                s = [float(x) for x in t["s"]]
                z = float(t.get("z", 0.0))
                raw = os.path.join(args.resume_quant, name + ".bin")
                with open(raw, "rb") as fb:
                    data = fb.read()
                ngroups = len(data) // (4 * max(1, len(s)))
                if ngroups <= 0 or len(s) != q:
                    print("REFUSED: tensor %s has inconsistent planes" % name, file=sys.stderr)
                    return 2
                nfloats = len(data) // 4
                vals = struct.unpack("<%df" % nfloats, data)
                # vals layout: q planes x ngroups floats in [0,1]
                plane_bytes = (ngroups + 7) // 8
                expect_pop = []
                for j in range(q):
                    bits = [1 if vals[j * ngroups + g] >= 0.5 else 0 for g in range(ngroups)]
                    expect_pop.append(sum(bits))
                    fp.write(_pack_bits(bits, ngroups))
                    fs.write(struct.pack("<f", s[j]))
                shat_sum = sum(x * 0.5 for x in s)
                zhats[name] = z + shat_sum
                tmeta.append({"name": name, "q": q, "ngroups": ngroups,
                              "plane_bytes": plane_bytes, "popcount": expect_pop})
            except (KeyError, OSError, struct.error) as e:
                print("REFUSED: tensor %s unreadable: %s" % (name, e), file=sys.stderr)
                return 2
    # Read-back verification: popcount each plane of planes.bin and compare
    # against the packed inputs (content check, not just lengths: this is
    # exactly what C++ unpack_bitplane_lsb8 will decode).
    with open(planes_path, "rb") as fp:
        blob = fp.read()
    off = 0
    for tm in tmeta:
        for j in range(tm["q"]):
            pb = blob[off:off + tm["plane_bytes"]]
            if len(pb) != tm["plane_bytes"]:
                print("REFUSED: planes.bin truncated at %s plane %d" % (tm["name"], j),
                      file=sys.stderr)
                return 2
            got = sum(bin(byte).count("1") for byte in pb)
            # Padding bits beyond ngroups are zero by construction (_pack_bits),
            # so the file popcount must equal the input popcount exactly.
            if got != tm["popcount"][j]:
                print("REFUSED: planes.bin content mismatch at %s plane %d "
                      "(popcount %d != %d)" % (tm["name"], j, got, tm["popcount"][j]),
                      file=sys.stderr)
                return 2
            off += tm["plane_bytes"]
    if off != len(blob):
        print("REFUSED: planes.bin has %d trailing bytes" % (len(blob) - off), file=sys.stderr)
        return 2
    with open(zhat_path, "wb") as fz:
        for name in sorted(zhats):
            fz.write(struct.pack("<f", float(zhats[name])))
    manifest = {
        "format": "hlq-cpu-v1",
        "wbits": args.wbits,
        "group_size": args.group_size,
        "source_quant_dir": args.resume_quant,
        "layout": "bitplane-LSB-first: plane j of each tensor at byte offset "
                  "j*plane_bytes, plane_bytes=(ngroups+7)//8; tensors sorted by "
                  "name, planes concatenated in manifest order",
        "mirror": "complement S-lut8[15-idx] for {0,1} LUTs; negation only for bipolar D.2",
        "tensors": tmeta,
        "files": ["planes.bin", "scales.f32", "zhat.f32"],
        "status": "PACKED-REAL (no measurement claims)",
    }
    with open(os.path.join(args.save_path, "hlq_cpu_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("HLQ-CPU packed: " + args.save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
