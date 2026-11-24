"""Revise ONE active plate so its forearms and hands read as a working man's.

Throwaway spike (src/ never imports spikes/). One image-edit call on the plate
alone - no mechanic photo, no uniform reference - at ~$0.014. Writes the draft
to --out and does NOT register it: registering is `castrol register-plate`,
after the draft has been through `prototype.py --only image` on real photos.

Why the plate and not the prompt: source photos are chest-up selfies that
almost never show a forearm, so the image edit has nothing to read the
mechanic's arms off and keeps the plate's - recoloured, still airbrushed and
hairless. Whatever the plate's arms look like is what nearly every video gets.

    uv run python spikes/plate_arms.py --uniform u1_tshirt \
        --background bg1_white_suv --out spikes/out/plate_arms/u1_bg1_a.png

SPENDS. Run one at a time: nothing here collides, but prototype uploads do.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import httpx

from castrol_pipeline.common import db, s3
from castrol_pipeline.config import get_settings
from castrol_pipeline.stages import vendors

#: One change, everything else frozen - the same Change / Preserve / Constrain
#: shape as the job prompt, because the card and the job prompt both depend on
#: this plate's geometry (invariant 6). Hands must not move: the card sits just
#: below them (invariant 27) and the avatar prompt parks them there.
#:
#: The face is left alone on purpose even though every job replaces it - an
#: edit that also "improves" the face is one that is not following "only".
#:
#: Grime is deliberately NOT asked for. Hair, veins, creases and short nails
#: read as a working man; grease in the nail lines reads as dirty in a brand
#: ad, and that is the client's call, not ours.
PROMPT = """\
Reproduce this image EXACTLY as-is. Same camera framing, same crop, same \
subject scale, same head and face, same shoulder line, same belt line, same \
pose, same arm and hand position, same finger positions, same background, \
same lighting, same uniform, the Castrol mark on the chest panel and the \
Castrol MAGNATEC overhead banner, identical in placement, size and colour.

Make EXACTLY ONE change: the bare skin of his forearms, wrists and hands. \
They should read as the arms of an Indian car mechanic in his thirties who \
works with his hands every day, photographed as he is. Give his forearms \
natural dark arm hair, resolved as individual hairs, denser along the outer \
forearm and thinning toward the wrist, with a little on the backs of the \
hands and fingers. Show the veins on the backs of his hands and along the \
forearms, real knuckle creases, tendons that move under the skin, and skin \
with its own texture, pores and slightly uneven tone, a little darker and \
drier on the knuckles. Keep his nails short, plain and practical, not \
manicured or polished. The arms keep exactly their current thickness, length, \
outline and position.

Do not move, rescale, repose or reshape the arms, hands or fingers. Do not \
change the face, the uniform, or anything else in the frame. Do not smooth, \
airbrush or even out the skin. Do not add dirt, grease, injuries, tattoos, \
jewellery or watches. Do not invent new text, logos or branding.\
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()

    with db.connection() as c:
        row = c.execute(
            "select s3_key from plates where active and uniform_id=%s and background_id=%s",
            (a.uniform, a.background),
        ).fetchone()
    if not row:
        raise SystemExit(f"no active plate for {a.uniform} + {a.background}")

    s = get_settings()
    body = {
        "model": s.image_edit_model_id,
        "prompt": PROMPT,
        "image_urls": [s3.get_storage().presigned_get_url(row["s3_key"])],
        "size": "9:16",
        "resolution": s.image_edit_resolution,
        "n": 1,
        "official_fallback": False,
    }
    with httpx.Client(timeout=60.0) as c:  # 60s: a short timeout orphans a paid task
        j = c.post(f"{vendors._image_base()}/images/generations",
                   headers=vendors._image_headers(), json=body).json()
    if j.get("code") != 200:  # invariant 14: 200 with an error inside
        raise SystemExit(f"submit rejected: {json.dumps(j)[:400]}")
    data = j["data"]
    task = (data[0] if isinstance(data, list) else data)["task_id"]
    print(f"submitted {task} ({row['s3_key']})")

    deadline = time.time() + 2700
    while time.time() < deadline:
        url = vendors.image_poll(task)
        if url:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            vendors.download(url, a.out, expect_image=True)  # invariant 15: copy at once
            print(f"wrote {a.out}")
            return
        time.sleep(6)
    raise SystemExit(f"timed out; task {task} may still complete - poll it, do not resubmit")


if __name__ == "__main__":
    main()
