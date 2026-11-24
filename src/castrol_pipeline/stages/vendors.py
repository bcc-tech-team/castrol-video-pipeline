"""Thin clients for the three paid providers.

Submit and poll are separate calls on purpose. Both remote stages submit, record
the vendor task id, and release the worker; a separate poller reconciles. A
worker blocked on a 20-minute video poll is a worker not doing the other 200 jobs,
and a claimed row that sits for 20 minutes is a row the reaper eventually
returns to the queue — which re-runs a call that is still in flight and bills
twice.

Nothing here reserves budget. `common/budget.py` is the only path to a paid
call and the stages hold that guard; putting it here too would double-meter.

Gateway quirks that are not optional to know:
  * Both gateways return HTTP 200 with `code != 200` for errors.
  * The image gateway's `data` is an ARRAY; the video gateway's `resultJson`
    is a JSON *string*.
  * The image gateway has no webhooks. Polling is the only completion signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from ..common.errors import (
    PipelineError,
    StageErrorCode,
    StageFailure,
    VendorOutOfCredits,
    VendorRejected,
    VendorThrottled,
    VendorTimeout,
    VendorUnavailable,
)
from ..common.logging import get_logger
from ..config import get_settings

log = get_logger(__name__)


def _is_throttle(status_code: int, body: Any) -> bool:
    """A 429, as either gateway may spell it. Pure.

    The video provider answers its 20-per-10s limit with HTTP 429; both
    gateways also return HTTP 200 with the real code in the body (invariant
    14), so a body `code` of 429 counts the same.
    """
    if status_code == 429:
        return True
    return isinstance(body, dict) and str(body.get("code", "")).strip() == "429"


def _is_out_of_credits(status_code: int, body: Any) -> bool:
    """A 402, or the gateway's "Credits insufficient" in a 200 body. Pure."""
    if status_code == 402:
        return True
    if not isinstance(body, dict):
        return False
    if str(body.get("code", "")).strip() == "402":
        return True
    if body.get("code") == 200:
        return False
    msg = str(body.get("msg") or body.get("message") or "").lower()
    return "insufficient" in msg and ("credit" in msg or "balance" in msg)


def _body(r: httpx.Response, what: str) -> dict[str, Any]:
    """The JSON body, or a throttle, or a retryable INTERNAL failure.

    A 429 is checked BEFORE parsing, because a rate limiter's refusal is often
    not JSON at all. Anything else unparseable stays what it always was - a
    retryable INTERNAL - and does not become a terminal rejection by default.
    """
    status = int(getattr(r, "status_code", 200))
    if status == 429:
        raise VendorThrottled(f"{what}: HTTP 429")
    if status == 402:
        raise VendorOutOfCredits(f"{what}: HTTP 402 {r.text[:200]}")
    try:
        body = r.json()
    except ValueError:
        raise StageFailure(
            f"{what}: HTTP {status}, body is not JSON: {r.text[:200]!r}",
            code=StageErrorCode.INTERNAL,
        ) from None
    if _is_throttle(status, body):
        raise VendorThrottled(f"{what}: {json.dumps(body)[:300]}")
    if _is_out_of_credits(status, body):
        raise VendorOutOfCredits(f"{what}: {json.dumps(body)[:300]}")
    return body if isinstance(body, dict) else {"data": body}


def _sniff_is_image(b: bytes) -> bool:
    return b.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"RIFF"))


def download(url: str, dst: Path, *, expect_image: bool = False) -> Path:
    """Copy a provider result to local disk immediately.

    Provider result URLs have an unmeasured TTL; never depend on one lasting.
    Our S3 copy is the system of record.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=180.0, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        if expect_image and not _sniff_is_image(r.content[:16]):
            # Retryable on purpose: the provider already produced a result, and
            # a 200 carrying HTML is its CDN having a bad moment, not a refusal.
            # VENDOR_REJECTED is terminal now, so borrowing it here would throw
            # away a finished render over one bad GET.
            raise StageFailure(
                f"{url[:80]} did not return an image (HTML error page?)",
                code=StageErrorCode.INTERNAL,
            )
        dst.write_bytes(r.content)
    return dst


# -------------------------------------------------- [A] voice provider TTS --


def voice_tts(
    text: str,
    *,
    voice_id: str,
    model_id: str,
    dst: Path,
    language: str | None = None,
) -> Path:
    """Synthesise to WAV. Synchronous — the provider returns bytes on the call.

    A DIRECT api, not one of the two gateways: the one deliberate exception,
    taken because that intersection has no voice-cloning Hindi lane.

    `language` is sent only when given. Hindi has never sent it and works
    (TECH_DESIGN: test before adding it), so None is the Hindi request byte for
    byte. The same voice id speaks every language - decided 2026-10-28.
    """
    body: dict[str, Any] = {
        "model_id": model_id,
        "transcript": text,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": 44100,
        },
    }
    if language:
        body["language"] = language
    s = get_settings()
    base = (s.tts_base_url or "https://api.cartesia.ai").rstrip("/")
    with httpx.Client(timeout=180.0) as c:
        r = c.post(
            f"{base}/tts/bytes",
            headers={
                "Authorization": f"Bearer {s.require('voice_api_key')}",
                "Cartesia-Version": s.voice_api_version,
                "Content-Type": "application/json",
            },
            json=body,
        )
    if r.status_code == 429:
        # Too many requests: put back without spending an attempt.
        raise VendorThrottled(f"voice provider 429: {r.text[:400]}")
    if r.status_code >= 500:
        # A bad moment, not an answer about this text. Kept separate from the
        # refusal below because VENDOR_REJECTED is terminal.
        raise VendorTimeout(f"voice provider {r.status_code}: {r.text[:400]}")
    if r.status_code != 200:
        raise VendorRejected(f"voice provider {r.status_code}: {r.text[:400]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(r.content)
    return dst


# ------------------------------------------------- [B] image provider edit --

#: Change / Preserve / Constrain, in that order, and the order is deliberate:
#: the model is told what the one change is, then everything that must survive
#: it, then the ways it is known to drift. Geometry preservation is invariant 6
#: - the card is burned in at a fixed fraction of frame height, so subject
#: scale drift lands it on the mechanic's hands.
#:
#: v3 (2026-09-16) adds a third paragraph to the CHANGE half, because "carry
#: over their facial hair" was not enough for a bearded mechanic. Naming a
#: feature tells the model the feature is there; it does not tell it to copy
#: the feature's structure, and this model answers an unqualified "beard" with
#: a beard-shaped mass - soft at the jawline, smeared into the lips, its
#: density and grey invented rather than read off the reference. The paragraph
#: therefore asks for the STRUCTURE by name (outline, edge, length, density,
#: patchiness, growth direction, grey) and for hair resolved as hair.
#:
#: It opens on the general case - an ordinary, unretouched photograph of an
#: ordinary man - because the same drift that softens a beard also plastics
#: the skin, and the constrain half now blocks both directly: smoothing the
#: skin and tidying the facial hair are the two ways "beautify" shows up here.
#: The clean-shaven sentence is there so the instruction cannot be read as a
#: reason to add hair that the reference photo does not have.
#:
#: Unlike AVATAR_PROMPT this text is hashed by VERSION, not by its own bytes,
#: so editing it does nothing until config.image_prompt_version is bumped.
_PROMPT_PRESERVE = """\
Reproduce this image EXACTLY as-is. Same camera framing, same crop, same \
subject scale, same head position, same shoulder line, same belt line, same \
pose, same hand position, same background, same lighting, same uniform \
geometry, the Castrol mark on the chest panel and the Castrol MAGNATEC \
overhead banner, identical in placement, size and colour. The chest panel's \
Castrol mark is the uniform's only branding, and the sleeves are plain.

Make EXACTLY ONE change: replace the person with the person in the second \
reference image. It is that person standing there, not the first person \
wearing their face - so carry over their face, apparent age, ethnicity, skin \
tone, facial hair, body build, proportions and posture, and keep all of it \
consistent with one another throughout the frame. Their HANDS, wrists and \
forearms are theirs too: the age, skin tone, thickness and hair of the hands \
must match that same person, not the hands in the first image. Keep their \
eyeglasses if they wear any.

The result is an ordinary, unretouched photograph of an ordinary man. Render \
his skin as real skin, with its own texture, pores, lines and marks, and \
render his facial hair as real hair. Take his beard, moustache and stubble \
from the second reference image exactly as they already are there: the same \
outline and jawline edge, the same length, density, patchiness, direction of \
growth and amount of grey, resolved as individual hairs that meet the skin at \
a clean edge, with the mouth and lips reading clearly through it. A \
clean-shaven man stays clean-shaven.\
"""

#: Sent ONLY when the job's plate has a uniform reference registered against it.
#:
#: The garment in the first image is being redrawn around a different body, so
#: the model reconstructs the uniform from what it can read off a figure it is
#: simultaneously changing - and fabric, stitching, collar shape and the printed
#: marks drift. The chest logo is what the video is for.
#:
#: It is still ONE change, and the clause has to say so in its first sentence.
#: The submit is a single generation with three references, and the identity
#: swap is the only edit being asked for; this paragraph is part of the PRESERVE
#: half - it says where to read the uniform's detail from, not that the uniform
#: is a second thing to change. Phrased as an instruction ("reproduce these
#: details") it contradicts "Make EXACTLY ONE change" two paragraphs above, and
#: a prompt that argues with itself measurably degrades output.
#:
#: The rest is scope. The third image is a flat shot on a plain background: its
#: framing, pose, lighting and background must not leak in, and neither must the
#: garment's fit or position, which come from the first image. Saying so
#: explicitly is what keeps the extra reference from re-opening invariant 6.
_PROMPT_UNIFORM = """\
That one change does not include the uniform, which stays exactly as it is. \
The third reference image is that same uniform, laid out flat on a plain \
background - use it as the reference for what the uniform already looks like: \
its exact fabric, colour, panel seams, collar and cuff shape, and the exact \
shape, proportion and colour of every printed logo and text mark on it. It \
tells you nothing else. The third image's framing, pose, lighting and \
background are irrelevant, and so is the way the garment is laid out in it - \
how the uniform sits on the body, its size in the frame and its place in the \
frame all come from the first image and do not change.\
"""

_PROMPT_CONSTRAIN = """\
Do not reframe. Do not zoom. Do not move or rescale the subject within the \
frame. Do not redesign, restyle, idealise or beautify anything. Do not \
smooth, airbrush or even out the skin. Do not tidy, trim, thin, reshape or \
fill in the facial hair. Do not invent new text, logos or branding.\
"""


def image_prompt(*, with_uniform_ref: bool) -> str:
    """The edit prompt for a two- or three-image submit.

    Two prompts rather than one that mentions an image which may not be there:
    a prompt referring to a third reference image on a two-image submit is a
    prompt the model has to guess at. Which one a job got is recoverable after
    the fact from its plate row - a plate either has a reference or it does
    not, and that fact is in the image stage's input_hash.
    """
    parts = [_PROMPT_PRESERVE]
    if with_uniform_ref:
        parts.append(_PROMPT_UNIFORM)
    parts.append(_PROMPT_CONSTRAIN)
    return "\n\n".join(parts)


#: The two-image prompt, word for word the one the prototype proved.
IMAGE_PROMPT = image_prompt(with_uniform_ref=False)


def _image_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_settings().require('image_api_key')}"}


def _image_base() -> str:
    return (get_settings().image_base_url or "https://api.apimart.ai/v1").rstrip("/")


def image_submit(
    plate_url: str,
    photo_url: str,
    *,
    model_id: str,
    uniform_ref_url: str | None = None,
) -> str:
    """Submit the person-replacement edit. Returns a task id.

    Order is load-bearing: the prompt addresses its inputs by ordinal, so the
    plate is first, the mechanic second, and the uniform reference - when the
    plate has one - third. Three references is well inside the ~4 cap
    (invariant 17), and none of them is generated: the uniform reference is
    client artwork, exactly like the plate.
    """
    s = get_settings()
    image_urls = [plate_url, photo_url]
    if uniform_ref_url:
        image_urls.append(uniform_ref_url)
    body = {
        "model": model_id,
        "prompt": image_prompt(with_uniform_ref=bool(uniform_ref_url)),
        "image_urls": image_urls,
        "size": "9:16",
        "resolution": s.image_edit_resolution,
        "n": 1,
        "official_fallback": False,
    }
    # 60s, not 15s: a short timeout ORPHANS jobs — the provider accepts and starts,
    # the client raises, and the result is keyed to a task_id nobody recorded.
    # That is a paid call with no way to collect it.
    with httpx.Client(timeout=60.0) as c:
        r = c.post(f"{_image_base()}/images/generations",
                   headers=_image_headers(), json=body)
    j = _body(r, "image provider submit")
    if j.get("code") != 200:
        raise VendorRejected(f"image provider submit rejected: {json.dumps(j)[:400]}")
    data = j["data"]
    return str((data[0] if isinstance(data, list) else data)["task_id"])


def image_poll(task_id: str) -> str | None:
    """Result URL, or None while still in flight. Raises on vendor failure."""
    with httpx.Client(timeout=60.0) as c:
        r = c.get(f"{_image_base()}/tasks/{task_id}", headers=_image_headers())
    j = _body(r, "image provider poll")
    d = j.get("data") or {}
    status = d.get("status")

    if status == "completed":
        res = d.get("result", {})
        images = res.get("images") or res.get("data") or []
        if not images:
            raise VendorRejected(f"image provider completed with no image: {json.dumps(d)[:400]}")
        url = images[0].get("url") if isinstance(images[0], dict) else images[0]
        return url[0] if isinstance(url, list) else str(url)

    if status in {"failed", "cancelled"}:
        # 11 of 20 observed failures on this model were content safety, and
        # swapping a real person into a branded plate is exactly the trigger.
        # Not retryable: the same inputs will trip the same filter.
        raise VendorRejected(f"image provider {status}: {json.dumps(d.get('error', d))[:400]}")

    return None


# ----------------------------------------------- [C] video provider avatar --


#: Motion direction for the avatar. NOT decorative — on kling-avatar-v2 the
#: prompt steers expression, head movement and hand gesture, and the field is
#: required (max 5000 chars).
#:
#: The comment that used to sit here said this text was r1 restored verbatim.
#: It has not been r1 since r4 stopped asking for hand gestures - r1 asks for
#: "natural open-palm hand gestures", which is the one thing the text below is
#: built to avoid. The revision history, as the renders actually recorded it:
#:
#:   r1 looped one gesture and smeared it; r2 clawed both hands across the
#:   chest panel; r3 produced clean open palms that still rose to chest level
#:   for no return. r4 stopped asking for gestures at all and named the rest
#:   position positively instead, which is the shape kept below.
#:
#: Two failure modes are permanent, both bought with paid renders:
#:
#:   * Never pair a placement with an exclusion naming the same region. "at
#:     chest height" plus "clear of the chest logo" is how r2 put both hands
#:     on the chest panel - given both, this model keeps the position and
#:     drops the exclusion.
#:   * SPEED must be constrained. Fast movement is what generative video
#:     smears, and r1's first render came back blurred for exactly that
#:     reason. "slow" is the guard; "calm" is a different guard, on intent.
#:
#: r5 (2026-09-16) REDUCES HEAD MOTION, at the client's request. r4's "subtle
#: head nods" is a request for a repeating movement, and this model repeats a
#: requested movement for the whole take rather than occasionally - the same
#: mechanism that looped r1's gesture, applied to the head. The head is now
#: told to stay level and face camera, moving only slightly.
#:
#: It is bounded, NOT frozen, and that is the same argument the hands get: a
#: motionless head over a moving mouth reads as a photograph with a talking
#: head pasted on. What replaces the nods is "a warm, engaged face" - with the
#: hands low and the head still, the eyes and mouth are all the life left, so
#: the prompt has to ask for them by name.
#:
#: r6 (2026-11-24) KEEPS HIM ALONE. Delivered renders (9391709420, 8571038306)
#: grew a second mechanic in the same uniform who walks in behind the car,
#: leans over it for most of the take and walks out - nowhere in the still.
#: "in his garage" describes a busy workplace, and this model fills a described
#: place with what usually happens there. The fix names the frame positively
#: rather than forbidding people: he is ALONE in it, and the background is part
#: of what stays unchanged, held still for the whole take. Written as what TO
#: do, because naming "other people" is how you get them (test_phrased_positively).
#:
#: This text is part of the video input_hash, so editing it regenerates. See
#: VideoStage._params.
AVATAR_PROMPT = (
    "An Indian auto mechanic in his Castrol work uniform, alone in the frame, "
    "speaking directly to camera in his garage. Calm, natural and slow, with "
    "clear articulation and a warm, engaged face, and a steady head that stays level and facing "
    "camera with only slight natural movement. His hands stay low and mostly "
    "still, one on each side of his body, apart from each other and clear of "
    "one another at all times, with only small slow movements that settle back "
    "to rest. Keep the existing framing, uniform, branding and background "
    "unchanged, with the garage behind him still for the whole take."
)


def _video_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {get_settings().require('video_api_key')}",
        "Content-Type": "application/json",
    }


def _video_base() -> str:
    return (get_settings().video_base_url or "https://api.kie.ai").rstrip("/")


#: The video provider's documented ceiling on the prompt field.
VIDEO_PROMPT_MAX_CHARS = 5000


def video_submit(
    image_url: str, audio_url: str, *, model_id: str, prompt: str = AVATAR_PROMPT
) -> str:
    """Submit the avatar render. Returns a task id. THIS TAKES 8-20 MINUTES."""
    if len(prompt) > VIDEO_PROMPT_MAX_CHARS:
        raise VendorRejected(
            f"Avatar prompt is {len(prompt)} chars, over the video "
            f"provider's {VIDEO_PROMPT_MAX_CHARS} "
            "limit. Caught before submit: a rejected submit on this model is a "
            "20-minute round trip to discover a typo."
        )
    with httpx.Client(timeout=120.0) as c:
        r = c.post(
            f"{_video_base()}/api/v1/jobs/createTask",
            headers=_video_headers(),
            json={"model": model_id,
                  "input": {"image_url": image_url,
                            "audio_url": audio_url,
                            "prompt": prompt}},
        )
    # A 429 here used to read as `code != 200` and raise VendorRejected, which
    # is terminal: one burst over the provider's 20-per-10s limit killed the
    # job outright. It is now VendorThrottled, which puts the run back.
    j = _body(r, "video provider submit")
    if j.get("code") != 200:
        raise VendorRejected(f"video provider submit rejected: {json.dumps(j)[:400]}")
    return str(j["data"]["taskId"])


def video_poll(task_id: str) -> str | None:
    """Result URL, or None while still in flight. Raises on vendor failure."""
    with httpx.Client(timeout=60.0) as c:
        r = c.get(f"{_video_base()}/api/v1/jobs/recordInfo",
                  params={"taskId": task_id}, headers=_video_headers())
    # A 429 body has no `data`, and used to read as "still in flight" - right
    # by accident, and silent, so nothing ever slowed down. It is now a
    # throttle the poller backs off on.
    j = _body(r, "video provider poll")
    d = j.get("data") or {}
    state = d.get("state")

    if state == "success":
        # resultJson is a JSON *string*, not an object.
        result = json.loads(d["resultJson"])
        urls = result.get("resultUrls") or []
        if not urls:
            raise VendorRejected(
                f"video provider success with no result url: {json.dumps(d)[:400]}"
            )
        return str(urls[0])

    if state in {"fail", "FAILED", "failed", "error", "ERROR"}:
        raise video_task_failure(state, d)

    return None


def video_balance_usd() -> float:
    """The video account's balance, in USD.

    The provider answers in its own credits; `budget.VIDEO_CREDITS_PER_USD`
    is the measured rate. Read defensively, because the shape of `data` is not
    documented: a bare number, or a dict holding one under a credit-ish key.
    Anything else raises, and the caller treats an unreadable balance as "not
    covered yet" rather than guessing.
    """
    from ..common.budget import VIDEO_CREDITS_PER_USD

    with httpx.Client(timeout=30.0) as c:
        r = c.get(f"{_video_base()}/api/v1/chat/credit", headers=_video_headers())
    j = _body(r, "video provider balance")
    data = j.get("data")
    if isinstance(data, dict):
        for key in ("credit", "credits", "balance", "remainCredits", "remaining"):
            if key in data:
                data = data[key]
                break
    try:
        credits = float(data)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise StageFailure(
            f"video provider balance unreadable: {json.dumps(j)[:200]}",
            code=StageErrorCode.INTERNAL,
        ) from None
    return credits / float(VIDEO_CREDITS_PER_USD)


def video_task_failure(state: str, data: dict[str, Any]) -> PipelineError:
    """The error for a video task the provider reports as failed. Pure.

    A 5xx `failCode` is the provider failing on its own side - Batch 1 saw
    "500 internal error, please try again later" on 188 renders in three bursts
    - so it is VendorUnavailable, which the retry policy retries. Anything else
    is a verdict on what we sent and stays VendorRejected, which it does not.
    "Audio size is too large" is one of those: a BYTE limit, not a duration
    limit, and if it appears the mp3 transcode did not happen.
    """
    code = data.get("failCode", "")
    msg = data.get("failMsg") or data.get("errorReason") or data.get("msg")
    text = f"video provider {state}: {code} {msg}"
    try:
        server_side = int(str(code).strip()) >= 500
    except ValueError:
        server_side = False
    return VendorUnavailable(text) if server_side else VendorRejected(text)


def poll_timeout(started_at_seconds: float, limit_seconds: float, vendor: str) -> None:
    """Raise once an in-flight task has outlived its ceiling.

    The poller is stateless per pass, so something has to notice a task the
    vendor will never finish. Without this a lost task sits `running` forever
    and the job never fails, never completes, and never appears in a report.
    """
    if started_at_seconds > limit_seconds:
        raise VendorTimeout(
            f"{vendor} task exceeded {limit_seconds:.0f}s without completing"
        )
