"""PAN2's checks of a rewriter's reply, ported from its prompt rewriter's assembly tool
(ifm-ai/PAN2@0fb907b6 `inference/pan-prompt-rewriter/scripts/prompt_tools.py`, Copyright 2026 MBZUAI, Apache-2.0):
the reply is one JSON object `{"refined_prompt": ...}` whose string decodes to the policy's seven-key positive caption
(`positive_structure`), within the encoder's soft limit. The messages are upstream's."""

from __future__ import annotations

import json
import re
from decimal import Decimal

#: The positive caption's encoder tokens the policy allows (`POSITIVE_SOFT_LIMIT`).
POSITIVE_SOFT_LIMIT = 1480
KEYS = ["video_style", "lighting", "perspective_view", "subjects_lifecycle", "init_frame_visual_description", "events",
        "quality"]
CAMERAS = {"static", "pan_left", "pan_right", "tilt_up", "tilt_down", "track_left", "track_right", "dolly_forward",
           "dolly_backward", "dolly_up", "dolly_down", "zoom_in", "zoom_out", "roll_clockwise", "roll_counterclockwise",
           "unstable"}
ID = re.compile(r"<\|Subject[1-9][0-9]*\|>")


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        require(key not in out, "Duplicate JSON key: " + key)
        out[key] = value
    return out


def loads(raw: str) -> object:
    def invalid(value: str) -> object:
        raise ValueError("Nonfinite JSON number: " + value)

    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid)


def ordered(value: object, keys: list[str], label: str) -> None:
    require(type(value) is dict and list(value) == keys, label + ": wrong keys/order")


def string(value: object, label: str) -> None:
    require(type(value) is str and bool(value.strip()), label + ": expected nonempty text")


def interval(value: object) -> tuple[Decimal, Decimal]:
    require(type(value) is str and re.fullmatch(r"\d+(?:\.\d)? to \d+(?:\.\d)?", value), "Invalid time interval")
    start, end = map(Decimal, str(value).split(" to "))
    require(0 <= start < end, "Interval must have positive duration")
    return start, end


def positive_structure(text: object, duration: Decimal) -> dict[str, object]:
    string(text, "refined_prompt")
    positive = loads(str(text))
    ordered(positive, KEYS, "Positive")
    assert isinstance(positive, dict)  # `ordered`: an object
    for key in ("video_style", "lighting", "perspective_view", "init_frame_visual_description"):
        string(positive[key], key)
        require(not ID.search(positive[key]), "Subject IDs only belong in declarations/action strings")
    require(len(positive["init_frame_visual_description"].split()) <= 200, "Initial description exceeds 200 words")
    subjects = positive["subjects_lifecycle"]
    require(type(subjects) is list, "subjects_lifecycle must be an array")
    ids = set()
    for i, subject in enumerate(subjects, 1):
        ordered(subject, ["subject_id", "display_name", "subject_attribute"], "Subject")
        require(subject["subject_id"] == "<|Subject" + str(i) + "|>", "Subject IDs must be sequential")
        for key in ("display_name", "subject_attribute"):
            string(subject[key], key)
            require(not ID.search(subject[key]), "Unexpected subject token in appearance")
        ids.add(subject["subject_id"])
    events = positive["events"]
    require(type(events) is list and bool(events), "Positive events must be nonempty")
    current = Decimal(0)
    for i, event in enumerate(events):
        ordered(event, ["event_index", "time_range_sec", "event_description", "camera_motion", "contained_actions"], "Event")
        require(type(event["event_index"]) is int and event["event_index"] == i, "Event indices must be sequential")
        start, end = interval(event["time_range_sec"])
        require(start == current and end <= duration, "Events must cover duration contiguously")
        current = end
        string(event["event_description"], "event_description")
        require(len(event["event_description"].split()) <= 200, "Event description exceeds 200 words")
        require(not ID.search(event["event_description"]), "No subject IDs in event prose")
        camera = event["camera_motion"]
        require(type(camera) is list and bool(camera), "Camera coverage must be nonempty")
        position = start
        for row in camera:
            ordered(row, ["time_range_sec", "camera_motion"], "Camera entry")
            a, b = interval(row["time_range_sec"])
            require(a == position and b <= end, "Camera must cover parent event contiguously")
            position = b
            tags = row["camera_motion"]
            require(type(tags) is list and bool(tags) and all(type(t) is str and t in CAMERAS for t in tags),
                    "Unknown positive camera tag")
            require(len(set(tags)) == len(tags) and ("static" not in tags or len(tags) == 1),
                    "Duplicate/conflicting camera tags")
        require(position == end, "Incomplete camera coverage")
        require(type(event["contained_actions"]) is list, "Actions must be an array")
        for action in event["contained_actions"]:
            ordered(action, ["time_range_sec", "action"], "Action")
            a, b = interval(action["time_range_sec"])
            require(start <= a < b <= end, "Action outside parent event")
            string(action["action"], "action")
            tokens = ID.findall(action["action"])
            require(tokens and action["action"].startswith(tokens[0]) and set(tokens) <= ids,
                    "Action needs a registered performer and registered subject references")
    require(current == duration, "Positive does not cover exact requested duration")
    quality = positive["quality"]
    ordered(quality, ["video_quality", "text_logos_overlays", "artifacts"], "Quality")
    string(quality["video_quality"], "video_quality")
    require(quality["artifacts"] is None, "Positive artifacts must be null")
    require(quality["text_logos_overlays"] is None or type(quality["text_logos_overlays"]) is str,
            "Positive text_logos_overlays must be string/null")
    require(not ID.search(json.dumps(quality)), "No subject IDs in quality")
    return positive


def authored(reply: str, duration: Decimal) -> str:
    """The positive caption of a rewriter's `reply`, as `assemble` takes an authored response: exactly
    `{"refined_prompt": <string>}`, its string the policy's caption for `duration` seconds. Raises ValueError, saying
    why, as upstream's checks do."""
    response = loads(reply)
    ordered(response, ["refined_prompt"], "Authored positive response")
    assert isinstance(response, dict)  # `ordered`: an object
    positive_structure(response["refined_prompt"], duration)
    return str(response["refined_prompt"])
