"""Map requested body names to contact-sensor columns in the sensor primary-body order."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.sensor import ContactSensor


def resolve_contact_indices(
    contact_sensor: ContactSensor,
    asset: Entity,
    name_keys: tuple[str, ...] | list[str],
) -> tuple[list[int], list[int], list[str]]:
    """Map requested bodies onto a contact sensor's output columns.

    Args:
        contact_sensor: the sensor whose per-body output columns are being indexed.
        asset: the robot entity the sensor's primaries resolve against.
        name_keys: the requested body names/patterns, in the desired output order.

    Returns:
        ``(sensor_column_indices, asset_body_ids, resolved_names)`` where
        ``sensor_column_indices[i]`` is the column of ``resolved_names[i]`` in the
        sensor's per-body outputs, and ``asset_body_ids[i]`` is its body id in
        ``asset.data.body_link_*``. All three lists are in ``name_keys`` order.
    """
    primary = contact_sensor.cfg.primary
    mode = getattr(primary, "mode", None)
    pattern = getattr(primary, "pattern", None)
    if mode == "geom":
        _, primary_names = asset.find_geoms(pattern)
    else:  # "subtree" / "body" (and any fallback) resolve against bodies.
        _, primary_names = asset.find_bodies(pattern)

    target_ids, target_names = asset.find_bodies(name_keys)
    name_to_column = {name: i for i, name in enumerate(primary_names)}

    missing = [n for n in target_names if n not in name_to_column]
    if missing:
        raise ValueError(
            f"Contact sensor '{contact_sensor.cfg.name}' primaries do not include "
            f"bodies {missing} (primary set: {list(primary_names)})."
        )

    sensor_column_indices = [name_to_column[n] for n in target_names]
    return sensor_column_indices, list(target_ids), list(target_names)
