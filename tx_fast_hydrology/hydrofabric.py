"""Load routing topology and channel attributes from a NextGen Hydrofabric."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


MIN_SLOPE = 1.0e-8
REQUIRED_GEOMETRY = (
    "So", "dx", "n", "Cs", "Bw", "Tw", "TwCC", "nCC",
)

# These are to check if the input is a valid NextGen Hydrofabric
FLOWPATH_TABLE = "flowpaths"
ATTRIBUTE_TABLE = "flowpath-attributes"
NEXUS_TABLE = "nexus"

ATTRIBUTE_COLUMNS = {
    "So": "So",
    "dx": "Length_m",
    "n": "n",
    "Cs": "ChSlp",
    "Bw": "BtmWdth",
    "Tw": "TopWdth",
    "TwCC": "TopWdthCC",
    "nCC": "nCC",
}


# ----------- Helper functions -----------------------
def _ids(values: pd.Series) -> pd.Series:
    return values.map(lambda value: "" if pd.isna(value) else str(value).strip())


def _read_table(
    connection: sqlite3.Connection,
    table_name: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Read required non-geometric columns from one NextGen GeoPackage table."""
    available = set(
        pd.read_sql_query(
            f'PRAGMA table_info("{table_name}")',
            connection,
        )["name"]
    )
    missing = set(columns) - available
    if missing:
        raise ValueError(
            f"NextGen table {table_name!r} is missing required columns: "
            f"{sorted(missing)}"
        )

    selected = ", ".join(f'"{column}"' for column in columns)
    return pd.read_sql_query(
        f'SELECT {selected} FROM "{table_name}"',
        connection,
    )
def _read_nextgen_geopackage(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with closing(sqlite3.connect(path)) as connection:
        table_names = set(
            pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type = 'table'",
                connection,
            )["name"]
        )
        required_tables = {FLOWPATH_TABLE, ATTRIBUTE_TABLE, NEXUS_TABLE}
        missing = required_tables - table_names
        if missing:
            raise ValueError(
                "The file is not a supported NextGen Hydrofabric; missing tables: "
                f"{sorted(missing)}"
            )

        flowpaths = _read_table(connection, FLOWPATH_TABLE, ("id", "toid"))
        attributes = _read_table(
            connection,
            ATTRIBUTE_TABLE,
            ("id", *ATTRIBUTE_COLUMNS.values()),
        )
        nexus = _read_table(connection, NEXUS_TABLE, ("id", "toid"))
    return flowpaths, attributes, nexus


def _nexus_downstream(
    flowpaths: pd.DataFrame,
    nexus: pd.DataFrame,
) -> list[str | None]:
    flowpath_ids = _ids(flowpaths["id"])
    nexus_ids = _ids(nexus["id"])
    nexus_targets = _ids(nexus["toid"])
    id_set = set(flowpath_ids)

    # It looks like the hydrofabrics can have multiple nexuses with the same id.
    # unless they point to different ids, this is ok.
    raw_nexus_map: dict[str, str] = {}
    for nexus_id, target in zip(nexus_ids, nexus_targets):
        if nexus_id in raw_nexus_map and raw_nexus_map[nexus_id] != target:
            raise ValueError(f"Nexus {nexus_id!r} has multiple downstream targets")
        raw_nexus_map[nexus_id] = target

    # NextGen Hydrofabric topology is:
    # flowpath.toid -> nexus.id -> nexus.toid (downstream flowpath).
    # Collapse the nexus part to leave only the flowpath relationship.
    nexus_map = {
        nexus_id: target if target in id_set else None
        for nexus_id, target in raw_nexus_map.items()
    }
    targets = _ids(flowpaths["toid"])
    unresolved = sorted(set(targets) - set(nexus_map))
    if unresolved:
        raise ValueError(
            "Flowpath `toid` values must reference IDs in the NextGen nexus "
            f"table. Missing examples: {unresolved[:5]}"
        )
    return [nexus_map[target] for target in targets]


def _geometry(
    flowpaths: pd.DataFrame,
    attributes: pd.DataFrame,
) -> dict[str, np.ndarray]:
    topology = flowpaths.copy()
    topology["_mc_id"] = _ids(topology["id"])

    # Join by flowpath id; attribute rows don't need to match topology row order.
    attrs = attributes.copy()
    attrs["_mc_id"] = _ids(attrs["id"])
    if attrs["_mc_id"].duplicated().any():
        duplicate = attrs.loc[attrs["_mc_id"].duplicated(), "_mc_id"].iloc[0]
        raise ValueError(f"Duplicate hydraulic attributes for flowpath {duplicate!r}")

    combined = topology[["_mc_id"]].merge(
        attrs.drop(columns=["id"]),
        on="_mc_id",
        how="left",
        validate="one_to_one",
    )

    result: dict[str, np.ndarray] = {}
    for model_name, nextgen_name in ATTRIBUTE_COLUMNS.items():
        values = pd.to_numeric(
            combined[nextgen_name], errors="coerce"
        ).to_numpy(np.float64)
        bad = ~np.isfinite(values)
        if bad.any():
            bad_ids = combined.loc[bad, "_mc_id"].head(5).tolist()
            raise ValueError(
                f"NextGen field {nextgen_name!r} contains missing/non-numeric "
                f"values for flowpaths {bad_ids}"
            )
        result[model_name] = values

    # Avoid undefined calculations for zero-slope reaches.
    result["So"] = np.maximum(result["So"], MIN_SLOPE)
    for field_name in ("dx", "n", "Bw", "Tw"):
        if np.any(result[field_name] <= 0.0):
            raise ValueError(f"Hydraulic field {field_name!r} must be positive")
    for field_name in ("Cs", "TwCC", "nCC"):
        if np.any(result[field_name] < 0.0):
            raise ValueError(f"Hydraulic field {field_name!r} cannot be negative")
    return result


def _check_acyclic(endnodes: np.ndarray) -> None:
    # Kahn's algorithm checks the graph for cycles. Outlets are excluded since
    # they are self-looping by definition
    node_count = endnodes.size
    indegree = np.zeros(node_count, dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(node_count)]
    for upstream, downstream in enumerate(endnodes):
        if upstream != downstream:
            indegree[downstream] += 1
            children[upstream].append(int(downstream))
    queue = [int(index) for index in np.flatnonzero(indegree == 0)]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for downstream in children[current]:
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                queue.append(downstream)
    if visited != node_count:
        raise ValueError("The selected Hydrofabric flowpath network contains a cycle")


@dataclass(frozen=True)
class HydrofabricNetwork:
    """Routing-ready view of a NextGen Hydrofabric flowpath network.

    reach_ids is the external Hydrofabric ID, while
    endnodes is the row index of its downstream reach, both indexed identically.
    """

    reach_ids: tuple[str, ...]
    endnodes: np.ndarray
    So: np.ndarray
    dx: np.ndarray
    n: np.ndarray
    Cs: np.ndarray
    Bw: np.ndarray
    Tw: np.ndarray
    TwCC: np.ndarray
    nCC: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        count = len(self.reach_ids)
        if len(set(self.reach_ids)) != count:
            raise ValueError("Hydrofabric flowpath IDs must be unique")
        endnodes = np.asarray(self.endnodes, dtype=np.int64)
        if endnodes.shape != (count,):
            raise ValueError("endnodes must contain one entry per flowpath")
        if np.any((endnodes < 0) | (endnodes >= count)):
            raise ValueError("endnodes contains an invalid flowpath index")
        object.__setattr__(self, "endnodes", endnodes)
        for name in REQUIRED_GEOMETRY:
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (count,):
                raise ValueError(f"{name} must contain one value per flowpath")
            object.__setattr__(self, name, values)
        _check_acyclic(endnodes)

    @property
    def size(self) -> int:
        return len(self.reach_ids)

    @property
    def outlets(self) -> tuple[str, ...]:
        return tuple(
            reach_id
            for index, reach_id in enumerate(self.reach_ids)
            if self.endnodes[index] == index
        )

    @classmethod
    def from_frames(
        cls,
        flowpaths: pd.DataFrame,
        attributes: pd.DataFrame,
        *,
        nexus: pd.DataFrame,
        reach_ids: Sequence[str | int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "HydrofabricNetwork":
        working = flowpaths.copy()
        working["id"] = _ids(working["id"])
        if working["id"].duplicated().any():
            duplicate = working.loc[working["id"].duplicated(), "id"].iloc[0]
            raise ValueError(f"Duplicate flowpath ID {duplicate!r}")

        all_downstream = _nexus_downstream(working, nexus)
        # Save full-fabric topology before subsetting. A selected reach whose
        # downstream neighbor is excluded will become a local outlet below.
        downstream_by_id = dict(zip(working["id"], all_downstream))

        if reach_ids is not None:
            selected = {str(value).strip() for value in reach_ids}
            available = set(working["id"])
            missing = selected - available
            if missing:
                raise ValueError(
                    f"{len(missing)} selected flowpath IDs are absent, e.g. "
                    f"{sorted(missing)[:5]}"
                )
            working = working.loc[working["id"].isin(selected)].copy()

        ids_in_order = tuple(working["id"].tolist())
        downstream_ids = [downstream_by_id[reach_id] for reach_id in ids_in_order]
        positions = {reach_id: index for index, reach_id in enumerate(ids_in_order)}

        endnodes = [
            positions.get(downstream, index)
            if downstream is not None else index
            for index, downstream in enumerate(downstream_ids)
        ]
        endnodes = np.asarray(endnodes, dtype=np.int64)

        geometry = _geometry(working, attributes)
        return cls(
            reach_ids=ids_in_order,
            endnodes=endnodes,
            metadata=dict(metadata or {}),
            **geometry,
        )


def load_hydrofabric(
    path: str | Path,
    *,
    reach_ids: Sequence[str | int] | None = None,
) -> HydrofabricNetwork:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.casefold() != ".gpkg":
        raise ValueError("NextGen Hydrofabric input must be a GeoPackage (.gpkg)")

    flowpaths, attributes, nexus = _read_nextgen_geopackage(source)
    return HydrofabricNetwork.from_frames(
        flowpaths,
        attributes,
        nexus=nexus,
        reach_ids=reach_ids,
        metadata={
            "source": str(source.resolve()),
            "flowpaths_layer": FLOWPATH_TABLE,
            "attributes_layer": ATTRIBUTE_TABLE,
            "nexus_layer": NEXUS_TABLE,
        },
    )
