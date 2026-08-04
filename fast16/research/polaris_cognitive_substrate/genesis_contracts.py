"""Polaris Embryo v0 的四个最小机器合同。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class PolarisPacket:
    packet_id: str
    packet_type: str
    payload: Any
    source: str
    epoch: int
    version: int = 1
    parents: tuple[str, ...] = ()
    status: str = "active"
    payload_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.payload_sha256 = canonical_sha256(self.payload)


@dataclass(frozen=True)
class OrganSpec:
    organ_id: str
    architecture: str
    input_ports: tuple[str, ...]
    output_port: str
    manifest_sha256: str
    lifecycle: str


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    organ_id: str
    input_packets: tuple[str, ...]
    output_packet: str
    status: str


@dataclass(frozen=True)
class CommitReceipt:
    commit_id: str
    accepted: bool
    proposal_id: str
    validation_packet: str
    superseded_packet: str
    committed_packet: str
    preserved_packets: tuple[str, ...]
    digest: str


@dataclass
class EmbryoState:
    task_id: str
    packets: dict[str, PolarisPacket] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    epoch: int = 0

    def add_packet(
        self,
        packet_id: str,
        packet_type: str,
        payload: Any,
        source: str,
        *,
        version: int = 1,
        parents: tuple[str, ...] = (),
        status: str = "active",
    ) -> PolarisPacket:
        if packet_id in self.packets:
            raise ValueError(f"packet_id重复: {packet_id}")
        missing = [parent for parent in parents if parent not in self.packets]
        if missing:
            raise ValueError(f"父packet不存在: {missing}")
        self.epoch += 1
        packet = PolarisPacket(
            packet_id=packet_id,
            packet_type=packet_type,
            payload=payload,
            source=source,
            epoch=self.epoch,
            version=version,
            parents=parents,
            status=status,
        )
        self.packets[packet_id] = packet
        self.events.append(
            {
                "epoch": self.epoch,
                "event": "packet.add",
                "packet_id": packet_id,
                "packet_type": packet_type,
                "parents": list(parents),
            }
        )
        return packet

    def supersede(self, packet_id: str, successor_id: str) -> None:
        packet = self.packets[packet_id]
        if packet.status != "active":
            raise ValueError(f"只能supersede active packet: {packet_id}")
        if successor_id not in self.packets:
            raise ValueError(f"successor不存在: {successor_id}")
        packet.status = "superseded"
        self.epoch += 1
        self.events.append(
            {
                "epoch": self.epoch,
                "event": "packet.supersede",
                "packet_id": packet_id,
                "successor_id": successor_id,
            }
        )

    def build_commit(
        self,
        *,
        commit_id: str,
        proposal: Proposal,
        validation_packet: str,
        superseded_packet: str,
        committed_packet: str,
        preserved_packets: tuple[str, ...],
    ) -> CommitReceipt:
        validation = self.packets[validation_packet]
        accepted = bool(validation.payload.get("passed"))
        identity = {
            "commit_id": commit_id,
            "accepted": accepted,
            "proposal": asdict(proposal),
            "validation_sha256": validation.payload_sha256,
            "superseded_packet": superseded_packet,
            "committed_packet": committed_packet,
            "committed_sha256": self.packets[committed_packet].payload_sha256,
            "preserved": {
                packet_id: self.packets[packet_id].payload_sha256
                for packet_id in preserved_packets
            },
        }
        return CommitReceipt(
            commit_id=commit_id,
            accepted=accepted,
            proposal_id=proposal.proposal_id,
            validation_packet=validation_packet,
            superseded_packet=superseded_packet,
            committed_packet=committed_packet,
            preserved_packets=preserved_packets,
            digest=canonical_sha256(identity),
        )
