from __future__ import annotations

from dataclasses import dataclass

REQUEST_API_VERSION = 1


@dataclass(frozen=True)
class RequestApiContract:
    version: int
    supports_reference_voice: bool
    supports_request_measurement: bool
    supports_named_voices: bool
    supports_voice_design: bool


def request_api_contract() -> RequestApiContract:
    return RequestApiContract(
        version=REQUEST_API_VERSION,
        supports_reference_voice=True,
        supports_request_measurement=True,
        supports_named_voices=False,
        supports_voice_design=True,
    )
