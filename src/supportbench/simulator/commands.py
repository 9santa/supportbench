from dataclasses import dataclass

from supportbench.simulator.models import CaseSeverity


@dataclass(frozen=True, slots=True)
class CreateSupportCaseCommand:
    world_id: str
    idempotency_key: str

    actor_user_id: str
    user_id: str
    service_id: str

    summary: str
    description: str
    severity: CaseSeverity
