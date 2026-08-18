from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from supportbench.api.dependencies import get_world_service
from supportbench.api.models import (
    CreateWorldRequest,
    CreateWorldResponse,
    DeleteWorldResponse,
)
from supportbench.api.worlds import (
    DemoWorldNotFoundError,
    WorldService,
)


router = APIRouter(
    prefix="/worlds",
    tags=["worlds"],
)


@router.post(
    "",
    response_model=CreateWorldResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_world(
    body: CreateWorldRequest,
    service: WorldService = Depends(get_world_service),
) -> CreateWorldResponse:
    world = service.create(scenario=body.scenario)

    return CreateWorldResponse(
        world_id=world.world_id,
        scenario=world.scenario,
    )


@router.delete(
    "/{world_id}",
    response_model=DeleteWorldResponse,
)
def remove_world(
    world_id: str,
    service: WorldService = Depends(get_world_service),
) -> DeleteWorldResponse:
    try:
        service.delete(
            world_id=world_id,
        )
    except DemoWorldNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return DeleteWorldResponse(
        world_id=world_id,
        deleted=True,
    )
