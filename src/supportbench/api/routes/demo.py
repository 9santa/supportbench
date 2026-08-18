from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

_API_DIRECTORY = Path(__file__).resolve().parents[1]

templates = Jinja2Templates(
    directory=_API_DIRECTORY / "templates",
)

router = APIRouter(
    include_in_schema=False,
)


@router.get(
    "/demo",
    response_class=HTMLResponse,
)
def demo(request: Request) -> Response:
    return templates.TemplateResponse(
        request=request,
        name="demo.html",
    )
