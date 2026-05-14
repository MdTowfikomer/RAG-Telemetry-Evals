from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_evaluation_service
from backend.services.evaluation_service import EvaluationService


router = APIRouter()


@router.get("/scores/stream")
async def scores_stream_endpoint(
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    return StreamingResponse(
        evaluation_service.score_event_stream(),
        media_type="text/event-stream",
    )
