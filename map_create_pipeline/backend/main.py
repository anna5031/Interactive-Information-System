from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import (
    ProcessFloorPlanRequest,
    ProcessFloorPlanResponse,
    StepTwoRecord,
    StepTwoSaveRequest,
    StepTwoStatus,
    StoredFloorPlanSummary,
)
from services.floorplan_pipeline import FloorPlanProcessingService
from services.step_two_repository import StepTwoRepository


DATA_DIR = PROJECT_ROOT / "data"

app = FastAPI(
    title="Floor Plan Processing API",
    description="프런트엔드에서 전달한 YOLO 라벨을 그래프로 변환하고 결과를 JSON으로 저장합니다.",
    version="1.0.0",
)

service = FloorPlanProcessingService(storage_root=DATA_DIR)
step_two_repository = StepTwoRepository(storage_root=DATA_DIR)


def _build_stored_floorplan_results() -> List[dict]:
    raw_results = service.list_results()
    step_two_records = step_two_repository.list_all()

    request_to_step_one_id = {}
    for record in step_two_records:
        request_id = record.get("requestId")
        step_one_id = record.get("id")
        if request_id and step_one_id:
            request_to_step_one_id[request_id] = step_one_id

    summaries: List[dict] = []
    seen_requests = set()
    for item in raw_results:
        request_id = item.get("request_id")
        if not request_id or request_id in seen_requests:
            continue
        seen_requests.add(request_id)

        step_one_id = request_to_step_one_id.get(request_id) or f"step_one_{request_id}"
        annotation_counts = item.get("annotation_counts") or {}

        summaries.append(
            {
                "stepOneId": step_one_id,
                "requestId": request_id,
                "createdAt": item.get("created_at"),
                "imageSize": item.get("image_size") or {},
                "classNames": item.get("class_names") or [],
                "sourceImagePath": item.get("source_image_path"),
                "graphSummary": item.get("graph_summary"),
                "annotationCounts": {
                    "boxes": annotation_counts.get("boxes", 0),
                    "walls": annotation_counts.get("walls", 0),
                    "doors": annotation_counts.get("doors", 0),
                },
                "yoloText": item.get("yolo_text") or "",
                "wallText": item.get("wall_text") or "",
                "doorText": item.get("door_text") or "",
                "imageUrl": item.get("image_url"),
                "imageDataUrl": None,
            }
        )

    summaries.sort(key=lambda entry: entry.get("createdAt") or "", reverse=True)
    return summaries


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post(
    "/api/floorplans/process",
    response_model=ProcessFloorPlanResponse,
    summary="YOLO 어노테이션을 그래프/객체 정보로 변환",
)
async def process_floorplan(payload: ProcessFloorPlanRequest) -> ProcessFloorPlanResponse:
    try:
        result = service.process(
            image_width=payload.image_width,
            image_height=payload.image_height,
            class_names=payload.class_names,
            source_image_path=payload.source_image_path,
            yolo_text=payload.yolo_text,
            wall_text=payload.wall_text,
            door_text=payload.door_text,
            image_data_url=payload.image_data_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProcessFloorPlanResponse(**result)


@app.get(
    "/api/floorplans/{request_id}/image",
    summary="저장된 원본 이미지 조회",
)
async def get_floorplan_image(request_id: str) -> FileResponse:
    image_info = service.get_image_info(request_id)
    if image_info is None:
        raise HTTPException(status_code=404, detail="저장된 이미지를 찾을 수 없습니다.")
    image_path, mime_type = image_info
    return FileResponse(image_path, media_type=mime_type, filename=image_path.name)


@app.get(
    "/api/floorplans",
    response_model=List[StoredFloorPlanSummary],
    summary="저장된 그래프 결과 목록 조회",
)
async def list_floorplan_results(request: Request) -> List[StoredFloorPlanSummary]:
    raw_results = _build_stored_floorplan_results()
    for entry in raw_results:
        image_url = entry.get("imageUrl")
        if image_url:
            entry["imageUrl"] = str(request.url_for("get_floorplan_image", request_id=entry["requestId"]))
    return [StoredFloorPlanSummary(**entry) for entry in raw_results]


@app.get(
    "/api/floorplans/by-step-one/{step_one_id}",
    response_model=StoredFloorPlanSummary,
    summary="Step One ID로 저장된 그래프 결과 조회",
)
async def get_floorplan_by_step_one(step_one_id: str, request: Request) -> StoredFloorPlanSummary:
    for item in _build_stored_floorplan_results():
        if item.get("stepOneId") == step_one_id:
            image_url = item.get("imageUrl")
            if image_url:
                item["imageUrl"] = str(request.url_for("get_floorplan_image", request_id=item["requestId"]))
            return StoredFloorPlanSummary(**item)
    raise HTTPException(status_code=404, detail="저장된 결과를 찾을 수 없습니다.")


@app.get(
    "/api/floorplans/{request_id}",
    response_model=ProcessFloorPlanResponse,
    summary="저장된 그래프 결과 조회",
)
async def get_floorplan(request_id: str, request: Request) -> ProcessFloorPlanResponse:
    try:
        result = service.get_result(request_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    metadata = result.get("metadata") or {}
    image_url = metadata.get("image_url")
    if image_url:
        metadata["image_url"] = str(request.url_for("get_floorplan_image", request_id=request_id))
    result["metadata"] = metadata
    return ProcessFloorPlanResponse(**result)


@app.get(
    "/api/step-two",
    response_model=List[StepTwoStatus],
    summary="2단계 저장 현황 조회",
)
async def list_step_two_results() -> List[StepTwoStatus]:
    records = step_two_repository.list_all()
    statuses: List[StepTwoStatus] = []
    for record in records:
        rooms = record.get("rooms") or []
        doors = record.get("doors") or []
        updated_at = record.get("updatedAt")
        rooms_have_base = all(
            (room.get("name") or "").strip() or (room.get("number") or "").strip() for room in rooms
        )
        has_base = rooms_have_base if rooms else True

        def _has_detail_entries(entries: list) -> bool:
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                key = (entry.get("key") or "").strip()
                value = (entry.get("value") or "").strip()
                if key or value:
                    return True
            return False

        has_details = any(_has_detail_entries(room.get("extra")) for room in rooms) or any(
            _has_detail_entries(door.get("extra")) for door in doors
        )

        statuses.append(
            StepTwoStatus(
                id=record.get("id"),
                request_id=record.get("requestId"),
                has_base=has_base,
                has_details=has_details,
                base_updated_at=updated_at if has_base else None,
                details_updated_at=updated_at if has_details else None,
                updated_at=updated_at,
            )
        )
    return statuses


@app.get(
    "/api/step-two/{step_one_id}",
    response_model=StepTwoRecord,
    summary="2단계 저장본 상세 조회",
)
async def get_step_two_result(step_one_id: str) -> StepTwoRecord:
    record = step_two_repository.get(step_one_id)
    if record is None:
        raise HTTPException(status_code=404, detail="저장된 2단계 결과가 없습니다.")
    return StepTwoRecord(**record)


@app.put(
    "/api/step-two/{step_one_id}",
    response_model=StepTwoRecord,
    summary="2단계 정보 저장",
)
async def save_step_two(step_one_id: str, payload: StepTwoSaveRequest) -> StepTwoRecord:
    record = step_two_repository.save(step_one_id, payload.model_dump(by_alias=True))
    return StepTwoRecord(**record)
