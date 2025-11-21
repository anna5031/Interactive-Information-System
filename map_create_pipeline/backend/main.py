from __future__ import annotations

import sys
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.dependencies import resolve_active_user_id
from api.schemas import (
    AutoCorrectLayoutRequest,
    AutoCorrectLayoutResponse,
    FloorPlanFloorSummary,
    FreeSpacePreviewOptions,
    FreeSpacePreviewRequest,
    FreeSpacePreviewResponse,
    GraphDataResponse,
    GraphUpdateRequest,
    PrepareGraphRequest,
    ProcessFloorPlanRequest,
    ProcessFloorPlanResponse,
    SaveStepOneRequest,
    SaveStepOneResponse,
    StepThreeRecord,
    StepThreeSaveRequest,
    StepThreeStatus,
    StoredFloorPlanSummary,
)
from configuration import get_auto_correction_settings
from services.auto_correction_service import FloorPlanAutoCorrectionService
from services.floorplan_pipeline import FloorPlanProcessingService
from services.inference_service import (
    DependencyNotAvailableError,
    TorchNotAvailableError,
    build_inference_service_from_config,
)
from services.step_three_repository import StepThreeRepository
from services.user_storage import UserScopedStorage, sanitize_user_id
from processing.corridor_pipeline import CorridorPipelineConfig


DATA_DIR = PROJECT_ROOT / "data"
USER_STORAGE = UserScopedStorage(DATA_DIR)

app = FastAPI(
    title="Floor Plan Processing API",
    description="프런트엔드에서 전달한 객체 감지 라벨을 그래프로 변환하고 결과를 JSON으로 저장합니다.",
    version="1.0.0",
)

service = FloorPlanProcessingService(storage_root=DATA_DIR, user_storage=USER_STORAGE)
AUTO_CORRECTION_SETTINGS = get_auto_correction_settings()
auto_correction_service = FloorPlanAutoCorrectionService(config=AUTO_CORRECTION_SETTINGS)
inference_service = build_inference_service_from_config()
step_three_repository = StepThreeRepository(storage_root=DATA_DIR, user_storage=USER_STORAGE)
DEFAULT_CORRIDOR_CONFIG = CorridorPipelineConfig()
BUILDING_REGISTRY_PATH = DATA_DIR / "building_registry.json"


class BuildingRegistrationRequest(BaseModel):
    building_name: str


class BuildingRegistrationResponse(BaseModel):
    building_name: str
    building_id: str
    is_new: bool


def _load_building_registry() -> dict[str, dict]:
    if not BUILDING_REGISTRY_PATH.exists():
        return {}
    try:
        raw = BUILDING_REGISTRY_PATH.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}


def _save_building_registry(registry: dict[str, dict]) -> None:
    BUILDING_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BUILDING_REGISTRY_PATH.open("w", encoding="utf-8") as fp:
        json.dump(registry, fp, ensure_ascii=False, indent=2)


def _generate_building_id(building_name: str, reserved_ids: set[str]) -> str:
    preferred = sanitize_user_id(building_name, default="").strip("_").lower()
    if preferred and preferred not in reserved_ids:
        return preferred

    suffix = 1
    while True:
        candidate = f"building-{suffix:04d}"
        if candidate not in reserved_ids:
            return candidate
        suffix += 1


def _build_stored_floorplan_results(user_id: str) -> List[dict]:
    raw_results = service.list_results(user_id=user_id)
    step_three_records = step_three_repository.list_all(user_id=user_id)

    request_to_step_one_id = {}
    for record in step_three_records:
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
                "objectDetectionText": item.get("object_detection_text") or "",
                "wallText": item.get("wall_text") or "",
                "wallBaseText": item.get("wall_base_text") or "",
                "doorText": item.get("door_text") or "",
                "imageUrl": item.get("image_url"),
                "imageDataUrl": None,
                "floorLabel": item.get("floor_label") or item.get("floorLabel"),
                "floorValue": item.get("floor_value") or item.get("floorValue"),
            }
        )

    summaries.sort(key=lambda entry: entry.get("createdAt") or "", reverse=True)
    return summaries


def _preview_options_to_config(options: Optional[FreeSpacePreviewOptions]) -> Optional[CorridorPipelineConfig]:
    if not options:
        return None
    door_probe = options.door_probe_distance or DEFAULT_CORRIDOR_CONFIG.door_probe_distance
    kernel = tuple(options.morph_open_kernel) if options.morph_open_kernel else DEFAULT_CORRIDOR_CONFIG.morph_open_kernel
    morph_iters = options.morph_open_iterations or DEFAULT_CORRIDOR_CONFIG.morph_open_iterations
    return CorridorPipelineConfig(
        door_probe_distance=int(door_probe),
        morph_open_kernel=(int(kernel[0]), int(kernel[1])),
        morph_open_iterations=int(morph_iters),
    )


def _absolute_image_url(request: Request, request_id: str, user_id: str) -> str:
    base_url = str(request.url_for("get_floorplan_image", request_id=request_id))
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}userId={user_id}"


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
    "/buildings/register",
    response_model=BuildingRegistrationResponse,
    summary="건물 이름을 안전한 데이터 폴더 ID와 매핑",
)
async def register_building(payload: BuildingRegistrationRequest) -> BuildingRegistrationResponse:
    building_name = (payload.building_name or "").strip()
    if not building_name:
        raise HTTPException(status_code=400, detail="building_name is required")

    registry = _load_building_registry()
    normalized_name = building_name
    entry = registry.get(normalized_name)
    now = datetime.utcnow().isoformat()

    if entry and isinstance(entry, dict) and entry.get("building_id"):
        entry["last_used_at"] = now
        registry[normalized_name] = entry
        _save_building_registry(registry)
        building_id = entry["building_id"]
        USER_STORAGE.resolve(building_id, create=True)
        return BuildingRegistrationResponse(building_name=normalized_name, building_id=building_id, is_new=False)

    reserved_ids = {
        item.get("building_id")
        for item in registry.values()
        if isinstance(item, dict) and item.get("building_id")
    }
    building_id = _generate_building_id(normalized_name, reserved_ids)
    registry[normalized_name] = {
        "building_id": building_id,
        "created_at": now,
        "last_used_at": now,
    }
    _save_building_registry(registry)
    USER_STORAGE.resolve(building_id, create=True)
    return BuildingRegistrationResponse(building_name=normalized_name, building_id=building_id, is_new=True)


@app.post(
    "/api/floorplans/process",
    response_model=ProcessFloorPlanResponse,
    summary="객체 감지 어노테이션을 그래프/객체 정보로 변환",
)
async def process_floorplan(request: Request, payload: ProcessFloorPlanRequest) -> ProcessFloorPlanResponse:
    user_id = resolve_active_user_id(request)
    try:
        preview_payload = None
        if payload.free_space_preview is not None:
            if hasattr(payload.free_space_preview, "model_dump"):
                preview_payload = payload.free_space_preview.model_dump(by_alias=True)
            else:
                preview_payload = payload.free_space_preview.dict(by_alias=True)
        scale_reference_payload = None
        if payload.scale_reference is not None:
            scale_reference_payload = payload.scale_reference.model_dump(by_alias=True)
        result = service.process(
            image_width=payload.image_width,
            image_height=payload.image_height,
            class_names=payload.class_names,
            source_image_path=payload.source_image_path,
            object_detection_text=payload.object_detection_text,
            wall_text=payload.wall_text,
            wall_base_text=payload.wall_base_text,
            door_text=payload.door_text,
            floor_label=payload.floor_label,
            floor_value=payload.floor_value,
            scale_reference=scale_reference_payload,
            user_id=user_id,
            image_data_url=payload.image_data_url,
            request_id=payload.request_id,
            skip_graph=payload.skip_graph,
            free_space_preview=preview_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProcessFloorPlanResponse(**result)


@app.post(
    "/api/floorplans/save-step-one",
    response_model=SaveStepOneResponse,
    summary="1단계 편집 내용을 서버에 저장 (그래프 생성 없음)",
)
async def save_step_one(request: Request, payload: SaveStepOneRequest) -> SaveStepOneResponse:
    user_id = resolve_active_user_id(request)
    preview_payload = None
    if payload.free_space_preview is not None:
        if hasattr(payload.free_space_preview, "model_dump"):
            preview_payload = payload.free_space_preview.model_dump(by_alias=True)
        else:
            preview_payload = payload.free_space_preview.dict(by_alias=True)
    scale_reference_payload = None
    if payload.scale_reference is not None:
        scale_reference_payload = payload.scale_reference.model_dump(by_alias=True)
    try:
        result = service.save_step_one(
            image_width=payload.image_width,
            image_height=payload.image_height,
            class_names=payload.class_names,
            source_image_path=payload.source_image_path,
            object_detection_text=payload.object_detection_text,
            wall_text=payload.wall_text,
            wall_base_text=payload.wall_base_text,
            door_text=payload.door_text,
            floor_label=payload.floor_label,
            floor_value=payload.floor_value,
            scale_reference=scale_reference_payload,
            user_id=user_id,
            image_data_url=payload.image_data_url,
            request_id=payload.request_id,
            free_space_preview=preview_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SaveStepOneResponse(**result)


@app.post(
    "/api/floorplans/{request_id}/prepare-graph",
    response_model=ProcessFloorPlanResponse,
    summary="저장된 Step1 텍스트를 기반으로 그래프 생성",
)
async def prepare_graph(
    request: Request,
    request_id: str,
    payload: Optional[PrepareGraphRequest] = None,
) -> ProcessFloorPlanResponse:
    user_id = resolve_active_user_id(request)
    preview_payload = None
    if payload and payload.free_space_preview is not None:
        if hasattr(payload.free_space_preview, "model_dump"):
            preview_payload = payload.free_space_preview.model_dump(by_alias=True)
        else:
            preview_payload = payload.free_space_preview.dict(by_alias=True)
    try:
        result = service.prepare_graph(request_id, user_id=user_id, free_space_preview=preview_payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProcessFloorPlanResponse(**result)


@app.post(
    "/api/floorplans/free-space-preview",
    response_model=FreeSpacePreviewResponse,
    summary="복도 자유 공간 마스크 미리보기 생성",
)
async def free_space_preview(payload: FreeSpacePreviewRequest) -> FreeSpacePreviewResponse:
    try:
        config = _preview_options_to_config(payload.options)
        result = service.generate_free_space_preview(
            image_width=payload.image_width,
            image_height=payload.image_height,
            object_detection_text=payload.object_detection_text,
            wall_text=payload.wall_text,
            door_text=payload.door_text,
            config=config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FreeSpacePreviewResponse(**result)


@app.post(
    "/api/floorplans/auto-correct",
    response_model=AutoCorrectLayoutResponse,
    summary="박스/벽 좌표를 자동 보정",
)
async def auto_correct_layout(payload: AutoCorrectLayoutRequest) -> AutoCorrectLayoutResponse:
    try:
        result = auto_correction_service.auto_correct(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AutoCorrectLayoutResponse(**result)


@app.post(
    "/api/floorplans/inference",
    summary="업로드된 이미지를 RF-DETR 모델로 객체 추론",
)
async def infer_floorplan_from_image(file: UploadFile = File(...)) -> dict:
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="이미지 파일을 읽지 못했습니다.") from exc

    if not image_bytes:
        raise HTTPException(status_code=400, detail="비어 있는 이미지가 전달되었습니다.")

    try:
        result = inference_service.infer_from_file(image_bytes, filename=file.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except DependencyNotAvailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except TorchNotAvailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"RF-DETR 추론에 실패했습니다: {exc}") from exc

    return result


@app.get(
    "/api/floorplans/{request_id}/image",
    summary="저장된 원본 이미지 조회",
)
async def get_floorplan_image(request: Request, request_id: str) -> FileResponse:
    user_id = resolve_active_user_id(request)
    image_info = service.get_image_info(request_id, user_id=user_id)
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
    user_id = resolve_active_user_id(request)
    raw_results = _build_stored_floorplan_results(user_id)
    for entry in raw_results:
        request_id = entry.get("requestId")
        if request_id:
            entry["imageUrl"] = _absolute_image_url(request, request_id, user_id)
    return [StoredFloorPlanSummary(**entry) for entry in raw_results]


@app.get(
    "/api/floorplans/floors",
    response_model=List[FloorPlanFloorSummary],
    summary="저장된 도면의 층 정보 목록 조회",
)
async def list_floorplan_floors(request: Request) -> List[FloorPlanFloorSummary]:
    user_id = resolve_active_user_id(request)
    raw_results = _build_stored_floorplan_results(user_id)
    floors: List[FloorPlanFloorSummary] = []
    for entry in raw_results:
        floors.append(
            FloorPlanFloorSummary(
                step_one_id=entry.get("stepOneId"),
                request_id=entry.get("requestId"),
                created_at=entry.get("createdAt"),
                floor_label=entry.get("floorLabel") or entry.get("floor_label"),
                floor_value=entry.get("floorValue") or entry.get("floor_value"),
            )
        )
    return floors


@app.get(
    "/api/floorplans/by-step-one/{step_one_id}",
    response_model=StoredFloorPlanSummary,
    summary="Step One ID로 저장된 그래프 결과 조회",
)
async def get_floorplan_by_step_one(step_one_id: str, request: Request) -> StoredFloorPlanSummary:
    user_id = resolve_active_user_id(request)
    for item in _build_stored_floorplan_results(user_id):
        if item.get("stepOneId") == step_one_id:
            image_url = item.get("imageUrl")
            if image_url:
                item["imageUrl"] = _absolute_image_url(request, item["requestId"], user_id)
            return StoredFloorPlanSummary(**item)
    raise HTTPException(status_code=404, detail="저장된 결과를 찾을 수 없습니다.")


@app.get(
    "/api/floorplans/{request_id}",
    response_model=ProcessFloorPlanResponse,
    summary="저장된 그래프 결과 조회",
)
async def get_floorplan(request: Request, request_id: str) -> ProcessFloorPlanResponse:
    user_id = resolve_active_user_id(request)
    try:
        result = service.get_result(request_id, user_id=user_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    metadata = result.get("metadata") or {}
    image_url = metadata.get("image_url")
    if image_url:
        metadata["image_url"] = _absolute_image_url(request, request_id, user_id)
    result["metadata"] = metadata
    return ProcessFloorPlanResponse(**result)


@app.get(
    "/api/floorplans/{request_id}/graph",
    response_model=GraphDataResponse,
    summary="그래프 데이터 조회",
)
async def get_floorplan_graph(request: Request, request_id: str) -> GraphDataResponse:
    user_id = resolve_active_user_id(request)
    try:
        graph_payload = service.get_graph_data(request_id, user_id=user_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GraphDataResponse(**graph_payload)


@app.put(
    "/api/floorplans/{request_id}/graph",
    response_model=GraphDataResponse,
    summary="그래프 데이터 저장",
)
async def save_floorplan_graph(request: Request, request_id: str, payload: GraphUpdateRequest) -> GraphDataResponse:
    user_id = resolve_active_user_id(request)
    try:
        updated_graph = service.save_graph(request_id, payload.graph.model_dump(), user_id=user_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GraphDataResponse(**updated_graph)


@app.delete(
    "/api/floorplans/{request_id}",
    summary="저장된 도면 삭제",
)
async def delete_floorplan(
    request: Request,
    request_id: str,
    step_one_id: Optional[str] = Query(default=None, alias="stepOneId"),
) -> dict:
    user_id = resolve_active_user_id(request)
    try:
        deletion_info = service.delete_result(request_id, user_id=user_id, step_one_id=step_one_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "deleted", **deletion_info}


@app.get(
    "/api/step-three",
    response_model=List[StepThreeStatus],
    summary="3단계 저장 현황 조회",
)
async def list_step_three_results(request: Request) -> List[StepThreeStatus]:
    user_id = resolve_active_user_id(request)
    records = step_three_repository.list_all(user_id=user_id)
    statuses: List[StepThreeStatus] = []
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
            StepThreeStatus(
                id=record.get("id"),
                request_id=record.get("requestId"),
                floor_label=record.get("floorLabel") or record.get("floor_label"),
                floor_value=record.get("floorValue") or record.get("floor_value"),
                has_base=has_base,
                has_details=has_details,
                base_updated_at=updated_at if has_base else None,
                details_updated_at=updated_at if has_details else None,
                updated_at=updated_at,
            )
        )
    return statuses


@app.get(
    "/api/step-three/{step_one_id}",
    response_model=StepThreeRecord,
    summary="3단계 저장본 상세 조회",
)
async def get_step_three_result(request: Request, step_one_id: str) -> StepThreeRecord:
    user_id = resolve_active_user_id(request)
    record = step_three_repository.get(step_one_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="저장된 3단계 결과가 없습니다.")
    return StepThreeRecord(**record)


@app.put(
    "/api/step-three/{step_one_id}",
    response_model=StepThreeRecord,
    summary="3단계 정보 저장",
)
async def save_step_three(request: Request, step_one_id: str, payload: StepThreeSaveRequest) -> StepThreeRecord:
    user_id = resolve_active_user_id(request)
    record = step_three_repository.save(step_one_id, payload.model_dump(by_alias=True), user_id=user_id)
    return StepThreeRecord(**record)
