from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator


class ObjectDetectionAnnotation(BaseModel):
    """프런트엔드에서 전달하는 객체 감지 박스 정보."""

    class_id: int = Field(..., ge=0, description="클래스 ID (예: 0=room, 1=stairs 등)")
    x_center: float = Field(..., ge=0.0, le=1.0, description="정규화된 중심 X 좌표")
    y_center: float = Field(..., ge=0.0, le=1.0, description="정규화된 중심 Y 좌표")
    width: float = Field(..., gt=0.0, le=1.0, description="정규화된 박스 폭")
    height: float = Field(..., gt=0.0, le=1.0, description="정규화된 박스 높이")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="선택적 신뢰도 점수 (없으면 1.0으로 처리)",
    )

class ScaleReferencePayload(BaseModel):
    x1: float = Field(..., ge=0.0, le=1.0, description="기준선 시작점 X(정규화)")
    y1: float = Field(..., ge=0.0, le=1.0, description="기준선 시작점 Y(정규화)")
    x2: float = Field(..., ge=0.0, le=1.0, description="기준선 끝점 X(정규화)")
    y2: float = Field(..., ge=0.0, le=1.0, description="기준선 끝점 Y(정규화)")
    length_meters: float = Field(
        ..., alias="lengthMeters", gt=0.0, description="기준선 실제 길이(미터)"
    )

    class Config:
        populate_by_name = True


class ProcessFloorPlanRequest(BaseModel):
    """객체 감지 라벨 텍스트나 어노테이션을 기반으로 그래프를 생성하는 요청 스키마."""

    annotations: Optional[List[ObjectDetectionAnnotation]] = Field(
        default=None,
        description="객체 감지 형식의 박스 리스트 (정규화 좌표). object_detection_text를 대신 사용할 수 있음.",
    )
    object_detection_text: Optional[str] = Field(
        default=None,
        description="객체 감지 텍스트 (room/stairs/elevator/door 박스)",
    )
    wall_text: Optional[str] = Field(
        default=None,
        description="벽 선분 텍스트 (x1 y1 x2 y2 ...)",
    )
    wall_base_text: Optional[str] = Field(
        default=None,
        serialization_alias="wallBaseText",
        validation_alias="wallBaseText",
        description="퍼센티지 필터 적용 전 원본 벽 선분 텍스트",
    )
    door_text: Optional[str] = Field(
        default=None,
        description="문 포인트 텍스트 (anchor 포함 형식)",
    )
    request_id: Optional[str] = Field(
        default=None,
        serialization_alias="requestId",
        validation_alias="requestId",
        description="기존 결과를 덮어쓸 때 사용할 요청 ID",
    )
    image_width: PositiveInt = Field(..., description="원본 이미지의 너비(px)")
    image_height: PositiveInt = Field(..., description="원본 이미지의 높이(px)")
    class_names: Optional[List[str]] = Field(
        default=None,
        description="클래스 이름 목록. 미제공 시 디폴트 클래스(방, 계단, 벽, 엘리베이터, 문)를 사용",
    )
    source_image_path: Optional[str] = Field(
        default=None,
        description="원본 이미지 파일 경로 (보관용 메타데이터)",
    )
    image_data_url: Optional[str] = Field(
        default=None,
        serialization_alias="imageDataUrl",
        validation_alias="imageDataUrl",
        description="Data URL(Base64) 형태의 원본 이미지. 제공 시 서버에 함께 저장됩니다.",
    )
    skip_graph: bool = Field(
        default=False,
        serialization_alias="skipGraph",
        validation_alias="skipGraph",
        description="true일 경우 그래프를 생성하지 않고 텍스트/메타데이터만 저장",
    )
    free_space_preview: Optional["FreeSpacePreviewData"] = Field(
        default=None,
        alias="freeSpacePreview",
        description="자유 공간 미리보기 결과(Stage1 비트마스크 묶음)",
    )
    floor_label: Optional[str] = Field(default=None, alias="floorLabel", description="도면 층 정보(표시용)")
    floor_value: Optional[str] = Field(default=None, alias="floorValue", description="도면 층 정보(식별자)")
    scale_reference: Optional[ScaleReferencePayload] = Field(
        default=None,
        alias="scaleReference",
        description="사용자가 지정한 기준선 정보",
    )

    @field_validator("class_names", mode="before")
    @classmethod
    def validate_class_names(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if not value:
            return None
        return value

    @model_validator(mode="after")
    def validate_input_payload(self) -> "ProcessFloorPlanRequest":
        annotations = self.annotations
        object_detection_text = self.object_detection_text

        if annotations is None and (object_detection_text is None or object_detection_text == ""):
            raise ValueError("annotations 또는 object_detection_text 중 하나는 반드시 제공해야 합니다.")
        if object_detection_text is not None:
            object.__setattr__(self, "wall_text", self.wall_text or "")
            object.__setattr__(self, "door_text", self.door_text or "")
        return self


class FreeSpacePreviewOptions(BaseModel):
    door_probe_distance: Optional[int] = Field(
        default=None,
        alias="doorProbeDistance",
        ge=1,
        le=200,
        description="문 주변 자유 공간을 탐색할 거리 (픽셀)",
    )
    morph_open_kernel: Optional[List[int]] = Field(
        default=None,
        alias="morphOpenKernel",
        description="전경 잡음을 제거할 오프닝 커널 크기 [width, height]",
    )
    morph_open_iterations: Optional[int] = Field(
        default=None,
        alias="morphOpenIterations",
        ge=1,
        le=10,
        description="모폴로지 오프닝 반복 횟수",
    )

    @field_validator("morph_open_kernel")
    @classmethod
    def validate_kernel(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("morphOpenKernel 은 [width, height] 2개 값을 포함해야 합니다.")
        width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("morphOpenKernel 값은 양수여야 합니다.")
        return value

    class Config:
        populate_by_name = True


class FreeSpacePreviewRequest(BaseModel):
    object_detection_text: Optional[str] = Field(
        default=None,
        description="객체 감지 텍스트",
        alias="objectDetectionText",
    )
    wall_text: Optional[str] = Field(
        default=None,
        description="벽 라인 텍스트",
        alias="wallText",
    )
    door_text: Optional[str] = Field(
        default=None,
        description="문 포인트 텍스트",
        alias="doorText",
    )
    image_width: PositiveInt = Field(..., alias="imageWidth")
    image_height: PositiveInt = Field(..., alias="imageHeight")
    options: Optional[FreeSpacePreviewOptions] = Field(default=None, description="마스크 생성 옵션")

    class Config:
        populate_by_name = True

    @model_validator(mode="after")
    def validate_source(self) -> "FreeSpacePreviewRequest":
        if not self.object_detection_text:
            raise ValueError("object_detection_text 값이 필요합니다.")
        return self


class GeometryPayload(BaseModel):
    type: str
    coordinates: list


class FloorPlanObject(BaseModel):
    id: int
    label: str
    corners: List[List[float]]
    centroid: GeometryPayload
    polygon: GeometryPayload
    properties: dict = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    type: Optional[str] = None
    pos: Optional[List[float]] = None
    attributes: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: Optional[float] = None
    attributes: dict = Field(default_factory=dict)


class GraphPayload(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    floor_label: Optional[str] = Field(default=None, alias="floorLabel")
    floor_value: Optional[str] = Field(default=None, alias="floorValue")
    request_id: Optional[str] = Field(default=None, alias="requestId")

    class Config:
        populate_by_name = True


class GraphDataResponse(BaseModel):
    request_id: str = Field(..., alias="requestId")
    graph: GraphPayload
    objects: Optional[dict] = None
    metadata: Optional[dict] = None

    class Config:
        populate_by_name = True


class GraphUpdateRequest(BaseModel):
    graph: GraphPayload


class AnnotationCounts(BaseModel):
    boxes: int = 0
    walls: int = 0
    doors: int = 0


class StoredFloorPlanSummary(BaseModel):
    step_one_id: str = Field(..., alias="stepOneId")
    request_id: str = Field(..., alias="requestId")
    created_at: datetime = Field(..., alias="createdAt")
    image_size: Dict[str, int] = Field(default_factory=dict, alias="imageSize")
    class_names: List[str] = Field(default_factory=list, alias="classNames")
    source_image_path: Optional[str] = Field(default=None, alias="sourceImagePath")
    graph_summary: Optional[Dict[str, int]] = Field(default=None, alias="graphSummary")
    annotation_counts: AnnotationCounts = Field(default_factory=AnnotationCounts, alias="annotationCounts")
    object_detection_text: str = Field(default="", alias="objectDetectionText")
    wall_text: str = Field(default="", alias="wallText")
    wall_base_text: str = Field(default="", alias="wallBaseText")
    door_text: str = Field(default="", alias="doorText")
    image_url: Optional[str] = Field(default=None, alias="imageUrl")
    image_data_url: Optional[str] = Field(default=None, alias="imageDataUrl")
    floor_label: Optional[str] = Field(default=None, alias="floorLabel")
    floor_value: Optional[str] = Field(default=None, alias="floorValue")

    class Config:
        populate_by_name = True


class FloorPlanFloorSummary(BaseModel):
    step_one_id: Optional[str] = Field(default=None, alias="stepOneId")
    request_id: str = Field(..., alias="requestId")
    created_at: Optional[datetime] = Field(default=None, alias="createdAt")
    floor_label: Optional[str] = Field(default=None, alias="floorLabel")
    floor_value: Optional[str] = Field(default=None, alias="floorValue")

    class Config:
        populate_by_name = True


class ProcessFloorPlanResponse(BaseModel):
    request_id: str
    created_at: datetime
    image_size: dict
    class_names: List[str]
    objects: dict
    graph: dict
    saved_files: dict
    input_annotations: List[dict]
    metadata: Optional[dict] = None


class StageOneBitmaskPayload(BaseModel):
    encoding: str
    shape: List[int]
    length: PositiveInt
    data: str

    @field_validator("encoding")
    @classmethod
    def validate_encoding(cls, value: str) -> str:
        if value not in {"bitpack-base64", "bitmask-base64"}:
            raise ValueError("encoding must be bitpack-base64")
        return value


class StageOneMaskSet(BaseModel):
    free_space: StageOneBitmaskPayload = Field(..., alias="freeSpace")
    door: StageOneBitmaskPayload = Field(..., alias="door")
    room: StageOneBitmaskPayload = Field(..., alias="room")
    wall: StageOneBitmaskPayload = Field(..., alias="wall")

    class Config:
        populate_by_name = True


class StageOneDoorMidpoint(BaseModel):
    midpoint_rc: Optional[List[float]] = Field(default=None, alias="midpointRc")
    midpoint_xy: Optional[List[float]] = Field(default=None, alias="midpointXy")
    room_ids: Optional[List[int]] = Field(default=None, alias="roomIds")

    class Config:
        populate_by_name = True


class StageOneDoorMidpointBundle(BaseModel):
    door_id: int = Field(..., alias="doorId")
    midpoints: List[StageOneDoorMidpoint] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StageOneArtifactBundle(BaseModel):
    version: int = Field(1, alias="version")
    width: PositiveInt = Field(..., alias="width")
    height: PositiveInt = Field(..., alias="height")
    masks: StageOneMaskSet = Field(..., alias="masks")
    skeleton: Optional[StageOneBitmaskPayload] = Field(default=None, alias="skeleton")
    door_midpoints: Optional[List[StageOneDoorMidpointBundle]] = Field(default=None, alias="doorMidpoints")

    class Config:
        populate_by_name = True


class FreeSpacePreviewResponse(BaseModel):
    image_size: Dict[str, int] = Field(..., alias="imageSize")
    free_space_ratio: float = Field(..., alias="freeSpaceRatio")
    config: Dict[str, Any]
    artifact_bundle: StageOneArtifactBundle = Field(..., alias="artifactBundle")

    class Config:
        populate_by_name = True


class StageOnePreviewBase(BaseModel):
    image_size: Dict[str, int] = Field(..., alias="imageSize")
    free_space_ratio: Optional[float] = Field(default=None, alias="freeSpaceRatio")
    config: Optional[Dict[str, Any]] = None
    artifact_bundle: Optional[StageOneArtifactBundle] = Field(default=None, alias="artifactBundle")

    class Config:
        populate_by_name = True


class FreeSpacePreviewData(StageOnePreviewBase):
    """1단계 미리보기 데이터를 API 요청/저장 시 재사용."""


class StoredPreviewSummary(StageOnePreviewBase):
    """서버에 저장된 미리보기 요약."""


class SaveStepOnePreviewPayload(StageOnePreviewBase):
    """프런트엔드가 1단계를 저장할 때 전달하는 미리보기 페이로드."""


class SaveStepOneRequest(BaseModel):
    object_detection_text: str = Field(..., alias="objectDetectionText")
    wall_text: str = Field(..., alias="wallText")
    wall_base_text: Optional[str] = Field(default=None, alias="wallBaseText")
    door_text: str = Field(..., alias="doorText")
    image_width: PositiveInt = Field(..., alias="imageWidth")
    image_height: PositiveInt = Field(..., alias="imageHeight")
    class_names: Optional[List[str]] = Field(default=None, alias="classNames")
    source_image_path: Optional[str] = Field(default=None, alias="sourceImagePath")
    image_data_url: Optional[str] = Field(default=None, alias="imageDataUrl")
    request_id: Optional[str] = Field(default=None, alias="requestId")
    free_space_preview: Optional[SaveStepOnePreviewPayload] = Field(default=None, alias="freeSpacePreview")
    floor_label: Optional[str] = Field(default=None, alias="floorLabel")
    floor_value: Optional[str] = Field(default=None, alias="floorValue")
    scale_reference: Optional[ScaleReferencePayload] = Field(
        default=None,
        alias="scaleReference",
        description="사용자가 지정한 기준선 정보",
    )

    class Config:
        populate_by_name = True

    @model_validator(mode="after")
    def validate_payload(self) -> "SaveStepOneRequest":
        if not self.object_detection_text:
            raise ValueError("object_detection_text 값이 필요합니다.")
        if not self.wall_text:
            raise ValueError("wall_text 값이 필요합니다.")
        if not self.door_text:
            raise ValueError("door_text 값이 필요합니다.")
        return self


class SaveStepOneResponse(BaseModel):
    request_id: str = Field(..., alias="requestId")
    created_at: datetime = Field(..., alias="createdAt")
    image_size: Dict[str, int] = Field(..., alias="imageSize")
    class_names: List[str] = Field(default_factory=list, alias="classNames")
    annotation_counts: Dict[str, int] = Field(default_factory=dict, alias="annotationCounts")
    preview: Optional[StoredPreviewSummary] = Field(default=None, alias="preview")
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True


class PrepareGraphRequest(BaseModel):
    free_space_preview: Optional[SaveStepOnePreviewPayload] = Field(default=None, alias="freeSpacePreview")

    class Config:
        populate_by_name = True


class AutoCorrectBoxPayload(BaseModel):
    id: str
    label_id: Optional[str] = Field(default=None, alias="labelId")
    type: Optional[str] = None
    x: float
    y: float
    width: float
    height: float
    meta: Optional[Dict] = None

    class Config:
        populate_by_name = True
        extra = "allow"


class AutoCorrectLinePayload(BaseModel):
    id: str
    label_id: Optional[str] = Field(default=None, alias="labelId")
    type: Optional[str] = None
    x1: float
    y1: float
    x2: float
    y2: float
    meta: Optional[Dict] = None

    class Config:
        populate_by_name = True
        extra = "allow"


class AutoCorrectLayoutRequest(BaseModel):
    boxes: List[AutoCorrectBoxPayload] = Field(default_factory=list)
    lines: List[AutoCorrectLinePayload] = Field(default_factory=list)
    image_width: Optional[int] = Field(default=None, alias="imageWidth")
    image_height: Optional[int] = Field(default=None, alias="imageHeight")

    class Config:
        populate_by_name = True


class AutoCorrectionStats(BaseModel):
    box_box_adjustments: int = Field(0, alias="boxBoxAdjustments")
    wall_box_snaps: int = Field(0, alias="wallBoxSnaps")
    wall_wall_snaps: int = Field(0, alias="wallWallSnaps")

    class Config:
        populate_by_name = True


class AutoCorrectLayoutResponse(BaseModel):
    boxes: List[AutoCorrectBoxPayload] = Field(default_factory=list)
    lines: List[AutoCorrectLinePayload] = Field(default_factory=list)
    stats: AutoCorrectionStats = Field(default_factory=AutoCorrectionStats)

    class Config:
        populate_by_name = True


class StepThreeExtraField(BaseModel):
    key: str = ""
    value: str = ""


class StepThreeRoomPayload(BaseModel):
    node_id: str = Field(..., alias="nodeId")
    graph_node_id: Optional[str] = Field(default=None, alias="graphNodeId")
    name: Optional[str] = ""
    number: Optional[str] = ""
    extra: List[StepThreeExtraField] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StepThreeDoorPayload(BaseModel):
    node_id: str = Field(..., alias="nodeId")
    graph_node_id: Optional[str] = Field(default=None, alias="graphNodeId")
    type: Optional[str] = ""
    custom_type: Optional[str] = Field(default="", alias="customType")
    extra: List[StepThreeExtraField] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StepThreeSaveRequest(BaseModel):
    request_id: Optional[str] = Field(default=None, alias="requestId")
    floor_label: Optional[str] = Field(default=None, alias="floorLabel")
    floor_value: Optional[str] = Field(default=None, alias="floorValue")
    rooms: List[StepThreeRoomPayload] = Field(default_factory=list)
    doors: List[StepThreeDoorPayload] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StepThreeRecord(BaseModel):
    id: str
    request_id: Optional[str] = Field(default=None, alias="requestId")
    floor_label: Optional[str] = Field(default=None, alias="floorLabel")
    floor_value: Optional[str] = Field(default=None, alias="floorValue")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")
    rooms: List[StepThreeRoomPayload] = Field(default_factory=list)
    doors: List[StepThreeDoorPayload] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StepThreeStatus(BaseModel):
    id: str
    request_id: Optional[str] = Field(default=None, alias="requestId")
    floor_label: Optional[str] = Field(default=None, alias="floorLabel")
    floor_value: Optional[str] = Field(default=None, alias="floorValue")
    has_base: bool = Field(..., alias="hasBase")
    has_details: bool = Field(..., alias="hasDetails")
    base_updated_at: Optional[datetime] = Field(default=None, alias="baseUpdatedAt")
    details_updated_at: Optional[datetime] = Field(default=None, alias="detailsUpdatedAt")
    updated_at: Optional[datetime] = Field(default=None, alias="updatedAt")

    class Config:
        populate_by_name = True
