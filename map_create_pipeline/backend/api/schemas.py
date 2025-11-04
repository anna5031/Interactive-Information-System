from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator


class YoloAnnotation(BaseModel):
    """프런트엔드에서 전달하는 YOLO 형식 박스 정보."""

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


class ProcessFloorPlanRequest(BaseModel):
    """YOLO 라벨 텍스트나 어노테이션을 기반으로 그래프를 생성하는 요청 스키마."""

    annotations: Optional[List[YoloAnnotation]] = Field(
        default=None,
        description="YOLO 형식 객체 리스트 (정규화 좌표). yolo_text를 대신 사용할 수 있음.",
    )
    yolo_text: Optional[str] = Field(
        default=None,
        description="YOLO 형식 텍스트 (room/stairs/elevator/door 박스)",
    )
    wall_text: Optional[str] = Field(
        default=None,
        description="벽 선분 텍스트 (x1 y1 x2 y2 ...)",
    )
    door_text: Optional[str] = Field(
        default=None,
        description="문 포인트 텍스트 (anchor 포함 형식)",
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
        alias="imageDataUrl",
        description="Data URL(Base64) 형태의 원본 이미지. 제공 시 서버에 함께 저장됩니다.",
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
        yolo_text = self.yolo_text

        if annotations is None and (yolo_text is None or yolo_text == ""):
            raise ValueError("annotations 또는 yolo_text 중 하나는 반드시 제공해야 합니다.")
        if yolo_text is not None:
            object.__setattr__(self, "wall_text", self.wall_text or "")
            object.__setattr__(self, "door_text", self.door_text or "")
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
    yolo_text: str = Field(default="", alias="yoloText")
    wall_text: str = Field(default="", alias="wallText")
    door_text: str = Field(default="", alias="doorText")
    image_url: Optional[str] = Field(default=None, alias="imageUrl")
    image_data_url: Optional[str] = Field(default=None, alias="imageDataUrl")

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


class StepTwoExtraField(BaseModel):
    key: str = ""
    value: str = ""


class StepTwoRoomPayload(BaseModel):
    node_id: str = Field(..., alias="nodeId")
    graph_node_id: Optional[str] = Field(default=None, alias="graphNodeId")
    name: Optional[str] = ""
    number: Optional[str] = ""
    extra: List[StepTwoExtraField] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StepTwoDoorPayload(BaseModel):
    node_id: str = Field(..., alias="nodeId")
    graph_node_id: Optional[str] = Field(default=None, alias="graphNodeId")
    type: Optional[str] = ""
    custom_type: Optional[str] = Field(default="", alias="customType")
    extra: List[StepTwoExtraField] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StepTwoSaveRequest(BaseModel):
    request_id: Optional[str] = Field(default=None, alias="requestId")
    rooms: List[StepTwoRoomPayload] = Field(default_factory=list)
    doors: List[StepTwoDoorPayload] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StepTwoRecord(BaseModel):
    id: str
    request_id: Optional[str] = Field(default=None, alias="requestId")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")
    rooms: List[StepTwoRoomPayload] = Field(default_factory=list)
    doors: List[StepTwoDoorPayload] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class StepTwoStatus(BaseModel):
    id: str
    request_id: Optional[str] = Field(default=None, alias="requestId")
    has_base: bool = Field(..., alias="hasBase")
    has_details: bool = Field(..., alias="hasDetails")
    base_updated_at: Optional[datetime] = Field(default=None, alias="baseUpdatedAt")
    details_updated_at: Optional[datetime] = Field(default=None, alias="detailsUpdatedAt")
    updated_at: Optional[datetime] = Field(default=None, alias="updatedAt")

    class Config:
        populate_by_name = True
