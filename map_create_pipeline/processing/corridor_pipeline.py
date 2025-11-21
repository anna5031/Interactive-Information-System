"""고정밀 복도 그래프 생성을 담당하는 파이프라인."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import networkx as nx
import numpy as np
from itertools import combinations
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label, center_of_mass
from scipy.signal import convolve2d
from scipy.spatial import KDTree
from shapely.geometry import LineString, Polygon, Point
from shapely.prepared import prep
from skimage.draw import polygon as draw_polygon
from skimage.morphology import skeletonize

try:
    from .util.find_slanted_line_102 import skeleton_to_segments  # type: ignore
except ImportError:  # pragma: no cover
    from util.find_slanted_line_102 import skeleton_to_segments  # type: ignore

try:
    from .floorplan_visualizer import FloorPlanVisualizer, EXPORT_IMAGE_FILENAME  # type: ignore
except ImportError:  # pragma: no cover
    from floorplan_visualizer import FloorPlanVisualizer, EXPORT_IMAGE_FILENAME  # type: ignore


CONNECTIVITY_STRUCTURE = np.ones((3, 3), dtype=np.int8)


def _normalize_kernel_size(size: Tuple[int, int]) -> Tuple[int, int]:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    return width, height


@dataclass
class CorridorPipelineConfig:
    """마스크 생성 및 문 연결 단계의 주요 하이퍼파라미터."""

    door_probe_distance: int = 10
    morph_open_kernel: Tuple[int, int] = (3, 3)
    morph_open_iterations: int = 1
    door_touch_kernel: Tuple[int, int] = (3, 3)


@dataclass
class FreeSpaceMaskBundle:
    """프리뷰/사후 처리를 위해 내보내는 마스크 묶음."""

    free_space: np.ndarray
    door_mask: np.ndarray
    room_mask: np.ndarray
    wall_mask: np.ndarray


@dataclass
class StageOneArtifacts(FreeSpaceMaskBundle):
    """1단계에서 생성해 2단계로 전달할 추가 아티팩트."""

    skeleton: Optional[np.ndarray] = None
    door_midpoints: Optional[List[Dict[str, Any]]] = None


class CorridorPipeline:
    """마스크·스켈레톤·A* 기반으로 통로 그래프를 구성하는 고정밀 생성기."""

    def __init__(
        self,
        parsed_objects,
        width,
        height,
        debug_dir: Optional[Path] = None,
        config: Optional[CorridorPipelineConfig] = None,
        precomputed_masks: Optional[FreeSpaceMaskBundle] = None,
        precomputed_stage_one: Optional[StageOneArtifacts] = None,
    ):
        """파서 결과와 이미지 크기를 받아 그래프/디버그 상태를 초기화한다."""
        self.objs = parsed_objects
        self.w, self.h = width, height
        self.G = nx.Graph()
        self.corridor_nodes_pos = {}
        self.corridor_endpoint_door_links = {}
        self.corridor_endpoint_room_links = {}
        self.door_endpoint_pixels = {}
        self.door_endpoint_rooms = {}
        self.newly_found_endpoint_coords: Set[Tuple[int, int]] = set()
        self.newly_found_corridor_nodes: Set[str] = set()
        self.door_to_corridor_nodes = defaultdict(list)
        self.corridor_kdtree = None
        self.debug_dir = debug_dir
        self.config = config or CorridorPipelineConfig()
        self._door_probe_distance = max(1, int(self.config.door_probe_distance))
        door_touch_size = _normalize_kernel_size(self.config.door_touch_kernel)
        self._door_touch_kernel = np.ones(door_touch_size, dtype=np.uint8)
        morph_open_size = _normalize_kernel_size(self.config.morph_open_kernel)
        self._morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_open_size)
        self._morph_open_iterations = max(1, int(self.config.morph_open_iterations))
        self._precomputed_stage_one = precomputed_stage_one
        if precomputed_stage_one is not None:
            self._precomputed_masks = precomputed_stage_one
            self._cached_masks = precomputed_stage_one
        else:
            self._precomputed_masks = precomputed_masks
            self._cached_masks = precomputed_masks

    def _prepare_polygon(self, corners: Sequence[Sequence[float]]) -> np.ndarray:
        """입력 꼭짓점 목록을 이미지 크기 안쪽으로 클램프한 ndarray 로 변환한다."""
        array = np.array(corners, dtype=float, copy=True)
        if array.size == 0:
            return array.reshape((-1, 2))
        array[:, 0] = np.clip(array[:, 0], 0.0, self.w - 1)
        array[:, 1] = np.clip(array[:, 1], 0.0, self.h - 1)
        return array

    def generate_free_space_masks(self) -> FreeSpaceMaskBundle:
        """자유 공간/도어/방/벽 마스크 묶음을 계산해 반환한다."""
        if self._cached_masks is not None:
            return self._cached_masks
        bundle = FreeSpaceMaskBundle(*self._create_free_space_mask_optimized())
        self._cached_masks = bundle
        return bundle

    def generate_stage_one_artifacts(self, include_midpoints: bool = True) -> StageOneArtifacts:
        """마스크/스켈레톤/문 중점 목록을 Stage1 아티팩트로 구성한다."""
        base_masks = self.generate_free_space_masks()
        skeleton = self._create_corridor_skeleton(base_masks.free_space, base_masks.door_mask)
        door_midpoints = self._collect_door_midpoints_payload() if include_midpoints else None
        artifacts = StageOneArtifacts(
            free_space=base_masks.free_space.copy(),
            door_mask=base_masks.door_mask.copy(),
            room_mask=base_masks.room_mask.copy(),
            wall_mask=base_masks.wall_mask.copy(),
            skeleton=skeleton.copy(),
            door_midpoints=door_midpoints,
        )
        self._precomputed_stage_one = artifacts
        self._cached_masks = artifacts
        return artifacts

    @staticmethod
    def _normalize_pair(value, cast_fn):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        result = []
        for component in value:
            try:
                result.append(cast_fn(component))
            except (TypeError, ValueError):
                return None
        return result

    def _collect_door_midpoints_payload(self) -> List[Dict[str, Any]]:
        """문 도어라인 중점을 직렬화 가능한 구조로 정리한다."""
        doors = self.objs.get('doors') or []
        payloads: List[Dict[str, Any]] = []
        for door in doors:
            door_id = door.get('id')
            if door_id is None:
                continue
            midpoint_infos = []
            for info in door.get('parallel_edge_midpoints') or []:
                rc_pair = info.get('midpoint_rc') or info.get('midpointRc')
                xy_pair = info.get('midpoint_xy') or info.get('midpointXy')
                rc_normalized = self._normalize_pair(rc_pair, int)
                if rc_normalized is None:
                    continue
                xy_normalized = self._normalize_pair(xy_pair, float) if xy_pair is not None else None
                room_ids = info.get('room_ids') or info.get('roomIds') or []
                normalized_rooms = sorted({
                    int(val)
                    for val in room_ids
                    if isinstance(val, (int, float))
                })
                midpoint_payload: Dict[str, Any] = {
                    'midpoint_rc': rc_normalized,
                }
                if xy_normalized is not None:
                    midpoint_payload['midpoint_xy'] = xy_normalized
                if normalized_rooms:
                    midpoint_payload['room_ids'] = normalized_rooms
                midpoint_infos.append(midpoint_payload)
            payloads.append({
                'door_id': int(door_id),
                'midpoints': midpoint_infos,
            })
        return payloads

    def _apply_precomputed_door_midpoints(self) -> None:
        """Stage1에서 전달된 도어 중점을 door 객체에 주입한다."""
        if not self._precomputed_stage_one:
            return
        payloads = self._precomputed_stage_one.door_midpoints
        if not payloads:
            return
        door_lookup: Dict[int, List[Dict[str, Any]]] = {}
        for entry in payloads:
            door_id_raw = entry.get('door_id') or entry.get('doorId')
            if door_id_raw is None:
                continue
            try:
                door_id = int(door_id_raw)
            except (TypeError, ValueError):
                continue
            normalized_midpoints: List[Dict[str, Any]] = []
            for info in entry.get('midpoints') or []:
                rc_pair = info.get('midpoint_rc') or info.get('midpointRc')
                xy_pair = info.get('midpoint_xy') or info.get('midpointXy')
                rc_normalized = self._normalize_pair(rc_pair, int)
                if rc_normalized is None:
                    continue
                xy_normalized = self._normalize_pair(xy_pair, float) if xy_pair is not None else None
                rooms = info.get('room_ids') or info.get('roomIds') or []
                normalized_rooms = sorted({
                    int(val)
                    for val in rooms
                    if isinstance(val, (int, float))
                })
                payload: Dict[str, Any] = {'midpoint_rc': rc_normalized}
                if xy_normalized is not None:
                    payload['midpoint_xy'] = xy_normalized
                if normalized_rooms:
                    payload['room_ids'] = normalized_rooms
                normalized_midpoints.append(payload)
            if normalized_midpoints:
                door_lookup[door_id] = normalized_midpoints
        if not door_lookup:
            return
        for door in self.objs.get('doors') or []:
            door_id = door.get('id')
            if door_id is None:
                continue
            try:
                lookup_id = int(door_id)
            except (TypeError, ValueError):
                continue
            if lookup_id in door_lookup:
                door['parallel_edge_midpoints'] = door_lookup[lookup_id]

    def build(self):
        """마스크 생성→스켈레톤→그래프 변환→후처리를 거쳐 최종 그래프를 반환한다."""
        # for i in range(len(self.objs.get('doors', []))):
        #     mask = self._create_free_space_mask_optimized(i)
        #     if self.debug_dir:
        #         self._save_mask(mask, f'free_space_mask_{i}.png')
        # i=26
        self._apply_precomputed_door_midpoints()
        masks = self.generate_free_space_masks()
        mask = masks.free_space
        if self.debug_dir:
            self._save_mask(mask, 'free_space_mask.png')
        # sys.exit(0)
        if self._precomputed_stage_one is not None and self._precomputed_stage_one.skeleton is not None:
            skeleton = self._precomputed_stage_one.skeleton.copy()
        else:
            skeleton = self._create_corridor_skeleton(mask, masks.door_mask)
            if self._precomputed_stage_one is not None:
                self._precomputed_stage_one.skeleton = skeleton.copy()
        self._remove_temporary_wall_edges()
        if self.debug_dir:
            self._save_mask(skeleton, 'skeleton.png', scale_to_uint8=True)
        # sys.exit(0)
        self._convert_skeleton_to_graph_optimized(
            skeleton,
            masks.door_mask,
            masks.room_mask,
            masks.wall_mask,
        )
        self._connect_objects_to_graph_optimized()
        self._simplify_graph_with_all_shortest_paths()

        if self.debug_dir:
            self._save_final_visualization()

        return self.G
    def _simplify_graph_with_all_shortest_paths(self) -> None:
        """터미널 노드 간 최단 경로를 유지하면서 불필요한 복도 노드를 제거한다."""
        # print("모든 최단 경로 보존 방식으로 그래프 단순화를 시작합니다...")

        # 1. 터미널 노드(반드시 연결되어야 할 중요 노드) 식별
        terminal_nodes = [
            n for n, data in self.G.nodes(data=True)
            if data.get('type') not in {'corridor'}
        ]

        if len(terminal_nodes) < 2:
            # print("터미널 노드가 부족하여 단순화를 건너뜁니다.")
            return

        # print(f"{len(terminal_nodes)}개의 터미널 노드 간 모든 최단 경로를 계산합니다...")
        
        essential_nodes = set(terminal_nodes)
        essential_edges = set()

        # 2. 모든 터미널 노드 쌍(combinations)에 대해 최단 경로 계산
        for start_node, end_node in combinations(terminal_nodes, 2):
            try:
                # Dijkstra 알고리즘으로 최단 경로를 찾음
                path = nx.dijkstra_path(self.G, source=start_node, target=end_node, weight='weight')
                
                # 3. 경로에 포함된 모든 노드와 엣지를 '필수' 요소로 추가
                essential_nodes.update(path)
                
                # 경로를 엣지 쌍으로 변환하여 추가
                for i in range(len(path) - 1):
                    # 엣지는 순서에 상관없이 저장하기 위해 정렬
                    edge = tuple(sorted((path[i], path[i+1])))
                    essential_edges.add(edge)

            except nx.NetworkXNoPath:
                # 두 터미널 사이에 경로가 없는 경우임
                # print(f"경고: 노드 {start_node}와 {end_node} 사이에 경로가 없습니다.")
                continue
        
        # 4. 필수적이지 않은 노드와 엣지 제거
        original_nodes = set(self.G.nodes())
        nodes_to_remove = original_nodes - essential_nodes
        
        # original_node_count = self.G.number_of_nodes()
        self.G.remove_nodes_from(list(nodes_to_remove))
        # simplified_node_count = self.G.number_of_nodes()
        
        # print(f"그래프 단순화 완료. 노드 수: {original_node_count} -> {simplified_node_count}")
        
        # 그래프 변경에 따른 의존적인 멤버 변수들 업데이트
        self.corridor_nodes_pos = {
            n: data['pos']
            for n, data in self.G.nodes(data=True)
            if 'pos' in data
        }
        if self.corridor_nodes_pos:
            self.corridor_kdtree = KDTree(list(self.corridor_nodes_pos.values()))
        else:
            self.corridor_kdtree = None
    def _create_free_space_mask_optimized(self):
        """벽/방/문 픽셀을 통합해 이동 가능 공간 마스크와 보조 마스크를 생성한다."""

        shape = (self.h, self.w)
        obstacle = np.zeros(shape, dtype=np.uint8)
        room_mask = np.zeros(shape, dtype=np.uint8)
        wall_mask = np.zeros(shape, dtype=np.uint8)
        door_mask = np.zeros(shape, dtype=np.uint8)
        door_boxes: List[np.ndarray] = []

        for key in ("rooms", "walls", "stairs", "elevators"):
            for obj in self.objs.get(key, []):
                poly_corners = self._prepare_polygon(obj['corners'])
                if poly_corners.size == 0:
                    continue
                rr, cc = draw_polygon(
                    np.round(poly_corners[:, 1]).astype(int),
                    np.round(poly_corners[:, 0]).astype(int),
                    shape,
                )
                obstacle[rr, cc] = 1
                if key == 'rooms':
                    room_mask[rr, cc] = 1
                if key == 'walls':
                    wall_mask[rr, cc] = 1

        for obj in self.objs.get('doors', []):
            poly_corners = self._prepare_polygon(obj['corners'])
            if poly_corners.size == 0:
                continue
            rr, cc = draw_polygon(
                np.round(poly_corners[:, 1]).astype(int),
                np.round(poly_corners[:, 0]).astype(int),
                shape,
            )
            door_mask[rr, cc] = 1
            obstacle[rr, cc] = 0
            room_mask[rr, cc] = 0
            door_boxes.append(poly_corners)

        door_mask_bool = door_mask.astype(bool)
        building_mask = binary_fill_holes(obstacle.astype(bool)).astype(np.uint8)
        free_space = building_mask & (1 - obstacle)
        free_space[door_mask_bool] = 0

        labeled_components, num_components = label(
            free_space.astype(bool),
            structure=CONNECTIVITY_STRUCTURE,
        )
        if num_components:
            door_touch_mask = cv2.dilate(
                door_mask,
                self._door_touch_kernel,
                iterations=1,
            ).astype(bool)
            unreachable_mask = np.zeros_like(free_space, dtype=bool)

            for component_idx in range(1, num_components + 1):
                component_mask = labeled_components == component_idx
                if not np.any(component_mask):
                    continue
                if np.any(door_touch_mask[component_mask]):
                    continue

                touches_border = (
                    np.any(component_mask[0, :])
                    or np.any(component_mask[-1, :])
                    or np.any(component_mask[:, 0])
                    or np.any(component_mask[:, -1])
                )
                if touches_border:
                    continue

                unreachable_mask |= component_mask

            if np.any(unreachable_mask):
                obstacle[unreachable_mask] = 1
                wall_mask[unreachable_mask] = 1
                free_space[unreachable_mask] = 0
                building_mask = binary_fill_holes(obstacle.astype(bool)).astype(np.uint8)
                free_space = building_mask & (1 - obstacle)
                free_space[door_mask_bool] = 0

        wall_mask_bool = wall_mask.astype(bool)
        probe_limit = self._door_probe_distance

        for corners in door_boxes:
            min_c = max(int(np.floor(corners[:, 0].min())), 0)
            max_c = min(int(np.ceil(corners[:, 0].max())), self.w - 1)
            min_r = max(int(np.floor(corners[:, 1].min())), 0)
            max_r = min(int(np.ceil(corners[:, 1].max())), self.h - 1)
            if min_c > max_c or min_r > max_r:
                continue

            patch_height = max_r - min_r + 1
            patch_width = max_c - min_c + 1
            if patch_height <= 0 or patch_width <= 0:
                continue

            door_patch = np.zeros((patch_height, patch_width), dtype=bool)
            rr_local, cc_local = draw_polygon(
                np.round(corners[:, 1] - min_r).astype(int),
                np.round(corners[:, 0] - min_c).astype(int),
                door_patch.shape,
            )
            door_patch[rr_local, cc_local] = True

            wall_patch = wall_mask_bool[min_r:max_r + 1, min_c:max_c + 1]
            overlap_mask = door_patch & wall_patch
            if not np.any(overlap_mask):
                continue

            overlap_coords = np.argwhere(overlap_mask)
            overlap_coords[:, 0] += min_r
            overlap_coords[:, 1] += min_c

            door_centroid = np.array([(min_c + max_c) / 2.0, (min_r + max_r) / 2.0])
            overlap_min_r, overlap_min_c = overlap_coords.min(axis=0)
            overlap_max_r, overlap_max_c = overlap_coords.max(axis=0)
            overlap_centroid = np.array([
                (overlap_min_c + overlap_max_c) / 2.0,
                (overlap_min_r + overlap_max_r) / 2.0,
            ])

            exit_vector = overlap_centroid - door_centroid
            norm = np.linalg.norm(exit_vector)
            if norm < 1e-6:
                continue
            exit_vector /= norm
            exit_vector = [round(exit_vector[0]), round(exit_vector[1])]

            is_vertical_exit = abs(exit_vector[1]) > abs(exit_vector[0])

            door_center_c = int(round(door_centroid[0]))
            door_center_r = int(round(door_centroid[1]))

            if is_vertical_exit:
                start_c = door_center_c
                start_r = min_r if exit_vector[1] < 0 else max_r
            else:
                start_r = door_center_r
                start_c = min_c if exit_vector[0] < 0 else max_c

            for i in range(1, probe_limit + 1):
                probe_c = int(start_c + exit_vector[0] * i)
                probe_r = int(start_r + exit_vector[1] * i)

                if not (0 <= probe_r < self.h and 0 <= probe_c < self.w):
                    break

                if free_space[probe_r, probe_c] == 1:
                    is_vertical_exit = abs(exit_vector[1]) > abs(exit_vector[0])
                    if is_vertical_exit:
                        pt1 = (min_c, probe_r if exit_vector[1] < 0 else max_r)
                        pt2 = (max_c, min_r if exit_vector[1] < 0 else probe_r)
                    else:
                        pt1 = (probe_c if exit_vector[0] < 0 else max_c, min_r)
                        pt2 = (min_c if exit_vector[0] < 0 else probe_c, max_r)

                    final_pt1 = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]))
                    final_pt2 = (max(pt1[0], pt2[0]), max(pt1[1], pt2[1]))

                    cv2.rectangle(
                        free_space,
                        final_pt1,
                        final_pt2,
                        color=1,
                        thickness=-1,
                    )
                    break

        free_space = cv2.morphologyEx(
            free_space,
            cv2.MORPH_OPEN,
            self._morph_open_kernel,
            iterations=self._morph_open_iterations,
        )

        free_space[door_mask_bool] = 1

        return free_space.astype(bool), door_mask_bool, room_mask.astype(bool), wall_mask.astype(bool)
    
    def _create_corridor_skeleton(self, mask, door_mask):
        """문을 포함한 자유 공간에서 스켈레톤을 추출해 통로 축을 얻는다."""
        # dist = distance_transform_edt(mask)
        # thresh = dist.max() * 0.1
        # with_door_mask = (dist>thresh)| door_mask
        # 좁은 복도 끝단에서 스켈레톤이 끊기는 문제를 막기 위해 한 번 skeletonize 한 뒤
        # 도어 마스크를 합쳐 다시 한 번 스켈레톤을 추출한다.
        skel = skeletonize(mask, method='zhang')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_skel = cv2.morphologyEx(skel.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        closed_skel = skeletonize(closed_skel.astype(bool), method='zhang')
        return closed_skel.astype(bool)

    def _remove_temporary_wall_edges(self) -> None:
        """임시 벽 카테고리로 태깅된 항목을 삭제해 후속 연산을 단순화한다."""
        tmp_categories = {'tmp_not_room_edge'}
        walls = self.objs.get('walls')
        if walls:
            filtered_walls = [
                wall
                for wall in walls
                if (wall.get('category') or '').lower() not in tmp_categories
            ]
            if len(filtered_walls) != len(walls):
                self.objs['walls'] = filtered_walls

        wall_segments = self.objs.get('wall_segments')
        if wall_segments:
            filtered_segments = [
                seg
                for seg in wall_segments
                if (seg.get('category') or '').lower() not in tmp_categories
            ]
            if len(filtered_segments) != len(wall_segments):
                self.objs['wall_segments'] = filtered_segments

    def _save_mask(self, mask: np.ndarray, filename: str, scale_to_uint8: bool = False) -> None:
        """마스크/스켈레톤 배열을 PNG로 저장해 중간 단계를 관찰한다."""
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.debug_dir / filename
        if scale_to_uint8:
            data = (mask.astype(np.uint8) * 255)
        else:
            data = (mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_path), data)

    def _save_final_visualization(self) -> None:
        """최종 그래프와 객체를 Matplotlib 시각화로 남겨 디버깅에 활용한다."""
        if not self.debug_dir:
            return

        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            visualizer = FloorPlanVisualizer(self.G, self.objs)
            output_path = self.debug_dir / EXPORT_IMAGE_FILENAME
            visualizer.show(
                path=None,
                output_path=output_path,
                show_window=False,
                annotate_connections=True,
                save_out=True,
            )
        except Exception as exc:
            print(f"경고: 최종 그래프 시각화를 저장하지 못했습니다: {exc}")

    def _astar_path(self, skeleton, start, end):
        """스켈레톤 격자에서 A*로 최단 경로 길이를 계산한다."""
        start, end = tuple(start), tuple(end)
        open_set = [(0, start)] # (f_score, node)
        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return g_score[end] # 경로 길이 반환

            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dy, current[1] + dx)
                
                if not (0 <= neighbor[0] < skeleton.shape[0] and 0 <= neighbor[1] < skeleton.shape[1]):
                    continue
                if not skeleton[neighbor[0], neighbor[1]]:
                    continue

                tentative_g_score = g_score[current] + np.hypot(dy, dx)
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
        return None # 경로 없음

    def _convert_skeleton_to_graph_optimized(self, skel, door_mask, room_mask, wall_mask):
        """ 키포인트 감지 및 A*를 이용한 정확한 엣지 가중치 계산 """
        self.corridor_endpoint_door_links = {}
        self.corridor_endpoint_room_links = {}
        self.door_endpoint_pixels = {}
        self.door_endpoint_rooms = {}
        self.newly_found_endpoint_coords = set()
        self.newly_found_corridor_nodes = set()
        self.door_to_corridor_nodes = defaultdict(list)

        def _new_endpoint_info() -> Dict[str, Set[int]]:
            return {'door_ids': set(), 'room_ids': set()}

        def _merge_endpoint_info(dest: Dict[str, Set[int]], src: Optional[Dict[str, Set[int]]]) -> None:
            if not src:
                return
            door_ids = src.get('door_ids')
            room_ids = src.get('room_ids')
            if door_ids:
                dest['door_ids'].update(int(val) for val in door_ids)
            if room_ids:
                dest['room_ids'].update(int(val) for val in room_ids)

        def _has_endpoint_links(info: Dict[str, Set[int]]) -> bool:
            return bool(info.get('door_ids') or info.get('room_ids'))

        room_polygons_raw: List[Tuple[int, Polygon]] = []
        for room in self.objs.get('rooms', []):
            poly = room.get('polygon')
            if isinstance(poly, Polygon) and not poly.is_empty:
                room_polygons_raw.append((room['id'], poly))
        prepared_rooms_for_midpoints = [
            (room_id, prep(poly))
            for room_id, poly in room_polygons_raw
        ]

        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbors = convolve2d(skel.astype(np.uint8), kernel, mode='same')
        interest_points = neighbors * skel

        structural_forbidden_mask = wall_mask.astype(bool) | room_mask.astype(bool)
        door_forbidden_mask = door_mask.astype(bool)

        # 도어 영역 위의 스켈레톤 교차점은 실제 복도 축과 방 사이의 연결부이므로
        # 제거 대상에서 제외한다. 그렇지 않으면 문 인접 복도에 노드가 생기지 않는다.
        junction_mask = (interest_points >= 13)
        junction_forbidden_mask = structural_forbidden_mask
        junction_points_to_remove = junction_mask & junction_forbidden_mask
        final_junction_mask = junction_mask & (~junction_points_to_remove)
        if self.debug_dir:
            debug_masks = {
                'junction_mask.png': junction_mask,
                'junction_forbidden_mask.png': junction_forbidden_mask,
                'junction_points_to_remove.png': junction_points_to_remove,
                'final_junction_mask.png': final_junction_mask,
            }
            for filename, mask in debug_masks.items():
                self._save_mask(mask, filename, scale_to_uint8=True)
        labeled_junctions, num_labels = label(final_junction_mask)
        if num_labels > 0:
            junction_coords_raw = center_of_mass(final_junction_mask, labeled_junctions, range(1, num_labels + 1))
            junction_coords_arr = np.asarray(junction_coords_raw, dtype=float)
            if junction_coords_arr.ndim == 1:
                junction_coords_arr = junction_coords_arr.reshape(1, -1)
            junction_coords = np.round(junction_coords_arr).astype(int)
        else:
            junction_coords = np.empty((0, 2), dtype=int)

        diagonal_segments = skeleton_to_segments(skel)
        diagonal_segment_points = []
        for seg in diagonal_segments:
            if len(seg) > 0:
                diagonal_segment_points.append(seg[0])  # 시작점
                diagonal_segment_points.append(seg[-1]) # 끝점
        if diagonal_segment_points:
            diagonal_segment_points_np = np.asarray(diagonal_segment_points, dtype=int)
            if diagonal_segment_points_np.ndim == 1:
                diagonal_segment_points_np = diagonal_segment_points_np.reshape(1, -1)
        else:
            diagonal_segment_points_np = np.empty((0, 2), dtype=int)
        # print(f"중복 제거 전 교차로 개수: {np.sum(final_junction_mask)}")
        # print(f"중복 제거 후 교차로 개수: {len(junction_coords)}")

        # junction_coords = np.argwhere(final_junction_mask)

        endpoint_mask = (interest_points == 11)
        original_endpoint_coords = np.argwhere(endpoint_mask)

        diagonal_combined_points = np.vstack((original_endpoint_coords, diagonal_segment_points_np))
        diagonal_unique_combined_points = np.unique(diagonal_combined_points, axis=0)
        # --- 2. 끝점을 '안전한 끝점'과 '후퇴 필요한 끝점'으로 분리 ---
        forbidden_mask = structural_forbidden_mask

        safe_endpoints = []
        retracing_needed_endpoints = []

        for r, c in diagonal_unique_combined_points:
            if structural_forbidden_mask[r, c]:
                retracing_needed_endpoints.append((r, c))
            elif door_forbidden_mask[r, c]:
                continue
            else:
                safe_endpoints.append((r, c))
        # --- 3. 후퇴가 필요한 끝점에 대해 후퇴 수행 ---
        newly_found_endpoints = []
        skel_for_tracing = skel.astype(bool).copy()

        for r_start, c_start in retracing_needed_endpoints:
            current_r, current_c = r_start, c_start

            encountered_door_mask = False

            for _ in range(np.sum(skel)):
                if door_forbidden_mask[current_r, current_c]:
                    encountered_door_mask = True
                    break
                if not forbidden_mask[current_r, current_c]:
                    newly_found_endpoints.append((current_r, current_c))
                    break
                skel_for_tracing[current_r, current_c] = False

                found_next = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        
                        next_r, next_c = current_r + dr, current_c + dc
                        
                        if not (0 <= next_r < self.h and 0 <= next_c < self.w):
                            continue
                        if skel_for_tracing[next_r, next_c]:
                            current_r, current_c = next_r, next_c
                            found_next = True
                            break
                    if found_next:
                        break
                if not found_next:
                    break
            if encountered_door_mask:
                continue
        if newly_found_endpoints:
            newly_found_endpoints_np = np.unique(np.array(newly_found_endpoints), axis=0)
        else:
            newly_found_endpoints_np = np.empty((0, 2), dtype=int)

        newly_found_coord_set: Set[Tuple[int, int]] = {
            (int(coord[0]), int(coord[1]))
            for coord in newly_found_endpoints_np.tolist()
        } if newly_found_endpoints_np.size > 0 else set()

        door_endpoint_map: Dict[Tuple[int, int], Dict[str, Set[int]]] = {}

        door_parallel_points: List[List[int]] = []
        door_parallel_info: Dict[Tuple[int, int], Dict[str, Set[int]]] = {}
        for door in self.objs.get('doors', []):
            door_id = door.get('id')
            if door_id is None:
                continue
            for info in door.get('parallel_edge_midpoints', []) or []:
                midpoint_rc = info.get('midpoint_rc')
                midpoint_xy = info.get('midpoint_xy')
                if not midpoint_rc or len(midpoint_rc) != 2:
                    continue
                row, col = int(midpoint_rc[0]), int(midpoint_rc[1])
                coord_tuple = (row, col)
                door_parallel_points.append([row, col])
                payload = door_parallel_info.setdefault(coord_tuple, _new_endpoint_info())
                payload['door_ids'].add(int(door_id))

                rooms_for_midpoint: Set[int] = set()
                if midpoint_xy and prepared_rooms_for_midpoints:
                    try:
                        midpoint_point = Point(float(midpoint_xy[0]), float(midpoint_xy[1]))
                    except (TypeError, ValueError):
                        midpoint_point = None
                    if midpoint_point is not None:
                        for room_id, prepared in prepared_rooms_for_midpoints:
                            if prepared.contains(midpoint_point) or prepared.touches(midpoint_point):
                                rooms_for_midpoint.add(int(room_id))

                if rooms_for_midpoint:
                    payload['room_ids'].update(rooms_for_midpoint)
                    info['room_ids'] = sorted(rooms_for_midpoint)
                    # print(f"Door {door_id} midpoint at {midpoint_rc} connects to rooms {info['room_ids']}")
                else:
                    info['room_ids'] = []

        if door_parallel_points:
            door_parallel_np = np.unique(np.array(door_parallel_points, dtype=int), axis=0)
            if newly_found_endpoints_np.size > 0:
                newly_found_endpoints_np = np.unique(
                    np.vstack([newly_found_endpoints_np, door_parallel_np]),
                    axis=0,
                )
            else:
                newly_found_endpoints_np = door_parallel_np

            parallel_coord_set = {
                (int(coord[0]), int(coord[1]))
                for coord in door_parallel_np.tolist()
            }
            if newly_found_coord_set:
                newly_found_coord_set |= parallel_coord_set
            else:
                newly_found_coord_set = parallel_coord_set

            for coord_tuple, payload in door_parallel_info.items():
                if coord_tuple not in parallel_coord_set:
                    continue
                dest_payload = door_endpoint_map.setdefault(coord_tuple, _new_endpoint_info())
                _merge_endpoint_info(dest_payload, payload)

        if safe_endpoints:
            safe_endpoints_np = np.array(safe_endpoints, dtype=int)
            safe_points_np = np.vstack([safe_endpoints_np, junction_coords])
        else:
            safe_points_np = junction_coords
        print(f"교차로 포함 전 안전한 키포인트 개수: {safe_points_np.shape[0]}", flush=True)

        if safe_points_np.size > 0:
            safe_points_np = merge_nearby_points(safe_points_np, threshold=3.0)
        else:
            dt = distance_transform_edt(skel.astype(bool))
            dt_threshold = max(2.0, float(dt.max()) * 0.05)
            dt_core_mask = (dt >= dt_threshold) & skel
            if np.any(dt_core_mask):
                core_coords = np.argwhere(dt_core_mask)
                target_count = max(1, int(np.ceil(core_coords.shape[0] / (25 * 25))))
                interval = max(1, core_coords.shape[0] // target_count)
                sampled = core_coords[::interval]
                safe_points_np = merge_nearby_points(sampled, threshold=5.0)
            else:
                safe_points_np = np.empty((0, 2), dtype=int)
        
        # --- 4. 장애물 내부에 위치한 키포인트 보정 및 제거 ---
        key_points_coords = safe_points_np
        skeleton_coords = np.argwhere(skel)
        if len(key_points_coords) == 0 and skeleton_coords.size > 0:
            key_points_coords = np.array([skeleton_coords[len(skeleton_coords) // 2]])

        if door_endpoint_map:
            door_pixel_map: Dict[Tuple[int, int], List[int]] = {}
            door_room_map: Dict[Tuple[int, int], List[int]] = {}
            for coord, payload in door_endpoint_map.items():
                coord_key = (int(coord[0]), int(coord[1]))
                door_ids_sorted = sorted({int(val) for val in payload.get('door_ids', set())}) if payload.get('door_ids') else []
                room_ids_sorted = sorted({int(val) for val in payload.get('room_ids', set())}) if payload.get('room_ids') else []
                if door_ids_sorted:
                    door_pixel_map[coord_key] = door_ids_sorted
                if room_ids_sorted:
                    door_room_map[coord_key] = room_ids_sorted
            self.door_endpoint_pixels = door_pixel_map
            self.door_endpoint_rooms = door_room_map
        else:
            self.door_endpoint_pixels = {}
            self.door_endpoint_rooms = {}
        self.newly_found_endpoint_coords = newly_found_coord_set

        node_map = {tuple(coord): f"corridor_{i}" for i, coord in enumerate(key_points_coords)}
        node_map_endpoints = {tuple(coord): f"door_endpoints_{i}" for i, coord in enumerate(newly_found_endpoints_np)}
        
        for coord, node_id in node_map.items():
            coord_tuple = (int(coord[0]), int(coord[1]))

            self.G.add_node(
                node_id,
                pos=(coord[1], coord[0]),
                type='corridor',
            )

            self.corridor_nodes_pos[node_id] = (coord[1], coord[0])
        for coord, node_id in node_map_endpoints.items():
            coord_tuple = (int(coord[0]), int(coord[1]))
            door_ids_list = self.door_endpoint_pixels.get(coord_tuple, [])
            room_ids_list = self.door_endpoint_rooms.get(coord_tuple, [])
            is_newly_found = coord_tuple in self.newly_found_endpoint_coords

            self.G.add_node(
                node_id,
                pos=(coord[1], coord[0]),
                type='door_endpoints',
                door_link_ids=door_ids_list,
                room_link_ids=room_ids_list,
                is_newly_found=is_newly_found,
            )

            if is_newly_found:
                self.newly_found_corridor_nodes.add(node_id)

            if door_ids_list:
                self.corridor_endpoint_door_links[node_id] = door_ids_list
                for door_id in door_ids_list:
                    self.door_to_corridor_nodes[door_id].append(node_id)

            if room_ids_list:
                self.corridor_endpoint_room_links[node_id] = room_ids_list

            self.corridor_nodes_pos[node_id] = (coord[1], coord[0])

        if len(key_points_coords) > 1:
            coords_list = [tuple(coord) for coord in key_points_coords.tolist()]
            coords_array = np.array(coords_list, dtype=float)
            tree = KDTree(coords_array)
            added_edges: Set[Tuple[int, int]] = set()

            def try_add_edge(i: int, j: int) -> bool:
                start_coord = coords_list[i]
                end_coord = coords_list[j]

                dist = self._astar_path(skel, np.array(start_coord), np.array(end_coord))
                if dist is None or dist == 0:
                    return False

                line_points = np.linspace(start_coord, end_coord, max(3, int(dist))).astype(int)
                for p in line_points[1:-1]:
                    pt = tuple(p)
                    if pt in node_map and pt not in (start_coord, end_coord):
                        return False

                start_pos = (start_coord[1], start_coord[0])
                end_pos = (end_coord[1], end_coord[0])
                if not self._is_line_clear(start_pos, end_pos):
                    return False

                self.G.add_edge(node_map[start_coord], node_map[end_coord], weight=dist)
                return True

            for i, start_coord in enumerate(coords_list):
                start_node = node_map[start_coord]
                current_degree = self.G.degree(start_node)

                neighbors_idx = tree.query_ball_point(coords_array[i], r=max(self.w, self.h))
                neighbors_idx = [idx for idx in neighbors_idx if idx != i]

                if neighbors_idx:
                    dists = np.linalg.norm(coords_array[neighbors_idx] - coords_array[i], axis=1)
                    neighbor_candidates = [idx for _, idx in sorted(zip(dists, neighbors_idx))]
                else:
                    distances_all = np.linalg.norm(coords_array - coords_array[i], axis=1)
                    neighbor_candidates = [idx for idx in np.argsort(distances_all) if idx != i]

                for j in neighbor_candidates:
                    edge_key = tuple(sorted((i, j)))
                    if edge_key in added_edges:
                        continue
                    if try_add_edge(i, j):
                        added_edges.add(edge_key)
                        current_degree += 1

                if current_degree == 0 and len(coords_list) > 1:
                    distances_all = np.linalg.norm(coords_array - coords_array[i], axis=1)
                    for j in np.argsort(distances_all):
                        if j == i:
                            continue
                        edge_key = tuple(sorted((i, j)))
                        if edge_key in added_edges:
                            continue
                        if try_add_edge(i, j):
                            added_edges.add(edge_key)
                            break

        # Remove corridor nodes that remain isolated before attaching other object nodes
        isolated_corridor_coords = [
            coord
            for coord, node_id in list(node_map.items())
            if self.G.has_node(node_id) and self.G.degree(node_id) == 0
        ]
        for coord in isolated_corridor_coords:
            node_id = node_map.pop(coord, None)
            if not node_id:
                continue
            if self.G.has_node(node_id):
                self.G.remove_node(node_id)
            self.corridor_nodes_pos.pop(node_id, None)

        mst_graph = nx.minimum_spanning_tree(self.G, weight='weight')
        self.G = mst_graph

        mst_node_coords = [coord for coord, node_id in node_map.items()]

        if mst_node_coords:
            mst_kdtree = KDTree(mst_node_coords)
            for endpoint_coord, endpoint_node_id in node_map_endpoints.items():
                
                distances, indices = mst_kdtree.query(endpoint_coord, k=len(mst_node_coords))

                # k=1인 경우 결과가 스칼라 값(np.int64 등)로 반환될 수 있으므로 1차원 배열로 변환
                distances = np.atleast_1d(distances)
                indices = np.atleast_1d(indices)

                candidates = []
                for idx in indices:
                    corridor_coord = mst_node_coords[int(idx)]
                    dist = self._astar_path(skel, np.array(endpoint_coord), np.array(corridor_coord))
                    if dist is not None and dist > 0:
                        candidates.append((dist, corridor_coord))
                        
                if not candidates:
                    continue

                sorted_candidates = sorted(candidates, key=lambda x: x[0])
                
                for dist, corridor_coord in sorted_candidates:
                    corridor_node_id = node_map[tuple(corridor_coord)]
                    start_pos = (endpoint_coord[1], endpoint_coord[0])
                    end_pos = (corridor_coord[1], corridor_coord[0])
                    
                    if self._is_line_clear(start_pos, end_pos):
                        self.G.add_edge(endpoint_node_id, corridor_node_id, weight=dist)
                        self.corridor_nodes_pos[endpoint_node_id] = start_pos
                        break
        if self.corridor_nodes_pos:
            self.corridor_kdtree = KDTree(list(self.corridor_nodes_pos.values()))

    def _is_line_clear(self, pos1, pos2):
        """문/방/계단/엘리베이터를 모두 고려해 선분이 막히는지 검사한다."""
        line = LineString([pos1, pos2])
        # R-tree와 같은 공간 인덱스를 사용하면 성능을 더 향상시킬 수 있음
        for key in ('rooms', 'walls', 'stairs', 'elevators'):
            for obj in self.objs.get(key, []):
                poly = obj.get('polygon')
                if poly is None or poly.is_empty:
                    continue
                if not poly.intersects(line):
                    continue
                if poly.touches(line):
                    # allow tangential contacts along room/wall boundaries
                    continue
                return False
        return True
    def _is_line_clear_without_stairs_elevators(self, pos1, pos2):
        """계단·엘리베이터를 무시한 채 방/벽 충돌 여부만 확인한다."""
        line = LineString([pos1, pos2])
        # R-tree와 같은 공간 인덱스를 사용하면 성능을 더 향상시킬 수 있음
        for key in ('rooms', 'walls'):
            for obj in self.objs.get(key, []):
                poly = obj.get('polygon')
                if poly is None or poly.is_empty:
                    continue
                if not poly.intersects(line):
                    continue
                if poly.touches(line):
                    # allow tangential contacts along room/wall boundaries
                    continue
                return False
        return True
    def _is_line_clear_without_rooms(self, pos1, pos2):
        """방을 제외한 장애물(벽/계단/엘리베이터)만 고려해 선분을 검사한다."""
        line = LineString([pos1, pos2])
        # R-tree와 같은 공간 인덱스를 사용하면 성능을 더 향상시킬 수 있음
        for key in ('walls', 'stairs', 'elevators'):
            for obj in self.objs.get(key, []):
                poly = obj.get('polygon')
                if poly is None or poly.is_empty:
                    continue
                if not poly.intersects(line):
                    continue
                if poly.touches(line):
                    # allow tangential contacts along room/wall boundaries
                    continue
                return False
        return True

    def _connect_objects_to_graph_optimized(self):
        """KDTree로 복도 노드를 찾고 문/방/계단/엘리베이터를 그래프에 연결한다."""
        if not self.corridor_nodes_pos: return

        corridor_node_ids = list(self.corridor_nodes_pos.keys())
        object_types = ['rooms', 'doors', 'stairs', 'elevators']

        door_nodes_info: List[Tuple[str, Dict[str, Any]]] = []

        for obj_type in object_types:
            for obj in self.objs.get(obj_type, []):
                node_id = f"{obj_type[:-1]}_{obj['id']}"
                centroid = obj['centroid']
                node_attrs = {
                    'pos': (centroid.x, centroid.y),
                    'type': obj_type[:-1],
                }

                if obj_type == 'doors':
                    node_attrs['door_id'] = obj['id']
                    node_attrs['connected_room_ids'] = obj.get('connected_room_ids', [])
                if obj_type == 'rooms':
                    node_attrs['rooms_id'] = obj['id']
                    node_attrs['connected_door_ids'] = obj.get('connected_door_ids', [])
                # if obj_type in ('rooms'):
                #     print(f"node {node_id}")
                self.G.add_node(node_id, **node_attrs)

                if obj_type == 'doors':
                    door_nodes_info.append((node_id, obj))
                    continue

                if obj_type == 'rooms':
                    # Room centroids now link via door boundary endpoints instead of directly to corridors
                    continue

                distances, indices = self.corridor_kdtree.query(
                    [centroid.x, centroid.y],
                    k=len(corridor_node_ids),
                )

                if not hasattr(indices, '__iter__'):
                    indices, distances = [indices], [distances]

                for dist, idx in zip(distances, indices):
                    candidate_node_id = corridor_node_ids[idx]
                    candidate_pos = self.corridor_nodes_pos[candidate_node_id]
                    if self._is_line_clear_without_stairs_elevators((centroid.x, centroid.y), candidate_pos):
                        self.G.add_edge(node_id, candidate_node_id, weight=dist)
                        break

        for node_id, obj in door_nodes_info:
            centroid = obj['centroid']
            door_pos = (float(centroid.x), float(centroid.y))
            door_pos_vec = np.array(door_pos, dtype=float)
            door_id = obj['id']
            connected_rooms = obj.get('connected_room_ids') or []
            corridor_targets = list(dict.fromkeys(self.door_to_corridor_nodes.get(door_id, [])))

            added_endpoint_edge = False
            for endpoint_node_id in corridor_targets:
                if not self.G.has_node(endpoint_node_id):
                    continue
                endpoint_pos = self.corridor_nodes_pos.get(endpoint_node_id)
                if endpoint_pos is None:
                    node_pos_attr = self.G.nodes[endpoint_node_id].get('pos')
                    if not node_pos_attr:
                        continue
                    endpoint_pos = (float(node_pos_attr[0]), float(node_pos_attr[1]))
                    self.corridor_nodes_pos[endpoint_node_id] = endpoint_pos
                if self.G.has_edge(node_id, endpoint_node_id):
                    continue
                endpoint_vec = np.array(endpoint_pos, dtype=float)
                weight = float(np.linalg.norm(door_pos_vec - endpoint_vec))
                self.G.add_edge(node_id, endpoint_node_id, weight=weight)
                added_endpoint_edge = True

            if added_endpoint_edge or connected_rooms:
                continue

            if len(corridor_node_ids) == 0:
                continue

            distances, indices = self.corridor_kdtree.query(
                [door_pos[0], door_pos[1]],
                k=min(5, len(corridor_node_ids)),
            )
            if not hasattr(indices, '__iter__'):
                indices, distances = [indices], [distances]

            for dist, idx in zip(distances, indices):
                candidate_id = corridor_node_ids[idx]
                candidate_pos = self.corridor_nodes_pos[candidate_id]
                if not self._is_line_clear(door_pos, candidate_pos):
                    continue
                if self.G.has_edge(node_id, candidate_id):
                    break
                self.G.add_edge(node_id, candidate_id, weight=dist)
                break

        if self.corridor_endpoint_room_links:
            for endpoint_node_id, room_ids in self.corridor_endpoint_room_links.items():
                if endpoint_node_id not in self.G:
                    continue
                endpoint_pos = self.corridor_nodes_pos.get(endpoint_node_id)
                if endpoint_pos is None:
                    node_pos_attr = self.G.nodes[endpoint_node_id].get('pos')
                    if not node_pos_attr:
                        continue
                    endpoint_pos = (float(node_pos_attr[0]), float(node_pos_attr[1]))
                endpoint_vec = np.array(endpoint_pos, dtype=float)
                # print(f"Processing door_endpoints node {endpoint_node_id} linking to rooms {room_ids}")
                for room_id in room_ids:
                    room_node_id = f"room_{room_id}"
                    if room_node_id not in self.G:
                        # print(f"  Skipping non-existent room node {room_node_id}")
                        continue
                    if self.G.has_edge(endpoint_node_id, room_node_id):
                        # print(f"  Edge already exists between {endpoint_node_id} and {room_node_id}, skipping")
                        continue
                    room_pos_attr = self.G.nodes[room_node_id].get('pos')
                    if not room_pos_attr:
                        # print(f"  Skipping room node {room_node_id} with no position")
                        continue
                    room_pos = (float(room_pos_attr[0]), float(room_pos_attr[1]))
                    room_vec = np.array(room_pos, dtype=float)
                    weight = float(np.linalg.norm(endpoint_vec - room_vec))
                    self.G.add_edge(endpoint_node_id, room_node_id, weight=weight)
                    # print(f"Connected door_endpoints node {endpoint_node_id} to room node {room_node_id} with weight {weight:.2f}")

        room_to_endpoint_nodes: Dict[int, List[str]] = defaultdict(list)
        for endpoint_node_id, room_ids in self.corridor_endpoint_room_links.items():
            if endpoint_node_id not in self.G:
                continue
            for room_id in room_ids:
                room_to_endpoint_nodes[int(room_id)].append(endpoint_node_id)

        for endpoint_nodes in room_to_endpoint_nodes.values():
            if len(endpoint_nodes) < 2:
                continue
            for i in range(len(endpoint_nodes)):
                node_a = endpoint_nodes[i]
                pos_a = self.corridor_nodes_pos.get(node_a)
                if pos_a is None:
                    pos_attr = self.G.nodes[node_a].get('pos')
                    if not pos_attr:
                        continue
                    pos_a = (float(pos_attr[0]), float(pos_attr[1]))
                    self.corridor_nodes_pos[node_a] = pos_a
                vec_a = np.array(pos_a, dtype=float)
                for j in range(i + 1, len(endpoint_nodes)):
                    node_b = endpoint_nodes[j]
                    if self.G.has_edge(node_a, node_b):
                        continue
                    pos_b = self.corridor_nodes_pos.get(node_b)
                    if pos_b is None:
                        pos_attr = self.G.nodes[node_b].get('pos')
                        if not pos_attr:
                            continue
                        pos_b = (float(pos_attr[0]), float(pos_attr[1]))
                        self.corridor_nodes_pos[node_b] = pos_b
                    if not self._is_line_clear_without_rooms(pos_a, pos_b):
                        continue
                    vec_b = np.array(pos_b, dtype=float)
                    weight = float(np.linalg.norm(vec_a - vec_b))
                    self.G.add_edge(node_a, node_b, weight=weight)


def generate_free_space_masks(
    parsed_objects,
    width: int,
    height: int,
    *,
    config: Optional[CorridorPipelineConfig] = None,
) -> FreeSpaceMaskBundle:
    """입력 객체를 바탕으로 자유 공간 마스크 묶음을 계산한다."""
    pipeline = CorridorPipeline(parsed_objects, width, height, config=config)
    return pipeline.generate_free_space_masks()


FloorPlanNavigationGraph = CorridorPipeline


def merge_nearby_points(points: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """주어진 좌표 배열에서 threshold 이하 거리에 있는 점들을 하나로 병합한다."""
    if points.size == 0:
        return points.astype(int)
    if len(points) == 1:
        return points.astype(int)

    coords = points.astype(float)
    tree = KDTree(coords)
    parents = np.arange(len(coords))

    def _find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def _union(a: int, b: int) -> None:
        root_a, root_b = _find(a), _find(b)
        if root_a == root_b:
            return
        parents[root_b] = root_a

    for idx, coord in enumerate(coords):
        neighbors = tree.query_ball_point(coord, threshold)
        for neighbor_idx in neighbors:
            if neighbor_idx <= idx:
                continue
            _union(idx, neighbor_idx)

    clusters: Dict[int, List[np.ndarray]] = defaultdict(list)
    for idx, coord in enumerate(coords):
        root = _find(idx)
        clusters[root].append(coord)

    merged_points = []
    for cluster_coords in clusters.values():
        averaged = np.rint(np.mean(cluster_coords, axis=0)).astype(int)
        merged_points.append(averaged)

    return np.unique(np.array(merged_points, dtype=int), axis=0)
