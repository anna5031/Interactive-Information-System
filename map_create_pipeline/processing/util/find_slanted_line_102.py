import numpy as np
from skimage.morphology import skeletonize
from skimage import io
from scipy.ndimage import convolve, distance_transform_edt
import matplotlib.pyplot as plt
import random
import sys

def _calculate_slope_discrete(p1, p2, dir=False):
    if len(p1) != 2 or len(p2) != 2:
        print(f"Error: p1={p1}, p2={p2}")
        raise ValueError(f"Points must be 2D coordinates")
    
    r1, c1 = int(p1[0]), int(p1[1])
    r2, c2 = int(p2[0]), int(p2[1])
    
    dx = c2 - c1
    dy = r2 - r1
    
    if dx == 0:
        if dir:
            return 'v', 'v'
        return 'v'  # 수직선
    elif dy == 0:
        if dir:
            return 'h', 'h'
        return 'h'  # 수평선
    elif abs(dy) == abs(dx):
        if dir:
            direction='r' if dx > 0 else 'l'
            direction += 'd' if dy > 0 else 'u'
            return dy / dx, direction
        return dy / dx  # 1.0 또는 -1.0
    else:
        if dir:
            direction='r' if dx > 0 else 'l'
            direction += 'd' if dy > 0 else 'u'
            return float(dy) / float(dx), direction
        return float(dy) / float(dx)

def _get_neighbors(p, image, visited=None):
    """특정 픽셀의 8-이웃 중 스켈레톤 픽셀을 찾음."""
    rows, cols = image.shape
    r, c = int(p[0]), int(p[1])
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            nr, nc = r + i, c + j
            if 0 <= nr < rows and 0 <= nc < cols and image[nr, nc]:
                if visited is None or not visited[nr, nc]:
                    neighbors.append((nr, nc))
    return neighbors

def trace_simple_path(start_point, skeleton, visited):
    """
    단순 경로 추적
    """
    path = [tuple(start_point)]
    current = tuple(start_point)
    visited[current] = True
    
    while True:
        neighbors = _get_neighbors(current, skeleton, visited)
        
        if len(neighbors) != 1:  # 끝점이거나 분기점
            break
        next_point = neighbors[0]
        visited[next_point] = True
        path.append(next_point)
        current = next_point
    return path
# def split_path_by_slope(path, threshold, skeleton_image, path_index):
def split_path_by_slope(path, threshold):
    if len(path) < 2:
        return []

    # --- 시각화 설정 시작 ---
    # plt.figure(figsize=(10, 8))
    # plt.imshow(skeleton_image, cmap='gray', alpha=0.2)
    # path_rows, path_cols = zip(*path)
    # plt.plot(path_cols, path_rows, 'b--', label=f'Input Path {path_index}', alpha=0.7, zorder=1)
    # plt.title(f'Visualizing Path Splitting (Path #{path_index})')
    # --- 시각화 설정 끝 ---

    segments = []
    current_segment = [path[0]]
    dir_list = ['ru', 'rd', 'lu', 'ld']
    current_segment_dir = None
    current_sgement_num = 1
    i = 1
    while i < len(path):
        current_segment.append(path[i])
        
        if len(current_segment) >= 3:
            p_a = current_segment[-3]
            p_b = current_segment[-2] 
            p_c = current_segment[-1]
            
            if len(current_segment) == 3:
                slope_ab, direction_ab = _calculate_slope_discrete(p_a, p_b, True)
                slope_bc, direction_bc = _calculate_slope_discrete(p_b, p_c, True)
                if direction_ab not in dir_list:
                    current_segment = [p_b, p_c]
                    current_sgement_num = 1
                    i += 1
                    continue
                else:
                    current_segment_dir = direction_ab
            else:
                slope_ab = _calculate_slope_discrete(p_a, p_b)
                slope_bc, direction_bc = _calculate_slope_discrete(p_b, p_c, True)
            if slope_ab != slope_bc or slope_ab in {'v', 'h'} or slope_bc in {'v', 'h'} or direction_bc != current_segment_dir:
                can_reconnect = False
                for j in range(1, min(threshold + 1, len(path) - i)):
                    if i + j >= len(path): break
                    p_future1, p_future2 = path[i + j - 1], path[i + j]
                    future_slope = _calculate_slope_discrete(p_future1, p_future2)
                    
                    if future_slope == slope_ab:
                        current_segment.extend(path[i + 1 : i + j + 1])
                        i += j
                        can_reconnect = True
                        current_sgement_num+=1
                        break
                
                if not can_reconnect:
                    segment_to_add = current_segment[:-1]
                    if current_sgement_num > 4:
                        segments.append(segment_to_add)
                        # --- 시각화: 분할된 선분 표시 ---
                        # seg_color = (random.random(), random.random(), random.random())
                        # seg_rows, seg_cols = zip(*segment_to_add)
                        # plt.plot(seg_cols, seg_rows, color=seg_color, linewidth=3, marker='o', markersize=3,
                        #          label=f'Segment {len(segments)}', zorder=2)
                        # plt.plot(p_b[1], p_b[0], 'rx', markersize=12, markeredgewidth=2,
                        #          label=f'Split Point @ {p_b}' if len(segments)==1 else "", zorder=3)
                        # --- 시각화 끝 ---
                        
                    current_segment = [p_b, p_c]
                    current_sgement_num = 1
            else:
                current_sgement_num+=1
        i += 1
    
    if current_sgement_num >4 and len(current_segment) >= 2:
        segments.append(current_segment)
        # --- 시각화: 마지막 남은 선분 표시 ---
        # seg_color = (random.random(), random.random(), random.random())
        # seg_rows, seg_cols = zip(*current_segment)
        # plt.plot(seg_cols, seg_rows, color=seg_color, linewidth=3, marker='o', markersize=3, 
        #          label=f'Segment {len(segments)} (Final)', zorder=2)
        # --- 시각화 끝 ---

    # --- 최종 플롯 표시 ---
    # plt.legend(loc='best')
    # plt.axis('on')
    # # 이미지 좌표계에 맞게 y축 뒤집기
    # ax = plt.gca()
    # ax.invert_yaxis()
    # plt.show()
    # --- 최종 플롯 표시 끝 ---
    
    return segments

def skeleton_to_segments(skeleton, threshold=5):
    """
    스켈레톤을 직선 세그먼트로 분할함.
    """
    if not isinstance(skeleton, np.ndarray) or skeleton.ndim != 2:
        raise ValueError("입력은 2D NumPy 배열이어야 함.")
    
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_map = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    neighbor_map[~skeleton] = 0
    
    endpoints = list(map(tuple, np.argwhere(neighbor_map == 1)))
    all_skeleton_points = list(map(tuple, np.argwhere(skeleton)))
    
    visited = np.zeros_like(skeleton, dtype=bool)
    all_paths = []
    
    for endpoint in endpoints:
        if not visited[endpoint]:
            path = trace_simple_path(endpoint, skeleton, visited)
            if len(path) > 1: all_paths.append(path)
    
    for point in all_skeleton_points:
        if not visited[point]:
            path = trace_simple_path(point, skeleton, visited)
            if len(path) > 1: all_paths.append(path)
            
    all_segments = []
    # for i, path in enumerate(iterable=all_paths):
    for path in all_paths:
        if len(path) >= 2:
            # segments = split_path_by_slope(path, threshold, skeleton, i)
            segments = split_path_by_slope(path, threshold)
            all_segments.extend(segments)
    
    return all_segments


# def filter_diagonal_segments(segments, diagonal_threshold=0.6):
#     diagonal_segments = []
#     for seg in segments:
#         if len(seg) < 2: continue
        
#         start_point, end_point = seg[0], seg[-1]
#         dr = abs(end_point[0] - start_point[0])
#         dc = abs(end_point[1] - start_point[1])
        
#         if dr == 0 or dc == 0: continue
            
#         aspect_ratio = float(min(dr, dc)) / float(max(dr, dc))
        
#         if aspect_ratio >= diagonal_threshold:
#             diagonal_segments.append(seg)
            
#     return diagonal_segments

if __name__ == '__main__':
    image_path = './sample_debug/free_space_mask.png'
    
    try:
        image_gray = io.imread(image_path, as_gray=True)
        print(f"이미지 로드 성공: {image_gray.shape}")
    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없음. 경로를 확인하셈.")
        print("임시 샘플 이미지를 생성하여 실행함.")
        image_gray = np.zeros((100, 100))
        image_gray[20:30, 10:80] = 1 # 수평선
        image_gray[10:90, 45:55] = 1 # 수직선
        image_gray[10:60, 10:60] = np.diag(np.ones(50)) # 대각선
        image_gray[70:90, 20:40] = 1 # 사각형
        sys.exit(1)
    
    binary_image = image_gray > 0.5
    
    dist = distance_transform_edt(binary_image)
    thresh = dist.max() * 0.1
    skeleton = skeletonize(dist > thresh, method='zhang')
    
    print(f"스켈레톤화 완료: {np.sum(skeleton)} 픽셀")
    final_segments = skeleton_to_segments(skeleton, threshold=3)
    print(f"초기 선분 탐지: {len(final_segments)}개")
    
    print("\n모든 경로 분할이 완료되었음. 최종 결과 플롯을 표시함.")
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(skeleton, cmap='gray')
    plt.title('Skeleton from Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(skeleton, cmap='gray', alpha=0.3)
    for segment in final_segments:
        color = (random.random(), random.random(), random.random())
        rows, cols = zip(*segment)
        plt.plot(cols, rows, color=color, linewidth=2, marker='o', markersize=2)
    plt.title(f'Final Diagonal Segments ({len(final_segments)} found)')
    plt.axis('on')
    
    plt.subplot(1, 3, 3)
    plt.imshow(binary_image, cmap='gray', alpha=0.5)
    for segment in final_segments:
        color = (random.random(), random.random(), random.random())
        rows, cols = zip(*segment)
        plt.plot(cols, rows, color=color, linewidth=3, alpha=0.7)
    plt.title('Diagonal Segments on Original')
    plt.axis('on')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 최종 결과 요약 ===")
    print(f"총 {len(final_segments)}개 선분 탐지됨.")