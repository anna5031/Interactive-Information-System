import numpy as np
import plotly.graph_objects as go

# 1. 격자 크기 (길이방향 X, 통로방향 Y)
grid_shape = (15, 40)  # Y=15 (위아래 깊이), X=40 (좌우 길이방향)

# 2. 지하 1층 (B1) 대합실 및 출구
b1_floor = np.zeros(grid_shape, dtype=int)
b1_floor[5:10, 5:35] = 1       # 대합실 중앙
b1_floor[6:9, 2:6] = 1         # 출구1
b1_floor[6:9, 35:38] = 1       # 출구3
b1_floor[4:6, 15:25] = 1       # 출구2
b1_floor[9:11, 33:38] = 1      # 출구4

# 3. 지하 2층 (B2) 승강장 보행 가능 영역
b2_floor = np.zeros(grid_shape, dtype=int)

# 좌우 승강장 전장 (길게)
b2_floor[3:6, 5:35] = 1        # 위쪽 승강장 (왼쪽)
b2_floor[9:12, 5:35] = 1       # 아래쪽 승강장 (오른쪽)

# 중앙 선로 제거 (중간 y=6~9)
b2_floor[6:9, 10:30] = 0       # 선로 영역 (보행 불가)

# 4. 좌표 추출
b1_points = np.argwhere(b1_floor == 1)
b2_points = np.argwhere(b2_floor == 1)

b1_x = b1_points[:, 1]
b1_y = b1_points[:, 0]
b1_z = np.zeros_like(b1_x)

b2_x = b2_points[:, 1]
b2_y = b2_points[:, 0]
b2_z = np.full_like(b2_x, -1)

# 5. Plotly 시각화
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=b1_x, y=b1_y, z=b1_z,
    mode='markers',
    marker=dict(size=3, color='blue'),
    name='B1 (지하1층)'
))

fig.add_trace(go.Scatter3d(
    x=b2_x, y=b2_y, z=b2_z,
    mode='markers',
    marker=dict(size=3, color='red'),
    name='B2 (지하2층, 선로 제거됨)'
))

fig.update_layout(
    title="지하철 3D 격자 - 실제 구조 기반 (X=길, Y=통로)",
    scene=dict(
        xaxis_title='X (길이방향)',
        yaxis_title='Y (플랫폼 깊이)',
        zaxis_title='Z (층)',
        yaxis=dict(autorange="reversed")  # 시각적 일치 위해 Y 반전
    ),
    legend=dict(x=0, y=1),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()