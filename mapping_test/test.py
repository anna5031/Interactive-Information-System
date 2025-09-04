import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 노드 정의
nodes = {
    "B5_platform_1": {"pos": (10, 20, -4), "type": "platform"},
    "B5_platform_2": {"pos": (15, 20, -4), "type": "platform"},
    "B4_escalator_1": {"pos": (10, 20, -3), "type": "escalator"},
    "B4_escalator_2": {"pos": (15, 20, -3), "type": "escalator"},
    "B3_concourse": {"pos": (12, 20, -2), "type": "concourse"},
    "B3_ticket_office": {"pos": (13, 22, -2), "type": "ticket_office"},
    "B3_exit_gate": {"pos": (14, 18, -2), "type": "gate"},
    "B1_exit_1": {"pos": (12, 25, 0), "type": "exit", "label": "역전시장 방면"},
    "B1_exit_2": {"pos": (15, 27, 0), "type": "exit", "label": "중앙시장 방면"},
    "B1_exit_3": {"pos": (18, 25, 0), "type": "exit", "label": "대전역 광장 방면"},
    "B1_exit_4": {"pos": (18, 18, 0), "type": "exit", "label": "KTX 방면"},
    "B1_exit_5": {"pos": (11, 15, 0), "type": "exit", "label": "동서관통로 방면"},
}

# 엣지 정의 (노드 연결)
edges = {
    ("B5_platform_1", "B4_escalator_1"),
    ("B5_platform_2", "B4_escalator_2"),
    ("B4_escalator_1", "B3_concourse"),
    ("B4_escalator_2", "B3_concourse"),
    ("B3_concourse", "B3_ticket_office"),
    ("B3_concourse", "B3_exit_gate"),
    ("B3_exit_gate", "B1_exit_1"),
    ("B3_exit_gate", "B1_exit_2"),
    ("B3_exit_gate", "B1_exit_3"),
    ("B3_exit_gate", "B1_exit_4"),
    ("B3_exit_gate", "B1_exit_5"),
}

# 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D 지하철 층별 구조 시각화")

# 노드 표시
for node_id, data in nodes.items():
    x, y, z = data["pos"]
    ax.scatter(x, y, z, s=50)
    ax.text(x, y, z, node_id, size=8)

# 엣지(선 연결) 표시
for n1, n2 in edges:
    x1, y1, z1 = nodes[n1]["pos"]
    x2, y2, z2 = nodes[n2]["pos"]
    ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray')

# 축 설정
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z (층)")
ax.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.show()