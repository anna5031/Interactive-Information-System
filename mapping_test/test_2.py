import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 노드 정의
nodes = {
    "B2_platform_up": {"pos": (10, 10, -2), "type": "platform"},
    "B2_platform_down": {"pos": (15, 10, -2), "type": "platform"},
    "B2_escalator": {"pos": (12, 12, -2), "type": "escalator"},
    "B2_stairs_1": {"pos": (9, 14, -2), "type": "stairs"},

    "B1_concourse": {"pos": (12, 12, -1), "type": "concourse"},
    "B1_gate": {"pos": (12, 10, -1), "type": "gate"},
    "B1_escalator": {"pos": (12, 12, -1), "type": "escalator"},
    "B1_stairs_1": {"pos": (9, 14, -1), "type": "stairs"},

    "B1_exit_1": {"pos": (8, 18, 0), "type": "exit"},
    "B1_exit_2": {"pos": (12, 20, 0), "type": "exit"},
    "B1_exit_3": {"pos": (16, 18, 0), "type": "exit"},
    "B1_exit_4": {"pos": (17, 14, 0), "type": "exit"},
}

# 엣지 정의 (노드 연결)
edges = {
    ("B2_platform_up", "B2_escalator"),
    ("B2_platform_down", "B2_escalator"),
    ("B2_escalator", "B1_escalator"),
    ("B2_stairs_1", "B1_stairs_1"),

    ("B1_escalator", "B1_concourse"),
    ("B1_concourse", "B1_gate"),
    ("B1_stairs_1", "B1_concourse"),

    ("B1_concourse", "B1_exit_1"),
    ("B1_concourse", "B1_exit_2"),
    ("B1_concourse", "B1_exit_3"),
    ("B1_concourse", "B1_exit_4"),
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