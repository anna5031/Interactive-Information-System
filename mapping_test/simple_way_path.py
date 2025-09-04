import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict, deque

plt.rcParams['font.family'] = 'AppleGothic'

# 1. 노드 정의
nodes = {
  "N1": {"coord": (220, 580), "info": "B2 키오스크", "floor": "B2"},
  "N2": {"coord": (372, 520), "info": "B2 에스컬레이터 아래", "floor": "B2"},
  "N3": {"coord": (603, 410), "info": "B2 에스컬레이터 아래", "floor": "B2"},
  "N4": {"coord": (290, 465), "info": "B2 에스컬레이터 중간", "floor": "B2"},
  "N5": {"coord": (520, 369), "info": "B2 에스컬레이터 동측", "floor": "B2"},

  "N6": {"coord": (442, 337), "info": "B1 대합실 서쪽", "floor": "B1"},
  "N7": {"coord": (534, 305), "info": "B1 대합실 중앙 서쪽", "floor": "B1"},
  "N8": {"coord": (353, 296), "info": "B1 중앙 대합실", "floor": "B1"},
  "N9": {"coord": (442, 264), "info": "B1 중앙 에스컬레이터", "floor": "B1"},
  "N10": {"coord": (600, 256), "info": "연결 노드", "floor": "B1"},
  "N11": {"coord": (468, 218), "info": "B1 대합실 중앙 동쪽", "floor": "B1"},
  "N12": {"coord": (530, 192), "info": "B1 동쪽 통로", "floor": "B1"},
  "N13": {"coord": (866, 324), "info": "B1 출구4 최종", "floor": "B1"},
  "N14": {"coord": (797, 348), "info": "B1 출구4 중간", "floor": "B1"}
}

# 2. 간선 정의
edges = [
  ("N1", "N2"),
  ("N2", "N3"),
  ("N2", "N6"),
  ("N4", "N8"),
  ("N3", "N7"),
  ("N5", "N9"),
  ("N10", "N6"),
  ("N10", "N7"),
  ("N10", "N8"),
  ("N10", "N9"),
  ("N10", "N11"),
  ("N10", "N12"),
  ("N10", "N13"),
  ("N10", "N14")
]

# 3. 그래프 생성
graph = defaultdict(list)
for a, b in edges:
  graph[a].append(b)
  graph[b].append(a)


# 4. 경로 탐색 함수 (BFS)
def bfs_path(start, goal):
  visited = set()
  queue = deque([(start, [start])])
  while queue:
    current, path = queue.popleft()
    if current == goal:
      return path
    visited.add(current)
    for neighbor in graph[current]:
      if neighbor not in visited:
        queue.append((neighbor, path + [neighbor]))
  return None


# 5. 이미지 로드
image_path = "./test.png"
img = Image.open(image_path)
width, height = img.size

# 6. 시작/도착 노드 설정 (수동 입력)
start_node = "N1"
goal_node = "N13"
path = bfs_path(start_node, goal_node)

# 7. 시각화
plt.figure(figsize=(width / 100, height / 100))
plt.imshow(img)

# 노드 출력
for node_id, data in nodes.items():
  x, y = data["coord"]
  plt.plot(x, y, 'ro')
  plt.text(x + 5, y - 5, node_id, fontsize=8, color='blue')

# 기본 간선
for n1, n2 in edges:
  x1, y1 = nodes[n1]["coord"]
  x2, y2 = nodes[n2]["coord"]
  plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1)

# 경로 시각화 (파란 실선)
if path:
  for i in range(len(path) - 1):
    x1, y1 = nodes[path[i]]["coord"]
    x2, y2 = nodes[path[i + 1]]["coord"]
    plt.plot([x1, x2], [y1, y2], 'b-', linewidth=3, alpha=0.8)
  print("경로:", " → ".join(path))
else:
  print("경로를 찾을 수 없습니다.")

plt.title(f"{start_node} → {goal_node} 경로 탐색")
plt.axis("on")
plt.xlim(0, width)
plt.ylim(height, 0)
plt.show()
