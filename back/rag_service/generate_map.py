
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

def generate_building_map(destination, output_path="building_map.png"):
    """기계공학과 건물 지도 생성"""

    building_layout = {
        "기계관 101호": (1, 1, "1층"),
        "기계관 203호": (2, 3, "2층"),
        "기계관 205호": (2, 5, "2층"),
        "기계관 301호": (3, 1, "3층"),
        "기계관 305호": (3, 5, "3층"),
        "기계관 412호": (4, 2, "4층"),
        "창의세미나실 A": (1, 8, "1층"),
        "대회의실": (2, 8, "2층"),
        "토론실 B": (3, 8, "3층"),
        "기계공학과 사무실": (1, 5, "1층")
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    floor_colors = {
        "1층": "#FFE6E6",
        "2층": "#E6F3FF",
        "3층": "#E6FFE6",
        "4층": "#FFFFE6"
    }

    for room, (floor, x, floor_name) in building_layout.items():
        color = floor_colors.get(floor_name, "#F0F0F0")

        if destination in room:
            color = "#FF6B6B"
            linewidth = 3
        else:
            linewidth = 1

        rect = patches.Rectangle((x-0.4, floor-0.4), 0.8, 0.8,
                               linewidth=linewidth, edgecolor='black',
                               facecolor=color)
        ax.add_patch(rect)

        ax.text(
            x,
            floor,
            room.split()[-1],
            ha="center",
            va="center",
            fontsize=8,
            weight="bold" if destination in room else "normal",
        )

    for floor_name, color in floor_colors.items():
        floor_num = int(floor_name[0])
        ax.text(
            0,
            floor_num,
            floor_name,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color),
        )

    if destination in building_layout:
        dest_floor, dest_x, dest_floor_name = building_layout[destination]

        ax.scatter(5, 1, c="green", s=100, marker="o", label="입구")
        ax.text(5, 0.7, "입구", ha="center", va="top", fontsize=10, weight="bold")

        if dest_floor_name != "1층":
            ax.annotate(
                "",
                xy=(9, dest_floor),
                xytext=(9, 1),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"),
            )
            ax.text(
                9.2,
                (dest_floor + 1) / 2,
                f"{dest_floor_name}\n계단",
                ha="left",
                va="center",
                fontsize=9,
            )

            ax.annotate(
                "",
                xy=(dest_x, dest_floor),
                xytext=(9, dest_floor),
                arrowprops=dict(arrowstyle="->", lw=2, color="blue"),
            )
        else:
            ax.annotate(
                "",
                xy=(dest_x, dest_floor),
                xytext=(5, 1),
                arrowprops=dict(arrowstyle="->", lw=2, color="blue"),
            )

        ax.scatter(dest_x, dest_floor, c="red", s=150, marker="*", label="목적지")

    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_title(f"기계공학과 건물 - {destination} 찾아가기", fontsize=16, weight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        destination = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "building_map.png"
        generate_building_map(destination, output_path)
        print(f"Map generated: {output_path}")
    else:
        print("Usage: python generate_map.py <destination> [output_path]")
