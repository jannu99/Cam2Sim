#!/usr/bin/env python3
import json
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw_car_rectangle(ax, cx, cy, yaw_deg, half_length=2.3, half_width=1.0, **kwargs):
    """
    Draw an oriented car rectangle centered at (cx, cy) with yaw in degrees.
    half_length, half_width are half-extents in 'meters' (same units as your coords).
    """
    yaw = math.radians(yaw_deg)

    # Local corners (car frame): +x forward, +y left (convention similar to CARLA)
    corners_local = np.array([
        [ half_length,  half_width],
        [ half_length, -half_width],
        [-half_length, -half_width],
        [-half_length,  half_width],
        [ half_length,  half_width],  # close polygon
    ])

    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s],
                    [s,  c]])

    corners_world = (rot @ corners_local.T).T
    corners_world[:, 0] += cx
    corners_world[:, 1] += cy

    ax.plot(corners_world[:, 0], corners_world[:, 1], **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Plot vehicle spawn positions from vehicle_data.json"
    )
    parser.add_argument(
        "--vehicle_json",
        default="maps/guerickestrae_alte_heide_munich_25_11_16/vehicle_data.json",
        help="Path to vehicle_data.json",
    )
    parser.add_argument(
        "--show_hero",
        action="store_true",
        help="Draw hero car (if present in JSON).",
    )
    parser.add_argument(
        "--max_spots",
        type=int,
        default=-1,
        help="Plot at most N parking spots (-1 = all).",
    )
    parser.add_argument(
        "--save",
        default="",
        help="If set, save figure to this path instead of only showing it.",
    )
    args = parser.parse_args()

    vehicle_json = Path(args.vehicle_json)
    if not vehicle_json.exists():
        raise FileNotFoundError(f"vehicle_data.json not found: {vehicle_json}")

    with vehicle_json.open("r") as f:
        data = json.load(f)

    spawn_positions = data.get("spawn_positions", [])
    hero_car = data.get("hero_car", None)

    if not spawn_positions:
        print("[WARN] No spawn_positions found in JSON.")
        return

    print(f"[INFO] Loaded {len(spawn_positions)} spawn segments from {vehicle_json}")

    # Limit number of spots if requested
    n_to_plot = len(spawn_positions) if args.max_spots < 0 else min(args.max_spots, len(spawn_positions))

    fig, ax = plt.subplots(figsize=(8, 8))

    xs_all = []
    ys_all = []

    for i, sp in enumerate(spawn_positions[:n_to_plot]):
        if "start" not in sp or "end" not in sp or "heading" not in sp:
            print(f"[WARN] spawn_positions[{i}] missing keys, skipping.")
            continue

        x0, y0, _ = sp["start"]
        x1, y1, _ = sp["end"]
        yaw = float(sp["heading"])
        side = sp.get("side", "unknown")

        # Collect for autoscaling later
        xs_all += [x0, x1]
        ys_all += [y0, y1]

        color = "tab:blue" if side == "right" else ("tab:orange" if side == "left" else "gray")

        # 1) Draw segment (parking line)
        ax.plot([x0, x1], [y0, y1], "-", color=color, linewidth=1.5, alpha=0.8)

        # 2) Draw car rectangle on midpoint
        mid_x = 0.5 * (x0 + x1)
        mid_y = 0.5 * (y0 + y1)

        draw_car_rectangle(
            ax,
            mid_x,
            mid_y,
            yaw_deg=yaw,
            half_length=2.3,
            half_width=1.0,
            color=color,
            linewidth=0.8,
        )

        # Optionally label first few to inspect
        if i < 10:
            ax.text(
                mid_x,
                mid_y,
                str(i),
                fontsize=8,
                ha="center",
                va="center",
                color="black",
            )

    # --- Draw hero car if requested ---
    if args.show_hero and hero_car is not None and "position" in hero_car and "heading" in hero_car:
        hx, hy, hz = hero_car["position"]
        hyaw = float(hero_car["heading"])

        xs_all.append(hx)
        ys_all.append(hy)

        # Hero marker
        ax.scatter([hx], [hy], marker="x", s=60, color="red", label="HERO pose")
        draw_car_rectangle(
            ax,
            hx,
            hy,
            yaw_deg=hyaw,
            half_length=2.3,
            half_width=1.0,
            color="red",
            linewidth=1.2,
        )
        ax.text(hx, hy, "HERO", color="red", fontsize=9, ha="left", va="bottom")
        print(f"[INFO] Hero car at ({hx:.2f}, {hy:.2f}), yaw={hyaw:.2f} deg")
    elif args.show_hero:
        print("[INFO] show_hero=True but hero_car not present or missing fields.")

    # --- Axes formatting ---
    if xs_all and ys_all:
        xmin, xmax = min(xs_all), max(xs_all)
        ymin, ymax = min(ys_all), max(ys_all)
        dx = xmax - xmin
        dy = ymax - ymin
        pad = 0.05 * max(dx, dy, 1.0)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (CARLA world units)")
    ax.set_ylabel("Y (CARLA world units)")
    ax.set_title("Vehicle spawn positions from vehicle_data.json")

    # Legend for sides
    handles = [
        plt.Line2D([0], [0], color="tab:blue", label="Right side"),
        plt.Line2D([0], [0], color="tab:orange", label="Left side"),
    ]
    if args.show_hero and hero_car is not None:
        handles.append(plt.Line2D([0], [0], color="red", label="Hero"))
    ax.legend(handles=handles, loc="best")

    if args.save:
        out_path = Path(args.save)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Figure saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
