import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw_car_rectangle(ax, cx, cy, yaw_deg, half_length=2.3, half_width=1.0, **kwargs):
    """
    Draw an oriented car rectangle centered at (cx, cy) with yaw in degrees.
    (Funzione disabilitata: non viene più usata per semplificare il plot come richiesto)
    """
    pass  # Disabilitata per semplificare il plot


def main():
    parser = argparse.ArgumentParser(
        description="Plot vehicle spawn positions from vehicle_data.json"
    )
    parser.add_argument(
        "--vehicle_json",
        default="maps/guerickestrae_alte_heide_munich_25_11_18/vehicle_data.json",
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

    fig, ax = plt.subplots(figsize=(10, 10))

    xs_all = []
    ys_all = []

    for i, sp in enumerate(spawn_positions[:n_to_plot]):
        if "start" not in sp or "end" not in sp or "heading" not in sp:
            print(f"[WARN] spawn_positions[{i}] missing keys, skipping.")
            continue

        # cluster_id (può arrivare come float da JSON → cast a int)
        car_id_raw = sp.get("cluster_id", i)
        if isinstance(car_id_raw, (float, np.floating)):
            car_id = int(car_id_raw)
        else:
            car_id = car_id_raw

        x0, y0, _ = sp["start"]
        # x1, y1, _ = sp["end"]  # non ci serve più per il centro

        side = sp.get("side", "unknown")
        mode = sp.get("mode", "parallel")

        # 🔴 QUI LA DIFFERENZA IMPORTANTE:
        # con la nuova build_spawn_positions_from_centroids
        # la macchina viene spawnata esattamente in 'start'
        cx = float(x0)
        cy = float(y0)

        xs_all.append(cx)
        ys_all.append(cy)

        # Colori: rosso = left, blu = right (come prima)
        if side == "right":
            color = "tab:blue"
        elif side == "left":
            color = "tab:red"
        else:
            color = "gray"

        # Marker diverso per parallel / perpendicular (opzionale ma utile)
        if mode == "perpendicular":
            marker = "s"  # square
        else:
            marker = "o"  # circle

        # Punto centrale del veicolo (posizione dove CARLA prova a spawnàre)
        ax.plot(
            [cx],
            [cy],
            marker=marker,
            markersize=6,
            linestyle='',
            color=color,
            zorder=5,
            alpha=0.8
        )

        # Etichetta con l'ID del cluster sopra il punto
        ax.text(
            cx,
            cy + 0.5,
            str(car_id),
            fontsize=9,
            ha="center",
            va="bottom",
            color="black",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
            zorder=10
        )

    # --- Hero car (rimane com'era) ---
    if args.show_hero and hero_car is not None and "position" in hero_car and "heading" in hero_car:
        hx, hy, hz = hero_car["position"]
        hyaw = float(hero_car["heading"])

        xs_all.append(hx)
        ys_all.append(hy)

        draw_car_rectangle(
            ax,
            hx,
            hy,
            yaw_deg=hyaw,
            half_length=2.3,
            half_width=1.0,
            color="red",
            linewidth=1.5,
            linestyle='-'
        )
        ax.text(hx, hy, "HERO", color="red", fontsize=9, ha="left", va="bottom", zorder=10)
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
    ax.set_title("Vehicle Spawn Positions (Simplified Point View)")

    # Legend per i lati + mode
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', markersize=8, label="Left side (parallel)"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=8, label="Right side (parallel)"),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:red', markersize=8, label="Left side (perp)"),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:blue', markersize=8, label="Right side (perp)"),
    ]
    if args.show_hero and hero_car is not None:
        handles.append(plt.Line2D([0], [0], color="red", linewidth=1.5, linestyle='-', label="Hero Car"))
    ax.legend(handles=handles, loc="best")

    if args.save:
        out_path = Path(args.save)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Figure saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()