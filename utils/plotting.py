import matplotlib
import matplotlib.pyplot as plt
from shapely import LineString
from shapely.geometry import Point
from matplotlib.widgets import Button
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from config import CLICK_DISTANCE_THRESHOLD, HERO_CAR_OFFSET_METERS, SPAWN_OFFSET_METERS, SPAWN_OFFSET_METERS_LEFT, SPAWN_OFFSET_METERS_RIGHT
from utils.debug import debug_hero_car_spawn, debug_hero_spawn_line_error, debug_spawn_line_distance, \
    debug_parking_area_created
from utils.map_data import generate_spawn_gdf, get_heading, latlon_to_carla, get_origin_lat_lon

hero_car_point, hero_car_heading, hero_car_display = None, None, None
selected_segments = []
click_points = []
all_spawns = []

matplotlib.use("TkAgg")

def close_event(event):
    plt.close()

def update_plot():
    ax.clear()
    
    # 1. Plot the base layers
    buildings.plot(ax=ax, color="lightgray")
    
    # --- VEHICLE DATA ---
    x_vehicles=[
        48.17541537, 48.17542563, 48.17539582, 48.17536592, 48.17530591, 48.17534776,
        48.17531037 ,48.17527863, 48.17533517, 48.17534232, 48.17536656, 48.17538138,
        48.17533185 ,48.17536004, 48.17527536, 48.1753022 , 48.17528241, 48.1753275,
        48.17526065 ,48.17525786, 48.17532335, 48.17514553, 48.17510287, 48.17497549,
        48.17492586 ,48.17494985, 48.17490047, 48.17492378, 48.1749216 , 48.17478934,
        48.17498228 ,48.17502563, 48.17506829, 48.17514266, 48.17520569, 48.1752803,
        48.17529853 ,48.17536087, 48.17535783, 48.17539085, 48.17540095, 48.1754051,
        48.17538525 ,48.17541378, 48.17540231, 48.1753997 , 48.17536298, 48.17537042,
        48.17537042 ,48.17536583, 48.17547446, 48.17549802, 48.17552143, 48.17557843,
        48.17572    ,48.17574933, 48.17579539, 48.17571643, 48.17571347, 48.17560418,
        48.17568355 ,48.1757439 , 48.1757066 , 48.1756427 , 48.175784  , 48.17562757,
        48.17580658 ,48.17567052, 48.17578039, 48.17567086, 48.1758451 , 48.17581399,
        48.17583507 ,48.17585729, 48.17575342, 48.17583414, 48.17592017, 48.17594306,
        48.1759189  ,48.17590112, 48.17596785, 48.17598611, 48.17600423, 48.17596763,
        48.17598438 ,48.17595054, 48.17594446, 48.17590099, 48.17608394, 48.17598275,
        48.17603535 ,48.17593758, 48.1760321 , 48.1760562 , 48.17592075, 48.17601235,
        48.1760604  ,48.17606079, 48.17599558, 48.17602524, 48.17617686, 48.1761265,
        48.17616735 ,48.17608856, 48.1762065 , 48.17615921, 48.17615329, 48.17628094,
        48.176307   ,48.17629864, 48.1763434 , 48.17626002, 48.17623147, 48.17631387,
        48.17634478 ,48.1762559 , 48.17639041, 48.17638788, 48.17644778, 48.17642359,
        48.17646424 ,48.17643075, 48.17640837, 48.1764773 , 48.17643416, 48.17638802,
        48.17662367 ,48.17660692, 48.17673285
        ]
    
    y_vheicles=[11.59564043, 11.59535726, 11.59543284, 11.59554422, 11.59568727, 11.59580022,
        11.59566228, 11.59571461, 11.59577522, 11.59581563, 11.59575186, 11.59571473,
        11.59584448, 11.59576533, 11.59597213, 11.59590394, 11.59594116, 11.59585365,
        11.59600986, 11.5960339 , 11.59588041, 11.5961269 , 11.59625818, 11.5965867,
        11.59672496, 11.59665748, 11.5967372 , 11.59712513, 11.59714274, 11.59697714,
        11.59709396, 11.59715788, 11.59720176, 11.59725148, 11.59733009, 11.59737279,
        11.59743917, 11.59756445, 11.59746212, 11.59813141, 11.5980917 , 11.59799571,
        11.59815788, 11.5978685 , 11.59796526, 11.59803773, 11.59816822, 11.59821249,
        11.59812752, 11.59824696, 11.59826646, 11.59841462, 11.59833001, 11.59838426,
        11.59854088, 11.5985501 , 11.59849407, 11.59839932, 11.59841357, 11.59839834,
        11.59835649, 11.59856612, 11.59851274, 11.59831212, 11.59857955, 11.59841469,
        11.59861468, 11.59848104, 11.59845941, 11.59833664, 11.59866425, 11.59859763,
        11.59853586, 11.59868662, 11.59843271, 11.59850909, 11.59862786, 11.59865634,
        11.59859243, 11.59869145, 11.5987654 , 11.59876506, 11.59867257, 11.59863678,
        11.59865794, 11.59873707, 11.59861431, 11.59856994, 11.59877938, 11.59880057,
        11.59869789, 11.59857811, 11.59883216, 11.59873817, 11.5987101 , 11.59866523,
        11.59887173, 11.59882509, 11.59882245, 11.59867891, 11.59885238, 11.59879226,
        11.59895325, 11.59873888, 11.59885005, 11.59880412, 11.59890391, 11.59906965,
        11.59893639, 11.59895367, 11.59899067, 11.59903822, 11.5989955 , 11.59905004,
        11.59901107, 11.59888511, 11.59906822, 11.59922127, 11.59907787, 11.59905533,
        11.59924179, 11.59921245, 11.59907175, 11.59926808, 11.59918396, 11.59913475,
        11.5992451,  11.59923005, 11.59922934]
    
    edges.plot(ax=ax, color="black", linewidth=1)
    
    if not spawn_gdf.empty:
        spawn_gdf.plot(ax=ax, color='red', linewidth=2)

    # 2. PLOT VEHICLES HERE (Outside the loop)
    # Note: I swapped x and y because 'x_vehicles' contains Latitude (48.x) 
    # and 'y_vehicles' contains Longitude (11.x). 
    # Matplotlib/GeoPandas expects (Lon, Lat).
    ax.scatter(y_vheicles, x_vehicles, color="green", zorder=5, label="Parked Vehicles")

    # 3. Handle Hero Car and Titles
    if hero_car_point is not None:
        ax.plot(hero_car_display.x, hero_car_display.y, 'o', color='blue', markersize=8, label="Hero-Car")
        fig.suptitle("Select Parking-Areas by clicking on the red parking areas.")
    else:
        fig.suptitle("Select a spawn point for the hero-car by clicking next to a street in the travel direction")
    
    # 4. Highlight Selected Segments
    for seg in selected_segments:
        ax.plot(*zip(*seg.coords), color="green", linewidth=4)
        # REMOVED: ax.scatter(...) was here. It caused duplication and dependency on selection.

    # 5. Final Formatting
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    legend_handles = [
        mpatches.Patch(color="lightgray", label="Buildings"),
        mlines.Line2D([], [], color="black", linewidth=2, label="Streets"),
        mlines.Line2D([], [], color="red", linewidth=2, label="Possible Parking Areas"),
        mlines.Line2D([], [], color="green", linewidth=4, label="Selected Parking Areas"),
        mlines.Line2D([], [], color="green", marker='o', linestyle='', label="Parked Vehicles"), # Added to legend
        mlines.Line2D([], [], color="blue", marker='o', linestyle='', markersize=8, label="Spawn Point"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.canvas.draw()

def create_plot(buildings_data, edges_data, address):
    global buildings, edges, hero_spawn_gdf, spawn_gdf, fig, ax, origin_lat, origin_lon
    buildings = buildings_data
    edges = edges_data

    hero_spawn_gdf = generate_spawn_gdf(edges, offset=HERO_CAR_OFFSET_METERS, offset_left=SPAWN_OFFSET_METERS_LEFT,  offset_right=SPAWN_OFFSET_METERS_RIGHT, override=True)
    spawn_gdf = generate_spawn_gdf(edges, offset=SPAWN_OFFSET_METERS, offset_left=SPAWN_OFFSET_METERS_LEFT,  offset_right=SPAWN_OFFSET_METERS_RIGHT,)

    print("spawn_gdf length:", len(spawn_gdf))
    print("spawn_gdf geometry head:", spawn_gdf.geometry.head())

    fig, ax = plt.subplots(figsize=(12, 12))
    origin_lat, origin_lon = get_origin_lat_lon(edges_data, address)
    fig.canvas.manager.set_window_title("Vehicle Spawn Position Selector")

    #button_ax = fig.add_axes([0.80, 0.01, 0.18, 0.06])  # [links, unten, Breite, Höhe]
    #close_button = Button(button_ax, "Close and Save", color="lightgray", hovercolor="gray")
    #close_button.on_clicked(close_event)

    update_plot()

def show_plot():
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.axis("off")
    plt.show(block=True)

def on_click(event):
    global hero_car_point, hero_car_heading, hero_car_display

    if event.inaxes != ax:
        return

    click_pt = Point(event.xdata, event.ydata)

    if hero_car_point is None:
        nearest_line, side, _ = find_nearest_hero_spawn_line(click_pt)
        if nearest_line is None:
            debug_hero_spawn_line_error()
            return
        proj_dist = nearest_line.project(click_pt)
        proj_point = nearest_line.interpolate(proj_dist)
        hero_car_display = proj_point
        hero_car_point = latlon_to_carla(origin_lat, origin_lon, proj_point.y, proj_point.x)
        coords = list(nearest_line.coords)
        start_lat, start_lon = coords[0][1], coords[0][0]
        end_lat, end_lon = coords[-1][1], coords[-1][0]
        heading = get_heading(start_lat, start_lon, end_lat, end_lon)
        if side == "left":
            heading = (heading + 180) % 360
        hero_car_heading = heading
        update_plot()
        debug_hero_car_spawn(hero_car_point, hero_car_heading, side)
        return

    click_points.append(click_pt)

    if len(click_points) == 2:
        p1, p2 = click_points
        nearest_line, side, street_id = find_nearest_spawn_line(p1)

        if nearest_line is None:
            click_points.clear()
            return

        segment = extract_line_segment(nearest_line, p1, p2)
        selected_segments.append(segment)

        start_lat, start_lon = segment.coords[0][1], segment.coords[0][0]
        end_lat, end_lon = segment.coords[-1][1], segment.coords[-1][0]

        heading = get_heading(start_lat, start_lon, end_lat, end_lon)

        if side == "left":
            heading = (heading + 180) % 360

        if event.button != 1:
            heading = (heading + 90) % 360

        carla_start = latlon_to_carla(origin_lat, origin_lon, start_lat, start_lon)
        carla_end = latlon_to_carla(origin_lat, origin_lon, end_lat, end_lon)

        all_spawns.append({
            "side": side,
            "street_id": street_id,
            "mode": "parallel" if event.button == 1 else "perpendicular",
            "start": carla_start,
            "end": carla_end,
            "heading": heading
        })
        update_plot()
        click_points.clear()
        debug_parking_area_created(side, carla_start, carla_end, heading)


def find_nearest_spawn_line(click_pt):
    return find_nearest_line(click_pt, spawn_gdf)

def find_nearest_hero_spawn_line(click_pt):
    return find_nearest_line(click_pt, hero_spawn_gdf)

def find_nearest_line(click_pt, gdf):
    min_dist = float('inf')
    nearest_geom = None
    side = None
    for idx, geom in enumerate(gdf.geometry):
        dist = geom.distance(click_pt)
        if dist < min_dist:
            min_dist = dist
            nearest_geom = geom
            side = gdf.iloc[idx]['side']
            street_id = None
    if min_dist < CLICK_DISTANCE_THRESHOLD:
        return nearest_geom, side, street_id
    else:
        debug_spawn_line_distance(min_dist)
        return None, None, None

def substring_line(line: LineString, start_dist: float, end_dist: float) -> LineString:
    if start_dist > end_dist:
        start_dist, end_dist = end_dist, start_dist

    coords = list(line.coords)
    result = []

    dist_travelled = 0.0
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        seg = LineString([p1, p2])
        seg_length = seg.length

        if dist_travelled + seg_length < start_dist:
            dist_travelled += seg_length
            continue
        if dist_travelled > end_dist:
            break

        seg_start = max(start_dist, dist_travelled)
        seg_end = min(end_dist, dist_travelled + seg_length)

        ratio_start = (seg_start - dist_travelled) / seg_length
        ratio_end = (seg_end - dist_travelled) / seg_length

        interp_start = seg.interpolate(ratio_start, normalized=True)
        interp_end = seg.interpolate(ratio_end, normalized=True)

        result.append(interp_start.coords[0])
        result.append(interp_end.coords[0])

        dist_travelled += seg_length

    return LineString(result)

def extract_line_segment(line, p1, p2):
    d1 = line.project(p1)
    d2 = line.project(p2)
    return substring_line(line, d1, d2)

def get_output(dist):
    if hero_car_point is None or not all_spawns:
        return None
    return {
        "offset": {
            "x": None,
            "y": None,
            "heading": None
        },
        "dist": dist,
        "hero_car": {"position": hero_car_point,
                     "heading": hero_car_heading} if hero_car_point else None,
        "spawn_positions": all_spawns,
    }