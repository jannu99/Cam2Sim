#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import cv2
import numpy as np
import rospy
import torch
import shutil 
from ultralytics import YOLO

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import tf2_ros
from tf2_geometry_msgs import do_transform_point


def quat_to_yaw(q):
    """Quaternion -> yaw (rad)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class MonoYOLOSegClusterNode:
    """
    YOLOv8-Seg detection + clustering.
    - Real-time: Detects, Clusters, Tracks, Logs to CSV.
    - Post-Process: Generates video AND Segmentation Maps folder at shutdown.
    - Final Output: Detailed CSV with centroids, orientation, side, and color.
    """

    def __init__(self):
        rospy.init_node("mono_yolo_cluster_orient_seg", anonymous=True)
        self.bridge = CvBridge()

        # === [NEW] Post-Processing Outputs ===
        self.output_video_path = rospy.get_param("~output_video", "final_recolored_video.mp4")
        
        # Folder for final cleaned segmentation maps
        self.output_maps_dir = rospy.get_param("~output_maps_dir", "final_maps")
        if os.path.exists(self.output_maps_dir):
            shutil.rmtree(self.output_maps_dir)
        os.makedirs(self.output_maps_dir)

        # Temp folder for raw frames
        self.temp_dir = os.path.join(os.getcwd(), "temp_raw_frames")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        
        self.history_data = [] # Stores metadata for video/map generation
        self.frame_count = 0

        # === Parametri YOLO / camera ===
        self.image_topic   = rospy.get_param("~image_topic", "/gmsl_camera/front_narrow/image_raw")
        self.model_path    = rospy.get_param("~model", "yolov8n-seg.pt")   
        self.device        = rospy.get_param("~device", "cuda" if torch.cuda.is_available() else "cpu")
        self.conf          = rospy.get_param("~conf", 0.4)
        self.print_coords  = rospy.get_param("~print_coords", False)

        self.fx = rospy.get_param("~fx", 969.63)
        self.fy = rospy.get_param("~fy", 999.94)
        self.cx = rospy.get_param("~cx", 336.15)
        self.cy = rospy.get_param("~cy", 324.73)

        self.frame_camera  = rospy.get_param("~camera_frame",  "front_6mm")
        self.frame_target  = rospy.get_param("~velodyne_frame","velodyne32")
        self.world_frame   = rospy.get_param("~world_frame",   "world")

        # === Parametri clustering ===
        self.max_ego_dist      = rospy.get_param("~max_ego_dist", 50.0)   # metri
        self.cluster_dist      = rospy.get_param("~cluster_dist", 2.0)    # metri
        self.use_ema           = rospy.get_param("~use_ema", True)
        self.ema_alpha         = rospy.get_param("~ema_alpha", 0.2)
        self.min_cluster_count = rospy.get_param("~min_cluster_count", 1)

        # === Classificatore orientazione ===
        self.orient_enable   = rospy.get_param("~enable_orientation", True)
        self.cls_model_path  = rospy.get_param("~cls_model", "/home/davide/catkin_ws/src/lidar_test/scripts/runs/classify/train2/weights/best.pt")
        self.cls_imgsz       = rospy.get_param("~cls_imgsz", 224)
        self.show_orient_label = rospy.get_param("~show_orient_label", True)

        if self.orient_enable:
            rospy.loginfo(f"[INIT] Loading orientation classifier: {self.cls_model_path}")
            try:
                self.cls_model = YOLO(self.cls_model_path)
                self.cls_model.to(self.device)
                if self.device == "cuda":
                    try:
                        self.cls_model.model.half()
                    except Exception: pass
                self.cls_names = self.cls_model.names
            except Exception as e:
                rospy.logwarn(f"[INIT] Failed to load orientation classifier: {e}")
                self.cls_model = None; self.cls_names = {}
        else:
            self.cls_model = None; self.cls_names = {}

        # === Stato Odom (ego) ===
        self.odom_pose = None
        rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=5)

        # === YOLO DETECTION (SEGMENTATION) ===
        rospy.loginfo(f"[INIT] Loading YOLO SEGMENTATION model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        if self.device == "cuda":
            try:
                self.model.model.half()
            except Exception: pass
        
        # === TF ===
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # === Clusters ===
        self.clusters = []
        self.next_cluster_id = 1

        # === Publisher ===
        self.pub_centroids = rospy.Publisher("/cloud_centroids", MarkerArray, queue_size=1)
        self.pub_debug_img = rospy.Publisher("/yolo_debug_image", Image, queue_size=1)
        self.pub_analysis_masks = rospy.Publisher("/car_color_analysis_masks", Image, queue_size=1)
        self.pub_masks = rospy.Publisher("/car_colored_masks", Image, queue_size=1)

        # === Subscriber immagini ===
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)

        # === Logging ===
        self.log_path = rospy.get_param("~output_file", "/home/davide/Desktop/yolo_centroids_clusters_orient_cls.txt")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log_file = open(self.log_path, "w")
        self.log_file.write("# timestamp, cluster_id, conf, x_world, y_world, z_world, ego_x, ego_y, ego_yaw, rgb_color\n")
        
        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(f"[LOG] Writing to {self.log_path}")
        rospy.loginfo(f"[READY] Recording raw frames to {self.temp_dir}")

    # -------------- Callbacks --------------
    def odom_callback(self, msg: Odometry):
        self.odom_pose = msg.pose.pose

    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return

        # 1. Save Raw Frame immediately
        frame_filename = os.path.join(self.temp_dir, f"frame_{self.frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, cv_img)

        detections_in_frame = []

        debug_img = cv_img.copy()
        instance_masks_img = np.zeros_like(cv_img, dtype=np.uint8)
        analysis_masks_img = np.zeros_like(cv_img, dtype=np.uint8)

        # === Ego overlay ===
        ego_x = ego_y = ego_yaw = float("nan")
        if self.odom_pose is not None:
            pos = self.odom_pose.position; ori = self.odom_pose.orientation
            ego_x, ego_y, ego_yaw = pos.x, pos.y, quat_to_yaw(ori)
            txt = f"EGO x={ego_x:.2f}, y={ego_y:.2f}"
            cv2.putText(debug_img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # === YOLO ===
        results = self.model.predict(cv_img, conf=self.conf, device=self.device, verbose=False, retina_masks=True)
        
        if not results or len(results[0].boxes) == 0:
            self.history_data.append({'frame_id': self.frame_count, 'detections': []})
            self.frame_count += 1
            self.publish_debug_image(debug_img, msg)
            return
        
        r = results[0]
        boxes = r.boxes
        has_masks = (r.masks is not None)

        # === TF Preload ===
        tf_ok = False
        try:
            t_cam_to_velo = self.tf_buffer.lookup_transform(self.frame_target, self.frame_camera, rospy.Time(0), rospy.Duration(0.2))
            t_velo_to_world = self.tf_buffer.lookup_transform(self.world_frame, self.frame_target, rospy.Time(0), rospy.Duration(0.2))
            tf_ok = True
        except Exception: pass

        H, W, _ = cv_img.shape
        half_W = W / 2.0

        for i, b in enumerate(boxes):
            if int(b.cls) != 2: continue 

            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            conf = float(b.conf.cpu().numpy())
            
            x1_i = max(0, int(x1)); y1_i = max(0, int(y1))
            x2_i = min(W, int(x2)); y2_i = min(H, int(y2))
            bbox_h = (y2 - y1); bbox_w = (x2 - x1)

            if bbox_h < 40 or bbox_w < 5: continue

            # Border check
            border = 20
            is_touching_border = (x1 < border or y1 < border or x2 > (W - border) or y2 > (H - border))

            # ===== 1. SEGMENTATION & COLOR =====
            rgb_detected = None
            current_obj_mask = None 
            mask_for_analysis = None 
            contours = [] 
            
            img_crop = cv_img[y1_i:y2_i, x1_i:x2_i]

            if has_masks and img_crop.size > 0:
                raw_mask = r.masks[i].data[0].cpu().numpy()
                if raw_mask.shape[:2] != (H, W):
                    raw_mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                
                # Full mask
                current_obj_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                
                # Save Contour for Post-Processing
                cnts, _ = cv2.findContours(current_obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    contours = cnts[0] # Take largest contour

                mask_crop = current_obj_mask[y1_i:y2_i, x1_i:x2_i]

                # Cut Logic
                h_crop, w_crop = mask_crop.shape
                mask_for_analysis = mask_crop.copy() 
                split_y = int(h_crop * 0.4)
                if split_y < h_crop: mask_for_analysis[0:split_y, :] = 0
                if cv2.countNonZero(mask_for_analysis) < 10: mask_for_analysis = mask_crop

                rgb_detected = self._get_dominant_color_mode(img_crop, mask_for_analysis)
                
                full_size_analysis_mask = np.zeros((H, W), dtype=np.uint8)
                full_size_analysis_mask[y1_i:y2_i, x1_i:x2_i] = mask_for_analysis
                mask_for_analysis = full_size_analysis_mask

            # ===== 2. CONDITIONAL 3D MATH =====
            cx_pix = (x1 + x2) / 2.0; cy_pix = (y1 + y2) / 2.0
            est_z = 8.0 * (200.0 / bbox_h)
            est_x = (cx_pix - self.cx) * est_z / self.fx
            est_y = (cy_pix - self.cy) * est_z / self.fy

            px = py = pz = float("nan")
            valid_tf = False
            if tf_ok:
                try:
                    p_cam = PointStamped()
                    p_cam.header.frame_id = self.frame_camera; p_cam.header.stamp = msg.header.stamp
                    p_cam.point.x = est_x; p_cam.point.y = est_y; p_cam.point.z = est_z
                    p_w = do_transform_point(do_transform_point(p_cam, t_cam_to_velo), t_velo_to_world)
                    px, py, pz = p_w.point.x, p_w.point.y, p_w.point.z
                    valid_tf = True
                except Exception: pass

            cid = None
            orient_pred = None
            side_label = "unknown"

            if valid_tf and not (math.isnan(px) or math.isnan(py)):
                dist_ok = True
                if not math.isnan(ego_x) and not math.isnan(ego_y):
                    if math.hypot(px - ego_x, py - ego_y) > self.max_ego_dist: dist_ok = False
                
                if dist_ok:
                    side_label = "left" if cx_pix < half_W else "right"
                    if not is_touching_border and self.orient_enable and self.cls_model:
                            try:
                                r_cls = self.cls_model.predict(img_crop, imgsz=self.cls_imgsz, device=self.device, verbose=False)[0]
                                c_idx = int(r_cls.probs.top1)
                                nm = self.cls_names.get(c_idx, "")
                                orient_pred = 0 if nm=="parallel" else 1 if nm=="perpendicular" else c_idx
                            except: pass

                    cid = self._assign_to_cluster(px, py, pz, conf, (x1, y1, x2, y2), orient_pred, side_label, rgb_detected, is_border=is_touching_border)
                    
                    if cid is not None:
                        self._log_row(msg.header.stamp.to_sec(), cid, conf, px, py, pz, ego_x, ego_y, ego_yaw)

            # --- SAVE HISTORY DATA ---
            if len(contours) > 0:
                detections_in_frame.append({
                    'cid': cid,
                    'contour': contours, 
                    'rgb_instant': rgb_detected
                })

            # --- PAINT MASKS (Real-time Feedback) ---
            if current_obj_mask is not None:
                paint_bgr = (0, 255, 0)
                if cid is not None:
                    cluster_data = next((c for c in self.clusters if c["id"] == cid), None)
                    if cluster_data:
                        win_rgb = self._get_winning_color(cluster_data)
                        if win_rgb: paint_bgr = (win_rgb[2], win_rgb[1], win_rgb[0])
                        elif rgb_detected: paint_bgr = (rgb_detected[2], rgb_detected[1], rgb_detected[0])
                elif rgb_detected is not None:
                    paint_bgr = (rgb_detected[2], rgb_detected[1], rgb_detected[0])
                
                instance_masks_img[current_obj_mask == 255] = paint_bgr
                if mask_for_analysis is not None: analysis_masks_img[mask_for_analysis == 255] = paint_bgr

            # --- Debug Graphics ---
            lbl = f"ID {cid}" if cid else "BORDER"
            if rgb_detected: lbl += f" RGB{rgb_detected}"
            cv2.rectangle(debug_img, (x1_i, y1_i), (x2_i, y2_i), (0,255,0), 2)
            cv2.putText(debug_img, lbl, (x1_i, y1_i-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        self.history_data.append({'frame_id': self.frame_count, 'detections': detections_in_frame})
        self.frame_count += 1

        self._publish_markers(rospy.Time.now())
        self.publish_debug_image(debug_img, msg)
        self.publish_mask_image(instance_masks_img, msg)
        self._publish_analysis_image(analysis_masks_img, msg)

    # -------------- Clustering --------------
    def _assign_to_cluster(self, px, py, pz, conf, bbox, orient_pred, side_label, rgb_pred, is_border):
        best_idx = None
        best_dist = float("inf")

        for i, c in enumerate(self.clusters):
            d = math.hypot(px - c["x"], py - c["y"])
            if d < self.cluster_dist and d < best_dist:
                best_dist = d; best_idx = i

        if is_border:
            if best_idx is not None:
                c = self.clusters[best_idx]
                c["last_conf"] = conf
                if rgb_pred:
                    if "color_votes" not in c: c["color_votes"] = {}
                    c["color_votes"][rgb_pred] = c["color_votes"].get(rgb_pred, 0) + 1
                return c["id"]
            else:
                return None

        if best_idx is not None:
            c = self.clusters[best_idx]
            if self.use_ema:
                a = self.ema_alpha
                c["x"] = (1.0 - a) * c["x"] + a * px
                c["y"] = (1.0 - a) * c["y"] + a * py
                c["z"] = (1.0 - a) * c["z"] + a * pz
            else:
                n = c["count"]
                c["x"] = (c["x"] * n + px) / (n + 1)
                c["y"] = (c["y"] * n + py) / (n + 1)
                c["z"] = (c["z"] * n + pz) / (n + 1)

            c["count"] += 1
            if orient_pred is not None:
                if orient_pred == 0: c["orient_parallel"] = c.get("orient_parallel", 0) + 1
                elif orient_pred == 1: c["orient_perpendicular"] = c.get("orient_perpendicular", 0) + 1
            if side_label == "left": c["side_left"] = c.get("side_left", 0) + 1
            elif side_label == "right": c["side_right"] = c.get("side_right", 0) + 1
            if rgb_pred:
                if "color_votes" not in c: c["color_votes"] = {}
                c["color_votes"][rgb_pred] = c["color_votes"].get(rgb_pred, 0) + 1
            return c["id"]
        else:
            cid = self.next_cluster_id
            self.next_cluster_id += 1
            par_votes = 1 if (self.orient_enable and orient_pred == 0) else 0
            perp_votes = 1 if (self.orient_enable and orient_pred == 1) else 0
            side_left = 1 if side_label == "left" else 0
            side_right = 1 if side_label == "right" else 0
            color_votes = {}
            if rgb_pred: color_votes[rgb_pred] = 1

            new_c = {
                "id": cid, "x": px, "y": py, "z": pz, "count": 1, "last_conf": conf, "last_bbox": bbox,
                "orient_parallel": par_votes, "orient_perpendicular": perp_votes,
                "side_left": side_left, "side_right": side_right, "color_votes": color_votes 
            }
            self.clusters.append(new_c)
            return cid

    # -------------- Helpers --------------
    def _get_dominant_color_mode(self, img_crop, mask_crop):
        pixels_bgr = img_crop[mask_crop == 255]
        if len(pixels_bgr) < 10: return None
        step = 5
        quantized = (pixels_bgr // step) * step 
        unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
        sorted_indices = np.argsort(-counts)
        unique_colors = unique_colors[sorted_indices]
        counts = counts[sorted_indices]
        k = 5
        top_colors = unique_colors[:k]
        top_counts = counts[:k]
        total_weight = np.sum(top_counts)
        if total_weight == 0: return None
        weighted_sum = np.sum(top_colors * top_counts[:, np.newaxis], axis=0)
        final_bgr = weighted_sum / total_weight
        return (int(final_bgr[2]), int(final_bgr[1]), int(final_bgr[0]))

    def _get_winning_color(self, c):
        votes = c.get("color_votes", {})
        if not votes: return None
        return max(votes, key=votes.get)

    def _publish_analysis_image(self, cv_img, msg):
        try:
            out = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
            out.header = msg.header
            self.pub_analysis_masks.publish(out)
        except Exception: pass

    def publish_debug_image(self, cv_img, msg):
        try:
            out = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
            out.header = msg.header
            self.pub_debug_img.publish(out)
        except Exception: pass

    def publish_mask_image(self, cv_img, msg):
        try:
            out = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
            out.header = msg.header
            self.pub_masks.publish(out)
        except Exception: pass

    def _publish_markers(self, now):
        marker_array = MarkerArray()
        for c in self.clusters:
            if c["count"] < self.min_cluster_count: continue
            mk = Marker()
            mk.header.frame_id = self.world_frame; mk.header.stamp = now
            mk.ns = "yolo_car_clusters"; mk.id = c["id"]
            mk.type = Marker.SPHERE; mk.action = Marker.ADD
            mk.pose.position.x = float(c["x"]); mk.pose.position.y = float(c["y"]); mk.pose.position.z = float(c["z"])
            mk.pose.orientation.w = 1.0; mk.scale.x = mk.scale.y = mk.scale.z = 0.4
            
            win_rgb = self._get_winning_color(c)
            if win_rgb: mk.color = ColorRGBA(win_rgb[0]/255.0, win_rgb[1]/255.0, win_rgb[2]/255.0, 1.0)
            else: mk.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
            mk.lifetime = rospy.Duration(1.0)
            marker_array.markers.append(mk)

            txt = Marker()
            txt.header.frame_id = self.world_frame; txt.header.stamp = now
            txt.ns = "yolo_car_clusters_id"; txt.id = 100000 + c["id"]
            txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
            txt.pose.position.x = mk.pose.position.x; txt.pose.position.y = mk.pose.position.y; txt.pose.position.z = mk.pose.position.z + 0.8
            txt.scale.z = 0.5; txt.color = ColorRGBA(1.0, 1.0, 0.0, 1.0); txt.text = f"ID {c['id']}"
            txt.lifetime = rospy.Duration(1.0)
            marker_array.markers.append(txt)

        if marker_array.markers: self.pub_centroids.publish(marker_array)

    def _log_row(self, t, cluster_id, conf, px, py, pz, ego_x, ego_y, ego_yaw):
        try:
            self.log_file.write(f"{t:.6f}, {cluster_id}, {conf:.3f}, {px:.3f}, {py:.3f}, {pz:.3f}, {ego_x:.3f}, {ego_y:.3f}, {ego_yaw:.3f}\n")
            self.log_file.flush()
        except Exception: pass

    # -------------- SHUTDOWN: GENERATE VIDEO & MAPS --------------
    def _on_shutdown(self):
        try:
            # === [RESTORED] Full CSV Logging Logic ===
            self.log_file.seek(0)
            self.log_file.truncate()
            self.log_file.write("# FINAL ACTIVE CLUSTER POSITIONS WITH ORIENTATION + SIDE + COLOR (MAJORITY)\n")
            self.log_file.write("# cluster_id, x, y, z, count, last_conf, orientation, side, rgb_color\n")

            for c in self.clusters:
                # Majority vote orientazione
                par  = c.get("orient_parallel", 0)
                perp = c.get("orient_perpendicular", 0)
                if not self.orient_enable or (par == 0 and perp == 0): orient_str = "unknown"
                elif par > perp: orient_str = "parallel"
                elif perp > par: orient_str = "perpendicular"
                else: orient_str = "unknown"

                # Majority vote lato
                sl = c.get("side_left", 0)
                sr = c.get("side_right", 0)
                if sl == 0 and sr == 0: side_str = "unknown"
                elif sl > sr: side_str = "left"
                elif sr > sl: side_str = "right"
                else: side_str = "unknown"
                
                # Majority vote colore
                win_rgb = self._get_winning_color(c)
                if win_rgb: color_str = f"{win_rgb[0]}-{win_rgb[1]}-{win_rgb[2]}"
                else: color_str = "unknown"

                self.log_file.write(
                    f"{c['id']}, "
                    f"{c['x']:.3f}, {c['y']:.3f}, {c['z']:.3f}, "
                    f"{c['count']}, {c['last_conf']:.3f}, "
                    f"{orient_str}, {side_str}, {color_str}\n"
                )
            
            self.log_file.close()
            rospy.loginfo("[LOG] Saved detailed final cluster positions.")
        except Exception as e:
            rospy.logwarn(f"[SHUTDOWN LOG] {e}")

        # === 2. GENERATE RECOLORED VIDEO & MAPS ===
        rospy.loginfo("=======================================")
        rospy.loginfo("[POST-PROCESS] Generating Video & Maps...")
        
        # Pre-compute final winning colors for all clusters
        final_colors = {}
        for c in self.clusters:
            rgb = self._get_winning_color(c)
            if rgb:
                # Store as BGR for OpenCV
                final_colors[c['id']] = (rgb[2], rgb[1], rgb[0])
            else:
                final_colors[c['id']] = (128, 128, 128) # Grey default

        # Setup Video Writer
        first_frame_path = os.path.join(self.temp_dir, "frame_000000.jpg")
        if not os.path.exists(first_frame_path):
            rospy.logwarn("[POST-PROCESS] No frames found. Skipping video.")
            return

        sample = cv2.imread(first_frame_path)
        H, W, _ = sample.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, 20.0, (W, H))

        total_frames = self.frame_count
        for i in range(total_frames):
            if i % 50 == 0: rospy.loginfo(f"Processing frame {i}/{total_frames}")

            img_path = os.path.join(self.temp_dir, f"frame_{i:06d}.jpg")
            img = cv2.imread(img_path)
            if img is None: continue

            # Create clean Segmentation Map (Black background)
            final_map_img = np.zeros_like(img)

            if i < len(self.history_data):
                frame_data = self.history_data[i]
                overlay = img.copy()

                for det in frame_data['detections']:
                    cid = det['cid']
                    contour = det['contour']
                    
                    color_bgr = (0, 255, 0) # Default Green
                    
                    if cid is not None:
                        color_bgr = final_colors.get(cid, (0, 255, 0))
                    else:
                        rgb_inst = det.get('rgb_instant')
                        if rgb_inst: color_bgr = (rgb_inst[2], rgb_inst[1], rgb_inst[0])
                        else: continue

                    # 1. Draw on Video Overlay
                    cv2.fillPoly(overlay, [contour], color_bgr)
                    
                    # 2. Draw on Segmentation Map (Black BG)
                    cv2.fillPoly(final_map_img, [contour], color_bgr)
                    
                    # Draw ID on Video
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        label = f"ID {cid}" if cid else ""
                        cv2.putText(img, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                # Blend Video
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # Write Video Frame
            out.write(img)
            
            # Save Segmentation Map
            map_filename = os.path.join(self.output_maps_dir, f"map_{i:06d}.png")
            cv2.imwrite(map_filename, final_map_img)

        out.release()
        rospy.loginfo(f"[SUCCESS] Video saved to {self.output_video_path}")
        rospy.loginfo(f"[SUCCESS] Maps saved to {self.output_maps_dir}")
        
        shutil.rmtree(self.temp_dir)
        rospy.loginfo("[CLEANUP] Temp frames deleted.")


if __name__ == "__main__":
    try:
        MonoYOLOSegClusterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass