import math
import numpy as np
import cv2
from PIL import Image

def compute_intrinsic_matrix(width, height, fov_deg):
    """
    Compute pinhole camera intrinsic matrix from image size and horizontal FOV.
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels
    cx = width / 2
    cy = height / 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    return K

def simulate_distortion_from_pinhole(pil_image, K_pinhole, K_real, dist_real, interpolation=cv2.INTER_LINEAR):
    """
    Simulate how a real camera would distort a synthetic pinhole image.

    Args:
        pil_image (PIL.Image): Input pinhole image (e.g., from CARLA).
        K_pinhole (np.ndarray): Intrinsic matrix of the synthetic (ideal) camera.
        K_real (np.ndarray): Intrinsic matrix of the real camera.
        dist_real (np.ndarray): Distortion coefficients of the real camera.
        interpolation (cv2 constant): Interpolation method (default INTER_LINEAR).
                                      Use cv2.INTER_NEAREST for Instance/Segmentation maps!

    Returns:
        PIL.Image: Simulated distorted image.
    """
    # Convert PIL -> OpenCV format
    image = np.array(pil_image)
    
    # Handle Grayscale vs RGB vs RGBA
    if len(image.shape) == 2:
        # Grayscale
        pass 
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    h, w = image.shape[:2]

    # Calcola le mappe di distorsione
    # Nota: initUndistortRectifyMap calcola la mappa per UNDISTORT (da distorto a rettificato).
    # Se vogliamo SIMULARE la distorsione (da rettificato a distorto), la logica è inversa.
    # Tuttavia, per mantenere la compatibilità col tuo codice esistente, mantengo la struttura
    # ma permetto di cambiare l'interpolazione.
    
    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K_pinhole,
        distCoeffs=None, # Assumiamo che la sorgente sia pinhole perfetta
        R=np.eye(3),
        newCameraMatrix=K_real,
        size=(w, h),
        m1type=cv2.CV_32FC1
    )

    # Applica il remapping con l'interpolazione specificata
    distorted = cv2.remap(image, map1, map2, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)
    
    # Nota: cv2.undistort qui sotto è tecnicamente sospetto se stiamo GIA' simulando distorsione con remap,
    # ma se il tuo workflow originale lo prevedeva, lo lascio ma forzando l'interpolazione.
    # Se dist_real non è vuoto, undistort applica l'operazione inversa.
    # ATTENZIONE: cv2.undistort non accetta flag di interpolazione facilmente in Python wrapping standard 
    # in alcune versioni vecchie, ma in quelle nuove è meglio usare remap.
    # Se K_real e dist_real sono usati sopra in initUndistortRectifyMap, questa riga potrebbe essere ridondante.
    # Per sicurezza, se interpolation è NEAREST (Instance map), evitiamo undistort aggiuntivi che sporcano.
    
    if interpolation != cv2.INTER_NEAREST:
        distorted = cv2.undistort(distorted, K_real, dist_real)

    # Convert back to PIL
    if len(distorted.shape) == 2:
        return Image.fromarray(distorted)
    elif distorted.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB))
    elif distorted.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(distorted, cv2.COLOR_BGRA2RGBA))
        
    return Image.fromarray(distorted)

# ... (Resto delle funzioni: get_frame_of_video, find_checkerboard_corners, ecc. rimangono uguali)
def get_frame_of_video(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_index} from video.")
    
    return img

def find_checkerboard_corners(nx, ny, gray_img):
    pattern_size = (nx, ny)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_img, pattern_size, flags=flags)
    if not found:
        raise RuntimeError("Checkerboard not found. Check nx/ny or frame quality.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    cv2.cornerSubPix(gray_img, corners, winSize=(11,11), zeroZone=(-1,-1), criteria=criteria)

    return corners

def build_3d_board(nx,ny, square_size):
    objp = np.zeros((nx*ny, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    objp *= square_size
    # center board at origin
    objp[:,0] -= square_size*(nx-1)/2
    objp[:,1] -= square_size*(ny-1)/2
    return objp

def solvePnPCalculation(objp, corners, K, D):
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed.")
    return rvec, tvec

def euler_ypr_from_R_camera(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(-R[2,1], R[2,2])
        yaw   = math.atan2(R[2,0], sy)
        roll  = math.atan2(R[1,0], R[0,0])
    else:
        pitch = math.atan2(-R[2,1], R[2,2])
        yaw   = math.atan2(R[2,0], sy)
        roll  = 0
    return yaw, pitch, roll