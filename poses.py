import numpy as np
import json

def compute_pose(position, target=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
    """
    Compute the camera pose matrix given the camera position, target, and up vector.
    """
    position = np.array(position, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    # Compute the forward vector (camera direction)
    forward = (target - position)
    forward /= np.linalg.norm(forward)

    # Compute the right vector
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        # Up vector is parallel to forward vector; choose a different up vector
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    # Recompute the true up vector
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # Build the rotation matrix
    rotation = np.vstack([right, up, -forward]).T

    # Build the pose matrix
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position

    return pose

def main():
    # List of positions for the cameras
    positions = [
        [10, 0, 0],
        [-10, 0, 0],
        [0, 10, 0],
        [0, -10, 0],
        [0, 0, 10],
        [0, 0, -10],
        
        [10, 10, 0],
        [10, -10, 0],
        [-10, 10, 0],
        [-10, -10, 0],
        
        [10, 0, 10],
        [10, 0, -10],
        [-10, 0, 10],
        [-10, 0, -10],
        
        [0, 10, 10],
        [0, 10, -10],
        [0, -10, 10],
        [0, -10, -10],
        
        [10, 10, 10],
        [10, 10, -10],
        [10, -10, 10],
        [10, -10, -10],
        [-10, 10, 10],
        [-10, 10, -10],
        [-10, -10, 10],
        [-10, -10, -10]
    ]
    
    # List to store the poses
    poses = []
    
    for idx, position in enumerate(positions):
        pose = compute_pose(position)
        poses.append({
            'position': position,
            'pose': pose.tolist()
        })
    
    # Optionally, assemble the JSON data
    data = {
        "backend": "CYCLES",
        "light_mode": "uniform",
        "fast_mode": False,
        "format_version": 6,
        "channels": ["R", "G", "B", "A", "D"],
        "scale": 0.5,
        "images": []
    }
    
    # Intrinsic matrix (assumed same for all images)
    intrinsic = [
        [711.1110599640117, 0.0, 256.0],
        [0.0, 711.1110599640117, 256.0],
        [0.0, 0.0, 1.0]
    ]
    
    for idx, item in enumerate(poses):
        image_data = {
            "intrinsic": intrinsic,
            "pose": item['pose'],
            "rgb": f"{idx:05d}_rgb.png",
            "depth": f"{idx:05d}_depth.png",
            "alpha": f"{idx:05d}_alpha.png",
            "max_depth": 5,
            "HW": [512, 512]
        }
        data['images'].append(image_data)
    
    # Save to JSON file
    with open('camera_data.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    # Print poses for verification
    for idx, item in enumerate(poses):
        print(f"Image {idx:05d}: Position {item['position']}")
        print("Pose Matrix:")
        for row in item['pose']:
            print(f"    {row}")
        print()

if __name__ == "__main__":
    main()
