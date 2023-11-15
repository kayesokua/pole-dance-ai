import cv2
import mediapipe as mp
import time

def measure_elapsed_time(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.3f} seconds")
        return result
    return wrapper

def create_output_dir_if_not_exists(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

@measure_elapsed_time
def extract_landmarks_from_videos(input_dir: str):
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        all_data = []
        for i, filename in enumerate(os.listdir(input_dir)):
            if filename.endswith(".mp4"):
                output_dir = f"../../data/interim/{filename[:-4]}/"
                output_csv = output_dir + "landmarks_rel.csv"
                output_frames = output_dir + "frames/"
                create_output_dir_if_not_exists(output_dir)
                create_output_dir_if_not_exists(output_frames)

                data = []
                cap = cv2.VideoCapture(os.path.join(input_dir, filename))
                frame_count = 0
                pose_frame_count = 0  # new counter for frames with pose landmarks
                missing_pose_count = 0
                fps = cap.get(cv2.CAP_PROP_FPS)
                while cap.isOpened():
                    ret, image = cap.read()
                    if not ret:
                        break
                    results = pose.process(image)

                    if results.pose_landmarks:
                        positions = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear','mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder','left_elbow', 'right_elbow', 'left_wrist', 'right_wrist','left_pinky', 'right_pinky', 'left_index', 'right_index','left_thumb', 'right_thumb', 'left_hip', 'right_hip','left_knee', 'right_knee', 'left_ankle', 'right_ankle','left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']
                        pose_data = {
                            'frame': pose_frame_count,  # use the new counter for frames with pose landmarks
                            'fps': round(fps,1)
                        }
                        for j, position in enumerate(positions):
                            landmark = results.pose_landmarks.landmark[j]
                            if landmark:
                                pose_data[f'{position}_x'] = landmark.x
                                pose_data[f'{position}_y'] = landmark.y
                                pose_data[f'{position}_z'] = landmark.z
                        pose_frame_count += 1  # increment the new counter
                    else:
                        missing_pose_count += 1

                    annotated_image = image.copy()
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    screenshot_path = os.path.join(output_frames, f"{pose_frame_count:05d}.png")  # use the new counter for screenshot file names
                    cv2.imwrite(screenshot_path, annotated_image)
                    if results.pose_landmarks:
                        data.append(pose_data)  # only append data for frames with pose landmarks
                df_landmarks_rel = pd.DataFrame(data)
                df_landmarks_rel.to_csv(output_csv, index=False)
                all_data.extend(data)
                print(f"Extracted {len(data)} frames from {filename} with {missing_pose_count} missing poses")
    return print(f"Extracted {len(all_data)} frames in total")

# if os.path.exists('../../data/external/'):
#     extract_landmarks_from_videos('../../data/external')