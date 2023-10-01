import csv
import matplotlib.pyplot as plt

def load_pose_landmarks_from_csv(csv_file):
    landmarks = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            landmark = {
                'Landmark': int(row[0]),
                'X': float(row[1]),
                'Y': float(row[2]),
                'Z': float(row[3])
            }
            landmarks.append(landmark)
    return landmarks

def plot_pose_landmarks(landmarks):
    x_values = [landmark['X'] for landmark in landmarks]
    y_values = [landmark['Y'] for landmark in landmarks]

    plt.scatter(x_values, y_values, s=20, marker='o')
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Pose Landmarks')
    plt.show()