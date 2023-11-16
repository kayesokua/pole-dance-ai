import os
import csv
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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

def plot_pose_xyzv_values(csv_path: str, image_path: str):
    plot_title = os.path.basename(csv_path).split('.')[0]    
    df = pd.read_csv(csv_path)
    orig_img = cv2.imread(image_path)

    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    v = df['v'].to_numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].set_title(plot_title)
    axes[0].imshow(orig_img)
    axes[0].axis('off')

    axes[1].set_title("Pose X,Y")
    axes[1].scatter(x[0:10], y[0:10], color=mcolors.CSS4_COLORS['red'], label="head")
    axes[1].scatter(x[11:22], y[11:22], color=mcolors.CSS4_COLORS['green'], label="upper body")
    axes[1].plot(x[23:28], y[23:28], color=mcolors.CSS4_COLORS['blue'], label="lower body", marker='P', linestyle='None')
    axes[1].plot(x[29:32], y[29:32], color=mcolors.CSS4_COLORS['magenta'], label="feet", marker='P', linestyle='None')
    axes[1].set_xlabel("x-axis")
    axes[1].set_ylabel("y-axis")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].invert_yaxis()
    axes[1].grid()

    # Pose Z Plot
    axes[2].set_title("Pose Z")
    axes[2].plot(z, color=mcolors.CSS4_COLORS['cornflowerblue'], label="z-axis")
    axes[2].set_xlabel("Pose Landmarks")
    axes[2].set_ylabel("z-axis")
    axes[2].set_ylim(min(z), max(z))
    axes[2].grid()

    # Visibility Plot
    axes[3].set_title("Visibility")
    axes[3].plot(v, color=mcolors.CSS4_COLORS['mediumpurple'], label="visibility")
    axes[3].set_xlabel("Pose Landmarks")
    axes[3].set_ylabel("visibility")
    axes[3].set_ylim(0, 1.5)
    axes[3].grid()

    fig.tight_layout()
    plt.show()

def batch_plot_pose_landmarks(directory, keyword):
    files = [f for f in os.listdir(directory) if f.startswith(keyword) and f.endswith('.csv')]
    files = sorted(files)
    if not files:
        print("No files found.")
        return

    cols = 4
    rows = -(-len(files) // cols)  # Ceiling division to ensure enough rows

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), squeeze=False)

    for i, file in enumerate(files):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        df = pd.read_csv(os.path.join(directory, file))

        if df.shape[1] != 4:
            print(f"File {file} does not have 4 columns.")
            continue

        x = df.iloc[:, 0].to_numpy()
        y = df.iloc[:, 1].to_numpy()

        ax.set_title(file[:-4])
        ax.plot(x[0:10], y[0:10], color=mcolors.CSS4_COLORS['red'], label="head", marker='P', linestyle='None')
        ax.plot(x[11:22], y[11:22], color=mcolors.CSS4_COLORS['green'], label="mid body", marker='P', linestyle='None')
        ax.plot(x[23:28], y[23:28], color=mcolors.CSS4_COLORS['blue'], label="lower body", marker='P', linestyle='None')
        ax.plot(x[29:32], y[29:32], color=mcolors.CSS4_COLORS['magenta'], label="feet", marker='P', linestyle='None')
        
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.grid()

    # Turn off axes that are not in use
    for j in range(i+1, rows*cols):
        fig.delaxes(axes[j // cols, j % cols])

    fig.tight_layout()
    plt.show()

def batch_plot_pose_landmarks_agg(directory, keyword):
    files = [f for f in os.listdir(directory) if f.startswith(keyword) and f.endswith('.csv')]
    files = sorted(files)
    if not files:
        print("No files found.")
        return

    cols = 4
    rows = -(-len(files) // cols)  # Ceiling division to ensure enough rows

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), squeeze=False)

    for i, file in enumerate(files):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        df = pd.read_csv(os.path.join(directory, file))

        if df.shape[1] != 4:
            print(f"File {file} does not have 4 columns.")
            continue
        x = df.iloc[:, 0].to_numpy()
        y = df.iloc[:, 1].to_numpy()
        
        head_x, head_y = df['x'].iloc[0:10].mean(), df['y'].iloc[0:10].mean()
        torso_x, torso_y = df['x'].iloc[[11,12,24,23]].mean(), df['y'].iloc[[11,12,24,23]].mean()

        handL_x, handL_y = df['x'].iloc[[15, 17, 19, 21]].mean(), df['y'].iloc[[15, 17, 19, 21]].mean()
        handR_x, handR_y = df['x'].iloc[[16, 18, 20, 22]].mean(), df['y'].iloc[[16, 18, 20, 22]].mean()
        
        footL_x, footL_y = df['x'].iloc[[27, 29, 31]].mean(), df['y'].iloc[[27, 29, 31]].mean()
        footR_x, footR_y = df['x'].iloc[[28, 30, 32]].mean(), df['y'].iloc[[28, 30, 32]].mean()
        
        ax.set_title(file[:-4])
        ax.plot(head_x, head_y, color=mcolors.CSS4_COLORS['red'], label="head", marker='P')
        ax.plot(torso_x, torso_y, color=mcolors.CSS4_COLORS['green'], label="upper body", marker='P')
        ax.plot(handR_x, handR_y, color=mcolors.CSS4_COLORS['blue'], label="upper body", marker='<')
        ax.plot(handL_x, handL_y, color=mcolors.CSS4_COLORS['blue'], label="upper body", marker='>')
        ax.plot(df['x'].iloc[[12, 14, 16]], df['y'].iloc[[12, 14, 16]], color=mcolors.CSS4_COLORS['blue'], label="right arm", marker='_')
        ax.plot(df['x'].iloc[[11, 13, 15]], df['y'].iloc[[11, 13, 15]], color=mcolors.CSS4_COLORS['blue'], label="left arm", marker='_')
        ax.plot(footR_x, footR_y, color=mcolors.CSS4_COLORS['violet'], label="lower body", marker='<')
        ax.plot(footL_x, footL_y, color=mcolors.CSS4_COLORS['violet'], label="lower body", marker='>')
        ax.plot(df['x'].iloc[[24, 26, 28]], df['y'].iloc[[24, 26, 28]], color=mcolors.CSS4_COLORS['magenta'], label="right legs", marker='_')
        ax.plot(df['x'].iloc[[23, 25, 27]], df['y'].iloc[[23, 25, 27]], color=mcolors.CSS4_COLORS['magenta'], label="left legs", marker='_')
        ax.plot(df['x'].iloc[[30,32]], df['y'].iloc[[30,32]], color=mcolors.CSS4_COLORS['violet'], label="right foot sole", marker='_')
        ax.plot(df['x'].iloc[[29, 31]], df['y'].iloc[[29, 31]], color=mcolors.CSS4_COLORS['violet'], label="left foot sole", marker='_')
        
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.grid()

    # Turn off axes that are not in use
    for j in range(i+1, rows*cols):
        fig.delaxes(axes[j // cols, j % cols])

    fig.tight_layout()
    plt.show()