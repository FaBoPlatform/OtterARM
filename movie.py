import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse
import cv2  # OpenCVをインポート
import constants  # constants.pyをインポート

# HDF5ファイルからデータを読み込む関数
def read_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        qpos = f['observations/qpos'][:]
        qvel = f['observations/qvel'][:]
        action = f['action'][:]
        images = {cam: f[f'observations/images/{cam}'][:] for cam in f['observations/images']}
    print("Loaded data from HDF5 file.")
    print("qpos shape:", qpos.shape)
    print("qvel shape:", qvel.shape)
    print("action shape:", action.shape)
    for cam, imgs in images.items():
        print(f"{cam} images shape:", imgs.shape)
    return qpos, qvel, action, images

# ビデオを生成する関数
def generate_video(qpos, qvel, action, images, output_path, camera_names):
    num_frames = qpos.shape[0]
    num_joints = qpos.shape[1]  # ジョイントの数を自動的に取得

    # フィギュアとサブプロットの作成
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    ax1, ax2, ax3, ax4 = axs

    # 初期フレームを設定（カメラ名を画像に描画）
    image_frames = []
    for cam_name in camera_names:
        img = images[cam_name][0]
        # OpenCVを使用してカメラ名を画像に描画
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGBからBGRに変換
        cv2.putText(img_bgr, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=2)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGRからRGBに戻す
        image_frames.append(img_rgb)
    img_combined = np.hstack(image_frames)
    ims_obj = ax1.imshow(img_combined)
    ax1.axis('off')  # 画像の軸を非表示にする

    # ラインオブジェクトのリストを初期化
    lines_qpos = []
    lines_qvel = []
    lines_action = []
    colors = plt.cm.tab20(np.arange(num_joints))  # 各ラインの色を設定

    # qposのラインを作成
    for i in range(num_joints):
        line_qpos, = ax2.plot([], [], label=f'qpos Joint {i}', color=colors[i])
        lines_qpos.append(line_qpos)

    # qvelのラインを作成
    for i in range(num_joints):
        line_qvel, = ax3.plot([], [], label=f'qvel Joint {i}', color=colors[i])
        lines_qvel.append(line_qvel)

    # actionのラインを作成
    for i in range(num_joints):
        line_action, = ax4.plot([], [], label=f'action Joint {i}', color=colors[i])
        lines_action.append(line_action)

    # 各プロットの設定
    ax2.set_xlim(0, num_frames)
    ax2.set_ylim(np.min(qpos), np.max(qpos))
    ax2.legend(loc='upper right')
    ax2.set_title('Joint Positions (qpos)')

    ax3.set_xlim(0, num_frames)
    ax3.set_ylim(np.min(qvel), np.max(qvel))
    ax3.legend(loc='upper right')
    ax3.set_title('Joint Velocities (qvel)')

    ax4.set_xlim(0, num_frames)
    ax4.set_ylim(np.min(action), np.max(action))
    ax4.legend(loc='upper right')
    ax4.set_title('Joint Actions')

    # フレームを更新する関数
    def update_frame(num):
        # 画像を取得し、カメラ名を描画
        image_frames = []
        for cam_name in camera_names:
            img = images[cam_name][num]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img_bgr, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255), thickness=2)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image_frames.append(img_rgb)
        img_combined = np.hstack(image_frames)
        ims_obj.set_data(img_combined)

        # qposのラインを更新
        for j in range(num_joints):
            lines_qpos[j].set_data(np.arange(num+1), qpos[:num+1, j])

        # qvelのラインを更新
        for j in range(num_joints):
            lines_qvel[j].set_data(np.arange(num+1), qvel[:num+1, j])

        # actionのラインを更新
        for j in range(num_joints):
            lines_action[j].set_data(np.arange(num+1), action[:num+1, j])

        ax1.set_title(f'Frame {num+1}/{num_frames}')
        # 返り値をリストとして返す
        return [ims_obj] + lines_qpos + lines_qvel + lines_action

    # アニメーションを作成
    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=num_frames,
        interval=50,
        blit=True
    )

    # 出力ディレクトリを確認または作成
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # ビデオを保存
    ani.save(output_path, writer='ffmpeg', fps=20)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate video from HDF5 data.')
    parser.add_argument('--name', type=str, help='Episode file name (e.g., episode_0.hdf5)')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., sort)')
    args = parser.parse_args()

    # constants.pyをインポートしてデータフォルダとカメラ名を取得
    data_folder = constants.DATA_DIR
    task_config = constants.TASK_CONFIGS[args.task]
    camera_names = task_config['camera_names']

    # タスクのデータディレクトリを構築
    task_data_dir = os.path.join(data_folder, args.task)

    # --nameオプションが指定されていない場合、最新のHDF5ファイルを探す
    if args.name is None:
        hdf5_files = [f for f in os.listdir(task_data_dir) if f.endswith('.hdf5')]
        if not hdf5_files:
            print(f"No HDF5 files found in {task_data_dir}")
            return
        # ファイルの更新日時でソートして最新のファイルを取得
        hdf5_files.sort(key=lambda x: os.path.getmtime(os.path.join(task_data_dir, x)), reverse=True)
        latest_file = hdf5_files[0]
        print(f"No --name specified. Using the latest file: {latest_file}")
        args.name = latest_file

    # ファイルパスを構築
    file_path = os.path.join(task_data_dir, args.name)

    # HDF5ファイルからデータを読み込む
    qpos, qvel, action, images = read_hdf5_file(file_path)

    # 出力ビデオのパスを構築
    output_dir = './mov/' + args.task + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # エピソード名から拡張子を除去してビデオファイル名を作成
    episode_base_name = os.path.splitext(args.name)[0]
    output_video_path = os.path.join(output_dir, f'{episode_base_name}.mp4')

    # ビデオを生成
    generate_video(qpos, qvel, action, images, output_video_path, camera_names)
    print(f"Video saved to {output_video_path}")

if __name__ == '__main__':
    main()
