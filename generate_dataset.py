import os
import glob
import subprocess

# Paths
ORIGINAL_DIR = "C:/video plagiarism/dataset/original"
PLAGIARISED_DIR = "C:/video plagiarism/dataset/plagiarised"

os.makedirs(PLAGIARISED_DIR, exist_ok=True)

def run_ffmpeg_command(cmd):
    """Executes FFmpeg commands silently."""
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def generate_attacks():
    original_videos = glob.glob(os.path.join(ORIGINAL_DIR, "*.mp4"))
    
    if not original_videos:
        print(f"[ERROR] No .mp4 files found in {ORIGINAL_DIR}")
        return

    print(f"[INFO] Found {len(original_videos)} original videos. Generating attack dataset...\n")

    for video_path in original_videos:
        filename = os.path.splitext(os.path.basename(video_path))[0]
        
        # -------------------------------------------------------------
        # ATTACK 1: Crop Edges (Remove 15% width & height)
        # -------------------------------------------------------------
        out_crop = os.path.join(PLAGIARISED_DIR, f"{filename}_attack_crop.mp4")
        cmd_crop = f'ffmpeg -y -i "{video_path}" -vf "crop=iw*0.85:ih*0.85" -c:a copy "{out_crop}"'
        if run_ffmpeg_command(cmd_crop):
            print(f"✔ Generated: {os.path.basename(out_crop)}")

        # -------------------------------------------------------------
        # ATTACK 2: Color Space Inversion / Dark Mode Filter
        # -------------------------------------------------------------
        out_color = os.path.join(PLAGIARISED_DIR, f"{filename}_attack_darkmode.mp4")
        cmd_color = f'ffmpeg -y -i "{video_path}" -vf "negate" -c:a copy "{out_color}"'
        if run_ffmpeg_command(cmd_color):
            print(f"✔ Generated: {os.path.basename(out_color)}")

        # -------------------------------------------------------------
        # ATTACK 3: Speed Adjustment (1.25x Speedup)
        # -------------------------------------------------------------
        out_speed = os.path.join(PLAGIARISED_DIR, f"{filename}_attack_speed.mp4")
        cmd_speed = f'ffmpeg -y -i "{video_path}" -vf "setpts=0.8*PTS" -filter:a "atempo=1.25" "{out_speed}"'
        if run_ffmpeg_command(cmd_speed):
            print(f"✔ Generated: {os.path.basename(out_speed)}")

        # -------------------------------------------------------------
        # ATTACK 4: Grayscale Conversion
        # -------------------------------------------------------------
        out_gray = os.path.join(PLAGIARISED_DIR, f"{filename}_attack_gray.mp4")
        cmd_gray = f'ffmpeg -y -i "{video_path}" -vf "hue=s=0" -c:a copy "{out_gray}"'
        if run_ffmpeg_command(cmd_gray):
            print(f"✔ Generated: {os.path.basename(out_gray)}")

    print("\n[SUCCESS] Attack dataset generated successfully!")

if __name__ == "__main__":
    generate_attacks()