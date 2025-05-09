import time
from deepcodec import VideoReader as DCVR
import sys
import os
import numpy as np
from scipy import stats
import glob

NUM_RUNS = 5  # Number of runs for averaging and confidence intervals
FIXED_THREADS = 16  # Fixed number of threads for all tests
SAMPLE_FPS = 1  # Sample at 1 frame per second

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    # Find all matching video files in the directory
    video_files = sorted(glob.glob(os.path.join(directory_path, "movie1080p.BluRay.1hour*.mp4")))
    
    if not video_files:
        print(f"No matching video files found in {directory_path}")
        sys.exit(1)
    
    for video_path in video_files:
        print(f"\n===== Testing file: {os.path.basename(video_path)} =====")
        
        # Get video metadata
        temp = DCVR(video_path, num_threads=1)
        num_frames = len(temp)
        video_fps = round(temp.get_fps())
        duration = num_frames / video_fps
        del temp
        
        minutes, seconds = divmod(duration, 60)
        print(f"Duration: {int(minutes)}m {int(seconds)}s, FPS: {video_fps}, Total frames: {num_frames}")
        
        # Calculate frame indices for 1 FPS sampling
        frame_step = video_fps // SAMPLE_FPS
        indices = list(range(0, num_frames, frame_step))
        print(f"Sampling {len(indices)} frames at {SAMPLE_FPS} FPS")

        # TorchCodec
        try:
            import torch
            from torchcodec.decoders import VideoDecoder

            times = []
            for _ in range(NUM_RUNS):
                start = time.time()
                device = "cpu"
                decoder = VideoDecoder(video_path, device=device, num_ffmpeg_threads=FIXED_THREADS)
                decoder.get_frames_at(indices=indices)
                elapsed = time.time() - start
                times.append(elapsed)
                del decoder  # Ensure decoder is released

            mean_time = np.mean(times)
            std_dev = np.std(times, ddof=1)
            n = len(times)
            if n >= 2:
                ci = stats.t.interval(0.95, n-1, loc=mean_time, scale=stats.sem(times))
                ci_str = f"{mean_time:.3f} ± {(ci[1] - ci[0])/2:.3f}"
            else:
                ci_str = f"{mean_time:.3f} ± 0.000"

            print(f"TorchCodec: {ci_str} sec (95% CI over {n} runs)")

        except Exception as e:
            print(f"TorchCodec failed: {str(e)}")

        # DeepCodec
        try:
            from deepcodec import VideoReader

            times = []
            for _ in range(NUM_RUNS):
                start = time.time()
                vr = VideoReader(video_path, num_threads=FIXED_THREADS)
                _ = vr.get_batch(indices)
                elapsed = time.time() - start
                times.append(elapsed)
                del vr

            mean_time = np.mean(times)
            std_dev = np.std(times, ddof=1)
            n = len(times)
            if n >= 2:
                ci = stats.t.interval(0.95, n-1, loc=mean_time, scale=stats.sem(times))
                ci_str = f"{mean_time:.3f} ± {(ci[1] - ci[0])/2:.3f}"
            else:
                ci_str = f"{mean_time:.3f} ± 0.000"

            print(f"DeepCodec: {ci_str} sec (95% CI over {n} runs)")

        except Exception as e:
            print(f"DeepCodec failed: {str(e)}")

        # Decord
        try:
            import decord
            from decord import VideoReader as DecordVideoReader
            from decord import cpu

            times = []
            for _ in range(NUM_RUNS):
                start = time.time()
                vr = DecordVideoReader(video_path, ctx=cpu(0), num_threads=FIXED_THREADS)
                _ = vr.get_batch(indices)
                elapsed = time.time() - start
                times.append(elapsed)
                del vr

            mean_time = np.mean(times)
            std_dev = np.std(times, ddof=1)
            n = len(times)
            if n >= 2:
                ci = stats.t.interval(0.95, n-1, loc=mean_time, scale=stats.sem(times))
                ci_str = f"{mean_time:.3f} ± {(ci[1] - ci[0])/2:.3f}"
            else:
                ci_str = f"{mean_time:.3f} ± 0.000"

            print(f"Decord:    {ci_str} sec (95% CI over {n} runs)")

        except Exception as e:
            print(f"Decord failed: {str(e)}")

if __name__ == "__main__":
    main()