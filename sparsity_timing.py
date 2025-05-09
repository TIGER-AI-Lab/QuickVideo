import time
from deepcodec import VideoReader as DCVR
import sys
import numpy as np
from scipy import stats

NUM_RUNS = 5  # Number of runs for averaging and confidence intervals

def main():
    video_path = sys.argv[1]
    num_threads = 16  # Fixed number of threads
    temp = DCVR(video_path, num_threads=1)
    num_frames = len(temp)
    fps = round(temp.get_fps())
    print(f"FPS is {fps}")
    del temp

    seconds_between_frames = [1, 2, 4, 8, 16]
    for seconds in seconds_between_frames:
        frame_interval = seconds * fps
        indices = list(range(0, num_frames, frame_interval))
        print(f"\n===== Sampling every {seconds} seconds (interval = {frame_interval} frames) =====")

        # TorchCodec
        try:
            import torch
            from torchcodec.decoders import VideoDecoder

            times = []
            for _ in range(NUM_RUNS):
                start = time.time()
                device = "cpu"
                decoder = VideoDecoder(video_path, device=device, num_ffmpeg_threads=num_threads)
                decoder.get_frames_at(indices=indices)
                elapsed = time.time() - start
                times.append(elapsed)
                del decoder

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
                vr = VideoReader(video_path, num_threads=num_threads)
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
            print(f"DeepCodec:  {ci_str} sec (95% CI over {n} runs)")
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
                vr = DecordVideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
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
            print(f"Decord:     {ci_str} sec (95% CI over {n} runs)")
        except Exception as e:
            print(f"Decord failed: {str(e)}")

if __name__ == "__main__":
    main()
