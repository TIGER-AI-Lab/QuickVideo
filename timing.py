import time
from deepcodec import VideoReader as DCVR
import sys
import numpy as np
from scipy import stats
from torchvision import transforms as T,


NUM_RUNS = 5  # Number of runs for averaging and confidence intervals

height = 448
width = 448

def main():
    video_path = sys.argv[1]
    max_num_threads = [2, 4, 8, 16, 32]
    temp = DCVR(video_path, num_threads=1)
    num_frames = len(temp)
    fps = round(temp.get_fps())
    print(f"FPS is {fps}")
    indices = list(range(0, num_frames, fps))
    del temp

    for thread in max_num_threads:
        print(f"\n===== Testing with {thread} threads =====")

        # TorchCodec
        try:
            import torch
            from torchcodec.decoders import VideoDecoder
            resize_transform = T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC, antialias=True)

            times = []
            decode_times = []
            combined_times = []

            for _ in range(NUM_RUNS):
                # Measure decoding time
                start_decode = time.time()
                device = "cpu"
                decoder = VideoDecoder(video_path, device=device, num_ffmpeg_threads=thread)
                b = decoder.get_frames_at(indices=indices)
                elapsed_decode = time.time() - start_decode
                decode_times.append(elapsed_decode)
                del decoder  # Ensure decoder is released

                # Measure resizing time
                start_resize = time.time()
                r = resize_transform(b)
                elapsed_resize = time.time() - start_resize

                del b
                del r
                # Track combined time (decode + resize)
                combined_time = elapsed_decode + elapsed_resize
                combined_times.append(combined_time)

            def calculate_ci(times):
                mean_time = np.mean(times)
                n = len(times)
                if n >= 2:
                    ci = stats.t.interval(0.95, n-1, loc=mean_time, scale=stats.sem(times))
                    return f"{mean_time:.3f} ± {(ci[1] - ci[0])/2:.3f}"
                return f"{mean_time:.3f} ± 0.000"

            decode_ci = calculate_ci(decode_times)
            combined_ci = calculate_ci(combined_times)

            print(f"Decode Only: {decode_ci} sec (95% CI over {len(decode_times)} runs)")
            print(f"Combined (Decode + Resize): {combined_ci} sec (95% CI over {len(combined_times)} runs)")

        except Exception as e:
            print(f"TorchCodec failed: {str(e)}")

        # DeepCodec
        try:
            from deepcodec import VideoReader

            times = []
            for _ in range(NUM_RUNS):
                start = time.time()
                vr = VideoReader(video_path, num_threads=thread, height=height, width=width)
                vr.interpolation = "LANCZOS"
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
                vr = DecordVideoReader(video_path, ctx=cpu(0), num_threads=thread, height=height, width=width)
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