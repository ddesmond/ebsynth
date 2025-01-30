import ebsynth as eb
from concurrent.futures import ThreadPoolExecutor
import time

STYLE_IMAGE = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/output000.jpg"
SOURCE1 = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/000.jpg"
TARGET1 = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/001.jpg"
SOURCE2 = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/002.jpg"
TARGET2 = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/003.jpg"

def run_ezsynth(style, source, target):
    ez = eb.Ebsynth(style_img=style, guides=[])
    ez.add_guide(source, target, 6.0)
    ez.run()

def main():
    # Prepare lists of source and target images for two sets
    SET1_SOURCES = [f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/{str(i).zfill(3)}.jpg" for i in range(0, 20, 2)]
    SET1_TARGETS = [f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/{str(i).zfill(3)}.jpg" for i in range(1, 21, 2)]
    SET2_SOURCES = [f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/{str(i).zfill(3)}.jpg" for i in range(20, 40, 2)]
    SET2_TARGETS = [f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/{str(i).zfill(3)}.jpg" for i in range(21, 41, 2)]

    # Serial Execution
    start_time = time.time()
    for source, target in zip(SET1_SOURCES, SET1_TARGETS):
        run_ezsynth(STYLE_IMAGE, source, target)
    for source, target in zip(SET2_SOURCES, SET2_TARGETS):
        run_ezsynth(STYLE_IMAGE, source, target)
    end_time = time.time()
    print(f"Serial execution time: {end_time - start_time}")

    # Parallel Execution
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        futures1 = [executor.submit(run_ezsynth, STYLE_IMAGE, source, target) for source, target in zip(SET1_SOURCES, SET1_TARGETS)]
        futures2 = [executor.submit(run_ezsynth, STYLE_IMAGE, source, target) for source, target in zip(SET2_SOURCES, SET2_TARGETS)]
        
        # Wait for all to complete
        for future in futures1:
            future.result()
        for future in futures2:
            future.result()

    end_time = time.time()
    print(f"Parallel execution time: {end_time - start_time}")

