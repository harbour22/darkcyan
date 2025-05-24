import logging
import logging.handlers
import json
import os
import os.path
import time
from datetime import datetime
import threading
from multiprocessing import Lock, Process, Value, Queue
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
import cv2
import yaml
from rich.progress import Progress, TextColumn

import darkcyan.yolo_proc
import darkcyan_utils.SignalMonitor as SignalMonitor
from darkcyan.config import Config

import numpy as np

def logger_thread(q):

    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


class DarkCyanSourceConfig:

    def __init__(self, source_name, source_path, keep_running) -> None:

        self.source_name = source_name
        self.source_path = source_path
        self.buffer_lock = Lock()
        self.source_fps = Value("f")
        self.inference_fps = Value("f")        
        self.keep_running = keep_running

def run(logging_queue):

    logger = logging.getLogger(__name__)

    with open(        
        Config.get_value("runtime_config_file"), "r"
    ) as f:
        app_config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(f"Loaded config file: {Config.get_value('config_file')}")


    video_sources = {}
    keep_running = Value("b", True)

    for source in app_config["sources"]:

        vsc = DarkCyanSourceConfig(
            app_config["sources"][source]["name"],
            app_config["sources"][source]["cv2_connection_string"],
            keep_running,
        )
        video_sources[source] = {"vs": vsc}
        logger.info(f"Loaded source: {source} with connection string {app_config['sources'][source]['cv2_connection_string']}")
    
    signal_monitor = SignalMonitor.SignalMonitor()

    smm = SharedMemoryManager()    
    smm.start()

    results_queue = Queue(5)
    
    for source in video_sources:
        process_config = video_sources[source]["vs"]
        infer_shared_memory = smm.SharedMemory(size=1280 * 960 * 3)
        status_shared_memory = smm.SharedMemory(size=100)

        status = f"Initialising {process_config.source_name}..."

        status_shared_memory.buf[: len(status)] = status.encode("utf-8")
        status_shared_memory.buf[99] = len(status)

        video_sources[source]["ishm"] = infer_shared_memory
        video_sources[source]["shm_status"] = status_shared_memory

        process = Process(
            name = process_config.source_name,
            target=darkcyan.yolo_proc.run,
            args=[
                logging_queue,
                source,
                process_config.source_name,
                process_config.source_path,
                process_config.buffer_lock,
                process_config.source_fps,
                process_config.inference_fps,
                infer_shared_memory,
                status_shared_memory,
                results_queue,
                process_config.keep_running,
            ],
        )
        video_sources[source]["process"] = process

    start_time = time.time()
    run_for = 60 * 60

    progress = Progress(TextColumn("[progress.descriptions]{task.description}"))

    with progress:


        for source in video_sources:

            process = video_sources[source]["process"]
            video_sources[source]["task"] = progress.add_task(
                f"[blue]{video_sources[source]['vs'].source_name}", start=False
            )
            logger.info(f"Starting Process {video_sources[source]['vs'].source_name}")
            process.start()
            logger.info(f"Started Process {video_sources[source]['vs'].source_name}")

        fourcc_fmt = cv2.VideoWriter_fourcc(*'X265')
        out_send = cv2.VideoWriter('appsrc is-live=true do-timestamp=true ! queue ! \
                                videoconvert ! queue ! vtenc_h265 realtime=true bitrate=2048 ! h265parse ! \
                                rtph265pay config-interval=1 pt=96 name=pay0 ! application/x-rtp,media=video,encoding-name=H265 ! queue ! \
                                udpsink host=127.0.0.1 port=5400 sync=false',
                                fourcc=fourcc_fmt, apiPreference=cv2.CAP_GSTREAMER, fps=25, frameSize = (800,600), isColor=True)
        while (
            ((time.time() - start_time) < run_for)
            and not signal_monitor.exit_now
            and keep_running.value
        ):

            for source in video_sources:
                vsc = video_sources[source]["vs"]
                str_len = video_sources[source]["shm_status"].buf[99]
                update_txt = f"[green] Camera [green]{vsc.source_name}. [green] Status [green]{bytes(video_sources[source]['shm_status'].buf[:str_len]).decode('utf-8')  }.  " \
                    f"[blue] Source FPS: [blue] {vsc.source_fps.value:.2f}, [blue] Infer FPS: [blue] {vsc.inference_fps.value:.2f}"
                progress.update(video_sources[source]["task"], description=update_txt)
                if(vsc.source_name=='front'):
                    # assuming you know shape & dtype in advance
                    frame_shape = (600, 800, 3)  
                    frame_dtype = np.uint8

                    # create numpy view into shared memory
                    frame_arr = np.ndarray(frame_shape, dtype=frame_dtype, buffer=video_sources[source]["ishm"].buf)                    
                    out_send.write(frame_arr)
            if(not results_queue.empty()):
                source_name, final_result_categories, final_result_boxes = results_queue.get(block=False)
                log.info(f"Result: {source_name}, {final_result_categories}, {final_result_boxes}")
                
            else:
                time.sleep(1/50)

        keep_running.value = False
        time.sleep(2)
        smm.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DarkCyan Vision Application')
    parser.add_argument('--config', '-c', help='Path to configuration file (overrides default)')
    args = parser.parse_args()

    logging_queue = Queue()

    fh = logging.handlers.TimedRotatingFileHandler(f'logs/vision-app-{datetime.now().strftime("%d%b%y")}-{os.getpid()}.log', when="midnight", backupCount=3)
    formatter = logging.Formatter("%(asctime)s %(name)-18s %(levelname)-8s %(processName)-12s %(message)s")
    fh.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(fh)

    lp = threading.Thread(target=logger_thread, args=(logging_queue,))
    lp.start()   

    log = logging.getLogger(__name__)
    qh = logging.handlers.QueueHandler(logging_queue)    
    log.setLevel(logging.INFO)
    
    # Update config file path if overridden
    if args.config:
        Config.config()["runtime_config_file"] = args.config        
        log.info(f"Using overridden config file: {args.config}")

    log.info("Starting main process")
    run(logging_queue)
    log.info("Ending main process")
    # And now tell the logging thread to finish up, too
    logging_queue.put(None)
    lp.join()
