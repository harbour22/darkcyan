import logging
import logging.handlers
import os
import os.path
import time
from datetime import datetime
import threading
from multiprocessing import Lock, Process, Value, Queue
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import yaml
from rich.progress import Progress, TextColumn

import darkcyan.yolo_proc
import darkcyan_utils.SignalMonitor as SignalMonitor
from darkcyan.config import Config

import cv2
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

    log = logging.getLogger(__name__)

    with open(        
        Config.get_value("config_file"), "r"
    ) as f:
        app_config = yaml.load(f, Loader=yaml.FullLoader)

    video_sources = {}
    keep_running = Value("b", True)

    for source in app_config["test_sources"]:
        print(f"{source}: {app_config['test_sources'][source]['name']}")
        vsc = DarkCyanSourceConfig(
            app_config["test_sources"][source]["name"],
            app_config["test_sources"][source]["cv2_connection_string"],
            keep_running,
        )
        video_sources[source] = {"vs": vsc}
    
    signal_monitor = SignalMonitor.SignalMonitor()

    smm = SharedMemoryManager()    
    smm.start()

    results_queue = Queue(50)
    
    for source in video_sources:
        process_config = video_sources[source]["vs"]
        infer_shared_memory = smm.SharedMemory(size=1280 * 960 * 3)
        status_shared_memory = smm.SharedMemory(size=100)

        status = f"Initialising {process_config.source_name}..."

        status_shared_memory.buf[: len(status)] = status.encode("utf-8")
        status_shared_memory.buf[99] = len(status)

        video_sources[source]["ishm"] = infer_shared_memory
        video_sources[source]["ishm_np"] = np.ndarray((960,1280,3), dtype=np.uint8, buffer=infer_shared_memory.buf)
        video_sources[source]["shm_status"] = status_shared_memory
        video_sources[source]["source_name"] = process_config.source_name

        process = Process(
            name = process_config.source_name,
            target=darkcyan.yolo_proc.run,
            args=[
                logging_queue,
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
            log.info(f"Starting Process {video_sources[source]['vs'].source_name}")
            process.start()
            log.info(f"Started Process {video_sources[source]['vs'].source_name}")
        
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
                infer_mapped_np = video_sources[source]["ishm_np"]
                cv2.imshow(video_sources[source]["source_name"],infer_mapped_np)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    keep_running.value = False

            if(not results_queue.empty()):
                try:
                    event_source, event_categories, event_boxes = results_queue.get(block=False)
                    log.debug(f"Result: {event_source} - {event_categories} ({event_boxes})")
                    
                    ## update screen
                    
                    for box in event_boxes:
                        infer_mapped_np = cv2.rectangle(infer_mapped_np, (box[0], box[1] ), (box[2], box[3]), (0, 0, 100), 1)

                    cv2.imshow(event_source,infer_mapped_np)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        keep_running.value = False
                except:
                    log.error("Results queue unexpectedly empty, continuing")
                    
            else:
                time.sleep(0.001)
            
            

        keep_running.value = False
        cv2.destroyAllWindows()
        time.sleep(2)
        smm.shutdown()


if __name__ == "__main__":

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
    
    run(logging_queue)
    
    # And now tell the logging thread to finish up, too
    logging_queue.put(None)
    lp.join()
