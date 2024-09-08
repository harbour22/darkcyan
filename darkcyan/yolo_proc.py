import logging,logging.handlers

import os
import traceback
import ast
from threading import Thread
from multiprocessing import Process

from multiprocessing import Value
from multiprocessing.shared_memory import SharedMemory

from sys import platform
import time

from darkcyan_utils.FPS import FPS

import darkcyan_utils.SignalMonitor as SignalMonitor

import cv2
from datetime import datetime


from queue import Queue, Empty, Full

import numpy as np

import contextlib

from ultralytics import YOLO


class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0):
        self.t = t

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        return time.time()    
    

        
class VideoSource:

    def __init__(self, source_name, source_path, source_fps, output_image_queue, model_imgsz, keep_running):
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
                
        self.source_name = source_name
        self.source_path = source_path        
       
        self.output_image_queue = output_image_queue
        self.model_imgsz = model_imgsz

        self.source_fps = source_fps
        self.fps = FPS()

        self.keep_running = keep_running
        self.stopped = False

        self.logger.debug('DarkCyanVideoStream created')
        
    def __repr__(self):
        return f"VideoSource object: {self.source_name}"    

    def update(self):
        
        if platform == "linux" or platform == "linux2":
            self.stream = cv2.VideoCapture(self.source_path, cv2.CAP_GSTREAMER)
            self.logger.info(f"Initialised {self.source_name} FVS Capture on Linux using Gstreamer")

        elif platform == "darwin":

            self.stream = cv2.VideoCapture(self.source_path, cv2.CAP_GSTREAMER)
            self.logger.info(f"Initialised {self.source_name} FVS Capture on MacOS using Gstreamer")
        else:
            self.logger.info(f"Initialised {self.source_name} FVS Capture on {platform}")
            self.stream = cv2.VideoCapture(self.source_path)

        # check the first frame
        (grabbed, frame) = self.stream.read()
        
        if(not grabbed):
            self.logger.info(f"{self.source_name} Unable to grab the first frame from the video stream, unable to proceed.  Check associated libraries are available / opencv has been built correctly")
            self.stop()
            return

        self.logger.info(f"{self.source_name} first frame size: {frame.shape}")
        frame = cv2.resize(frame, (self.model_imgsz[1],self.model_imgsz[0]), interpolation = cv2.INTER_AREA)

        self.fps.start()
        failure_count = 0
        while not self.stopped and self.keep_running.value and failure_count < 25:
            frame_reader_pf = Profile()
            with frame_reader_pf:
                (grabbed, original_frame) = self.stream.read()
            
            if(frame_reader_pf.t>1):
                self.logger.info(f"Frame reader for {self.source_name} took {frame_reader_pf.t} seconds, more than expected.  We will continue")
            
            if not grabbed:
                self.logger.info(f"FVS for {self.source_name} has unexpectedly reached the end of the video stream, we will gracefully end and notify the other threads")
                failure_count += 1
                continue
            failure_count = 0
            self.fps.update()
            self.source_fps.value = self.fps.fps()
            try:
                # We do the resizing / prep in this thread to improve performace on the inference thread (it's more computationally expensive)
                resized_frame = cv2.resize(original_frame, (self.model_imgsz[1],self.model_imgsz[0]), interpolation = cv2.INTER_AREA)                
                            
            except:
                traceback.print_exc()
                continue
            
            # Check if the queue is full, if so take the oldest item and add this
            if(self.output_image_queue.full()):
                try:
                    self.output_image_queue.get_nowait()
                except Empty:
                    self.logger.info(f"Unexpected non-fatal race condition for {self.source_name}, we took long enough to clear a space that the infer thread got there")

            q_pf = Profile()
            
            self.output_image_queue.put((original_frame, resized_frame))
                    
    
        self.fps.stop()  

        self.logger.info(f'{self.source_name} FVS ended (failure count: {failure_count})  Stream read approx. FPS: {self.fps.fps():.2f}')     

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def stop(self):
        # indicate that the thread should be stopped
        self.logger.info(f'{self.source_name} FVS stop called')         
        self.stopped = True    
        time.sleep(1)    
        self.stream.release()
        cv2.destroyAllWindows()

class ObjectDetection(object):

    def __init__(self, source_name, inference_fps, image_source_queue, infer_shared_memory, buffer_lock, status_shared_memory, results_queue, keep_running) -> None:
        
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
            
        self.source_name = source_name
        self.fps_feedback = inference_fps
        self.fps = FPS()
        self.stopped = False
        self.infer_shared_memory = infer_shared_memory
        self.status_shared_memory = status_shared_memory
        self.keep_running = keep_running
        self.buffer_lock = buffer_lock
        self.results_queue = results_queue
        self.imgsz = (640,480)

        if(platform=="darwin"):
            self.device='mps'
            self.logger.info('Enabling MPS support for macOS')         
        else:
            self.device='0'

        #self.model = YOLO('/Users/chris/developer/darkcyan_data/engines/yolov9_4.15_large-det.mlpackage', task='detect', verbose=False)
        self.model = YOLO('yolov8x.mlpackage', task='detect', verbose=False)
        


        self.logger.debug('Warming yolo detection engine for image size: ' + str(self.imgsz))
        detection_engine_pf = Profile()

        test_img = np.random.randint(low=0, high=255, size=(640, 480, 3), dtype='uint8')
        with detection_engine_pf:
            try:
                results = self.model.predict(source=test_img, imgsz=(640,480), device=self.device, conf=0.4, iou=0.45, stream=True, verbose=False)                    
            except:
                traceback.print_exc()
                self.stop()
        self.logger.debug (f"First warmup completed in :{detection_engine_pf.dt * 1E3:.1f}ms")
        with detection_engine_pf:
            results = self.model.predict(source=test_img, device=self.device, conf=0.4, iou=0.45)                    
        self.logger.debug (f"Second warmup completed in :{detection_engine_pf.dt * 1E3:.1f}ms")

        #self.results_queue = Queue(5)
        self.image_source_queue = image_source_queue

    def infer(self):
        # keep looping infinitely
        self.fps.start()
        time_since_last_image = time.time()
        
        while not self.stopped and self.keep_running.value:
                   
            try:
                
                ## If we don't get an image on one of the queues for 15 seconds, exit.  This is a weird state / we should always have images from all queues at this point
                ( original_frame, inference_img ) = self.image_source_queue.get(timeout=15)
                
                if(time.time() - time_since_last_image > 10):
                    self.logger.error(f"Inference has not received an image in over 15 seconds.  Exiting")
                    self.stop()
                    break

                time_since_last_image = time.time()
                 
                orig_h,orig_w,orig_d = original_frame.shape

                results = self.model.predict(source=inference_img, device=self.device, conf=0.3, iou=0.40, stream=True, verbose=False)    
                                
                self.fps_feedback.value = self.fps.fps()               

                result_boxes = []
                result_scores = []
                result_classid = []
                result_categories = []
                result_tracks = []

                for result in results:

                    if(not result.boxes.xyxy.nelement()):                        
                        continue
                    else:
                        boxes = result.boxes.xyxy.cpu()
                        confs = result.boxes.conf.cpu()
                        cls_ints = result.boxes.cls.cpu()
                        track_ids = [0]*len(boxes)
                        for box, track_id, conf, cls_int in zip(boxes, track_ids, confs, cls_ints):
                            x1, y1, x2, y2 = box
                            cls_int = cls_int.int().item()
                            
                            w_ratio = orig_w / inference_img.shape[1]
                            h_ratio = orig_h / inference_img.shape[0]
                            xyxy=[0,0,0,0]
                            xyxy[0] = (x1*w_ratio).int().item()
                            xyxy[2] = (x2*w_ratio).int().item()
                            xyxy[1] = (y1*h_ratio).int().item()
                            xyxy[3] = (y2*h_ratio).int().item()
                            
                            result_boxes.append(xyxy)

                            result_scores.append(conf.item())
                            result_classid.append(cls_int)
                            result_tracks.append(track_id)
                            if(self.model.names==None):                                
                                category = f'class{cls_int}'
                            else:                                        
                                category = self.model.names[cls_int]
                                                            
                            result_categories.append(category)

                            original_frame = cv2.rectangle(original_frame, (xyxy[0], xyxy[1] ), (xyxy[2], xyxy[3]), (255, 255, 255), 3)
                                                        
                if(len(result_categories))>0:                                                      
                    status = f'{self.source_name} can see {result_categories}'
                else:
                    status = f'{self.source_name} can see nothing'

                self.status_shared_memory.buf[:len(status)] = status.encode('utf-8')
                self.status_shared_memory.buf[99] = len(status)

                if(len(result_categories)!=1000):

                    output_frame = original_frame
                    locked = self.buffer_lock.acquire(block=False)
                    if(not locked):
                        self.logger.debug("[WARN] Buffer lock is held, not writing results to shared memory")
                    else:                                                  
                        shm_arr = np.ndarray(output_frame.shape, dtype=output_frame.dtype, buffer=self.infer_shared_memory.buf)
                        shm_arr[:] = output_frame[:]
                        self.buffer_lock.release()
                        try:
                            if(self.results_queue.full()):
                                self.logger.debug("[WARN] Results queue is full, clearing some space")
                                self.results_queue.get(block=False)
                            self.results_queue.put(( self.source_name, result_categories, result_boxes ), block=False )
                            self.logger.debug(f"Added {result_categories} to results queue")
                        except Full:
                            self.logger.debug("[WARN] Not able to add to the results queue, continuing.  This is wholly unexpected")
                                            

                self.fps.update()                           
                

            except Empty:
                self.logger.debug("[WARN] No more frames to infer from after 15 seconds, exiting")
                self.stop()
                
            except Full:
                self.logger.debug("[WARN] Not able to add to the results queue, continuing.  This is wholly unexpected")
                continue
            except:    
                traceback.print_exc()
                self.stop()

        
        self.fps.stop()
        self.logger.debug("[INFO] infer approx. FPS: {:.2f}".format(self.fps.fps()))    


    def start(self):
        # start a thread to read frames from the video queue and infer
        t = Thread(target=self.infer, args=())
        t.daemon = True
        t.start()
        return self
    
    def stop(self):
        # indicate that the thread should be stopped
        self.logger.info(f'{self.source_name} Infer Thread stop called')         
        self.stopped = True    
        time.sleep(1)    

def run(logging_queue, source_name, source_path, buffer_lock, source_fps, inference_fps, infer_shared_memory, status_shared_memory, results_queue, keep_running):

    qh = logging.handlers.QueueHandler(logging_queue)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)

    output_image_queue = Queue(5)
    inference_engine = ObjectDetection(source_name, inference_fps, output_image_queue, infer_shared_memory, buffer_lock, status_shared_memory, results_queue, keep_running)
    image_stream = VideoSource(source_name, source_path, source_fps, output_image_queue, inference_engine.imgsz, keep_running)

    image_stream.start()
    inference_engine.start()
    try:
        while( keep_running.value and not image_stream.stopped and not inference_engine.stopped ):
            time.sleep(1)
    except:
        pass
    keep_running.value = False
    image_stream.stop()
    inference_engine.stop()


