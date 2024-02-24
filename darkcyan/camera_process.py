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
import logging,logging.handlers

from queue import Queue, Empty, Full

import numpy as np

import contextlib

import coremltools as ct
from PIL import Image 

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
    

        
class DarkCyanVideoSource:

    def __init__(self, source_name, source_path, source_fps, output_image_queue, model_imgsz, keep_running):
        self.logger = logging.getLogger("darkcyan")

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
                imageRGB = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)                
                pill_img = Image.fromarray(imageRGB)
                            
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
            
            self.output_image_queue.put((original_frame, pill_img))
                    
    
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

class DarkCyanObjectDetection(object):

    def __init__(self, source_name, inference_fps, image_source_queue, infer_shared_memory, buffer_lock, status_shared_memory, results_queue, keep_running) -> None:
        
        self.logger = logging.getLogger("darkcyan")
        self.source_name = source_name
        self.fps_feedback = inference_fps
        self.fps = FPS()
        self.stopped = False
        self.infer_shared_memory = infer_shared_memory
        self.status_shared_memory = status_shared_memory
        self.keep_running = keep_running
        self.buffer_lock = buffer_lock
        self.results_queue = results_queue

        if(platform=="darwin"):
            self.device='mps'
            self.logger.info('Enabling MPS support for macOS')         
        else:
            self.device='0'

        self.model = ct.models.MLModel('/Users/chris/developer/darkcyan_data/engines/yolov8_4.11_large-det.mlpackage')

        self.categories = ast.literal_eval(self.model.user_defined_metadata['names'])
        self.imgsz = ast.literal_eval(self.model.user_defined_metadata['imgsz'])
        
        self.logger.debug('Warming coreml detection engine for image size: ' + str(self.imgsz))
        detection_engine_pf = Profile()
        pill_img = Image.fromarray(np.zeros((self.imgsz[0],self.imgsz[1],3), np.uint8))
        with detection_engine_pf:
            out_dict = self.model.predict({'image':pill_img})
        
        self.logger.debug (f"First warmup completed in :{detection_engine_pf.dt * 1E3:.1f}ms")
        with detection_engine_pf:
            out_dict = self.model.predict({'image':pill_img})
            self.logger.debug(out_dict)           
        self.logger.debug (f"Second warmup completed in :{detection_engine_pf.dt * 1E3:.1f}ms")

        self.results_queue = Queue(5)
        self.image_source_queue = image_source_queue

    def infer(self):
        # keep looping infinitely
        self.fps.start()
        time_since_last_image = time.time()

        while not self.stopped and self.keep_running.value:
                   
            try:
                
                ## If we don't get an image on one of the queues for 15 seconds, exit.  This is a weird state / we should always have images from all queues at this point
                ( original_frame, pill_img ) = self.image_source_queue.get(timeout=15)
                
                if(time.time() - time_since_last_image > 10):
                    self.logger.error(f"Inference has not received an image in over 15 seconds.  Exiting")
                    self.stop()
                    break

                time_since_last_image = time.time()
                 
                h,w,d = original_frame.shape
                #h,w = image_raw.shape

                out_dict = self.model.predict({'image':pill_img})
                                
                self.fps_feedback.value = self.fps.fps()               

                result_boxes = []
                result_confidences = []
                result_categories = []
                final_result_boxes = []
                final_result_categories = []
                final_result_confidences = []

                for coord in out_dict['coordinates']:

                    xyxy=[0,0,0,0]
                    xyxy[0] = int(coord[0]*w - (coord[2]*w)/2)
                    xyxy[1] = int(coord[1]*h - (coord[3]*h)/2)
                    xyxy[2] = int(coord[0]*w + (coord[2]*w)/2)
                    xyxy[3] = int(coord[1]*h + (coord[3]*h)/2)

                    result_boxes.append(xyxy)

                for conf in out_dict['confidence']:
                    
                    max_confidence = np.max(conf)
                    max_confidence_idx = np.argmax(conf)
                    result_confidences.append(max_confidence)
                    result_categories.append(f'{self.categories[max_confidence_idx]}({max_confidence:.2f}%)')

                for idx, confidence in enumerate(result_confidences):
                    if(confidence>.3):
                        xyxy = result_boxes[idx]

                        final_result_boxes.append(xyxy)
                        final_result_categories.append(result_categories[idx])
                        final_result_confidences.append(confidence)

                        original_frame = cv2.rectangle(original_frame, (xyxy[0], xyxy[1] ), (xyxy[2], xyxy[3]), (0, 255, 0), 1)
                                                        
                if(len(final_result_categories))>0:                                                      
                    status = f'{self.source_name} can see {final_result_categories}'
                else:
                    status = f'{self.source_name} can see nothing'

                self.status_shared_memory.buf[:len(status)] = status.encode('utf-8')
                self.status_shared_memory.buf[99] = len(status)
                
                if(len(final_result_categories)>0):
                    #cv2.imwrite(f'output_{self.source_name}.png', original_frame)

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
                            self.results_queue.put(( final_result_categories, final_result_boxes ), block=False )
                            self.logger.debug(f"Added {final_result_categories} to results queue")
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

def run(source_name, source_path, buffer_lock, source_fps, inference_fps, infer_shared_memory, status_shared_memory, results_queue, keep_running):

    output_image_queue = Queue(5)
    inference_engine = DarkCyanObjectDetection(source_name, inference_fps, output_image_queue, infer_shared_memory, buffer_lock, status_shared_memory, results_queue, keep_running)
    image_stream = DarkCyanVideoSource(source_name, source_path, source_fps, output_image_queue, inference_engine.imgsz, keep_running)

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


