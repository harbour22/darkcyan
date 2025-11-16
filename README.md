# darkcyan-tools

## yolo export for macos coreml
```bash
yolo export model=yolov8_4.11_large-det.pt format=coreml imgsz=640,480  nms=true
```
## apple gstreamer pipeline
```bash
gst-launch-1.0 rtspsrc location=rtsp://<uid>:<pwd>@reolink4k-frontgarden.private:554/h265Preview_01_main latency=200 drop-on-latency=true ! rtph265depay ! h265parse ! vtdec_hw ! videorate ! video/x-raw,framerate=10/1 ! videoscale ! video/x-raw,width=640,height=480 ! autovideosink

gst-launch-1.0 rtspsrc user-id=<pwd> user-pw=<uid> location=rtsp://reolink4k-frontgarden.private:554/h265Preview_01_main latency=200 drop-on-latency=true ! rtph265depay ! h265parse ! vtdec_hw ! videorate ! video/x-raw,framerate=10/1 ! videoscale ! video/x-raw,width=640,height=480 ! fakesink
```

## Potential Poetry Fix on Linux
```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

## Testing app

For the frontend we need to run this command
```bash
npm run dev
```
and for the backend

```bash
uvicorn server:app --reload
```