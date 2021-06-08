#!/usr/bin/env python3
from bleak import BleakClient
from bleak import discover
from neosensory_python import NeoDevice
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import asyncio
import time

def notification_handler(sender, data):
    print("{0}: {1}".format(sender, data))

nnPathDefault = str((Path(__file__).parent / Path('face-detection-retail-0004.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=False)
args = parser.parse_args()

if not Path(nnPathDefault).exists():
    import sys
    print(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setFps(40)

# Define a neural network that will make predictions based on the source frames
nn = pipeline.createMobileNetDetectionNetwork()
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(args.nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
camRgb.preview.link(nn.input)

# Create outputs
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
if args.sync:
    nn.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

nnOut = pipeline.createXLinkOut()
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# MobilenetSSD label texts
labelMap = ["", "face"]


async def initialize_buzz(blacklist_addresses = None):
    buzz_addr = "not yet set"  # e.g. "EB:CA:85:38:19:1D"
    devices = await discover()
    for d in devices:
        if str(d).find("Buzz") > 0:
            print("    Found a Buzz! " + str(d) +
             "\r\nAddress substring: " + str(d)[:17])
            # set the address to a found Buzz
            if(blacklist_addresses is None): # register the first buzz you find if there's no blacklisted addresses
                buzz_addr = str(d)[:17]
                break
            elif(not(str(d)[:17] in blacklist_addresses)): # otherwise the first one you find not in the blacklist
                buzz_addr = str(d)[:17]
                break
            else:
                print("    Ignoring already registered Buzz: " + str(d)[:17])

    if(buzz_addr == "not yet set"):
        print("No buzzes found. Dividing by zero. Get ready to explode")
        exit()

    client = BleakClient(buzz_addr)
    
    try:
        await client.connect()

        my_buzz = NeoDevice(client)

        await asyncio.sleep(1)

        x = await client.is_connected()
        print("    Connection State: {0}\r\n".format(x))    

        await asyncio.sleep(1)

        await my_buzz.request_developer_authorization()

        await my_buzz.accept_developer_api_terms()

        await my_buzz.pause_device_algorithm()

        await my_buzz.clear_motor_queue()
    

        async def send_vibration_frame(motor_vibrate_frame):
            try:
                await my_buzz.vibrate_motors(motor_vibrate_frame)                
            except KeyboardInterrupt:
                await my_buzz.resume_device_algorithm()
                exit()
        return send_vibration_frame, buzz_addr

    except Exception as e:
        print(e)
        client.disconnect()
        exit()

  

async def run(loop):
# Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        # Start pipeline

        #await initialize_buzz()
        (send_vibration_frame, buzz_address) = await initialize_buzz()
        await send_vibration_frame([200, 50, 50, 0])

        device.startPipeline()

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        startTime = time.monotonic()
        counter = 0
        detections = []
        frame = None

        def frameNorm(frame, bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        while True:
            if args.sync:
                # Use blocking get() call to catch frame and inference result synced
                inRgb = qRgb.get()
                inDet = qDet.get()
            else:
                # Instead of get (blocking), we use tryGet (nonblocking) which will return the available data or None otherwise
                inRgb = qRgb.tryGet()
                inDet = qDet.tryGet()

            if inRgb is not None:
                frame = inRgb.getCvFrame()
                cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

            if inDet is not None:
                detections = inDet.detections
                counter += 1

            if frame is not None:
                for detection in detections:
                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                cv2.imshow("name", frame)                

            if cv2.waitKey(1) == ord('q'):
                break        
            
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
