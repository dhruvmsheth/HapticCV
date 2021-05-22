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
import sys

'''
Spatial detection network demo.
    Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
'''

def notification_handler(sender, data):
    print("{0}: {1}".format(sender, data))

# MobilenetSSD label texts
labelMap = ["", "person"]
syncNN = True

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('face-detection-retail-0004.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
colorCam = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutRgb = pipeline.createXLinkOut()
xoutNN = pipeline.createXLinkOut()
xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")


colorCam.setPreviewSize(300, 300)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Setting node configs
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(255)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Create outputs

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

colorCam.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    colorCam.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)


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

        async def stop_vibration_frame():
            try:
                time.sleep(0.01)
                await my_buzz.stop_motors()                 
            except KeyboardInterrupt:
                #await my_buzz.resume_device_algorithm()
                exit()    

        async def send_vibration_frame(motor_vibrate_frame):
            try:
                await my_buzz.vibrate_motors(motor_vibrate_frame)                
            except KeyboardInterrupt:
                await my_buzz.resume_device_algorithm()
                exit()
        return send_vibration_frame, buzz_addr, stop_vibration_frame

    except Exception as e:
        print(e)
        client.disconnect()
        exit()
   

async def run(loop):
# Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        neo = False
        #await initialize_buzz()
        (send_vibration_frame, buzz_address, stop_vibration_frame) = await initialize_buzz()
        await send_vibration_frame([255, 255, 255, 255])
        # Start pipeline
        device.startPipeline()

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        frame = None
        detections = []

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)

        while True:
            inPreview = previewQueue.get()
            inNN = detectionNNQueue.get()
            depth = depthQueue.get()

            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            frame = inPreview.getCvFrame()
            depthFrame = depth.getFrame()

            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            detections = inNN.detections
            if len(detections) != 0:
                boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                roiDatas = boundingBoxMapping.getConfigData()

                for roiData in roiDatas:
                    roi = roiData.roi
                    roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                    topLeft = roi.topLeft()
                    bottomRight = roi.bottomRight()
                    xmin = int(topLeft.x)
                    ymin = int(topLeft.y)
                    xmax = int(bottomRight.x)
                    ymax = int(bottomRight.y)

                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


            # If the frame is available, draw bounding boxes on it and show the frame
            height = frame.shape[0]
            width  = frame.shape[1]
            if frame is not None:
                for detection in detections:
                    neo = True
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = labelMap[detection.label]
                    except:
                        label = detection.label
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    Zdepth = int((detection.spatialCoordinates.z) / 21)
                    ZDepth = Zdepth + 50
                    Xframe = int(detection.spatialCoordinates.x)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
 
                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("rgb", frame)

            if neo is True:
                for detection in detections:
                    if -400 < Xframe < -150:
                        await send_vibration_frame([ZDepth, 0, 0, 0])
                        await stop_vibration_frame()
                    if -150 < Xframe < 0:
                        await send_vibration_frame([0, ZDepth, 0, 0])
                        await stop_vibration_frame()

                    if 0 < Xframe < 150:
                        await send_vibration_frame([0, 0, ZDepth, 0])
                        await stop_vibration_frame()

                    if 150 < Xframe < 400:
                        await send_vibration_frame([0, 0, 0, ZDepth])
                        await stop_vibration_frame()



            if cv2.waitKey(1) == ord('q'):
                break
            
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
