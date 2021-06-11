# Face Detection Model integration with Buzz

### face-detection-retail-0004 Model, which is a state-of-the-art face detection model that performs inference at high FPS, with low energy consumption suited for our use-case.

This is a pretrained OpenVINO Model, which is an Optimised MobileNet SSD Model used to run on Edge AI devices.
Convolutional Architecture of the Model we'll be using -
The network features a default MobileNet backbone that includes depth-wise convolutions to reduce the amount of computation for the 3x3 convolution block.

![image](https://user-images.githubusercontent.com/67831664/121692483-c1a3ef80-cae5-11eb-9763-f357caa452f3.png)


Face detector based on SqueezeNet light (half-channels) as a backbone with a single SSD for indoor/outdoor scenes shot by a front-facing camera. The backbone consists of fire modules to reduce the number of computations. The single SSD head from 1/16 scale feature map has nine clustered prior boxes.

![image](https://user-images.githubusercontent.com/67831664/121692523-cb2d5780-cae5-11eb-951b-6833969b07a1.png)


These images show how Face detection Model accurately detects my face. This model is running on the OAK-D, and the left preview shows the Depth Map of the corresponding RGB image. Bounding Boxes are resized to be displayed on Depth Map as well, for a better view of the Detected object in Spatial Plane. Also, the X,Y and Z coordinates will be further used to send signals to the Buzz to indicate which motor has to be vibrated. This will be shown ahead.
![image](https://user-images.githubusercontent.com/67831664/121692576-d7b1b000-cae5-11eb-8691-d834104a33bb.png)
![image](https://user-images.githubusercontent.com/67831664/121692635-e4360880-cae5-11eb-99fa-f010dd8d0433.png)
![image](https://user-images.githubusercontent.com/67831664/121692655-eb5d1680-cae5-11eb-9a67-b3ece235d468.png)


The pedestrian detection and buzz alerting demo would work out in the following manner -


OAK-D retrieves spatial information from each object detected in terms of metre, and then, I use this value to trigger the intensity of vibration of motors on buzz. So, as the person approaches closer to the bicycle, the intensity of the motor on buzz increases, and as it moves away, the intensity decreases. The formula used to map the intensity of buzz motors from 1 to 255 is as follows -

Here, int(detection.spatialCoordinates.z) is in millimetres. The OAK-D cannot compute depth information for distances less than or equal to 0.35metre. 

![image](https://user-images.githubusercontent.com/67831664/121692719-fdd75000-cae5-11eb-8a88-2eb563d53748.png)

