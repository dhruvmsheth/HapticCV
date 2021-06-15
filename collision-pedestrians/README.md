# Pedestrian and Vehicle Detection Model
### Pedestrian-and-vehicle-detector-adas which is a low-latency, high- accuracy model used to detect Pedestrians and Vehicles on Road trained on ADAS dataset for a better classified use-case

How is it different than face-detection? Well, this model features an additional class trained on an ADAS(Advanced Driver Assistance Systems) dataset, which includes Vehicles and Automobiles. So, while face-detection is meant for the category where the population of the region where you're driving a cycle is more, and you're more afraid of pedestrian accidents than automobile accidents. While, this model is more robust, and can be used in a wider use-case. So, lets say that you're in a region where the accident rate is high with automobiles as well as pedestrians. Here, the pedestrian and vehicle model suits the best!

Convolutional Architecture of the Model 
![image](https://user-images.githubusercontent.com/67831664/122098346-5c763400-ce2e-11eb-94d1-f6d15d2be912.png)
-

Pedestrian and vehicle detection network based on MobileNet v1.0 + SSD. MobileNet is a type of convolutional neural network designed for mobile and embedded vision applications. They are based on a streamlined architecture that uses depthwise separable convolutions to build lightweight deep neural networks that can have low latency for mobile and embedded devices.

Specifications of the Model -
![image](https://user-images.githubusercontent.com/67831664/122098381-66983280-ce2e-11eb-8643-49badfe7cc9b.png)

Tested on challenging internal datasets with 1001 pedestrian and 12585 vehicles to detect.
An example of the pedestrian and vehicle adas model inference results on an example image -
![image](https://user-images.githubusercontent.com/67831664/122098403-6bf57d00-ce2e-11eb-87ca-97f25782214f.png)

So, this definitely demonstrates how accurate this model is! This model potrays around 14-15 FPS as compared to 30 FPS of face-detection model. But, as I said, it depends on the use-case. this model does some additional work as compared to the first model, also has high accuracy.

Here's an example of person detection demo:
![image](https://user-images.githubusercontent.com/67831664/122098495-8cbdd280-ce2e-11eb-99c7-c4e662ca0b4f.png)

If the Person is closer than a specified threshold, the detection turns red, indicating that this might lead to a collision.
![image](https://user-images.githubusercontent.com/67831664/122098518-934c4a00-ce2e-11eb-9caf-9a5eafe95e11.png)

The Yellow detection indicates that the Person is a bit far away from the given threshold, but may cause an accident if it comes any closer. Similarly, a green detection indicates that everything's safe, enjoy your ride! These detections are converted to increasing intensities of motor on buzz. So closer the person is, the higher will be the frequesncy and intensity of vibration of motor. This can be heard in the video below.
Video - https://youtu.be/XbszFFpESYg
