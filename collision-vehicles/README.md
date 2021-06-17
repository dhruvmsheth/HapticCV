# Vehicle Detection ADAS model to detect Vehicles.

vehicle-detection-adas demo to detect different types of Vehicles like Cars, truck, lorries, vans etc. Also useful for motorcycle detection. This demo can be used in Highways or areas prone to automobile-cycle accidents.
How is this model different from the other two? This model doesn't detect pedestrians, but detects the most proominient threat, vehicles. Trucks and Cars always have been the source to most cycle accidents. Many of such accidents take place on crossroads, flyovers or roundabouts. Sometimes, even near pavements. This model also, helps the cyclist navigate better in heavy traffic conditions and understand the rear view of the rider through haptic impulses.
Imagine the situation to be something similar to this.

![image](https://user-images.githubusercontent.com/67831664/122394867-0cb17d00-cf94-11eb-928b-d3ec923788d1.png)


The Field Of View(FOV) for the OAK-D is pretty large, allowing the camera to observe the vehicles in multi-directions in the back.
- Field of View of RGB camera - 68.7938 deg
- Field of View of Mono Cameras - 73.5 deg (nearly 1280 pixels)

This is a vehicle detection network based on an SSD framework with tuned MobileNet v1 as a feature extractor. Since this uses a MobileNet v1 Architecture, the Convolutional Architecture is same as the second model. They are based on a streamlined architecture that uses depthwise separable convolutions to build lightweight deep neural networks that can have low latency for mobile and embedded devices. -
![image](https://user-images.githubusercontent.com/67831664/122395145-54380900-cf94-11eb-91bd-bf7933e66c16.png)


Tested on a challenging internal dataset with 3000 images and 12585 vehicles to detect.


Example -
![image](https://user-images.githubusercontent.com/67831664/122395184-5bf7ad80-cf94-11eb-8644-3b8b4075a560.png)


Replicating the Demo:

Just as all other demos, replicating the demo here, is pretty straightforward.

```python
$ cd collision-vehicles/
$ python3 buzz_sptial_car.py
```

That's it? Yes! This runs the demo for you, once you have cloned the repo following the instructions in the previous section.

Since Vehicle detection wasn't possible indoors like the other two, finally I decided to go out and get a Roadtest! So, the car detection demo was a part of the the actual roadtest, and I have picked a few examples from the demo to display the accuracy of the model running real time on OAK-D:

![image](https://user-images.githubusercontent.com/67831664/122395233-6619ac00-cf94-11eb-96c5-e3d2b3b13777.png)
![image](https://user-images.githubusercontent.com/67831664/122395266-6ca82380-cf94-11eb-8f0f-a8180eb7a9fe.png)
