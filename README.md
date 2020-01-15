# Tezze-Game

![image](https://user-images.githubusercontent.com/42900802/72404582-d4b16500-3723-11ea-948f-3e17b50af35f.png)

![image](https://user-images.githubusercontent.com/42900802/72404480-81d7ad80-3723-11ea-85e3-60b345bf6b61.png)

Using openCV, python to help solve Tezze puzzles. The photos of the tezze are taken via a gopro. A demo could be seen [here](https://www.youtube.com/watch?v=5TFHbwq4wUM&feature=youtu.be)-don't mind skipping about half a minute of this demo.

### Main idea:
After a photo is taken, we find the circles and the individual segments within that circle. Then those individual segments could be rotated about their respective circle's center. After every rotation, a reassignment of the segments to the circles happens.

##### Inner Lobe:
![image](https://user-images.githubusercontent.com/42900802/72405780-0c221080-3728-11ea-9ea1-132a888d376e.png)

The area where the two circles overlap is the lob. In my code structure, this area is also considered to be a circle. The segments with in this circle could be rotated by either of the outer circles.

##### 60:
![image](https://user-images.githubusercontent.com/42900802/72406175-5657c180-3729-11ea-8103-80a15d61d79b.png)

A rotation is only of 60 degrees. If you rotate a circle, you need a complete inner lobe for the other circle to be able to rotate. A rotation from one complete inner lobe to another complete inner lobe is only physically possible when you rotates between lobes by an angle of 60 degrees.

### How to run?
1. Connect your PC to a gopro's wifi.
2. Go to the terminal and type "python3 Game.py"
3. Take a photo of the tezze.
4. Once the tezze is processed, you can use the keys "a","s","d","w". 

### Issues
![image](https://user-images.githubusercontent.com/42900802/72406796-7be5ca80-372b-11ea-8bc8-c047e6bb8eee.png)

I haven't been able to resolve these 2 issues:

1. Noise: After a couple of rotaions, the segments lose their shape. This is because of approximations in shape processing around rotation which leads to some noise which adds up over time. 

2. Uneven rotation: All the segments within a circle do not rotate by the same amount. I know that there is some underlying data structure bug, but haven't been able to resolve it yet. 
