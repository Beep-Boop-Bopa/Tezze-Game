import numpy as np
import cv2
import math
import time
import statistics as st
import threading as th
import imutils
from Segment import Segment,Circle
import os

def Var_calc(Img,STD_array,start,end):
    for i in range(start,end):
        STD_array[i]=[i,cv2.meanStdDev( Img[i] )[1] ]

class Game:
    def __init__(self):
        self.Canny_Upper=100
        self.Canny_Lower=50
        self.Scale=1
        self.FullFrame=0
        self.GameFrame=0
        self.Cardinality=1 #0 means flat, 1 means vertical
        self.game_Center=0
        self.FullFrame_center=0
        self.Circles=[]  #2 circles put in here
        self.Max_radius=0#Reassigned during Initialisation
        self.Min_radius=0#Reassigned during Initialisation
        self.colors=[]   #Of length 4 for the 4 colors we have
        self.lobes=[]    #One sublist for each circle
        self.Dist_Circles=0
    def setup(self, img):
        self.ScaleDown(img)
        self.CutGameBoard()
        self.findColors()#Later Parts are relying on this to work properly..but the segments in the same lob will get the same answer even if it is wrong in general
        self.FindCircles()
        self.FindSegmentsInEachCircle()
        self.findCommonSegments()
#        self.draw()

    def ExtractLobes(self):
        for circle in self.Circles:
            for lobe in self.lobes:
                if(circle.Intersect(lobe,self.Dist_Circles,self.getEuclidDist(circle.getGameCenter(),lobe.getGameCenter()))):
                    lobe.setID(circle.getID())
                    lobe.setIndvCenter((lobe.getGameCenter()[0]-circle.getGameCenter()[0]+circle.getIndvCenter()[0],lobe.getGameCenter()[1]-circle.getGameCenter()[1]+circle.getIndvCenter()[1]))
                    circle.addLobe(lobe)
                    cv2.circle(circle.getImage(),lobe.getIndvCenter(),lobe.getRadius(),(255,255,255),1)

    def findSegmentsInGrameFrame(self):
        blank_image = np.zeros((self.GameFrame.shape[1],self.GameFrame.shape[0],3), np.uint8)
        for circle in self.Circles:
            circle.draw(blank_image)

    def findCommonSegments(self):
        C1B = np.zeros(self.Circles[0].getImage().shape, np.uint8)
        C2B = np.zeros(self.Circles[1].getImage().shape, np.uint8)
        for Segment1 in self.Circles[0].getSegments():
            for Segment2 in self.Circles[1].getSegments():
                if(Segment1.equals(Segment2)):
                    cv2.drawContours(C1B,[Segment1.getContour()],-1,[255,255,255],cv2.FILLED);
                    cv2.drawContours(C2B,[Segment2.getContour()],-1,[255,255,255],cv2.FILLED);
                    break
        cv2.imshow("C1B",C1B)
        cv2.imshow("C2B",C2B)
        cv2.waitKey(0)

    def FindSegmentsInEachCircle(self):
        id=0
        for circle in self.Circles:
            img=cv2.cvtColor(circle.getImage(), cv2.COLOR_BGR2GRAY)   #We can change this to self.getGameFrame? 
            ret,img = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours=sorted(contours, key=lambda z: cv2.contourArea(z),reverse=True)

            #will have to filter though the contours whose centroids lie within the radius of the circle???
            LargestContour=contours[0]
            Distances=[]
            for Sublist in LargestContour:
                Distances.append(self.getEuclidDist(Sublist[0],circle.getIndvCenter()))
            #The radius of the circles is modified and set to the largest distance between the current center point and all of the pts on the contour
            circle.setRadius(sorted(Distances, key=lambda z: z,reverse=True)[0])

            #Might need to move this section before the one chunk above?
            New_Img =  np.zeros((circle.getImage().shape[0],circle.getImage().shape[1],3), np.uint8)
            CircleImg =  New_Img.copy()
            cv2.circle(CircleImg,circle.getIndvCenter(),circle.getRadius(),(255,255,255),-1)

            for c in contours:
                if(cv2.contourArea(c)>5):
                    Left = tuple(c[c[:, :, 0].argmin()][0])
                    Right = tuple(c[c[:, :, 0].argmax()][0])
                    Top = tuple(c[c[:, :, 1].argmin()][0])
                    Bottom = tuple(c[c[:, :, 1].argmax()][0])

                    OverLap =  np.zeros((circle.getImage().shape[0],circle.getImage().shape[1],3), np.uint8)#Change to self.GameFrame
                    cv2.drawContours(OverLap,[c],-1,[255,255,255],cv2.FILLED);
                    OverLap=cv2.bitwise_and(OverLap,CircleImg)
                    OverLap=cv2.cvtColor(OverLap, cv2.COLOR_BGR2GRAY)
                    contours1, hierarchy = cv2.findContours(OverLap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    OverLapArea=0
                    for cnts in contours1:
                        OverLapArea+=cv2.contourArea(cnts)
                    if(OverLapArea/cv2.contourArea(c)>0.3):
                        Contour_img=circle.getImage()[Top[1]:Bottom[1],Left[0]:Right[0]].copy()
                        Sum=[0,0,0]
                        No=0
                        for x in range(Top[1],Bottom[1]):
                            for j in range(Left[0],Right[0]):
                                if(cv2.pointPolygonTest(c,(j,x),0)>=0):
                                    Sum[0]+=Contour_img[x-Top[1]][j-Left[0]][0]
                                    Sum[1]+=Contour_img[x-Top[1]][j-Left[0]][1]
                                    Sum[2]+=Contour_img[x-Top[1]][j-Left[0]][2]
                                    No+=1
                        mean = [Sum[0]/No,Sum[1]/No,Sum[2]/No]
                        Color=self.findClosetColor(mean)
                        M = cv2.moments(c)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        Seg=Segment((cX,cY),Color,c,id,circle.getID(),circle)##Segment ID will depend on a lot more than this?????
                        id+=1
                        circle.addSeg(Seg)
                        cv2.drawContours(New_Img,[c],-1,[255,255,255],cv2.FILLED)
            circle.setImage(cv2.bitwise_and(New_Img,circle.getImage()))#A little bit of code here

    def FindCircles(self):    
        Circles=self.HoughCircles(self.GameFrame,self.Min_radius,self.Max_radius,self.game_Center)
        if(Circles is not None):
            sorted(Circles, key=lambda Circle: Circle.getDistance())
            self.Circles=Circles[0:2]
            for circle in self.Circles:
                #cv2.circle(self.GameFrame, circle.getGameCenter(), circle.getRadius(), [255,255,255], 1)
                y0=circle.getGameCenter()[1]-circle.getRadius()-5 if(circle.getGameCenter()[1]-circle.getRadius()>0) else 0
                y1=circle.getGameCenter()[1]+circle.getRadius()+5 if(circle.getGameCenter()[1]+circle.getRadius()<self.GameFrame.shape[0]) else self.GameFrame.shape[0]###not correcy
                x0=circle.getGameCenter()[0]-circle.getRadius()-5 if(circle.getGameCenter()[0]-circle.getRadius()>0) else 0
                x1=circle.getGameCenter()[0]+circle.getRadius()+5 if(circle.getGameCenter()[0]+circle.getRadius()<self.GameFrame.shape[1]) else self.GameFrame.shape[1]
                Circle=self.GameFrame[y0:y1,x0:x1]
                blank_image = np.zeros((Circle.shape[1],Circle.shape[0],3), np.uint8)
                newCenter=(circle.getIndvCenter()[0]+5,circle.getIndvCenter()[1]+5)
                cv2.circle(blank_image, newCenter, circle.getRadius(), [255,255,255], -1)
                circle.setImage(cv2.bitwise_and(blank_image,Circle))
            self.Circles=sorted(self.Circles, key=lambda x: x.getGameCenter()[0])
            self.Circles[0].setID(0)
            self.Circles[1].setID(1)
            self.Dist_Circles=self.getEuclidDist(self.Circles[0].getGameCenter(),self.Circles[1].getGameCenter())
#            self.lobes=Circles[2:len(Circles)]
#            for circle in self.lobes:
#                cv2.circle(self.GameFrame, circle.getGameCenter(), circle.getRadius(), [255,255,255], 1)

        else:
            print("Not enough Circles found")

    def CutGameBoard(self):
        Canny=self.getCannyImage(self.FullFrame)
        contours, hierarchy = cv2.findContours(Canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = np.vstack(contours[i] for i in range(len(contours)) if( cv2.contourArea(contours[i])>50))

        Left = tuple(c[c[:, :, 0].argmin()][0])
        Right = tuple(c[c[:, :, 0].argmax()][0])
        Top = tuple(c[c[:, :, 1].argmin()][0])
        Bottom = tuple(c[c[:, :, 1].argmax()][0])

        self.GameFrame=self.FullFrame[Top[1]:Bottom[1],Left[0]:Right[0]].copy()
        #Making the Game horizontal if vertical
        if(Bottom[1]-Top[1]<Right[0]-Left[0]):
            self.Cardinality=0
        else:
            self.FullFrame = imutils.rotate_bound(self.FullFrame, 90)
            self.GameFrame = imutils.rotate_bound(self.GameFrame, 90)

        #Finding some extra parameters
        self.game_Center=(self.GameFrame.shape[1]//2,self.GameFrame.shape[0]//2)
        x=self.game_Center[0]+Left[0]
        y=self.game_Center[1]+Top[1]
        self.FullFrame_center=(x,y)

        self.Min_radius=self.GameFrame.shape[0]//3
        self.Max_radius=self.GameFrame.shape[1]//2

        DeNoised_pic = self.GameFrame.copy()
        DeNoised_pic = DeNoised_pic.reshape((-1,3))

        No_OF_threads=3
        previous_Index=0
        length=len(DeNoised_pic)//No_OF_threads
        Remainder=len(DeNoised_pic)%No_OF_threads
        Threads={}
        STD_RGB=[None]*len(DeNoised_pic)
        for i in range(No_OF_threads):
            start_Index=previous_Index
            last_index=previous_Index+length if(i!=0) else previous_Index+length+Remainder
            previous_Index=last_index
            Threads[i]=th.Thread(target=Var_calc,args=[DeNoised_pic,STD_RGB,start_Index,last_index])
            Threads[i].start()

        for i in range(No_OF_threads):
            Threads[i].join()

        STD_RGB=sorted(STD_RGB, key=lambda x: x[1])

        for i in range(round(len(STD_RGB)*0.375)):
            DeNoised_pic[STD_RGB[i][0]]=[0,0,0]

        self.GameFrame = DeNoised_pic.reshape((self.GameFrame.shape))

    def findClosetColor(self,color):
        Dist=[]
        for i in range(len(self.colors)):
            Dist.append([ self.getEuclidDist(color,self.colors[i],1),self.colors[i]])
        Dist=sorted(Dist, key=lambda z: z[0])
        return Dist[0][1]


    def findColors(self):
        img=self.GameFrame.copy()
        Z = img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        Sums=[]
        for i in range(5):
            Sums.append([i,sum(center[i])])
        Sums=sorted(Sums, key=lambda z: z[1],reverse=True)

        self.colors=[]

        for i in range(4):
            self.colors.append(center[Sums[i][0]])
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        #cv2.imshow('res2',res2)        
        #cv2.moveWindow("res2", 500, 150)
        
    def draw(self):
        cv2.imshow("Board",self.GameFrame)
        cv2.moveWindow("Board", 150, 150)
        cv2.imshow("Full",self.FullFrame)
        cv2.moveWindow("Full", 800, 150)
        cv2.imshow("Circle1",self.Circles[0].getImage())
        cv2.moveWindow("Circle1", 500, 50)
        cv2.imshow("Circle2",self.Circles[1].getImage())
        cv2.moveWindow("Circle2", 500, 350)
        cv2.waitKey(0)

    def ScaleDown(self,img,Scale=8):
        self.Scale=1/Scale
        self.FullFrame=cv2.resize(cv2.imread(img),None,fx=self.Scale,fy=self.Scale)

    def getEuclidDist(self,Point1,Point2,ThreeD_bool=0):
        X_diff=(Point1[0]-Point2[0])**2
        Y_diff=(Point1[1]-Point2[1])**2
        Sum=X_diff+Y_diff
        if(ThreeD_bool):
            Z_diff=(Point1[2]-Point2[2])**2
            Sum+=Z_diff
        return (Sum)**.5

    def getCannyImage(self,image,Upper=0,Lower=0,Canny=1,data=0):
        image=image.copy()
        if(image.shape[2]==3):
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if(Upper==0):
            Upper=self.Canny_Upper
            Lower=self.Canny_Lower
        if(Canny):
            GREY=cv2.GaussianBlur(image,(3,3),0)
            return cv2.Canny(GREY,Lower,Upper)
        else: #For thresholded image
            _,image = cv2.threshold(image,5,255,cv2.THRESH_BINARY_INV)
            blank_image = np.zeros(image.shape, np.uint8)
            cv2.circle(blank_image,data[1],data[0]-8,(255,255,255),-1)
            image=cv2.bitwise_and(blank_image,image)
            kernel = np.ones((3,3),np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image=cv2.Canny(image,Lower,Upper)
            #image = cv2.erode(image,kernel,iterations = 1)
            cv2.imshow("Voila",image)
            cv2.waitKey(0)
            return image

    def HoughCircles(self,image,min,max,center,Canny=1,data=0):
        Canny=self.getCannyImage(image,0,0,Canny,data)
        circles = cv2.HoughCircles(Canny,cv2.HOUGH_GRADIENT,1,50,
                            param1=10,param2=10,minRadius=min,maxRadius=max)
        if(circles is not None):
            circles = np.int32(np.around(circles))
            Circles=[]
            for i in circles[0,:]:
                Circles.append(   Circle(     (i[0],i[1]),   i[2],    self.getEuclidDist(center,(i[0],i[1])),  -1)  )
            return Circles
        else:
            return None


def main():
    Tezze=Game();
    str="Photos"
    arr = os.listdir(str)
    for photo in arr:
        Photo=str+"/"+photo
        Tezze.setup(Photo);

main()