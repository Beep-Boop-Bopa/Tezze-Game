import math
import numpy as np
import cv2

class Segment:
    def __init__(self,centroid,color,contour,id,circleNo,circle,lobe=-1):
        self.Centroid=centroid
        self.Color=color
        self.Contour=contour
        self.ID=id
        self.Lobe=lobe
        self.CircleNo=circleNo
        self.Circle=circle


    def setCentroid(self,centroid):
        self.Centroid=centroid
    def setLobe(self,lobe):
        self.Lobe=lobe
    def setCircleNo(self,circleNo):
        self.CircleNo=circleNo
    def setCircle(self,Circle):
        self.Circle=Circle
    def getCentroid(self):
        return self.Centroid
    def getColor(self):
        return self.Color
    def getLobe(self):
        return self.Lobe
    def getID(self):
        return self.ID
    def getContour(self):
        return self.Contour
    def getCircleNo(self):
        return self.CircleNo
    def getCircle(self):
        return self.Circle
    def getAngle(self):
        CircleCenter=self.Circle.getIndvCenter()
        SegmentCenter=self.getCentroid()
        return math.atan2(abs(CircleCenter[1]-SegmentCenter[1]),abs(CircleCenter[0]-SegmentCenter[0]))

    def equals(self,Segment):

        Angle1=self.getAngle()
        Angle2=Segment.getAngle()
        x_Diff1=self.getCentroid()[0]-self.Circle.getIndvCenter()[0]
        x_Diff2=Segment.getCentroid()[0]-Segment.Circle.getIndvCenter()[0]
        Y_Diff1=self.getCentroid()[1]-self.Circle.getIndvCenter()[1]
        Y_Diff2=Segment.getCentroid()[1]-Segment.Circle.getIndvCenter()[1]
        Color_sum1=int(self.getColor()[0])+int(self.getColor()[1])+int(self.getColor()[2])
        Color_sum2=int(Segment.getColor()[0])+int(Segment.getColor()[1])+int(Segment.getColor()[2])
        Area1=cv2.contourArea(self.getContour())
        Area2=cv2.contourArea(Segment.getContour())
        
        Upper=1.5
        Lower=0.5

#        print(" x_Diff1 ",x_Diff1," x_Diff2 ",x_Diff2," Y_Diff1 ",Y_Diff1," Y_Diff2 ",Y_Diff2," Color_sum1 ",Color_sum1," Color_sum2 ",Color_sum2," Area1 ",Area1," Area2 ",Area2)
#        print(np.sign(0))
        if(np.sign(x_Diff1)==1 and np.sign(x_Diff2)==-1):# and (np.sign(Y_Diff1)==np.sign(Y_Diff2) or (      np.sign(Y_Diff1)==0 or np.sign(Y_Diff2)==0     )  ) ):
#            if(abs(x_Diff1/x_Diff2)<Upper and abs(x_Diff1/x_Diff2)>Lower and abs(Y_Diff1/Y_Diff2)<Upper and abs(Y_Diff1/Y_Diff2)>Lower):
 #               if(Angle1/Angle2<Upper and Angle1/Angle2>Lower):
            if(Color_sum1==Color_sum2):
 #               if(Area1/Area2<Upper and Area1/Area2>Lower):
                if(Area1==Area2):
                    return True
        return False

    def FindLobe(self,lobe):
        pass
    def FindCircle(self,lobe):
        pass
    def IN(self,contour):
        pass

class Circle():
    def __init__(self,gamecenter,radius,distanceFromFrame,id,image=0):
        self.GameCenter=gamecenter
        self.Radius=int(radius)
        self.IndividualCenter=(radius,radius)
        self.ID=id #Make sure 0 is the left, 1 is the right..-1 none? and 2 for both?
        self.DistanceFromFrame=distanceFromFrame #Make sure 0 is the left, 1 is the right
        self.Image=image
        self.Segments=[]
        self.Lobes=[]
    def setgameCenter(self,gamecenter):
        self.GameCenter=gamecenter
    def setIndvCenter(self,individualCenter):
        self.IndividualCenter=individualCenter
    def setImage(self,image):
        self.Image=image
    def setRadius(self,radius):
        self.Radius=int(radius)
    def setID(self,id):
        if(self.ID==-1):
            self.ID=id
        elif(self.ID==0 or self.ID==1):
            self.ID=2
    def getDistance(self):
        return self.DistanceFromFrame
    def getGameCenter(self):
        return self.GameCenter
    def getIndvCenter(self):
        return self.IndividualCenter
    def getRadius(self):
        return self.Radius
    def getImage(self):
        return self.Image
    def getID(self):
        return self.ID
    def getSegments(self):
        return self.Segments

    def Intersect(self,lobe,Dist_Circles,Dist_lobe_circle):
        Radius_ratio=1.3
        Dist_ratio=1.3
        if(self.getRadius()/lobe.getRadius()<Radius_ratio and lobe.getRadius()/self.getRadius()<Radius_ratio):
            if(Dist_Circles/Dist_lobe_circle<Dist_ratio and Dist_lobe_circle/Dist_Circles<Dist_ratio):
                return True
        return False

    def draw(self,frame):
        pass

    def addSeg(self,Seg):
        self.Segments.append(Seg)

    def addLobe(self,lobe):
        self.Lobes.append(lobe)

    def removeLobe(self,lobe):
        self.Lobes.remove(lobe)

    def commonSegs(self,Circle):#Circle.CommonSegs(Lobe_Circle)
        pass#First find the overlapping area's contour. Then find the the lobes in that circle 