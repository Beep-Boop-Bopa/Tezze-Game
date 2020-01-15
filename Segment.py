import math
import numpy as np
import cv2
import imutils
import sys

class Segment:
    def __init__(self,centroid,contour,Image=0,color=-1,circleNo=-1,circle=-1,id=-1):
        self.Centroid=centroid
        self.Color=color
        self.Contour=contour
        self.ID=id
        self.CircleNo=circleNo
        self.Circle=circle
        self.Image=Image

    def __eq__(self, other): 
        if not isinstance(other, Segment):
            raise Exception('Segment being compared with another type {}'.format(type(other)))
        elif(self.ID == other.ID):
            return True
        return False

    def setImage(self,img):
        self.Image=img

    def setCentroid(self,centroid):
        self.Centroid=centroid

    def setColor(self,color):
        self.Color=color

    def setContour(self,cnts):
        self.Contour=cnts.copy()

    def setCircleNo(self,circleNo):
        self.CircleNo=circleNo

    def setCircle(self,circle):
        self.Circle=Circle

    def setID(self,id):
        if(self.ID==-1):
            self.ID=id
        else:
            raise Exception('Reassigning of Ids of the Segment')
    def getCentroid(self):
        return self.Centroid
    def getColor(self):
        return self.Color
    def getID(self):
        return self.ID
    def getContour(self):
        return self.Contour
    def getCircleNo(self):
        return self.CircleNo
    def getCircle(self):
        return self.Circle
    def getImage(self):
        return self.Image

    def RotatePt(self,o,p,r):
        p = np.atleast_2d(p)
        p=np.squeeze((r @ p.T - r @ o.T + o.T).T)
        p=[int(p[0]),int(p[1])]
        p = np.atleast_2d(p)
        return p

    def rotateSeg(self,Center,Down=0):
        Angle=60
        if(Down):
            Angle*=-1
        Angle = np.deg2rad(Angle)
        r = np.array([[np.cos(Angle), -np.sin(Angle)],[np.sin(Angle),  np.cos(Angle)]])
        o = np.atleast_2d(Center)

        pts=[]
        old=self.getContour()
        for p in self.getContour():
            pts.append(self.RotatePt(o,p[0],r))
        pts=np.asarray(pts)
        self.setContour(pts)

        M = cv2.moments(self.getContour())
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        Center=(cX,cY)

 #       OldCenter=self.getCentroid()
#        if(self.getID()==14 or self.getID()==16 or self.getID()==19 or self.getID()==20 or self.getID()==21 or self.getID()==27 or self.getID()==28):
#            print(OldCenter[0]," ",Center[0]," ",OldCenter[1]," ",Center[1])
  #          print(self.getContour()[0][0],old[0][0])
        self.setCentroid(Center)

    def ImageRefresh(self):
            cnt=self.getContour()

            Left=tuple(cnt[cnt[:, :, 0].argmin()][0])
            Right=tuple(cnt[cnt[:, :, 0].argmax()][0])
            Top=tuple(cnt[cnt[:, :, 1].argmin()][0])
            Bottom=tuple(cnt[cnt[:, :, 1].argmax()][0])

            shape=(Bottom[1]-Top[1]+40,Right[1]-Left[1]+40)
            blank_image = np.zeros(shape, np.uint8)

            OldCenter=self.getCentroid()
            Blank_Center=(shape[1]//2,shape[0]//2)
            Change=[    Blank_Center[0]-OldCenter[0],Blank_Center[1]-OldCenter[1]    ]
            Cnt=cnt+Change
            cv2.drawContours(blank_image,[Cnt],-1,[255,255,255],-1)
            self.setImage(blank_image)

            return Top,Bottom,Left,Right,cnt

class Circle():
    def __init__(self,gamecenter,radius,distanceFromFrame,id=-1):
        self.GameCenter=gamecenter
        self.Radius=int(radius)
        self.ID=id #Make sure 0 is the left, 1 is the right..-1 none? and 2 for both?
        self.DistanceFromFrame=distanceFromFrame #Make sure 0 is the left, 1 is the right
        self.Segments=[]
    def setgameCenter(self,gamecenter):
        self.GameCenter=gamecenter
    def setRadius(self,radius):
        self.Radius=int(radius)
    def getDistance(self):
        return self.DistanceFromFrame
    def getGameCenter(self):
        return self.GameCenter
    def getRadius(self):
        return self.Radius
    def getID(self):
        return self.ID
    def setID(self,iD):
        self.ID=iD
    def getSegments(self):
        return self.Segments
    def draw(self,frame):
        for seg in self.Segments:
            cv2.drawContours(frame,[seg.getContour()],-1,seg.getColor(),cv2.FILLED)
    def addSeg(self,Seg):
        if(Seg not in self.Segments):
            Seg.setCircleNo(self.getID())
            Seg.setCircle(self)
            self.Segments.append(Seg)
    def RemoveSeg(self,seg1):
        i=0
        for seg2 in self.Segments:
            if(seg1==seg2):
                self.Segments.pop(i)
                break
            i+=1