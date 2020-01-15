import numpy as np
import cv2
import math
import time
import statistics as st
import threading as th
import imutils
from Segment import Segment,Circle
import os
from goprocam import GoProCamera, constants
from time import time
import socket

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
        self.Segments=[]   #0 for left, 1 for right, 2 for center
        self.lobes=[]    #One sublist for each circle
        self.Dist_Circles=0
    def setup(self, img):
        self.ScaleDown(img)
        self.CutGameBoard()
        self.findColors()#Later Parts are relying on this to work properly..but the segments in the same lob will get the same answer even if it is wrong in general
        self.FindCircles()
        if(self.FindSegments()):
            while(True):
                self.draw()
                key=cv2.waitKey(0)
                if(key==ord('q')):
                    break
                if(key==ord('a')):
                    self.Rotate(0,0)
                elif(key==ord('d')):
                    self.Rotate(0,1)
                elif(key==ord('w')):
                    self.Rotate(1,0)
                elif(key==ord('s')):
                    self.Rotate(1,1)
                #print("self.Segments ", len(self.Segments), " self.Circles[0] ", len(self.Circles[0].getSegments()), " self.Circles[1] ", len(self.Circles[1].getSegments()), "self.Circles[2] ", len(self.Circles[2].getSegments()) )
        else:
            print("not enough Segments found")
    
    def Rotate(self,CircleID,Direction=0):
        List=[]
        for seg in self.Circles[CircleID].getSegments():
            OldCenter=seg.getCentroid()
            if(seg.getCircleNo()==CircleID):
                List.append(seg.getID())
                seg.rotateSeg(self.Circles[CircleID].getGameCenter(),Direction)
                self.AssignCircle(seg,-1,seg.getCircleNo())
                self.Segments[seg.getID()]=seg
            else:
                print("ShitShow1")
        for seg in self.Circles[2].getSegments():
            if(seg.getCircleNo()==2):
                List.append(seg.getID())
                seg.rotateSeg(self.Circles[CircleID].getGameCenter(),Direction)
                self.AssignCircle(seg,-1,2)
                self.Segments[seg.getID()]=seg
            else:
                print("ShitShow2")
        if(len(List)!=22):
            pass
        #print("Circle Assignment is off by ", 22-len(List))

    def draw(self):
        blank_image1 = np.zeros(self.GameFrame.shape, np.uint8)
        blank_image2 = np.zeros(self.GameFrame.shape, np.uint8)
        blank_image3 = np.zeros(self.GameFrame.shape, np.uint8)
        for seg in self.Segments:
            Center=(seg.getCentroid()[0],seg.getCentroid()[1])
            cv2.drawContours(blank_image1,[seg.getContour()],-1,seg.getColor(),cv2.FILLED)
            if(seg.getCircleNo()==0):
                cv2.putText(blank_image3,str(seg.getID()),Center,cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1,cv2.LINE_AA)
                cv2.drawContours(blank_image2,[seg.getContour()],-1,(0,0,255),cv2.FILLED)
            elif(seg.getCircleNo()==1):
                cv2.putText(blank_image3,str(seg.getID()),Center,cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1,cv2.LINE_AA)
                cv2.drawContours(blank_image2,[seg.getContour()],-1,(0,255,0),cv2.FILLED)
            elif(seg.getCircleNo()==2):
                cv2.putText(blank_image3,str(seg.getID()),Center,cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),1,cv2.LINE_AA)
                cv2.drawContours(blank_image2,[seg.getContour()],-1,(255,0,0),cv2.FILLED)
        cv2.imshow('Segments',blank_image1)
        cv2.imshow('Circle Allocation',blank_image2)
        cv2.imshow('IDs',blank_image3)
        cv2.moveWindow('Segments', 120, 200) 
        cv2.moveWindow('Circle Allocation', 880, 200)
        cv2.moveWindow('IDs', 500, 200)
#            if(seg.getID()==14 or seg.getID()==16 or seg.getID()==19 or seg.getID()==20 or seg.getID()==21 or seg.getID()==27 or seg.getID()==28):

    def AssignCircle(self,Seg,id=-1,old_crc=-1):
        Upper=0.95
        Lower=0.8
        ##Finding the right Stuff
        i=-1
        Dist1=self.getEuclidDist(Seg.getCentroid(),self.Circles[0].getGameCenter())/self.Circles[0].getRadius()
        Dist2=self.getEuclidDist(Seg.getCentroid(),self.Circles[1].getGameCenter())/self.Circles[1].getRadius()
        cnt=Seg.getContour()
        if(Dist1<Upper or Dist2<Upper):
            if(Dist1<Upper and Dist2<Upper):
                i=2
            elif(Dist1<Upper and Dist2>Lower):
                i=0
            elif(Dist2<Upper and Dist1>Lower):
                i=1
        if(i!=-1): #If it does belong to a circle somewhere
            if(id!=-1):#For initialisation
                Seg.setID(id)
                
                Top,Bottom,Left,Right,cnt=Seg.ImageRefresh()
                Sum=[0,0,0]
                No=0
                
                Contour_img=self.GameFrame[Top[1]:Bottom[1],Left[0]:Right[0]]

                for x in range(Top[1],Bottom[1]):
                    for j in range(Left[0],Right[0]):
                        if(cv2.pointPolygonTest(cnt,(j,x),0)>=0):
                            Sum[0]+=Contour_img[x-Top[1]][j-Left[0]][0]
                            Sum[1]+=Contour_img[x-Top[1]][j-Left[0]][1]
                            Sum[2]+=Contour_img[x-Top[1]][j-Left[0]][2]
                            No+=1
                Seg.setColor(self.findClosetColor([Sum[0]/No,Sum[1]/No,Sum[2]/No]))
                self.Segments.append(Seg)
            if(old_crc!=-1):
                self.Circles[old_crc].RemoveSeg(Seg)
            self.Circles[i].addSeg(Seg)
        return i

    def FindSegments(self):
        img=cv2.cvtColor(self.GameFrame, cv2.COLOR_BGR2GRAY) 
        ret,img = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
        conts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        id=0
        self.Segments=[]
        Game_area=self.GameFrame.shape[0]*self.GameFrame.shape[1]
        for cnt in conts:
            if(cv2.contourArea(cnt)>5 and cv2.contourArea(cnt)/Game_area<0.06):
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid=(cX,cY)
                Seg=Segment(centroid,cnt)
                if(self.AssignCircle(Seg,id)!=-1):
                    id+=1
        if(30==len(self.Segments)):
            return True
        else:
            return False

    def FindCircles(self):    
        Circles=self.HoughCircles(self.GameFrame,self.Min_radius,self.Max_radius,self.game_Center)
        if(Circles is not None):
            sorted(Circles, key=lambda Circle: Circle.getDistance())
            self.Circles=Circles[0:2]
            self.Circles=sorted(self.Circles, key=lambda x: x.getGameCenter()[0])
            self.Circles[0].setID(0)
            self.Circles[1].setID(1)
            Middle_X=self.Circles[0].getGameCenter()[0]+self.Circles[1].getGameCenter()[0]
            Middle_Y=self.Circles[0].getGameCenter()[1]+self.Circles[1].getGameCenter()[1]
            Center=(Middle_X//2,Middle_Y//2)
            r=self.Circles[0].getRadius()
            self.Circles.append(Circle(Center,r,self.getEuclidDist(self.game_Center,Center),2))
            self.Dist_Circles=self.getEuclidDist(self.Circles[0].getGameCenter(),self.Circles[1].getGameCenter())
        else:
            print("Not enough Circles found")

    def CutGameBoard(self):
        Canny=self.getCannyImage(self.FullFrame)
        contours, hierarchy = cv2.findContours(Canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        c = np.vstack(contours[i] for i in range(len(contours)) if( cv2.contourArea(contours[i])>50))

        Left = tuple(c[c[:, :, 0].argmin()][0])
        Right = tuple(c[c[:, :, 0].argmax()][0])
        Top = tuple(c[c[:, :, 1].argmin()][0])
        Bottom = tuple(c[c[:, :, 1].argmax()][0])

        self.GameFrame=self.FullFrame[Top[1]:Bottom[1],Left[0]:Right[0]].copy()
        #Making the Game horizontal if vertical
        '''
        Find the min bounding rect. Identify the longer edges. Then get the angle of these longer edges wrt horizontal.
        '''
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
        dist= [int(Dist[0][1][0]),int(Dist[0][1][1]),int(Dist[0][1][2])]
        return dist

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

    def ScaleDown(self,img,Scale=8):
        #self.Scale=1/Scale
        #self.FullFrame=cv2.resize(cv2.imread(img),None,fx=self.Scale,fy=self.Scale)
        gpCam=GoProCamera.GoPro()
        t=time()
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        gpCam.livestream("start")
        gpCam.video_settings(res='1080p',fps='24')
        gpCam.gpControlSet(constants.Stream.WINDOW_SIZE,constants.Stream.WindowSize.R720)
        cap=cv2.VideoCapture("udp://10.5.5.9:8554",cv2.CAP_FFMPEG)
        while True:
            ret,frame=cap.read()    
            self.FullFrame=frame.copy()
            shape=self.FullFrame.shape
            cv2.line(frame, (shape[1],shape[0]//2), (0,shape[0]//2), [200,200,200],1)
            cv2.line(frame, (shape[1]//2,shape[0]), (shape[1]//2,0), [200,200,200],1)
            cv2.putText(frame,"Align the tezze game with the center of the frame and press q",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow("GoPro OpenCV",frame)
            if time()-t>=2.5:#Sending back that signal to keep the connection open
                sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(),("10.5.5.9",8554))
                t=time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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
                Circles.append(   Circle(     (i[0],i[1]),   i[2],    self.getEuclidDist(center,(i[0],i[1])))  )
            return Circles
        else:
            return None

def main():
    #Tezze=Game()
    #str="Photos"
    #arr = os.listdir(str)
    #for photo in arr:
    #    Photo=str+"/"+photo
    #    Tezze.setup(Photo)s
    while(True):
        Tezze=Game()
        Tezze.setup(0)
        cv2.destroyAllWindows()
        print("Press x to exit or any other key to take an other photo")
        key=cv2.waitKey(0)
        if(key==ord('x')):
            break
main()