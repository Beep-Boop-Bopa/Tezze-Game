import numpy as np
import cv2
import math
import time
import statistics as st
import threading as th
import imutils


def Var_calc(Img,STD_array,start,end):
    for i in range(start,end):
        STD_array[i]=[i,np.var(Img[i])]

class Game:
    def __init__(self):
        self.Canny_Upper=150
        self.Canny_Lower=75
        self.Scale=1
        self.FullFrame=0
        self.GameFrame=0
        self.Cardinality=1 #0 means flat, 1 means vertical
        self.game_Center=0
        self.FullFrame_center=0
        self.Circles=[]
        self.Max_radius=0
        self.Min_radius=0
        self.colors=[]
        self.Segments=[]
        self.lobes=[]

    def setup(self, img="Photos/3.jpg"):
        Colored=cv2.imread(img)
        self.FindScale()
        Colored=cv2.resize(Colored,None,fx=self.Scale,fy=self.Scale)
        self.FullFrame=Colored
        self.CutGameBoard()
        self.findColors()
        self.FindCircles()
        self.FindSegments()
        self.ExtractLobes()
        self.AssignLobes()
        self.draw()

    def ExtractLobes(self):
        pass

    def AssignLobes(self):
        pass

    def FindSegments(self):
        for i in range(2):
            Crc=self.Circles[i]
            Canny=self.getCannyImage(self.Circles[i][3])
            contours, hierarchy = cv2.findContours(Canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self.segments=[[],[]]
            New_Img =  np.zeros((self.Circles[i][3].shape[0],self.Circles[i][3].shape[1],3), np.uint8)
            contours=sorted(contours, key=lambda z: cv2.contourArea(z),reverse=True)
            LargestContour=contours[0]
            Distances=[]
            for Sublist in LargestContour:
                Distances.append(self.getEuclidDist(Sublist[0],self.Circles[i][0]))
            self.Circles[i][1]=sorted(Distances, key=lambda z: z,reverse=True)[0]
            CircleImg =  np.zeros((self.Circles[i][3].shape[0],self.Circles[i][3].shape[1],3), np.uint8)
            cv2.circle(CircleImg,self.Circles[i][0],int(self.Circles[i][1]),(255,255,255),-1)
            for c in contours:
                if(cv2.contourArea(c)>15):
                    Left = tuple(c[c[:, :, 0].argmin()][0])
                    Right = tuple(c[c[:, :, 0].argmax()][0])
                    Top = tuple(c[c[:, :, 1].argmin()][0])
                    Bottom = tuple(c[c[:, :, 1].argmax()][0])
                    Width=Bottom[1]-Top[1]
                    Length=Left[0]-Right[0]
                    if(Length/Width<=3 and Width/Length<=3):#Introduce area measures as well
                        _,_,w,h = cv2.boundingRect(c)
                        Area=w*h
                        if(cv2.contourArea(c)>250 or cv2.contourArea(c)/Area>0.45):
                            OverLap =  np.zeros((self.Circles[i][3].shape[0],self.Circles[i][3].shape[1],3), np.uint8)
                            cv2.drawContours(OverLap,[c],-1,[255,255,255],cv2.FILLED);
                            OverLap=cv2.bitwise_and(OverLap,CircleImg)
                            contours1, hierarchy = cv2.findContours(self.getCannyImage(OverLap,5,10), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            Area=0
                            for cnts in contours1:
                                Area+=cv2.contourArea(cnts)

                            M = cv2.moments(c)
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            Distances=[]
                            for Sublist in c:
                                Distances.append(self.getEuclidDist(Sublist[0],self.Circles[i][0]))
                            Smallest_Distance=sorted(Distances, key=lambda z: z)[0]

                            if(Smallest_Distance < self.Circles[i][1] and Area/cv2.contourArea(c)>0.25):
                                Contour_img=self.Circles[i][3][Top[1]:Bottom[1],Left[0]:Right[0]].copy()
                                Sum=[0,0,0]
                                No=0
                                for x in range(Top[1],Bottom[1]):
                                    for j in range(Left[0],Right[0]):
                                        if(cv2.pointPolygonTest(c,(j,x),0)>0):
                                            Sum[0]+=Contour_img[x-Top[1]][j-Left[0]][0]
                                            Sum[1]+=Contour_img[x-Top[1]][j-Left[0]][1]
                                            Sum[2]+=Contour_img[x-Top[1]][j-Left[0]][2]
                                            No+=1
                                cv2.drawContours(New_Img,[c],-1,[255,255,255],cv2.FILLED);
                                New_Img=cv2.bitwise_and(New_Img,self.Circles[i][3])
                                mean = [Sum[0]/No,Sum[1]/No,Sum[2]/No]
                                Color=self.findClosetColor(mean)
                                self.segments[i].append([c,Color])
#                            print("mean ", np.uint32(mean), " Color ",Color)
#                            cv2.imshow("Contour",Contour_img)
#                            cv2.imshow("New_Img",New_Img)
#                            cv2.moveWindow("Contour", 800, 150)
#                            cv2.waitKey(0)
            cv2.imshow("New_Img",CircleImg)
            self.Circles[i][3]=New_Img

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

    def FindCircles(self):    
        Distances=self.HoughCircles(self.GameFrame,self.Min_radius,self.Max_radius,self.game_Center)
        if(Distances is not None):
            sorted(Distances, key=lambda x: x[2])
            self.Circles=Distances[0:2]
            i=0
            for center,Radius,_ in self.Circles:
                #Do the mask thing here
                y0=center[1]-Radius if(center[1]-Radius>=0) else 0
                y1=center[1]+Radius if(center[1]+Radius<=self.GameFrame.shape[0]) else self.GameFrame.shape[0]###not correcy
                x0=center[0]-Radius if(center[0]-Radius>=0) else 0
                x1=center[0]+Radius if(center[0]+Radius<=self.GameFrame.shape[1]) else self.GameFrame.shape[1]
                Circle=self.GameFrame[y0:y1,x0:x1]
                self.Circles[i][0]=(Radius//1,Radius//1)
                blank_image = np.zeros((Circle.shape[0],Circle.shape[1],3), np.uint8)
                cv2.circle(blank_image, self.Circles[i][0], Radius-1, [255,255,255], -1)
                self.Circles[i].append(cv2.bitwise_and(blank_image,Circle))
                i=+1
        else:
            print("Not enough Circles found")

    def CutGameBoard(self):
        Canny=self.getCannyImage(self.FullFrame)
        contours, hierarchy = cv2.findContours(Canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #Can be replaced by drawing a convex hull around them?
        #size of contour points
        length = len(contours)
        #concatinate poits form all shapes into one array
        c = np.vstack(contours[i] for i in range(length))
        hull = cv2.convexHull(c)
        uni_hull = []
        uni_hull.append(hull)
        blank_image = np.zeros((self.FullFrame.shape[0],self.FullFrame.shape[1],3), np.uint8)
        blank_image=cv2.drawContours(blank_image,uni_hull,-1,[255,255,255],-1);
        self.FullFrame=cv2.bitwise_and(blank_image,self.FullFrame)

        Left = tuple(c[c[:, :, 0].argmin()][0])
        Right = tuple(c[c[:, :, 0].argmax()][0])
        Top = tuple(c[c[:, :, 1].argmin()][0])
        Bottom = tuple(c[c[:, :, 1].argmax()][0])

        self.GameFrame=self.FullFrame[Top[1]:Bottom[1],Left[0]:Right[0]].copy()

        if(Bottom[1]-Top[1]<Right[0]-Left[0]):
            self.Cardinality=0
        else:
            self.FullFrame = imutils.rotate_bound(self.FullFrame, 90)
            self.GameFrame = imutils.rotate_bound(self.GameFrame, 90)

        self.game_Center=(self.GameFrame.shape[1]//2,self.GameFrame.shape[0]//2)
        x=self.game_Center[0]+Left[0]
        y=self.game_Center[1]+Top[1]
        self.FullFrame_center=(x,y)

        Width=self.GameFrame.shape[0]
        length=self.GameFrame.shape[1]

        self.Min_radius=Width//3
        self.Max_radius=length//2

        Colored_pic = self.GameFrame.copy()
        Colored_pic = Colored_pic.reshape((-1,3))

        No_OF_threads=4
        previous_Index=0
        length=len(Colored_pic)//No_OF_threads
        Remainder=len(Colored_pic)%No_OF_threads
        Threads={}
        STD_RGB=[None]*len(Colored_pic)
        for i in range(No_OF_threads):
            start_Index=previous_Index
            last_index=previous_Index+length if(i!=0) else previous_Index+length+Remainder#len(Colored_pic)
            previous_Index=last_index
            Threads[i]=th.Thread(target=Var_calc,args=[Colored_pic,STD_RGB,start_Index,last_index])
            Threads[i].start()

        for i in range(No_OF_threads):
            Threads[i].join()

        STD_RGB=sorted(STD_RGB, key=lambda x: x[1])

        for i in range(round(len(STD_RGB)*0.375)):
            Colored_pic[STD_RGB[i][0]]=[0,0,0]

        self.GameFrame = Colored_pic.reshape((self.GameFrame.shape))

    def draw(self):
        while(True):
            key=cv2.waitKey(1)
            cv2.imshow("Canny",self.GameFrame)
            cv2.moveWindow("Canny", 150, 150)
            cv2.imshow("Circle1",self.Circles[0][3])
            cv2.moveWindow("Circle1", 500, 50)
            cv2.imshow("Circle2",self.Circles[1][3])
            cv2.moveWindow("Circle2", 500, 350)
            if(key==ord('q')):
                break
        cv2.destroyAllWindows()

    def FindScale(self,scale=8):
        self.Scale=1/scale

    def HoughCircles(self,image,min,max,center):
        Canny=self.getCannyImage(image)
        circles = cv2.HoughCircles(Canny,cv2.HOUGH_GRADIENT,1,100,
                            param1=50,param2=30,minRadius=min,maxRadius=max)
        if(circles is not None):
            circles = np.int32(np.around(circles))
            Distances=[]
            for i in circles[0,:]:
                Distances.append([(i[0],i[1]),i[2],self.getEuclidDist(center,(i[0],i[1]))])
            return Distances
        else:
            return None


    def getEuclidDist(self,Point1,Point2,ThreeD_bool=0):
        X_diff=(Point1[0]-Point2[0])**2
        Y_diff=(Point1[1]-Point2[1])**2
        Sum=X_diff+Y_diff
        if(ThreeD_bool):
            Z_diff=(Point1[2]-Point2[2])**2
            Sum+=Z_diff
        return (Sum)**.5

    def getCannyImage(self,image,Upper=0,Lower=0):
        GREY=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        GREY=cv2.GaussianBlur(GREY,(3,3),0)
        if(Upper==0):
            Upper=self.Canny_Upper
            Lower=self.Canny_Lower
        return cv2.Canny(GREY,Lower,Upper)


def main():
    Tezze=Game();
    Tezze.setup("Photos/0.jpg");
    Tezze.setup("Photos/1.jpg");
    Tezze.setup("Photos/2.jpg");
    Tezze.setup("Photos/4.jpg");
    Tezze.setup("Photos/5.jpg");

main()