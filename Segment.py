class Segment:
    def __init__(self,centroid,color,contour,id,circle,lobe=-1):
        self.Centroid=centroid
        self.Color=color
        self.Contour=contour
        self.ID=id
        self.Lobe=lobe
        self.Circle=circle

    def setCentroid(self,centroid):
        self.Centroid=centroid
    def setLobe(self,lobe):
        self.Lobe=lobe
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
    def getCircle(self):
        return self.Circle

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

    def Intersect(self,lobe,Dist_Circles,Dist_lobe_circle):
        Radius_ratio=1.3
        Dist_ratio=1.3
        if(self.getRadius()/lobe.getRadius()<Radius_ratio and lobe.getRadius()/self.getRadius()<Radius_ratio):
            if(Dist_Circles/Dist_lobe_circle<Dist_ratio and Dist_lobe_circle/Dist_Circles<Dist_ratio):
                return True
        return False

    def addLobe(self,lobe):
        self.Lobes.append(lobe)

    def removeLobe(self,lobe):
        self.Lobes.remove(lobe)

    def commonSegs(self,Circle):#Circle.CommonSegs(Lobe_Circle)
        pass#First find the overlapping area's contour. Then find the the lobes in that circle 