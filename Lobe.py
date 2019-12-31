class Lobe(Segment):
    def __init__(self,centroid,color,contour,id,lobe=0):
        self.Centroid=centroid
        self.Color=color
        self.Contour=contour
        self.ID=id
        self.Lobe=lobe

    def setCentroid(self,centroid):
        self.Centroid=centroid
    def setLobe(self,lobe):
        self.Lobe=lobe

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