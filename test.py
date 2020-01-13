import numpy as np

def RotatePt(origin,pt,Rotation_matrix):
    p = np.atleast_2d(p)
    return np.squeeze((r @ (p.T-o.T) + o.T).T)

def rotate(points,Center,Down=0):
    #Getting the angle
    Angle=90
    if(Down):
        Angle=-90
    Angle = np.deg2rad(Angle)

    #Getting the rotation matric
    r = np.array([[np.cos(Angle), -np.sin(Angle)],
                    [np.sin(Angle),  np.cos(Angle)]])
    o = np.atleast_2d(Center)

    #For pts of contours
    pts=[]
    for p in points:
        self.RotatePt(o,p,r)
        pts.append([self.RotatePt(o,p[0],r)])

points=[[[100,0]]]
origin=(0,100)

new_points = rotate(points,origin,1)
new_points=( round(new_points[0]), round(new_points[1]) )
print(new_points)