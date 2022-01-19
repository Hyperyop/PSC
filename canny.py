from scipy import ndimage
from scipy.ndimage.filters import convolve
import cv2
from scipy import misc
import numpy as np
from PIL import Image, ImageEnhance
from math import tan


class cannyEdgeDetector:
    def __init__(self, img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.img = img
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return 
    
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
       # print(g.shape)
        return g
    
    def sobel_filters(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    def grad(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)
        return np.sign(Ix)*np.sign(Iy)#*(Ix*Ix+Iy*Iy)

    

    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z

    def threshold(self, img):

        highThreshold = img.max() * self.highThreshold;
        lowThreshold = highThreshold * self.lowThreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self, img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img
    
    def detect(self):
        imgs_final = []    
        self.img_smoothed = convolve(self.img, self.gaussian_kernel(self.kernel_size, self.sigma))
        self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
        self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
        self.thresholdImg = self.threshold(self.nonMaxImg)
        img_final = self.hysteresis(self.thresholdImg)
        self.imgs_final.append(img_final)

        return self.imgs_final
    
image = Image.open("resized_images3/3.png")
image=image.resize((128,128),Image.BICUBIC)
img=image
image=np.asarray(img , dtype=np.float32)
image=np.copy(image)
image/=255

detector= cannyEdgeDetector(image,sigma = 20, lowthreshold=0.1, highthreshold=0.2)
image_signe = detector.grad(detector.img)
from matplotlib import pyplot as plt
image_canny=detector.detect()

image_canny=np.transpose(image_canny)
image_canny=np.flipud(image_canny)
#print(image_canny.shape)
image_canny=np.rot90(image_canny,k=3)

#image_signe=np.expand_dims(image_signe, 2)
print(np.transpose(image_signe).shape)
#image_canny=image_canny*np.rot90(image_signe,k=1)
#image_signe=image_canny
image_canny=np.reshape(image_canny, (128,128))

image_signe=np.transpose(image_signe)
plt.imshow(image_canny,cmap = 'gray')
image_canny=np.transpose(image_canny)
class unionfind:
    def __init__(self,n):
        self.uf=np.zeros([n, n, 3],dtype=int)
        for i in range(n):
            for j in range(n):
                self.uf[i][j][0] = i
                self.uf[i][j][1] = j
                self.uf[i][j][2] = 1
        return
    def find(self,i, j):
        if self.uf[i][j][0] != i or self.uf[i][j][1] != j:
            c = self.find(self.uf[i][j][0], self.uf[i][j][1])
            self.uf[i][j][0] = c[0]
            self.uf[i][j][1] = c[1]
            return c
        
        return (i, j)
    def union(self,i, j, k, l):
        i, j = self.find(i, j)
        k, l = self.find(k, l)
        
        if (i, j) == (k, l):
            return
        
        if self.uf[i][j][2] < self.uf[k][l][2]:
            i, j, k, l = k, l, i, j
    
        self.uf[k][l][0] = i
        self.uf[k][l][1] = j
        self.uf[k][l][2] = self.uf[i][j][2] + self.uf[k][l][2]
        
        self.uf[i][j][2] = self.uf[i][j][2] + self.uf[k][l][2]
    
    def adjacent(self,i, j, n):
        a = []
        for k in range(-1, 2):
            for l in range(-1, 2):
                if 0 <= i + k < n and 0 <= j + l < n and (k, l) != (0, 0):
                    a.append((i+k, j+l))
        
        return a
# uf = np.zeros([224, 224, 3],dtype=np.uint8)

# for i in range(224):
#     for j in range(224):
#         uf[i][j][0] = i
#         uf[i][j][1] = j
#         uf[i][j][2] = 1

# def find(i, j):
#     if uf[i][j][0] != i or uf[i][j][1] != j:
#         c = find(uf[i][j][0], uf[i][j][1])
#         uf[i][j][0] = c[0]
#         uf[i][j][1] = c[1]
#         return c
    
#     return (i, j)

# def union(i, j, k, l):
#     i, j = find(i, j)
#     k, l = find(k, l)
    
#     if (i, j) == (k, l):
#         return
    
#     if uf[i][j][2] < uf[k][l][2]:
#         i, j, k, l = k, l, i, j

#     uf[k][l][0] = i
#     uf[k][l][1] = j
#     uf[k][l][2] = uf[i][j][2] + uf[k][l][2]
    
#     uf[i][j][2] = uf[i][j][2] + uf[k][l][2]

# def adjacent(i, j, n):
#     a = []
#     for k in range(-1, 2):
#         for l in range(-1, 2):
#             if 0 <= i + k < n and 0 <= j + l < n and (k, l) != (0, 0):
#                 a.append((i+k, j+l))
    
#     return a
dim=128
uf=unionfind(dim)
for i in range(dim):
    for j in range(dim):
        if image_signe[i][j] == 0 or image_canny[i][j] == 0:
            continue
        for v in uf.adjacent(i, j,dim):
            if image_signe[i][j]== image_signe[v[0]][v[1]]:
                uf.union(i, j, v[0], v[1])

dico = {}

for i in range(dim):
    for j in range(dim):
        k, l = uf.find(i, j)
        
        if (k, l) in dico:
            dico[(k, l)].append((i, j))
        
        else:
            dico[(k, l)] = [(k, l)]

arcs = list(dico.items())

for i in range(len(arcs)):
    if len(arcs[i][1]) == 1:
        arcs[i] = arcs[i][1]
    else:
        arcs[i] = arcs[i][1][1:]
#print(arcs)
#Algorithm 1 : arc dont le D est +

def getConvexity1(arc):
    N=len(arc)
    left=arc[0]
    right=arc[N-1]
    current_x=left[0]
    area_O=0
    for i in range(1,N):
        if arc[i][0]!=current_x: 
            area_O+=abs(arc[i][1]-left[1])
            #print('cum', area_O)
            current_x=arc[i][0]
    area_bb=(1+abs(right[0]-left[0]))*(1+abs(right[1]-left[1]))
    area_u=area_bb-N-area_O
    #print("get 1 bb", area_bb)
    #print("get 1 u", area_u)
    #print("get 1 O", area_O)
    if area_u>area_O:
        return 1
    elif area_u<area_O:
        return -1
    else: 
        return 0
    
#Algoritm 2 ; arc dont le D est - 

def getConvexity2(arc):
    arc=sorted([(a[0],dim-a[1])for a in arc], key=lambda k:[k[0],k[1]])
    y_sort=sorted(arc,key=lambda k:[k[1],k[0]])
    N=len(arc)
    left=arc[0]
    right=arc[N-1]
    current_x=left[0]
    #area under the curb
    area_u=0
    for i in range(1,N):
        if arc[i][0]!=current_x: 
            area_u+=abs(arc[i][1]-y_sort[0][1])
            #print("cum u", area_u, i)
            current_x=arc[i][0]
    area_bb=(1+abs(right[0]-left[0]))*(1+abs(y_sort[0][1]-y_sort[N-1][1]))
    area_O=area_bb-N-area_u
    #print("bb", area_bb)
    #print("u", area_u)
    #print("O", area_O)
    #print(arc)
    if area_O>area_u:
        return -1
    elif area_O<area_u:
        return 1
    else: 
        return 0
    
quadrant = [[] for i in range(4)]
arc_length_min=50
for i in range(len(arcs)):
    a = int(image_signe[arcs[i][0][0]][arcs[i][0][1]])
    b = 0
    
    if a == 1 and len(arcs[i]) > arc_length_min:
        
        b = getConvexity1(arcs[i])
        
        if b==1:   
            #plt.plot([a[0] for a in arcs[i]], [a[1] for a in arcs[i]], '.')
            quadrant[0].append(arcs[i])
        if b==-1: 
            #plt.plot([a[0] for a in arcs[i]], [a[1] for a in arcs[i]], '.')
            quadrant[2].append(arcs[i])
    if a == -1 and len(arcs[i]) > arc_length_min:
        #flip along the y axis and sort ( change coordinate system)
        #arcs[i]=sorted([(a[0],224-a[1])for a in arcs[i]], key=lambda k:[k[0],k[1]])
        b = getConvexity2(arcs[i])
        print(getConvexity2(arcs[i]))
        if b==1:   
            #plt.plot([a[0] for a in arcs[i]], [a[1] for a in arcs[i]], '.')
            quadrant[1].append(arcs[i])
        if b==-1:
            #plt.plot([a[0] for a in arcs[i]], [a[1] for a in arcs[i]], '.')
            quadrant[3].append(arcs[i])
    #if a*b != 0: 
        #quadrant[(a+1)+ (b+1)//2].append(arcs[i])
        #plt.plot(arcs[i])
        #plt.plot([a[0] for a in arcs[i]], [a[1] for a in arcs[i]], '.')
#print(quadrant)
arc_test = [(4,0),(4,1),(3,2),(2,2),(2,3),(2,4),(1,4),(0,4)]
arc_test2 = [(0,4),(1,4),(2,4),(2,3),(2,2),(3,2),(4,1),(4,0)]
arc_test3= [(0,0),(0,1),(1,0),(2,0),(2,1),(2,2),(3,2),(4,3),(4,4)]
arc_x=[4,4,3,2,2,2,1,0]
arc_y=[0,1,2,2,3,4,4,4]

arc_test4=[(a[0],4-a[1]) for a in arc_test3]
arc_test5=[(4-a[0],4-a[1]) for a in arc_test3]
arc_test5.reverse()
#print(arc_test5)
print("blablabla")


#plt.plot([a[0] for a in quadrant[1][2]], [a[1] for a in quadrant[1][2]], '.')
#realquad= sorted([(a[0],224-a[1])for a in quadrant[3][1]], key=lambda k:[k[0],k[1]])
#print(getConvexity2(quadrant[3][1]))
#plt.plot([a[0] for a in realquad], [a[1] for a in realquad], '.')

#plt.plot([a[0] for a in arc_test3], [a[1] for a in arc_test3], '.')
#plt.plot([a[0] for a in arc_test4], [a[1] for a in arc_test4], '.')
#plt.plot([a[0] for a in arc_test3], [a[1] for a in arc_test3], '.')
#print(getConvexity2(arc_test3))
#print(getConvexity2(arc_test5))
#print(getConvexity2(arc_test2))
#plt.plot(arc_x, arc_y,'.')


def generateMidpoints(arc1,arc2,accuracy):
    N1=len(arc1)//2
    N2=len(arc2)//2
    midpoints=[((arc1[N1][0]+arc2[N2][0])//2,(arc1[N1][1]+arc2[N2][1])//2)]
    if (arc1[N1][0]-arc2[N2][0])==0:
        slope=1000
    else:
        slope= (arc1[N1][1]-arc2[N2][1])/(arc1[N1][0]-arc2[N2][0])
    slope2=np.empty((len(arc1),len(arc2)))
    print('slope',slope)
    for i in range(1, len(arc1)-1):
        for j in range(1, len(arc2)-1):
            if ((arc1[i][0]-arc2[j][0])==0):
                slope2[i][j]=0
            else:
                slope2[i][j]=(arc1[i][1]-arc2[j][1])/(arc1[i][0]-arc2[j][0])                                 
                #print('2', slope2)
        k=np.argmin(np.abs([a-slope for a in slope2[i]]))
        if (abs(slope2[i][k]-slope)<abs(accuracy*slope)):
            midpoints.append([(arc1[i][0]+arc2[k][0])//2,(arc1[i][1]+arc2[k][1])//2])
                    
    return midpoints

def getSlope(midpoints):
    middle=len(midpoints)//2
    S=[]
    for i in range(middle):
        if (midpoints[middle+i][0]-midpoints[i][0]==0):
            slope =1000
        else:
            slope=(midpoints[middle+i][1]-midpoints[i][1])/(midpoints[middle+i][0]-midpoints[i][0])
        S.append(slope)

    s=np.median(S)
    s.astype(int)
    return s

def getCenter(midpoints1, midpoints2):
    H1=np.zeros((1,2))
    H2=np.zeros((1,2))
    H1[0][0]=np.median([a[0] for a in midpoints1])
    H1[0][1]=np.median([a[1] for a in midpoints1])
    
    H2[0][0]=np.median([a[0] for a in midpoints2])
    H2[0][1]=np.median([a[1] for a in midpoints2])
    H1=H1.astype(int)
    H2=H2.astype(int)
    print('H',H1,H2)
    C=[0,0]
    t1=getSlope(midpoints1)
    t2=getSlope(midpoints2)
    
    print('t1,t2',t1,t2)
    C[0]=-(H2[0][1]-t2*H2[0][0]-H1[0][1]+t1*H1[0][0])//(t2-t1)
    C[1]=-(t1*H2[0][1]-t2*H1[0][1]+t1*t2*(H1[0][0]-H2[0][0]))//(t2-t1)
    return C




midpoints1 = generateMidpoints(quadrant[0][1],quadrant[1][0],0.002)
midpoints2=generateMidpoints(quadrant[2][0],quadrant[1][0],0.002)
C=getCenter(midpoints1,midpoints2)
print(C)
#print(midpoints2)

plt.plot([a[0] for a in midpoints1], [a[1] for a in midpoints1], '.')
plt.plot([a[0] for a in midpoints2], [a[1] for a in midpoints2], '.')
plt.plot([a[0] for a in quadrant[0][1]], [a[1] for a in quadrant[0][1]], '.')
plt.plot([a[0] for a in quadrant[1][0]], [a[1] for a in quadrant[1][0]], '.')
plt.plot([a[0] for a in quadrant[2][0]], [a[1] for a in quadrant[2][0]], '.')
plt.plot(C[0],C[1],'.')

x_array=list(range(250))
y_array=[0.32*a+87 for a in x_array]
#plt.plot(x_array, y_array)
x2_array=list(range(50,150))
y2_array=[-2.66*a+358 for a in x2_array]
#plt.plot(x2_array, y2_array)
