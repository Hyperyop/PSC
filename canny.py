from asyncio.windows_events import NULL
from scipy import ndimage
from scipy.ndimage.filters import convolve
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import statistics

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
       # if not self.test:
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

        strong_i, strong_j = np.where(img >= highThreshold) #liste des indices des points brillants sous forme de booléans
        
        zeros_i, zeros_j = np.where(img < lowThreshold)  #liste des points trop faibles

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
        self.img_smoothed = convolve(self.img, self.gaussian_kernel(self.kernel_size, self.sigma)) #applique le filtre gausssien
        self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
        self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
        self.thresholdImg = self.threshold(self.nonMaxImg)
        img_final = self.hysteresis(self.thresholdImg)
        self.imgs_final.append(img_final)

        return self.imgs_final
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
class ellipseDetector:
    arc_length_min=75
    def __init__(self,image,dim=128,test=False):
        self.p=[]
        self.test=test
        #image.show()
        image=image.resize((dim,dim),Image.BICUBIC)

        #image = image.resize((256,256),Image.BILINEAR)
        image=np.asarray(image , dtype=np.float32)
  

            

        image/=255 #Les valeurs de l'array sont entre  0 et 1
        if not self.test:
            plt.imshow(image)
            plt.show()

        detector = cannyEdgeDetector(image,sigma = 20, lowthreshold=0.1, highthreshold=0.3)
        #L'écart-type sigma a été choisi par dichotomie à 20. On peut le modifier. Prochaine étape déterminer automatiquement les paramètres de Canny

        image_signe = detector.grad(detector.img)

        #plt.imshow(image_signe,cmap= 'gray')
        image_canny=detector.detect()

        image_canny=np.transpose(image_canny)
        image_canny=np.flipud(image_canny)
        #if not self.test:
        # print(image_canny.shape)
        image_canny=np.rot90(image_canny,k=3)


        image_canny=np.reshape(image_canny, (dim,dim))

        image_signe=np.transpose(image_signe)
        #llnr0
        plt.imshow(image_canny,cmap = 'gray')
        image_canny=np.transpose(image_canny)
        self.image_canny=image_canny
        self.dim=dim
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

        self.arcs = list(dico.items())

        for i in range(len(self.arcs)):
            if len(self.arcs[i][1]) == 1:
                self.arcs[i] = self.arcs[i][1]
            else:
                self.arcs[i] = self.arcs[i][1][1:]
        self.quadrant = [[] for i in range(4)]
        self.stock = [[] for i in range(4)]
        for i in range(len(self.arcs)):
            a = int(image_signe[self.arcs[i][0][0]][self.arcs[i][0][1]])
            b = 0
            
            if a == 1:
                b = self.getConvexity1(self.arcs[i])
                if b==1:   
                    #plt.plot([a[0] for a in self.arcs[i]], [a[1] for a in self.arcs[i]], '.')
                    if len(self.arcs[i]) > len(self.stock[0]):
                        self.stock[0] = self.arcs[i]
                    if len(self.arcs[i]) > self.arc_length_min:
                        self.quadrant[0].append(self.arcs[i])
                if b==-1: 
                    #plt.plot([a[0] for a in self.arcs[i]], [a[1] for a in self.arcs[i]], '.')
                    if len(self.arcs[i]) > len(self.stock[2]):
                        self.stock[2] = self.arcs[i]
                    if len(self.arcs[i]) > self.arc_length_min:
                        self.quadrant[2].append(self.arcs[i])
                        
            if a == -1:
                #flip along the y axis and sort ( change coordinate system)
                #self.arcs[i]=sorted([(a[0],224-a[1])for a in self.arcs[i]], key=lambda k:[k[0],k[1]])
                b = self.getConvexity2(self.arcs[i])

                if b==1:   
                    #plt.plot([a[0] for a in self.arcs[i]], [a[1] for a in self.arcs[i]], '.')
                    if len(self.arcs[i]) > len(self.stock[1]):
                        self.stock[1] = self.arcs[i]
                    if len(self.arcs[i]) > self.arc_length_min:
                        self.quadrant[1].append(self.arcs[i])
                if b==-1:
                    #plt.plot([a[0] for a in self.arcs[i]], [a[1] for a in self.arcs[i]], '.')
                    if len(self.arcs[i]) > len(self.stock[3]):
                        self.stock[3] = self.arcs[i]
                    if len(self.arcs[i]) > self.arc_length_min:
                        self.quadrant[3].append(self.arcs[i])
            
        for i in range(4):
            if len(self.quadrant[i]) == 0 and len(self.stock[i]) > 0:
                self.quadrant[i].append(self.stock[i])

    
        
    
    #if not self.test:
    # print(arcs)




    #Algorithm 1 : arc dont le D est +

    def getConvexity1(self,arc):
        N=len(arc)
        left=arc[0]
        right=arc[N-1]
        current_x=left[0]
        area_O=0
        
        for i in range(1,N):
            if arc[i][0]!=current_x: 
                area_O+=abs(arc[i][1]-left[1])
                current_x=arc[i][0]
        
        area_bb=(1+abs(right[0]-left[0]))*(1+abs(right[1]-left[1]))
        area_u=area_bb-N-area_O

        if area_u>area_O:
            return 1
        elif area_u<area_O:
            return -1
        else: 
            return 0
        
    #Algoritm 2 ; arc dont le D est - 

    def getConvexity2(self,arc):
        arc=sorted([(a[0],self.dim-a[1])for a in arc], key=lambda k:[k[0],k[1]])
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
                #if not self.test:
                # print("cum u", area_u, i)
                current_x=arc[i][0]
        area_bb=(1+abs(right[0]-left[0]))*(1+abs(y_sort[0][1]-y_sort[N-1][1]))
        area_O=area_bb-N-area_u
        #if not self.test:
        # print("bb", area_bb)
        #if not self.test:
        # print("u", area_u)
        #if not self.test:
        # print("O", area_O)
        #if not self.test:
        # print(arc)
        if area_O>area_u:
            return -1
        elif area_O<area_u:
            return 1
        else: 
            return 0


    
    
    

        #if a*b != 0: 
            #quadrant[(a+1)+ (b+1)//2].append(arcs[i])
            #plt.plot(arcs[i])
            #plt.plot([a[0] for a in arcs[i]], [a[1] for a in arcs[i]], '.')
    #if not self.test:
    # print(quadrant)
    # arc_test = [(4,0),(4,1),(3,2),(2,2),(2,3),(2,4),(1,4),(0,4)]
    # arc_test2 = [(0,4),(1,4),(2,4),(2,3),(2,2),(3,2),(4,1),(4,0)]
    # arc_test3= [(0,0),(0,1),(1,0),(2,0),(2,1),(2,2),(3,2),(4,3),(4,4)]
    # arc_x=[4,4,3,2,2,2,1,0]
    # arc_y=[0,1,2,2,3,4,4,4]

    # arc_test4=[(a[0],4-a[1]) for a in arc_test3]
    # arc_test5=[(4-a[0],4-a[1]) for a in arc_test3]
    # arc_test5.reverse()
    # #if not self.test:
    # print(arc_test5)

    # arc_test3 = quadrant[0][1]
    # arc_test4 = quadrant[1][0]
    # arcfus = arc_test3 + arc_test4
    # #if not self.test:
    # print('arctest', arc_test3)
    # xt,yt= [a[0] for a in arcfus], [a[1] for a in arcfus]


    # arc_test = [(4,0),(4,1),(3,2),(2,2),(2,3),(2,4),(1,4),(0,4)]
    # arc_test2 = [(0,4),(1,4),(2,4),(2,3),(2,2),(3,2),(4,1),(4,0)]

    # x = np.array([arc_test[i][0] for i in range(len(arc_test))])
    # y = np.array([arc_test[i][1] for i in range(len(arc_test))])

    # space = np.linspace(0, 4, 20)

    # e = 0.5
    # circle_x = np.array([3*np.cos(i)+e*(random()-0.5) for i in space])
    # circle_y = np.array([np.sin(i)+e*(random()-0.5) for i in space])

    # for arc in quadrant[0]:
    #     x1, y1 = [c[0] for c in arc], [c[1] for c in arc]
    #     plt.plot(x1, y1)

    # plt.show()
    # x1, y1 = [quadrant[0][0][i][0] for i in range(len(quadrant[0][0]))], [quadrant[0][0][i][1] for i in range(len(quadrant[0][0]))]
    # x2, y2 = [quadrant[1][0][i][0] for i in range(len(quadrant[1][0]))], [quadrant[1][0][i][1] for i in range(len(quadrant[1][0]))]
    # x3, y3 = [quadrant[2][0][i][0] for i in range(len(quadrant[2][0]))], [quadrant[2][0][i][1] for i in range(len(quadrant[2][0]))]
    # plt.plot(x1, y1)
    # plt.plot(x2, y2)
    # plt.plot(x3, y3)

    # plt.show()

    def check1(self,sa, sb):
        tolerance = 3
        
        v = (max(sa) >= max(sb)) and (min(sb) <= min(sa))
        v = v and (max(sb) - min(sa) <= tolerance)
        
        return v
        

    def verticale(self,a, b):
        xa = [a[i][0] for i in range(len(a))]
        xb = [b[i][0] for i in range(len(b))]
        
        return self.check1(xa, xb)
        
        
    def horizontale(self,a, b):
        ya = [a[i][1] for i in range(len(a))]
        yb = [b[i][1] for i in range(len(b))]
        
        return self.check1(ya, yb)

    def diagonale(self,a, b):
        return self.horizontale(a, b) and self.verticale(b, a)

    def antediagonale(self,a, b):
        return self.horizontale(a, b) and self.verticale(a, b)

    def valid_triplet(self,i, j, k, a, b, c): #i < j < k
        
        if i == 0 and j == 1 and k == 2:
            return self.verticale(a, b) and self.diagonale(c, a) and self.horizontale(c, b)
        
        if i == 0 and j == 1 and k == 3:
            return self.verticale(a, b) and self.horizontale(c, a) and self.antediagonale(c, b)
        
        if i == 0 and j == 2 and k == 3:
            return self.diagonale(b, a) and self.horizontale(c, a) and self.verticale(c, b)
        
        if i == 1 and j == 2 and k == 3:
            return self.horizontale(b, a) and self.antediagonale(c, a) and self.verticale(c, b)

    def reglin_ellipse(self,x, y):
        n = len(x)
        m = np.zeros((n , 5))
        
        for i in range(n):
            x0, y0 = x[i], y[i]
            m[i] = np.array([2*x0*y0, y0**2, 2*x0, 2*y0, 1])
        
        b = np.array([-x[i]**2 for i in range(n)])
        
        z = np.linalg.lstsq(m, b) 
        
        average = sum([-x[i]**2 for i in range(n)]) / n
        
        TSS = sum([(b[i]-average)**2 for i in range(n)])
        
        r2 = 1 - z[1][0] / TSS
        
        if not self.test:
            print("coefficient de correlation = ", r2)
        
        z0 = list(z[0])
        z0.insert(0, 1)
        return z0, r2

    def plot_ellipse(self,p):
        
        a, b, c, d, e, f = p[0], p[1], p[2], p[3], p[4], p[5]
        
        xmin, xmax = 0, self.dim
        ymin, ymax = 0, self.dim
        x = np.linspace(xmin, xmax, 1000)
        y = np.linspace(ymin, ymax, 1000)

        x, y = np.meshgrid(x, y)
        plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*2*x + e*2*y + f), [0], colors = "magenta")
        

    def find_ellipse(self):
        for i in range(4) :
            for j in range(i+1,4):
                for k in range(j+1,4):
                    for a in self.quadrant[i] :
                        for b in self.quadrant[j] :
                            for c in self.quadrant[k] :
                                if not self.valid_triplet(i, j, k, a, b, c):
                                    continue
                                arcfus = a + b + c
                                xt,yt= [d[0] for d in arcfus], [d[1] for d in arcfus]
                                plt.scatter(xt, yt, s = 15)
                                truc = self.reglin_ellipse(xt, yt)
                                if not self.test:
                                    self.plot_ellipse(truc[0])
                                    plt.show()
                                
    #find_ellipse(quadrant)
                                
    def find_best_ellipse(self):
        m = 0
        d = True
        for i in range(4) :
            for j in range(i+1,4):
                for k in range(j+1,4):
                    for a in self.quadrant[i] :
                        for b in self.quadrant[j] :
                            for c in self.quadrant[k] :
                                arcfus = a + b + c
                                xt,yt= [d[0] for d in arcfus], [d[1] for d in arcfus]
                                
                                truc = self.reglin_ellipse(xt, yt)
                                
                                if d:
                                    p = truc[0]
                                    x0, y0 = xt, yt
                                    d = False
                                
                                if truc[1] > m :
                                    m = truc[1]
                                    p = truc[0]
                                    x0, y0 = xt, yt
        if not self.test:
            print(p)
        if not self.test:
            print(m)
        plt.scatter(x0, y0, s = 15)
        if not self.test:
            self.plot_ellipse(p)
            plt.show()

    #find_best_ellipse(quadrant)

    def find_average_ellipse(self):
        compteur = 0
        p = np.array([0. for i in range(6)])
        for i in range(4) :
            for j in range(i+1,4):
                for k in range(j+1,4):
                    for a in self.quadrant[i] :
                        for b in self.quadrant[j] :
                            for c in self.quadrant[k] :
                                arcfus = a + b + c
                                xt,yt= [d[0] for d in arcfus], [d[1] for d in arcfus]

                                param = self.reglin_ellipse(xt, yt)[0]
                                
                                if param[1]**2 - 4*param[0]*param[2] < 0:
                                    compteur += 1

                                    p += np.array(param)
                                
                                plt.scatter(xt, yt, s = 10, color = "blue")
        if not self.test:
            print(p)
        if not self.test:
            self.plot_ellipse(p/compteur)
            plt.show()

    #find_average_ellipse(quadrant)

    

    def find_median_ellipse(self):
        stock = []
        r0 = 0
        for i in range(4) :
            for j in range(i+1,4):
                for k in range(j+1,4):
                    for a in self.quadrant[i] :
                        for b in self.quadrant[j] :
                            for c in self.quadrant[k] :
                                arcfus = a + b + c
                                xt,yt= [d[0] for d in arcfus], [d[1] for d in arcfus]

                                param, r = self.reglin_ellipse(xt, yt)
                                
                                if r > r0:
                                    param0 = param
                                    r0 = r
                                
                                if param[1]**2 - 4*param[0]*param[2] < 0 and r > 0.998:
                                    stock.append(param)
                                
                                plt.scatter(xt, yt, s = 1, color = "red")
        
        p = []
        
        if len(stock) > 0:
            for i in range(6):
                p.append(statistics.median([stock[j][i] for j in range(len(stock))]))
        else:
            p = param0
        if not self.test:
            self.plot_ellipse(p)
            plt.show()

    def get_lines(self):
        lignex = []
        ligney = []

        for a in self.arcs:
            if len(a) > 10:
                for p in a:
                    lignex.append(p[0])
                    ligney.append(p[1])


    #plt.scatter(lignex, ligney, s =0.1 , color = "blue")

    #find_median_ellipse(quadrant)

    def find_median_ellipse2(self):
        stock = []
        param0=0
        r0 = 0
        for i in range(4) :
            for j in range(i+1,4):
                for k in range(j+1,4):
                    for a in self.quadrant[i] :
                        for b in self.quadrant[j] :
                            for c in self.quadrant[k] :
                                if not self.valid_triplet(i, j, k, a, b, c):
                                    continue
                                arcfus = a + b + c
                                xt,yt= [d[0] for d in arcfus], [d[1] for d in arcfus]

                                param, r = self.reglin_ellipse(xt, yt)
                                
                                if r > r0:
                                    param0 = param
                                    r0 = r
                                
                                if param[1]**2 - 4*param[0]*param[2] < 0 and r > 0.998:
                                    stock.append(param)
                                
                                    #plt.scatter(xt, yt, s = 0.3, color = "red")
        
        p = []
        
        if len(stock) > 0:
            for i in range(6):
                p.append(statistics.median([stock[j][i] for j in range(len(stock))]))
        else:
            if param0==0:
                return 0
            p = param0
        self.p=p
        if not self.test:
            self.plot_ellipse(p)
            plt.show()


    #plt.scatter(lignex, ligney, s =1 , color = "blue")


    def all_ellipse(self):
        accu = []
        for i in range(4):
            for a in self.quadrant[i]:
                accu += a
        
        xt, yt = [d[0] for d in accu], [d[1] for d in accu]
        param = self.reglin_ellipse(xt, yt)[0]
        
        #plt.scatter(xt, yt, s = 10, color = "blue")
        if not self.test:
            self.plot_ellipse(param)
            plt.show()
        

    #all_ellipse(quadrant)
    def perimetre(self):
        epsilon=1e-2
        result=0
        p=self.p
        if not self.test:
            print(self.p)
        accuracy=400
        A= np.zeros((accuracy,accuracy))
        xmin, xmax = 0, self.dim
        ymin, ymax = 0, self.dim
        x = np.linspace(xmin, xmax, accuracy)
        y = np.linspace(ymin, ymax, accuracy)
        for i in range(accuracy):
            for j in range(accuracy):
                # if (abs(p[0]*x**2 + p[1]*y**2 + p[2]*x*y + p[3]*2*x + p[4]*2*y + p[5])<epsilon):
                #     result+=1
                A[i,j]=abs(p[0]*x[i]**2 + p[2]*y[j]**2 + p[1]*x[i]*y[j] + p[3]*2*x[i] + p[4]*2*y[j] + p[5])

        A/=np.max(A)
        if not self.test:
            plt.imshow(A)
            plt.show()
        for i in range(accuracy):
            for j in range(accuracy):
                if (A[i,j]<epsilon):
                    result+=1
        if not self.test:
            print("La longueur du périmetre est {}".format(self.dim**2*result/accuracy**2))
        return result
    def is_in_ellipse_cartesian(self, x, y): #works because a = 1 > 0
        return self.p[0]*x**2 + self.p[2]*y**2 + self.p[1]*x*y + self.p[3]*2*x + self.p[4]*2*y + self.p[5] <= 0

    def is_in_ellipse_polar(self,q, x, y): #q = cx, cy, a, b, r
        x1 = np.cos(q[4])*(x-q[0])+np.sin(q[4])*(y-q[1])
        y1 = -np.sin(q[4])*(x-q[0])+np.cos(q[4])*(y-q[1])
        
        return x1**2/q[2]**2 + y1**2/q[3]**2 <= 1
        

    def symetrical_difference(self, q):
        xmin, xmax = 0, self.dim
        ymin, ymax = 0, self.dim
        
        union, inter = 0, 0
        
        for i in range(xmax-xmin):
            for j in range(ymax-ymin):
                a, b = self.is_in_ellipse_cartesian( i, j), self.is_in_ellipse_polar(q, i, j)
                union += a or b
                inter += a and b
        
        return inter / union