import canny 
from PIL import Image
import json
import pandas as pd
A = pd.read_csv("annotation_of_0_100(1).csv")
result = 0
memecounter=0

for i in range(30,60):
    image = Image.open("resized_images3/{}".format(A["filename"][i]))
    Test=canny.ellipseDetector(image,test=True)
    if(Test.find_median_ellipse2()==0):
        memecounter+=1
        continue
    Test.perimetre()
    temp = json.loads(A["region_shape_attributes"][i])
    q=[]
    del temp["name"]
    for k in temp.items():
        q.append(k[1]/224*Test.dim)
    print(q)
    a =Test.symetrical_difference(q)
    result+=0 if a<0.6 else a
    memecounter += 1 if a<0.6 else 0

result=result/(30-memecounter)
print(result)