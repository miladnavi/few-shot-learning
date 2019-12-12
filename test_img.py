#%%
from PIL import Image, ImageFilter


im = Image.open('3409.png')
# %%
width, heigth = im.size
width1 = int(width/2)
heigth1 = int(heigth)
img2 = Image.new(mode='L', size= (width1, heigth), color=0)
img3 = Image.new(mode='L', size= (width1, heigth), color=0)
k = 0
for i in range(width - 1):
    for j in range(heigth):    
        pixel = im.getpixel((i,j))
        if i % 2 == 0:
            img2.putpixel((k,j), pixel)   
        else:
            img3.putpixel((k,j), pixel)
    if i % 2 == 0:
        k+=1
    if k == width1:
        break 

#img4 = img2.resize(size = (100,100), resample=Image.BILINEAR)
#img5 = img3.resize(size = (100,100), resample=Image.BILINEAR)
img4 = img2.filter(ImageFilter.SMOOTH())
img4.show()
img5 = img3.filter(ImageFilter.SMOOTH())

img5.show()
# %%
