import numpy as np
from PIL import Image

def get_dataset():
    path = "/Users/ClaudiaEspinoza/Desktop/Patter Recognition/duck.jpg"
    im = Image.open(path)
    im2 = Image.open(path).convert('L')
    im2.save('bnw.png')
    data = np.array(im)
    data2 = np.array(im2)
    im.close()
    im2.close()
    print(data.size)
    print(data.shape)
    print(data)
    print(data2.size)
    print(data2.shape)
    print(data2)

    for i in range(13816):
        for j in range(5946):
            if(data2[i][j] < 240):
                data2[i][j] = 0
    im_out = Image.fromarray(data2)
    im_out.save("y2.png")






get_dataset()
