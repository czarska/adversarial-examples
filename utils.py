import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
def make_picture(image, old_image, tag, number, mode="mnist"):
    m = {0: "airplane", 1: "auto", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
    if tag == 0:
        name = "img/" + mode + "/" + str(number) + "/org.png"
        image = np.array(old_image, dtype='float')
        if mode == "cifar10":
            pixels = image.reshape((24, 24, 3))
            plt.imshow(pixels)
        else:
            pixels = image.reshape((28, 28))
            plt.imshow(pixels, cmap='gray')
        plt.show()
        plt.savefig(name)
    if mode == "cifar10":
        name = "img/" + mode + "/" + str(number) + "/" + m[tag] + ".png"
    else:
        name = "img/" + mode + "/" + str(number) + "/" + str(tag) + ".png"
    image = np.array(image, dtype='float')
    if mode == "cifar10":
        pixels = image.reshape((24, 24, 3))
        plt.imshow(pixels)
    else:
        pixels = image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
    plt.show()
    plt.savefig(name)

def stats(predicts, labels):
    temp = [0]*len(predicts[0])
    avreage = 0
    for i in range(len(predicts[0])):
        for j in range(10):
            if np.argmax(labels[i]) != j and predicts[j][i] == j:
                temp[i] += 1
                avreage += 1
    best = (len(predicts[0])-temp.count(0))/len(predicts[0])
    worst = temp.count(9) / len(predicts[0])
    avreage = avreage/(len(predicts[0])*9)
    print("best:", best, " worst:", worst, " avreage:", avreage)

def stats_cari(norms, was):
    best = 0.0
    worst = 0.0
    avreage = 0.0
    for j in range(len(norms[0])):
        b = 1000.0
        w = 0.0
        for i in range(10):
            if i != was[j]:
                avreage += norms[i][j]
                b = min(b, norms[i][j])
                w = max(w, norms[i][j])
        best += b
        worst += w
    avreage /= (9*len(norms[0]))
    best /= len(norms[0])
    worst /= len(norms[0])
    print("best:", best, " worst:", worst, " avreage:", avreage)

def heat(predicts, labels, name=""):
    ar = np.zeros([10,10])
    was = np.zeros([10,10])
    for i in range(len(predicts[0])):
        for j in range(10):
            p = np.argmax(labels[i])
            if p != j:
                was[p][j] += 1
                if predicts[j][i] == j:
                    ar[p][j] += 1
    for i in range(10):
        for j in range(10):
            if was[j][i] != 0:
                ar[j][i] = ar[j][i]/was[j][i]
                ar[j][i] *= 100
    print(ar)
    plt.imshow(ar, cmap='hot', interpolation='nearest')
    plt.show()
    n = name + "heat.png"
    plt.savefig(n)

def heat_cari(norms, was, name=""):
    ar = [np.zeros([10, 10]), np.zeros([10, 10])]
    for i in range(len(norms[0])):
        for j in range(10):
            p = was[i]
            if p != j:
                ar[1][p][j] += 1
                ar[0][p][j] += norms[j][i]
    for i in range(10):
        for j in range(10):
            if ar[1][j][i] != 0:
                ar[0][j][i] = ar[0][j][i]/ar[1][j][i]
                ar[0][j][i] *= 10
    print(ar[0])
    plt.imshow(ar[0], cmap='hot', interpolation='nearest')
    plt.show()
    n = name + "heat_cari.png"
    plt.savefig(n)



def disc(image):
    for i in range(len(image[0])):
        image[0][i] = int(image[0][i]*255)
        image[0][i] = float(image[0][i])
        image[0][i] /= 255
    return image