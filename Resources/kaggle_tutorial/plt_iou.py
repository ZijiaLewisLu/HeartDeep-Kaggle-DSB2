import matplotlib.pyplot as plt
import numpy as np

ONLINE = ['o%d.csv'%(i) for i in range(1,4)]
VALIDATE = ['v%d.csv'%(i) for i in range(1,3)]

def read_concat(fname):
    output = []
    if not isinstance(fname,list):
        fname = [fname]
    for f in fname:
        data = np.loadtxt(f, delimiter = ',').astype('float')
        output += list(data)

    return output

def plot_data(data, img_name, show = True):
    plt.plot(data,'o')
    plt.savefig(img_name)
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    data = read_concat(ONLINE)
    plot_data(data,"ONLINE_IOU.jpg")

    data = read_concat(VALIDATE)
    plot_data(data,"VALIDATE_IOU.jpg")
