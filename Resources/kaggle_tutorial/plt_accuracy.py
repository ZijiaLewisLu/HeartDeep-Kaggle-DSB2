import matplotlib.pyplot as plt
import numpy as np

FILE = 'accuracy.csv'
TAG = ['index', 'edv', 'esv', 'dedv', 'desv']

def read_file(fname,tag):
    csv = np.loadtxt(fname,delimiter = ',').astype('float')
    csv = csv.T
    data_dict = {}

    for idx, k in enumerate(tag):
        data_dict[k] = csv[idx]
    return data_dict


def get_ef(data_dict,tag):
    edv, esv = tag[1], tag[2]
    dedv, desv = tag[3], tag[4]

    data_dict['ef'] = data_dict[edv]-data_dict[esv]
    data_dict['ef'] /= data_dict[edv]+1e-5

    data_dict['def'] = data_dict[dedv]-data_dict[desv]
    data_dict['def'] /= data_dict[dedv]+1e-5
    return data_dict


def plot_pair(data_dict, pair, show=False):
    a,b = pair[0], pair[1]
    mx = np.max(dd[a])
    my = np.max(dd[b])
    plt.plot(dd[a],dd[b],'o')
    plt.xlabel(a)
    plt.ylabel(b)
    x = np.linspace(0, max(mx,my), 10000)
    y = x
    plt.plot(x,y,'-',color='red')
    plt.savefig("%s-%s.jpg"%(a,b))
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    dd = read_file(FILE,TAG)
    dd = get_ef(dd,TAG)
    pairs = [['edv','dedv'],['esv','desv'],['ef','def']]
    for p in pairs:
        plot_pair(dd,p,show=True)
