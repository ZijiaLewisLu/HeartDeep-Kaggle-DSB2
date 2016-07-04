import pickle as pk

with open('validate.pk','r') as f:
    img = pk.load(f)
    ll = pk.load(f)

N = img.shape[0]/2

with open('v1.pk','w') as f:
    pk.dump(img[:N],f)
    pk.dump(ll[:N],f)

with open('v2.pk','w') as f:
    pk.dump(img[N:],f)
    pk.dump(ll[N:],f)
