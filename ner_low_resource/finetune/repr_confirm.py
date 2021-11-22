import os

dir = '/srv/share4/ychen3411/project00_model/'
L = 'ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu,qu,cdo,ilo,xmf,mi,mhr,tk,gn'
L = L.split(',')
for idx in range(240):
    for lang in L:
        tmp = dir + str(idx) + '/repr_{}.npz.npy'.format(lang)


        cond =  os.path.exists(tmp)
        if not cond:
            print(idx)

