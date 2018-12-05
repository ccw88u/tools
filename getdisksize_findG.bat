import os, glob

dlist = glob.glob('./*')
for i in dlist:
    of = os.popen('du -hs %s' % i)
    for line in of.readlines():
        line = line.rstrip()
        tlst = line.split('\t', 1)
        ##['163M', './nclcdr']
        if len(tlst) == 2:
            if tlst[0].find('G') != -1:
                print line
    of.close()
