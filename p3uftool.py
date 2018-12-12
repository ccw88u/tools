import os

def randomsleep(nums=1):
    pwchar = '1234'
    rs = ''
    import random
    for x in range(nums):
        rs += pwchar[random.randint(0, len(pwchar) - 1)]
    return rs

## 將{'a':3, 'b':2, ....} 等排序, 最多在前面
def sortdic(dic):
    rlst = []
    for k in dic.keys():
        rlst.append((dic[k], k))
    rlst.sort()
    rlst.reverse()
    return rlst

# dictionary sort (5, 'key')
def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort()
    backitems.reverse()
    return [(backitems[i][1], backitems[i][0]) for i in range(0, len(backitems))]

# dictionary key: 1


def put2dicnums(rdic, key, addnum=1):
    if not rdic.get(key, ''):
        rdic[key] = addnum
    else:
        pn = rdic.get(key)
        pn += addnum
        rdic[key] = pn
    return rdic


def put2dicnum(rdic, key, val):

    if not rdic.get(key):
        rdic[key] = [val]
    else:
        lst = rdic.get(key)
        lst.append(val)
        rdic[key] = lst
    return rdic

# dictionary


def put2dic(rdic, key, val, droprepeat=0):
    if not rdic.get(key, ''):
        if type(val) == str:
            rdic[key] = [val]
        elif type(val) == list:
            rdic[key] = val
    else:
        if type(rdic[key]) == str:
            addlst = [rdic[key], val]
            if droprepeat == 1:
                addlst = droprepeatlist(addlst)
            rdic[key] = addlst
        elif type(rdic[key]) == list:
            addlst = rdic[key] + [val]
            if droprepeat == 1:
                addlst = droprepeatlist(addlst)
            rdic[key] = addlst
    return rdic


def droprepeatlist(runlst, fmt=''):
    retlist = []
    allwords = []
    for v in runlst:
        if fmt == 'pair':
            if v[1] not in allwords:
                allwords.append(v[1])
                retlist.append(v)
        elif fmt == 'first':
            if v[0] not in allwords:
                allwords.append(v[0])
                retlist.append(v)
        else:
            v = v.strip()
            if v not in retlist:
                retlist.append(v)
    return retlist


def timeobv(ti, obvlastime=0, dispmsg=0):
    import time
    nowtime = int(time.time())
    if obvlastime > 0:
        if dispmsg == 1:
            print('%s:%s secs' % (ti, (nowtime - obvlastime)))
    return nowtime


def ps(fn, fv=''):
    print(fn, fv)


def save_pickle_file(pickle_file, param):
    from pickle import dump

    dump(param, open(pickle_file, 'wb'))


def load_pickle_file(pickle_file):
    from pickle import load

    dumparam = load(open(pickle_file, 'rb'))
    return dumparam

# 切割資料處理


def splitfunc(str, spchar=';//,', spc='//'):
    allst = []

    num = 0
    for v in spchar.split(spc):
        if num == 0:
            allst = str.split(v)
        else:
            allst = runsplit(allst, spc=v)
        num += 1
    return allst


def runsplit(strlst, spc=''):
    rlst = []
    for v in strlst:
        for value in v.split(spc):
            if value:
                rlst.append(value)
    return rlst


def csvline(file):
    import re
    L, cont = [], 1
    # read csv line
    while cont:
        s = file.readline()
        if not s:
            break
        s = (L and ',"' or ',') + s.replace('\r\n', '\n')
        L2 = re.findall(r',("(?:[^"]|"")*"?|[^",\n]*)', s)
        if L2[-1].endswith('\n'):
            L2[-1] += '"'  #
        else:
            cont = 0
        for i in range(len(L2)):  # replace '""' to '"'
            if L2[i].startswith('"') and L2[i].endswith('"'):
                L2[i] = L2[i][1:-1].replace('""', '"')
        if L:
            L = L[:-1] + [L[-1] + L2[0]] + L2[1:]  # merge last entry
        else:
            L = L2
    return L


# 區分成幾等份
def chunks_split(arr, m=2):
    import math
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

# random 取 rtblst 中的項目
# bookmax => 取幾個資料
# bookmax = 0  => 取1個
# bookmax = 3  => 取4個


def getrandom_fromlst(rtblst, bookmax=3):

    import random
    bookmax = bookmax
    dbrklst = []

    avoidunlimitcircle = 0
    # avoidunlimitcircle => 避免無圖檔資料時. 成為無窮迴圈
    while(len(dbrklst) < (bookmax + 1) and avoidunlimitcircle < 100):
        r = rtblst[random.randint(0, len(rtblst) - 1)]
        if r not in dbrklst:
            dbrklst.append(r)
        avoidunlimitcircle += 1

    return dbrklst


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def chunks_nums(arr, m=10):
    return [arr[i:i+m] for i in range(0, len(arr), m)]

# 比較檔案是否是在定義的時間範圍內
def countfiledate(ckfile, difftime):
    flag = 0 
    import os, time
    todayint = int(time.mktime(time.localtime()))
    if os.path.isfile(ckfile):
        stats = os.stat(ckfile)
        lastmod_date = time.localtime(stats[8])
        filetime = int(time.mktime(lastmod_date))
        ##ps('diff', (filetime - todayint))
        ## 代表在時間範圍內 
        if (todayint - filetime) < difftime:
            flag = 1
    return flag    
