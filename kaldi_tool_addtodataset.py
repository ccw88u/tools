import glob, os
import sys
import re
import time

# 收集聲音資料集後, 產生生成kaldi format file: spk2utt / utt2spk / text / wav.scp 準備做force alignment decode
#command
# python3 kaldi_tool_addtodataset.py $dir1 $userid
# python3 kaldi_tool_addtodataset.py grammarsentence_wav ponddyperson1

# spk2utt
# NEWG10_M0201F NEWG10_M0201F_10_01
# NEWG10_M0202F NEWG10_M0202F_10_02

# utt2spk
# NEWG10_M0201F_10_01 NEWG10_M0201F
# NEWG10_M0202F_10_02 NEWG10_M0202F
# NEWG10_M0203F_10_03 NEWG10_M0203F

# text
# NEWG10_M0201F_10_01 ta men dou shi da xue sheng ma
# NEWG10_M0202F_10_02 wo bu shi

# wav.scp
# NEWG10_M0201F_10_01 {path}/Test_Wav/001.wav
# NEWG10_M0202F_10_02 {path}/Test_Wav/002.wav

# When finish the command, need to change wav.scp {relative path} to replace the file relative path.
# such as    replace {relative path} => /XXX/YYY/ZZZ/

def main():
    dealdir, userid = sys.argv[1:]
    dealfiles = glob.glob('%s/*.wav' % dealdir)
    dealfiles.sort()

    wf = open('spk2utt', 'w')
    runint = 1
    for fp in dealfiles:
        fpname = os.path.basename(fp)        
        wf.write('NEWG10_M%06dF NEWG10_M%06dF_%s\n' % (runint, runint, fpname.split('.')[0]))
        runint += 1
    wf.close()

    wf = open('utt2spk', 'w')
    runint = 1
    for fp in dealfiles:
        fpname = os.path.basename(fp)        
        wf.write('NEWG10_M%06dF_%s NEWG10_M%06dF\n' % (runint, fpname.split('.')[0], runint))
        runint += 1
    wf.close()

    runint = 1
    wf = open('text', 'w')
    for fp in dealfiles:
        fpname = os.path.basename(fp)
        #os.path.split(name)  => ('./ebook/022926/book', 'nchul-tb-022926-z00-000-0195.jpg')
        fh, ft = os.path.split(fp)
        ft_head = ft.split('.', 1)[0]
        ft_txt = ft_head + '.txt'
        filetxtpath = fh + '/' + ft_txt

        filetxt = 'auto decode'
        if os.path.isfile(filetxtpath):
            yestone_filter_lst, notone_filter_lst = readtxtcontent(filetxtpath)
        wf.write('%s_%s %s\n' % (userid, fpname.split('.')[0], ' '.join(notone_filter_lst)))
        runint += 1
    wf.close()

    runint = 1
    wf = open('wav.scp', 'w')
    for fp in dealfiles:
        fpname = os.path.basename(fp)        
        wf.write('NEWG10_M%06dF_%s {relative path}/%s\n' % (runint, fpname.split('.')[0], fp))
        runint += 1
    wf.close()

# 我們 都 不 喜歡 看 電視 。 美美 喜歡 看 書 ， 我 喜歡 聽 音樂 。 你 呢
def readtxtcontent(filetxtpath):
    lines = []
    wf = open(filetxtpath)
    for line in wf.readlines():
        line = line.strip()
        linelst = line.split()
        linev = ''.join(linelst)
        lines.append(linev)

    retline = ''.join(lines)

    engpat = re.compile(r'[a-z|ḿ|é|ń|ō|è|ā|á|ǎ|à|ē|é|ě|è|ī|í|ǐ|ì|ō|ó|ǒ|ò|ū|ú|ǔ|ù|ǖ|ǘ|ǚ|ǜ|Ā|Á|Ǎ|À|Ē|É|Ě|È|Ī|Í|Ǐ|Ì|Ō|Ó|Ǒ|Ò|Ū|Ú|Ǔ|Ù|Ǖ|Ǘ|Ǚ|Ǜ]+')
    from pypinyin import pinyin, lazy_pinyin, Style
    # ['sui', 'sui', 'jin', 'he', 'fu', 'yu', 'guan']
    pinzin_notone_lst = lazy_pinyin(retline)
    #ps('pinzin_notone_lst', pinzin_notone_lst)

    # 只取有中文的 音調
    notone_filter_lst = []
    for lv in pinzin_notone_lst:     
        lv = lv  
        if lv and re.search(engpat, lv):
            #ps('lv-1', lv)
            notone_filter_lst.append(lv)

    
    pinzin_yestone_lst = pinyin(retline)
    #ps('pinzin_yestone', pinzin_yestone)
    # 只取有中文的音調
    yestone_filter_lst = []
    for lv in pinzin_yestone_lst:   
        #ps('lv', lv) 
        lv = lv[0]     
        if lv and re.search(engpat, lv):
            #ps('lv-2', lv)   
            yestone_filter_lst.append(lv)

    if len(notone_filter_lst) != len(yestone_filter_lst):
        print('retline', retline)
        ps('notone_filter_lst', notone_filter_lst)
        ps('yestone_filter_lst', yestone_filter_lst)

    return yestone_filter_lst, notone_filter_lst


def ps(fn,fv=''):
    print(fn, fv)

if __name__ == '__main__': main()
