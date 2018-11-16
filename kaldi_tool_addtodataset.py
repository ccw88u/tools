import glob
import sys
import re
import time

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

def main():
    dealdir, userid = sys.argv[1]
    dealfiles = glob.glob('%s/*.wav' % dealdir)
    dealfiles.sort()

    wf = open('spk2utt', 'w')
    for fp in dealfiles:
        fpname = os.path.basename(fp)        
        wf.write('%s %s_%s' % (userid, userid, fpname.split('.')[0]))
    wf.close()

    wf = open('utt2spk', 'w')
    for fp in dealfiles:
        fpname = os.path.basename(fp)        
        wf.write('%s_%s %s' % (userid, fpname.split('.')[0], userid))
    wf.close()

    wf = open('text', 'w')
    for fp in dealfiles:
        fpname = os.path.basename(fp)        
        wf.write('%s_%s %s' % (userid, fpname.split('.')[0], 'auto decode'))
    wf.close()

    wf = open('wav.scp', 'w')
    for fp in dealfiles:
        fpname = os.path.basename(fp)        
        wf.write('%s_%s {relative path}/%s' % (userid, fpname.split('.')[0], fp))
    wf.close()

def ps(fn,fv=''):
    print(fn, fv)

if __name__ == '__main__': main()