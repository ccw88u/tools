

from p3mltool import confusion_matrix_label_topN

samples = '''
25798        1830        931        2891        134
2001        25707        2321        1975        540
1229        2911        22527        3386        337
2322        1566        2476        45990        393
292        508        483        942        6506
'''   

samples1 = '''
2290        64        56        134
105        2039        102        76
26        49        1301        110
108        66        106        2630
'''

#confusion_matrix_label_topN(samples, topN=4)
confusion_matrix_label_topN(samples1, topN=4)