

from p3mltool import confusion_matrix_label_topN

samples = '''
25798        1830        931        2891        134
2001        25707        2321        1975        540
1229        2911        22527        3386        337
2322        1566        2476        45990        393
292        508        483        942        6506
'''   

samples1 = '''
28035        1159        693        1584        113
957        28778        1426        1068        315
856        1735        26021        1556        222
1416        1141        1627        48241        322
215        302        321        531        7362
'''

#confusion_matrix_label_topN(samples, topN=4)
confusion_matrix_label_topN(samples1, topN=4)