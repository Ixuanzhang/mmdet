#-*-coding:utf-8-*-
# Reference link:https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx
# Cpu
# Written by xuan zhang

import numpy as np
import math

def py_cpu_nms(dets,thresh):
    # pure(纯净的) python nms baseline
    # 首先数据赋值和计算对应矩形框的面积
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2-x1+1) * (y2-y1+1)
    order = scores.argsort()[::-1]  # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下,即从大到小的排序
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引order[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。

    keep = []    # keep用于存放，NMS后剩余的方框
    # order会剔除遍历过的方框，和合并过的方框
    while order.size > 0:
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = order[0]   # every time the first is the biggst, and add it directly
        keep.append(i)
        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。

        # 计算交集的左上角和右下角坐标
        # 这里要注意，计算的是x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 根据IOU计算公式计算
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法
        inds = np.where(ovr <= thresh)[0]

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        order = order[inds + 1]

    return keep

def py_cpu_soft_nms(dets, thresh):
    pass

def py_cpu_dual_nms(dets, N1_thresh, N2_thresh, a=2.6, b=1.1, c=0.1): #N2_thresh:经典nms的阈值  dets:一个array
    D = []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep_index = [i for i in range(len(dets))]
    while len(keep_index) > 0:
        T = []
        i = keep_index[0]
        bbox = dets[i]  # 取出dets中的第一个元素
        T.append(bbox)
        keep_index.remove(i)

        xx1 = np.maximum(x1[i], x1[keep_index[:]])
        yy1 = np.maximum(y1[i], y1[keep_index[:]])
        xx2 = np.minimum(x2[i], x2[keep_index[:]])
        yy2 = np.minimum(y2[i], y2[keep_index[:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 根据IOU计算公式计算
        ovr = np.zeros(len(dets))
        for number,index in enumerate(keep_index):
            ovr[index] = inter[number] / (areas[i]+areas[index]-inter[number])
        inds = np.where(ovr >= N1_thresh)[0]

        for ind in inds:
            T.append(dets[ind])
            keep_index.remove(ind)

        T_array = np.array(T)
        s_sum = sum(T_array[:, 4])
        density = len(T_array)
        s_m = T_array[np.argmax(T_array[:, 4]), 4]

        if s_sum > f(density,s_m,a,b,c):
            '''print("afdasdfsafs:  ")
            print(s_sum)
            print(f(density,s_m,a,b,c)) '''

            sub_keep_indexs = py_cpu_nms(T_array, N2_thresh)   # 要留下的T中的框的索引
            for sub_keep_index in sub_keep_indexs:
                D.append(T_array[sub_keep_index])

    return np.array(D)

def f(density, s_m, a, b, c):
    result = b * math.exp(-s_m) * (density**(1-a*s_m-c*density))
    return result
'''
if __name__ == "__main__":
     bboxs = np.array([[4624,698,4809,816,0.7],[4652,732,4846,827,0.54],[4646,848,4782,1007,0.89],[4669,865,4818,1042,0.67],[4688,883,4779,1024,0.54]])
     thresh = 0.5
     b = py_cpu_dual_nms(bboxs,0.4,thresh)
     print(b)
'''
def multiclass_dual_nms(bboxs_results,N1_thresh, N2_thresh, a=2.6, b=1.1, c=0.1):
    dual_nms_result=[]
    for i in range(len(bboxs_results)):
        if bboxs_results[i].size == 0:
            # print(i)
            dual_nms_result.append(bboxs_results[i])
        else:
            sub_result = py_cpu_dual_nms(bboxs_results[i],N1_thresh,N2_thresh)
            if sub_result.size ==0:
                dual_nms_result.append(np.empty([0,5],dtype = np.float32))
            else:
                dual_nms_result.append(sub_result)
    return dual_nms_result

