# sklearn
from sklearn import metrics
from sklearn.metrics import *
from sklearn.preprocessing import OneHotEncoder
# others
import numpy as np
import config
from config import *


def count_data(labels, is_print=True, not_zero=True):
    """
    统计数据数量
    :param labels: 数据标签
    :param is_print: 若为True则自动输出结果
    :param not_zero: 若为True则不显示数量为0的类别
    :return:
    """
    data_nums = {k:0 for k in TYPE_LABELS.keys()}
    for label in labels:
        data_nums[LABELS_TYPE[int(label)]] += 1
    data_num = sum(data_nums.values())
    if not_zero:
        data_nums = {k:v for k, v in data_nums.items() if v!=0}
    if is_print:
        print('总数：{}\n{}\n'.format(data_num, data_nums))
    return data_num, data_nums



# 统计分类结果
def result_metrics(score_train, score_test, labels_train, labels_test, method, is_print=False):
    precs,recalls, ths = precision_recall_curve(labels_test, score_test[:,1])
    precs_,recalls_, ths_ = precision_recall_curve(-(labels_test-1), score_test[:,0])
    f1s = 2*precs*recalls/(precs+recalls)
    if not config.testing:
        th = ths[f1s.argmax()]
        pred_test = (score_test[:,1]>th).astype("int")
        pred_train = (score_train[:,1]>th).astype("int")
    else:
        th = config.threshold
        pred_test = (score_test[:,1]>th).astype("int")
        pred_train = (score_train[:,1]>th).astype("int")
    confusion_train = confusion_matrix(labels_train, pred_train, labels=range(labels_train.max()+1))
    confusion_test = confusion_matrix(labels_test, pred_test, labels=range(labels_train.max()+1))
    acc_train = accuracy_score(labels_train, pred_train)
    acc_test = accuracy_score(labels_test, pred_test)
    prec_train = precision_score(labels_train, pred_train, average=None)
    prec_test = precision_score(labels_test, pred_test, average=None)
    recall_train = recall_score(labels_train, pred_train, average=None)
    recall_test = recall_score(labels_test, pred_test, average=None)
    spec_train = np.array([confusion_train[1,1]/(confusion_train[1,1]+confusion_train[1,0]),confusion_train[0,0]/(confusion_train[0,0]+confusion_train[0,1])])
    spec_test = np.array([confusion_test[1,1]/(confusion_test[1,1]+confusion_test[1,0]),confusion_test[0,0]/(confusion_test[0,0]+confusion_test[0,1])])
    f1_train = f1_score(labels_train, pred_train, average=None)
    f1_test = f1_score(labels_test, pred_test, average=None)
    auc_train = roc_auc_score(OneHotEncoder(sparse=False).fit_transform(labels_train), score_train, average=None)
    auc_test = roc_auc_score(OneHotEncoder(sparse=False).fit_transform(labels_test), score_test, average=None)
    # f1_train = f1_score(labels_train, pred_train, average='binary')
    # f1_test = f1_score(labels_test, pred_test, average='binary')
    # auc_train = roc_auc_score(OneHotEncoder(sparse=False).fit_transform(labels_train), score_train, average="macro")
    # auc_test = roc_auc_score(OneHotEncoder(sparse=False).fit_transform(labels_test), score_test, average="macro")
    if is_print:
        print('{} Test 混淆矩阵 \n{}'.format(method, confusion_test))
        print('{} Train 准确率\t{:>.5f}'.format(method, float(acc_train)))
        print('{} Test 准确率 \t{:>.5f}'.format(method, float(acc_test)))
        print('{} Train 精确率\t{}'.format(method, [prec_train, prec_train.mean()]))
        print('{} Test 精确率 \t{}'.format(method, [prec_test, prec_test.mean()]))
        print('{} Train 召回率\t{}'.format(method, [recall_train, recall_train.mean()]))
        print('{} Test 召回率 \t{}'.format(method, [recall_test,recall_test.mean()]))
        print('{} Train 特异性\t{}'.format(method, [spec_train, spec_train.mean()]))
        print('{} Test 特异性\t{}'.format(method, [spec_test, spec_test.mean()]))
        print('{} Train F1分数\t{}'.format(method, [f1_train, f1_train.mean()]))
        print('{} Test F1分数 \t{}'.format(method, [f1_test, f1_test.mean()]))
        print('{} Train AUC   \t{}'.format(method, auc_train))
        print('{} Test AUC   \t{}'.format(method, auc_test))
        print('{} threshold   \t{}'.format(method, th))
        # print('{} Train F1分数\t{:>.3f}'.format(method, float(f1_train)))
        # print('{} Test F1分数 \t{:>.3f}'.format(method, float(f1_test)))
        # print('{} Train AUC   \t{:>.3f}'.format(method, float(auc_train)))
        # print('{} Test AUC   \t{:>.3f}'.format(method, float(auc_test)))
    return {'confusion_matrix': confusion_test,
            'acc_train': acc_train,
            'acc_test': acc_test,
            'prec_train': prec_train,
            'prec_test': prec_test,
            'recall_train': recall_train,
            'recall_test': recall_test,
            'spec_train': spec_train,
            'spec_test': spec_test,
            'f1_train':f1_train,
            'f1_test':f1_test,
            'auc_train': auc_train,
            'auc_test': auc_test,
            "precs": precs,
            "recalls": recalls,
            "precs_": precs_,
            "recalls_": recalls_,
            "threshold": th      
        }, pred_test
