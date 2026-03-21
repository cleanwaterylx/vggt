import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

def draw_roc_curve(gts_preds_list):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, confusion_matrix
    
    # gts, preds = gts_preds_list[0]
    # fpr, tpr, thresholds = roc_curve(gts, preds)
    # plt.plot(fpr, tpr, label='vggt 3 ROC curve (AUC = {:.4f})'.format(roc_auc_score(gts, preds)))
    
    for pair in gts_preds_list:
        gts, preds, label = pair
        fpr, tpr, thresholds = roc_curve(gts, preds)
        plt.plot(fpr, tpr, label=label + 'ROC curve (AUC = {:.4f})'.format(roc_auc_score(gts, preds)))
        print(f"{label} Confusion Matrix:")
        threshold = 0.5
        y_preds = (preds >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(gts, y_preds).ravel()
        print(f"{label} TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    plt.plot([0, 1], [0, 1], 'k--')  # 添加对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
def draw_pr_curve(gts_preds_list):
    # 多个pr曲线绘制
    import matplotlib.pyplot as plt
    
    # gts, preds = gts_preds_list[0]
    # precision, recall, thresholds = precision_recall_curve(gts, preds)
    # plt.plot(recall, precision, label='vggt 3 PR curve (AP = {:.4f})'.format(average_precision_score(gts, preds)))
    
    gts, preds = gts_preds_list[0]
    precision, recall, thresholds = precision_recall_curve(gts, preds)
    plt.plot(recall, precision, label='vggt 3+1 PR curve (AP = {:.4f})'.format(average_precision_score(gts, preds)))
    
    gts, preds = gts_preds_list[1]
    precision, recall, thresholds = precision_recall_curve(gts, preds)
    plt.plot(recall, precision, label='doppelganger++ PR curve (AP = {:.4f})'.format(average_precision_score(gts, preds)))
    
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

if __name__ == "__main__":
    # Load the data from the .npy file
    # data_vggt_four_img_1 = np.load('vggt_fourimg_add13_layer05111723_epoch1_focalloss_31.npy', allow_pickle=True).item()
    # data_vggt_four_img_2 = np.load('vggt_fourimg_add13_layer05111723_epoch2_focalloss_31.npy', allow_pickle=True).item()
    # data_vggt_four_img_3 = np.load('vggt_fourimg_add13_layer05111723_epoch3_focalloss_31.npy', allow_pickle=True).item()
    # data_vggt_four_img_4 = np.load('vggt_fourimg_add13_layer05111723_epoch4_focalloss_31.npy', allow_pickle=True).item()
    # data_vggt_four_img_5 = np.load('vggt_fourimg_add13_layer05111723_epoch5_focalloss_31.npy', allow_pickle=True).item()
    data_vggt_four_img_focalloss_3 = np.load('result/vggt_fourimg_add13_layer05111723_epoch3_focalloss_31.npy', allow_pickle=True).item()
    data_vggt_four_img_focalloss_12 = np.load('result/vggt_fourimg_add13_layer05111723_epoch12_focalloss_dopp_adam_31.npy', allow_pickle=True).item()
    data_dopp = np.load('../doppelgangers-plusplus/eval_visym_ap0.9917_auc0.9902_prec85_1.0000_recall95_0.9130.npy', allow_pickle=True).item()
    
    
    # draw_roc_curve([(np.array(data_vggt_four_img_1['gts'])[:, 3], np.array(data_vggt_four_img_1['preds'])[:, 3], 'vggt epoch1 '),
    #                 (np.array(data_vggt_four_img_2['gts'])[:, 3], np.array(data_vggt_four_img_2['preds'])[:, 3], 'vggt epoch2 '),
    #                 (np.array(data_vggt_four_img_3['gts'])[:, 3], np.array(data_vggt_four_img_3['preds'])[:, 3], 'vggt epoch3 '),
    #                 (np.array(data_vggt_four_img_4['gts'])[:, 3], np.array(data_vggt_four_img_4['preds'])[:, 3], 'vggt epoch4 '),
    #                 (np.array(data_vggt_four_img_5['gts'])[:, 3], np.array(data_vggt_four_img_5['preds'])[:, 3], 'vggt epoch5 '),
    #                  (np.array(data_dopp['gts']), np.array(data_dopp['preds']), 'doppelganger++ ')])
    draw_roc_curve([(np.array(data_vggt_four_img_focalloss_3['gts'])[:, 3], np.array(data_vggt_four_img_focalloss_3['preds'])[:, 3], 'vggt epoch3 focal loss '),
                    (np.array(data_vggt_four_img_focalloss_12['gts'])[:, 3], np.array(data_vggt_four_img_focalloss_12['preds'])[:, 3], 'vggt epoch12 focal loss '),
                     (np.array(data_dopp['gts']), np.array(data_dopp['preds']), 'doppelganger++ ')])

    draw_pr_curve([(np.array(data_vggt_four_img_focalloss_3['gts'])[:, 3], np.array(data_vggt_four_img_focalloss_3['preds'])[:, 3], 'vggt epoch3 focal loss '),
                    (np.array(data_vggt_four_img_focalloss_12['gts'])[:, 3], np.array(data_vggt_four_img_focalloss_12['preds'])[:, 3], 'vggt epoch12 focal loss '),(data_dopp['gts'], data_dopp['preds'])])
    quit()
    
    # Extract predictions and ground truths
    preds = np.array(data_vggt['preds'])
    gts = np.array(data_vggt['gts'])
    
    # Calculate True Positives, False Positives, True Negatives, and False Negatives
    tp = np.sum((preds >= 0.5) & (gts == 1))
    fp = np.sum((preds >= 0.5) & (gts == 0))
    tn = np.sum((preds < 0.5) & (gts == 0))
    fn = np.sum((preds < 0.5) & (gts == 1))
    
    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')

    
    # Calculate AUC and AP
    auc_score = roc_auc_score(gts, preds)
    ap_score = average_precision_score(gts, preds)
    
    
    
    print(f"Average Precision: {ap_score:.4f}")
    print(f"AUC: {auc_score:.4f}")
    
    precision, recall, thresholds = precision_recall_curve(gts, preds)

    from scipy.interpolate import interp1d

    f = interp1d(recall[::-1], precision[::-1])
    prec_at_recall = float(f(0.85))
    print("Prec@Recall>=0.85:", prec_at_recall)

    # Recall@Prec>=0.99
    f = interp1d(precision, recall)
    recall_at_prec = float(f(0.99))
    print("Recall@Prec>=0.99:", recall_at_prec)
    
    