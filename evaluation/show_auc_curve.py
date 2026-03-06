import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

def draw_roc_curve(gts_preds_list):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve   
    
    gts, preds = gts_preds_list[0]
    fpr, tpr, thresholds = roc_curve(gts, preds)
    plt.plot(fpr, tpr, label='vggt ROC curve (AUC = {:.4f})'.format(roc_auc_score(gts, preds)))
    
    gts, preds = gts_preds_list[1]
    fpr, tpr, thresholds = roc_curve(gts, preds)
    plt.plot(fpr, tpr, label='dopp++ ROC curve (AUC = {:.4f})'.format(roc_auc_score(gts, preds)))
    
    gts, preds = gts_preds_list[2]
    fpr, tpr, thresholds = roc_curve(gts, preds)
    plt.plot(fpr, tpr, label='vggt four img ROC curve (AUC = {:.4f})'.format(roc_auc_score(gts, preds)))
    
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
    
    gts, preds = gts_preds_list[0]
    precision, recall, thresholds = precision_recall_curve(gts, preds)
    plt.plot(recall, precision, label='vggt PR curve (AP = {:.4f})'.format(average_precision_score(gts, preds)))
    
    gts, preds = gts_preds_list[1]
    precision, recall, thresholds = precision_recall_curve(gts, preds)
    plt.plot(recall, precision, label='dopp++ PR curve (AP = {:.4f})'.format(average_precision_score(gts, preds)))
    
    gts, preds = gts_preds_list[2]
    precision, recall, thresholds = precision_recall_curve(gts, preds)
    plt.plot(recall, precision, label='vggt four img PR curve (AP = {:.4f})'.format(average_precision_score(gts, preds)))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

if __name__ == "__main__":
    # Load the data from the .npy file
    data_vggt = np.load('vggt_two_img.npy', allow_pickle=True).item()
    data_vggt_four_img = np.load('vggt_four_img.npy', allow_pickle=True).item()
    data_dopp = np.load('../doppelgangers-plusplus/eval_visym_ap0.9917_auc0.9902_prec85_1.0000_recall95_0.9130.npy', allow_pickle=True).item()
    
    # print(data)
    
    draw_roc_curve([(data_vggt['gts'], data_vggt['preds']), (data_dopp['gts'], data_dopp['preds']), (data_vggt_four_img['gts'], data_vggt_four_img['preds'])])
    draw_pr_curve([(data_vggt['gts'], data_vggt['preds']), (data_dopp['gts'], data_dopp['preds']), (data_vggt_four_img['gts'], data_vggt_four_img['preds'])])
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
    
    