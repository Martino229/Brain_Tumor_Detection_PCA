import numpy as np



def confusion_matrix(y_true, y_pred):
    y_true= np.array(y_true)
    y_pred =np.array(y_pred)

    #In questo passaggio sono scartate le anomalie
    valid_indices= y_pred != -1
    if not np.any(valid_indices):
        return None

    y_t=y_true[valid_indices]
    y_p=y_pred[valid_indices]


    TN=np.sum((y_t==0) & (y_p==0)) #Paziente sano individuato correttamente
    TP=np.sum((y_t==1) & (y_p==1)) #Paziente malato individuato correttamente
    FP=np.sum((y_t==0) & (y_p==1)) #Paziente sano etichettato come malato
    FN=np.sum((y_t==1) & (y_p==0)) #Paziente malato etichettato come sano CASO GRAVE

    return np.array([[TN,FP],[FN,TP]])

def results(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    if matrix is None:
        return None

    TN, FP = matrix[0,0], matrix[0,1]
    FN, TP = matrix[1,0], matrix[1,1]

    total=np.sum(matrix)

    accuracy= (TP+TN)/total if total>0 else 0.0
    recall = TP/(TP+FN) if TP+FN>0 else 0.0
    precision = TP/(TP+FP) if TP+FP>0 else 0.0
    specificity = TN/(TN+FP) if TN+FP>0 else 0.0

    f1_score = 0.0
    if (precision+recall) >0:
        f1_score= 2*(precision*recall)/(precision+recall)


    return {
        "Matrix": matrix,
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision,
        "Specificity": specificity,
        "F1_score": f1_score
    }
def report(y_true, y_pred):
    metrics = results(y_true, y_pred)
    if metrics is None:
        print("Impossibile generare il report: solo anomalie rilevate.")
        return

    matrix = metrics["Matrix"]
    print("\n--- REPORT DIAGNOSTICO COMPLETO ---")
    print(f"Veri Negativi (TN): {matrix[0, 0]} | Falsi Positivi (FP): {matrix[0, 1]}")
    print(f"Falsi Negativi (FN): {matrix[1, 0]} | Veri Positivi (TP): {matrix[1, 1]}")
    print("-" * 35)
    print(f"Accuratezza: {metrics['Accuracy'] * 100:.2f}% | Sensibilità: {metrics['Recall'] * 100:.2f}%")
    print(f"Specificità: {metrics['Specificity'] * 100:.2f}% | Precisione: {metrics['Precision'] * 100:.2f}%")
    print("-----------------------------------\n")


