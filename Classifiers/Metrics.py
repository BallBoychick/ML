class Metrics:

    def My_Confusion_matrix(y_test, y_pred):
        m = len(set(y_test))
        size = len(y_test)
        matrix = dict()

        for class_name in range(m):
            matrix[class_name] = [0 for k in range(m)]
        for i in range(size):
            actual_class = y_test[i]
            pred_class = y_pred[i]
            matrix[actual_class][pred_class] += 1
        print("Confusion Matrix :")
        for key, value in matrix.items():
            print("Actual %-13s %-15d %-15d %-15d" % (key, *value))
        matrix = dict(list(zip(set(y_test), matrix.values())))
        print(matrix)
        return matrix
