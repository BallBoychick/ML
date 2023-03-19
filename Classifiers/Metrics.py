class Metrics:  
    def my_accuracy_score(y_test, pred_train):
        pass #equal

# def accuracy_score_p(y_true, y_pred, *, normalize=True, sample_weight=None):

#     # Compute accuracy for each possible representation
#     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
#     check_consistent_length(y_true, y_pred, sample_weight)
#     if y_type.startswith("multilabel"):
#         differing_labels = count_nonzero(y_true - y_pred, axis=1)
#         score = differing_labels == 0
#     else:
#         score = y_true == y_pred

#     return _weighted_sum(score, sample_weight, normalize)
    def my_precision(y_predicted, y_test):
        return (y_predicted == y_test).sum()