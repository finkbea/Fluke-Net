import csv

def calc_AP_5(class_predictions, true_class):
    """
    For calculating the AP@5 score for the predictions on a single 
      datapoint. Averaging these scores over our dev set will 
      give us the full MAP@5 score

    class_predictions: 5 most confident predictions, ordered
      most confident -> least confident

    true_class: True class of the predicted image
    """

    assert(len(class_predictions)==5)
    
    # Search for true_class in class_predictions
    true_class_location = None
    for i in range(5):
        if class_predictions[i] == true_class:
            true_class_location = i+1
            break
    
    # Average Precision if true_class was not found
    AP_5 = 0
    
    # If true_class was found, AP@5 is the precision at that point
    if not true_class_location is None:
        AP_5 = float(1)/true_class_location
    return AP_5