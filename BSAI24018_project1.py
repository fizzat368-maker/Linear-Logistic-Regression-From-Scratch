import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression #import_for_comparison

print("=" * 60)
print("         QUESTION 1: LINEAR REGRESSION")
print("=" * 60)

# Load dataset (from CSV as instructor requested)
df1 = pd.read_csv(r'C:\Users\User\Desktop\PAI\BostonHousing.csv')
df1.fillna(df1.mean(), inplace=True) #no_extreme_outliers,modify_directly_inside

X1 = df1.drop(columns=['medv']).values  #drop_col_return_copy_dataframe
y1 = df1['medv'].values #select_from_original_med_col_and_convert_to_numpy_array

#normalise(large_values_donot_dominate)
X1_mean = X1.mean(axis=0) #mean_of_each_col_seperately_featurewise
X1_std = X1.std(axis=0) #standard_dev,how_spread_out_each_feature_is
X1_std[X1_std == 0] = 1 #if_0_var,replace_with_1_to_avoid_error_NAN
X1_norm = (X1 - X1_mean) / X1_std #normalise_to_avoid_dominance_and_speed_convergence 

# Train/test split (80/20)
np.random.seed(42)
idx1 = np.random.permutation(len(X1_norm)) #shuffles_without_it,gives_biased_eval<premute_doesnt_modify_original
split1 = int(0.8 * len(X1_norm)) #split_data
X1_train = X1_norm[idx1[:split1]] #features_for_training
X1_test = X1_norm[idx1[split1:]]
y1_train = y1[idx1[:split1]] #outputs_for_training
y1_test = y1[idx1[split1:]]

print(f"\nDataset shape : {df1.shape}")
print(f"Train size    : {X1_train.shape}")
print(f"Test  size    : {X1_test.shape}")


def q1_cost(X, y, w, b):
    n = len(y) #num_of_training_samples_used_to_avg_error
    y_hat = np.clip(X @ w + b, -1e6, 1e6) #predict_output_for_all_samples_at_once
    return (1 / n) * np.sum((y - y_hat) ** 2) #sqr_act_predicted_dif_bc_it_makes_positive_large_errors_penalised_more,avg_so_loss

def q1_gradients(X, y, w, b):
    n = len(y)
    error = y - (X @ w + b) #how_far_each_prediction_is_from_actual,+ve_if_underpredicted,-ve_if_overpredicted
    dw = (-2 / n) * (X.T @ error) #how_much_toCHnage_each_weight_to_reduce_prediction_error,under_weight_inc
    db = (-2 / n) * np.sum(error) #grad_for_bias
    return dw, db

def converged(w_new, w_old, eps=1e-4):
    return np.linalg.norm(w_new - w_old) <= eps #if_change_smaller_than_0.0001_stop,
#weight_change_to_check_if_model_has_truly_stopped_learning 

def q1_train(X, y, lr=0.01, max_iter=10000, eps=1e-4):
    w = np.zeros(X.shape[1]) #initial_all_w=0
    b = 0.0 #all_bias,convex_so_lead_to_same_solution
    hist = [] #empty_list_to_record_loss
    for i in range(max_iter):
        w_old = w.copy() #save_current_w_before_updating,copy_for_compare
        dw, db = q1_gradients(X, y, w, b)
        w = w - lr * dw #update_weights_in_direction_that_reduces_loss,go_downhill,lr_controls_step_size
        b = b - lr * db #update_bias_similarly
        cost = q1_cost(X, y, w, b)
        if not np.isfinite(cost): #detects_nan_or_inf
            print(f"  lr={lr}: diverged at iteration {i+1}")
            return None, None, []
        hist.append(cost)
        if converged(w, w_old, eps): #stop_if_weight_stable
            print(f"  lr={lr}: converged at iteration {i+1}, loss={cost:.4f}")
            return w, b, hist
    print(f"  lr={lr}: max iterations reached, loss={hist[-1]:.4f}")
    return w, b, hist

def q1_predict(X, w, b): #calc_predictions_for_each_input_feature
    predictions = []
    for i in range(len(X)):
        pred = 0
        for j in range(len(w)):
            pred += X[i, j] * w[j]
        predictions.append(pred + b)
    return np.array(predictions)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) #avg_sqred_dif


print("\nTraining with different learning rates...")
lr_list = [0.0001, 0.001, 0.01, 0.05, 1.0]
q1_results = {}

for lr in lr_list:
    #training_features,targets,curr_lr,max_iteration,stop_if_converged
    w, b, hist = q1_train(X1_train, y1_train, lr=lr, max_iter=5000) 
    q1_results[lr] = (w, b, hist)
                            #list_of_loss_vals


plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red', 'orange', 'purple']
for lr, color in zip(lr_list, colors): #pairs_lr_and_color
    _, _, hist = q1_results[lr] #ignore_w_and_b,consider_loss_history_only
    if len(hist) > 0: #if_successful(converged)
        plt.plot(hist, label=f'lr = {lr}', color=color)
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')
plt.title('Training Loss vs Iterations (All Learning Rates)')
plt.legend()
plt.tight_layout()
plt.savefig('q1_loss_all_lr.png', dpi=150)
plt.show()

print("\nObservations:")
print("  lr=0.0001 : very slow, barely converges in 5000 iterations")
print("  lr=0.001  : slow but stable")
print("  lr=0.01   : best choice, smooth and converges fast")
print("  lr=0.05   : fast but slight oscillation early on")
print("  lr=1.0    : diverges immediately")


best_w1, best_b1, best_hist1 = q1_results[0.01] #select_best_model_with_lr_0.01
if best_w1 is None:
    best_w1, best_b1, best_hist1 = q1_results[0.001]

print("\nLearned parameters (lr=0.01):")
feature_names1 = df1.drop(columns=['medv']).columns.tolist()
for name, weight in zip(feature_names1, best_w1): #print_each_feature_and_learned_weight
    print(f"  {name:>8}: {weight:>8.4f}")
print(f"  {'bias':>8}: {best_b1:>8.4f}")

y1_pred_train = q1_predict(X1_train, best_w1, best_b1) #predict_on_training_data_tocheck_if_model_learned_well
y1_pred_test = q1_predict(X1_test, best_w1, best_b1) #check_if_model_generalizes_to_unseen_data

print(f"\nEvaluation:")
print(f"  Train MSE : {mse(y1_train, y1_pred_train):.4f}")
print(f"  Test MSE  : {mse(y1_test, y1_pred_test):.4f}")
print(f"  Test RMSE : {np.sqrt(mse(y1_test, y1_pred_test)):.4f}")

# Sample predictions and failure cases
print(f"\nSample predictions (15 samples):")
print(f"  {'#':>3} {'Actual':>10} {'Predicted':>10} {'Error':>10}") 
for i, (a, p) in enumerate(zip(y1_test[:15], y1_pred_test[:15])):  #pair_first_15_actual_and_predicted_values
    print(f"  {i+1:>3} {a:>10.2f} {p:>10.2f} {abs(a-p):>10.2f}")

errors1 = np.abs(y1_test - y1_pred_test) #compute_absolute_error_for_every_test_sample
fail1 = errors1 > 10 #true_where_exceeds_10,BOOlean_array_to_identify_failures
print(f"\nFailure cases (error > 10): {fail1.sum()} out of {len(y1_test)}")
if fail1.sum() > 0:
    print(f"  {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    for a, p, e in zip(y1_test[fail1], y1_pred_test[fail1], errors1[fail1]): #actual,predicted,error_only_for_failutre_cases
        print(f"  {a:>10.2f} {p:>10.2f} {e:>10.2f}")

#compare_with_sklearn
sk_lr = LinearRegression()
sk_lr.fit(X1_train, y1_train) #train_sklearn_model_on_same_data_to_compare_predictions_and_metrics
y1_sk = sk_lr.predict(X1_test) #our_model_predictions_vs_sklearn_predictions
sk_mse1 = mse(y1_test, y1_sk) #our_equation

print(f"\nComparison with sklearn:")
print(f"  {'Metric':>10} {'Our Model':>12} {'sklearn':>12} {'Diff':>10}")
print(f"  {'MSE':>10} {mse(y1_test, y1_pred_test):>12.4f} {sk_mse1:>12.4f} {abs(mse(y1_test, y1_pred_test)-sk_mse1):>10.4f}")

#classification_metrics
def bin_prices(prices):
    bins = np.zeros(len(prices), dtype=int) #create_array_of_0_1_per_sample
    bins[prices >= 17] = 1 
    bins[prices >= 25] = 2
    return bins

y1_test_cls = bin_prices(y1_test) #convert_continuous_actual_prices_to_class_labels
y1_pred_cls = bin_prices(y1_pred_test) #convert_continuous_predicted_prices_to_class_labels
n_cls1 = 3

accuracy1 = np.mean(y1_test_cls == y1_pred_cls) 
print(f"\nClassification metrics (binned: Low<17, Medium 17-25, High>25):")
print(f"  Accuracy: {accuracy1:.4f} ({accuracy1*100:.2f}%)")

cm1 = np.zeros((n_cls1, n_cls1), dtype=int)
for t, p in zip(y1_test_cls, y1_pred_cls): #t==p,diagnol,correct_prediction
    cm1[t][p] += 1

print(f"\n  Confusion matrix (rows=Actual, cols=Predicted):")
print(f"  {'':>12} {'Low':>8} {'Medium':>8} {'High':>8}")
for i, name in enumerate(['Low', 'Medium', 'High']):
    print(f"  {name:>12}", end='')
    for j in range(n_cls1):
        print(f"{cm1[i][j]:>8}", end='')
    print()

print(f"\n  {'Class':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
f1_list1 = []
for c in range(n_cls1):
    tp = cm1[c, c]
    fp = cm1[:, c].sum() - tp
    fn = cm1[c, :].sum() - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    f1_list1.append(f1)
    print(f"  {['Low','Medium','High'][c]:>10} {prec:>12.4f} {rec:>10.4f} {f1:>10.4f}")
print(f"  {'Macro F1':>10} {'':>12} {'':>10} {np.mean(f1_list1):>10.4f}")




print("\n" + "=" * 60)
print("       QUESTION 2: LOGISTIC REGRESSION")
print("=" * 60)

df2 = pd.read_csv(r'C:\Users\User\Desktop\PAI\data.csv')
df2.drop(columns=['id', 'Unnamed: 32'], inplace=True) #keeping_would_add_noise_and_missing_vals
df2['diagnosis'] = (df2['diagnosis'] == 'M').astype(int) #convert_to_binary_1_for_M_and_0_for_B,our_model_is_binary_classifier

X2 = df2.drop(columns=['diagnosis']).values #remove_target_col_and_convert_to_numpy_array
y2 = df2['diagnosis'].values #extract_target_col_as_numpy_array_for_training_and_evaluation

# Normalize features
X2_mean = X2.mean(axis=0)
X2_std = X2.std(axis=0)
X2_std[X2_std == 0] = 1
X2_norm = (X2 - X2_mean) / X2_std #without_it,slow_or_failed_convergance

# Train/test split (80/20)
np.random.seed(42)
idx2 = np.random.permutation(len(X2_norm)) #shuffle_randomnly_to_avoid_biased_split,permute_doesnt_modify_original
split2 = int(0.8 * len(X2_norm)) #split_data_into_80_for_training_and_20_for_testing
X2_train = X2_norm[idx2[:split2]] #training_features_for_logistic_regression
X2_test = X2_norm[idx2[split2:]] #testing_features_to_evaluate_generalization
y2_train = y2[idx2[:split2]] #training_labels_for_logistic_regression
y2_test = y2[idx2[split2:]] #testing_labels_to_evaluate_predictions

print(f"\nDataset shape : {df2.shape}")
print(f"Train size    : {X2_train.shape}")
print(f"Test  size    : {X2_test.shape}")
print(f"Class balance — Train: {y2_train.sum()} M / {(y2_train==0).sum()} B")
print(f"Class balance — Test : {y2_test.sum()} M / {(y2_test==0).sum()} B")

def sigmoid(z):
    z = np.clip(z, -500, 500) #limit_value_range_to_avoid_overflow_in_exp,large_values_cause_inf_NaN
    return 1 / (1 + np.exp(-z)) #sigmoid_maps_any_real_number_to_0_1

def q2_loss(X, y, w, b):
    n = len(y) #num_of_samples_to_avg_loss_over
    y_hat = np.clip(sigmoid(X @ w + b), 1e-9, 1 - 1e-9) #predicted_probabilities_clipped_to_avoid_log_of_0_or_1,which_causes_inf_or_nan
    return -(1 / n) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) #binary_cross_entropy_loss
    #penalises_wrong_predictions_more_as_probabilities_get_more_confidently_wrong

def q2_gradients(X, y, w, b):
    n = len(y)
    error = sigmoid(X @ w + b) - y #difference_between_predicted_probabilities_and_actual_labels,
    #positive_if_overpredicting_positive_class,negative_if_underpredicting
    dw = (1 / n) * (X.T @ error) #gradient_for_weights,how_much_to_change_each_weight_to_reduce_loss,
    #positive_if_increasing_weight_increases_loss
    db = (1 / n) * np.sum(error) #gradient_for_bias,how_much_to_change_bias_to_reduce_loss,
    return dw, db

def q2_train(X, y, lr=0.1, max_iter=10000, eps=1e-4):
    w = np.zeros(X.shape[1]) #initial_weights_all_zero,convex_loss_leads_to_same_solution_from_any_start
    b = 0.0
    hist = []
    for i in range(max_iter):
        w_old = w.copy() #save_current_weights_before_update_to_check_convergence
        dw, db = q2_gradients(X, y, w, b) 
        w = w - lr * dw #update_weights_in_direction_that_reduces_loss,lr_controls_step_size
        b = b - lr * db #update_bias_similarly
        loss = q2_loss(X, y, w, b) 
        if not np.isfinite(loss):
            print(f"  lr={lr}: diverged at iteration {i+1}")
            return None, None, []
        hist.append(loss) #record_loss_history_for_plotting_and_analysis
        if converged(w, w_old, eps):
            print(f"  lr={lr}: converged at iteration {i+1}, loss={loss:.4f}")
            return w, b, hist
    print(f"  lr={lr}: max iterations reached, loss={hist[-1]:.4f}")
    return w, b, hist

def q2_predict(X, w, b, threshold=0.5):
    probs = sigmoid(X @ w + b) #predicted_probabilities_for_positive_class,values_between_0_and_1
    return (probs >= threshold).astype(int), probs #classify_as_1_if_prob_exceeds_threshold_else_0,also_return_probabilities_for_analysis

def metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)) #true_positives,correctly_predicted_positive_cases
    tn = np.sum((y_pred == 0) & (y_true == 0)) #true_negatives,correctly_predicted_negative_cases
    fp = np.sum((y_pred == 1) & (y_true == 0)) #false_positives,incorrectly_predicted_positive_cases
    fn = np.sum((y_pred == 0) & (y_true == 1)) #false_negatives,incorrectly_predicted_negative_cases
    acc = (tp + tn) / len(y_true) #accuracy,overall_correct_predictions_over_total_samples
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0 #precision,how_many_predicted_positives_are_actually_positive,avoid_division_by_zero
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0 #recall,how_many_actual_positives_are_captured_by_model,avoid_division_by_zero
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0 #F1-score,harmonic_mean_of_precision_and_recall,avoid_division_by_zero
    return acc, prec, rec, f1 

# (b) Train and plot loss
print("\nTraining logistic regression (lr=0.1)...")
best_w2, best_b2, loss_hist2 = q2_train(X2_train, y2_train, lr=0.1, max_iter=5000) 
#train_logistic_regression_with_learning_rate_0.1_and_record_loss_history

plt.figure(figsize=(8, 5))
plt.plot(loss_hist2, color='steelblue')
plt.xlabel('Iterations')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Training Loss vs Iterations')
plt.tight_layout()
plt.savefig('q2_loss.png', dpi=150)
plt.show()

# Evaluate on test set
y2_pred_test, y2_probs = q2_predict(X2_test, best_w2, best_b2) #default_threshold=0.5,also_get_probabilities_for_analysis
a2, p2, r2, f2 = metrics(y2_test, y2_pred_test) #calculate_classification_metrics_on_test_set_using_threshold_0.5

print(f"\nEvaluation (threshold=0.5):")
print(f"  Accuracy : {a2:.4f}")
print(f"  Precision: {p2:.4f}")
print(f"  Recall   : {r2:.4f}")
print(f"  F1-Score : {f2:.4f}")

# (c) 5-fold cross-validation
def k_fold_cv(X, y, k=5, lr=0.1, max_iter=3000):
    n = len(X)
    fold_size = n // k #size_of_each_fold_for_validation,integer_division_to_get_equal_folds
    indices = np.random.permutation(n) #shuffle_indices_to_randomize_fold_assignment,permute_doesnt_modify_original
    acc_list, prec_list, rec_list, f1_list = [], [], [], [] #lists_to_record_metrics_for_each_fold
    print(f"\n{k}-Fold Cross Validation:") 
    for fold in range(k):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size] #indices_for_current_validation_fold
        train_idx = np.concatenate([indices[:fold * fold_size], 
                                    indices[(fold + 1) * fold_size:]]) #indices_for_training_folds_combined
        w, b, _ = q2_train(X[train_idx], y[train_idx], lr=lr, max_iter=max_iter) #train_on_training_folds_only
        if w is None:
            continue
        y_val_pred, _ = q2_predict(X[val_idx], w, b) #predict_on_validation_fold_using_trained_model_from_training_folds
        a, p, r, f = metrics(y[val_idx], y_val_pred) #calculate_metrics_on_validation_fold
        acc_list.append(a)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f)
        print(f"  Fold {fold+1}: Acc={a:.4f}, Prec={p:.4f}, Rec={r:.4f}, F1={f:.4f}")
    print(f"\n  Avg Accuracy : {np.mean(acc_list):.4f} +/- {np.std(acc_list):.4f}")
    print(f"  Avg Precision: {np.mean(prec_list):.4f} +/- {np.std(prec_list):.4f}")
    print(f"  Avg Recall   : {np.mean(rec_list):.4f} +/- {np.std(rec_list):.4f}")
    print(f"  Avg F1-Score : {np.mean(f1_list):.4f} +/- {np.std(f1_list):.4f}")

np.random.seed(42)
k_fold_cv(X2_norm, y2, k=5)

# (d) Compare with sklearn
sk_log = LogisticRegression(max_iter=5000) #train_sklearn_logistic_regression_on_same_data_to_compare_metrics_and_predictions
sk_log.fit(X2_train, y2_train) #train_sklearn_model_on_training_data
y2_sk = sk_log.predict(X2_test) #predict_on_test_data_using_sklearn_model_to_compare_with_our_model_predictions

sk_a, sk_p, sk_r, sk_f = metrics(y2_test, y2_sk)

print(f"\nComparison with sklearn:")
print(f"  {'Metric':>12} {'Our Model':>12} {'sklearn':>12}")
print(f"  {'Accuracy':>12} {a2:>12.4f} {sk_a:>12.4f}")
print(f"  {'Precision':>12} {p2:>12.4f} {sk_p:>12.4f}")
print(f"  {'Recall':>12} {r2:>12.4f} {sk_r:>12.4f}")
print(f"  {'F1-Score':>12} {f2:>12.4f} {sk_f:>12.4f}")

# (e) Threshold tuning
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
th_acc, th_prec, th_rec, th_f1 = [], [], [], []

print(f"\nThreshold tuning:")
print(f"  {'Threshold':>10} {'Accuracy':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
for th in thresholds:
    y_th, _ = q2_predict(X2_test, best_w2, best_b2, threshold=th)
    a, p, r, f = metrics(y2_test, y_th)
    th_acc.append(a)
    th_prec.append(p)
    th_rec.append(r)
    th_f1.append(f)
    print(f"  {th:>10.1f} {a:>10.4f} {p:>12.4f} {r:>10.4f} {f:>10.4f}")

# Plot threshold metrics
plt.figure(figsize=(9, 5))
plt.plot(thresholds, th_acc, marker='o', label='Accuracy', color='blue')
plt.plot(thresholds, th_prec, marker='s', label='Precision', color='green')
plt.plot(thresholds, th_rec, marker='^', label='Recall', color='red')
plt.plot(thresholds, th_f1, marker='D', label='F1-Score', color='orange')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Classification Threshold')
plt.xticks(thresholds)
plt.legend()
plt.tight_layout()
plt.savefig('q2_threshold.png', dpi=150)
plt.show()

print("\nThreshold discussion:")
print("  Lower threshold (0.3) = higher recall, more false positives")
print("  Higher threshold (0.7) = higher precision, misses more cases")
print("  Threshold 0.5 = balanced trade-off")
print("  For medical diagnosis = lower threshold preferred since")
print("    missing a cancer case is worse than a false alarm")

print("\n" + "=" * 60)
print("            ALL QUESTIONS COMPLETE")
print("=" * 60)