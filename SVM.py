# load dataset and analyze for features
import pandas as pd
df = pd.read_csv('CancerData.csv')
print(df) # for Windows Command Prompt, print() function not necessary

# make categorical variables to categorical type
cat_vars = []
df[cat_vars] = df[]

# check for and count missing values in the dataset
print(df.isna().sum())

# since no missing values, no need to impute them
# now, get count plot for the breast cancer outcome
from matplotlib import pyplot as plt
import seaborn as sns
ax = sn.countplot(x='', data=df)
plt.show()

# split dataset into training and testing datasets
from sklearn.model_selection import train_test_split
X = df.iloc[]
y = df.iloc['Classification']
# split as 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
'''
random_state parameter ensures that the train_test function will reproduce the same train and test data (precision).
It can be set to any integer.
'''

# fit SVM model with training data
from sklearn.svm import SVC
svm = SVC(C=1, kernel='linear', random_state=1)
svm.fit(X=X_train, y=y_train)

# perform classification prediction using testing dataset from fitted SVM model
y_pred = svm.predict(X=X_test)
print(y_pred)

# evaluate classification prediction from fitted SVM model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# confusion matrix
confusion_matrix(y_true=y_test, y_pred=y_pred)

# fitted SVM model accuracy
accuracy_score(y_true=y_test, y_pred=y_pred)

'''
In the confusion matrix, the trace (main diagonal; from top left to bottom right) indicate correct predictions
The elements of the other diagonal indicate incorrect predictions, e.g., false positives and false negatives
'''

# plot receiver operating characteristic (ROC) curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
from bioinfokit.analys import stat
y_score = svm.decision_function(X=X_test)
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)
auc = roc_auc_score(y_true=y_test, y_score=y_score)
# plot ROC
stat.roc(fpr=fpr, auc=auc, shade_auc=True, per_class=True,legendpos='upper center', legendanchor=(0.5, 1.08), legendcols=3)

# plot area of precision-recall curve (PRC); integrate the function with respective lower and upper bounds (AUPRC)
from sklearn.metrics import precision_recall_curve, average_precision_score, plot_precision_recall_curve
import matplotlib.pyplot as plt
average_precision = average_precision_score(y_true=y_test, y_score=y_score)
# plot AUPRC
disp = plot_precision_recall_curve(estimator=svm, X=X_test, y=y_test)
disp.ax_.set_title('2-class Precision-Recall curve: 'AP={0:0.2f}'.format(average_precision))
plt.show()
