import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# تولید داده نمونه (در کاربرد واقعی، باید داده‌های واقعی جمع‌آوری شود)
data = {
    'student_id': np.arange(1, 101),
    'absences': np.random.randint(0, 20, 100),  # تعداد غیبت‌ها
    'delays': np.random.randint(0, 10, 100),  # تعداد تأخیرها
    'learning_disorder': np.random.choice([0, 1], 100, p=[0.7, 0.3])  # اختلال یادگیری (1: دارد، 0: ندارد)
}
df = pd.DataFrame(data)

# ویژگی‌ها و برچسب هدف
X = df[['absences', 'delays']]
y = df['learning_disorder']

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ایجاد مدل و آموزش آن
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# پیش‌بینی و ارزیابی مدل
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# تابع پیش‌بینی اختلال یادگیری برای دانش‌آموز خاص
def predict_learning_disorder(absences, delays):
    prediction = model.predict([[absences, delays]])[0]
    return 'Learning Disorder' if prediction == 1 else 'No Learning Disorder'

# تست مدل
print(predict_learning_disorder(15, 5))
