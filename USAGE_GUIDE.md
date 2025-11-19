# دليل الاستخدام - Spark ML Boston Housing

## 📋 المحتويات

1. [الملفات المتاحة](#الملفات-المتاحة)
2. [كيفية تشغيل الكود](#كيفية-تشغيل-الكود)
3. [شرح الكود المحسّن](#شرح-الكود-المحسّن)
4. [التخصيص للبيانات الخاصة بك](#التخصيص-للبيانات-الخاصة-بك)

---

## الملفات المتاحة

### 1. `improved_spark_ml_boston.py` ⭐
**الكود المحسّن كـ Python script**
- يمكن تشغيله مباشرة من terminal
- منظم في دوال قابلة لإعادة الاستخدام
- يحتوي على جميع التحسينات

### 2. `Improved_Spark_ML_Boston_Housing.ipynb`
**النوتبوك المحسّن**
- للاستخدام في Jupyter Notebook
- يحتوي على شروحات وتوثيق
- مناسب للتعلم والعرض

### 3. `original_notebook.ipynb`
**النوتبوك الأصلي**
- للمقارنة مع النسخة المحسّنة

---

## كيفية تشغيل الكود

### الطريقة 1: تشغيل Python Script

```bash
# 1. التأكد من تثبيت المتطلبات
pip install -r requirements.txt

# 2. تحضير ملف البيانات
# ضع ملف "Boston House Price Data.csv" في نفس المجلد

# 3. تشغيل الكود
python improved_spark_ml_boston.py
```

### الطريقة 2: تشغيل Jupyter Notebook

```bash
# 1. تشغيل Jupyter
jupyter notebook

# 2. فتح النوتبوك
# افتح ملف: Improved_Spark_ML_Boston_Housing.ipynb

# 3. تشغيل الـ cells واحدة تلو الأخرى
# أو Run All من قائمة Cell
```

### الطريقة 3: استخدام Google Colab

```python
# 1. ارفع النوتبوك إلى Google Drive
# 2. افتحه باستخدام Google Colab
# 3. ثبّت PySpark في أول cell:

!pip install pyspark

# 4. ارفع ملف البيانات:
from google.colab import files
uploaded = files.upload()

# 5. شغّل باقي الـ cells
```

---

## شرح الكود المحسّن

### البنية العامة

```python
# 1. INITIALIZATION - تهيئة Spark
spark = initialize_spark()

# 2. DATA LOADING - تحميل البيانات
df = load_data(spark, "Boston House Price Data.csv")

# 3. EXPLORATION - استكشاف البيانات
check_missing_values(df)
visualize_target(df)

# 4. FEATURE ENGINEERING - هندسة الميزات
df_fe = feature_engineering(df)

# 5. DATA SPLITTING - تقسيم البيانات
train_df, val_df, test_df = split_data(df_fe)

# 6. OUTLIER HANDLING - معالجة القيم الشاذة
bounds = calculate_outlier_bounds(train_df, numeric_cols)
train_cap = cap_outliers(train_df, bounds)

# 7. STANDARDIZATION - التطبيع
stats_row = calculate_standardization_stats(train_cap, numeric_cols)
train_final = standardize(train_cap, numeric_cols, stats_row)

# 8. MODEL TRAINING - تدريب النماذج
lr_model = train_linear_regression(...)
dt_model = train_decision_tree(...)
rf_model = train_random_forest_with_cv(...)  # مع CrossValidator

# 9. EVALUATION - التقييم
results = evaluate_model(...)
compare_models([lr_results, dt_results, rf_results])

# 10. FEATURE IMPORTANCE - أهمية الميزات
importance_df = analyze_feature_importance(best_rf, feature_cols)

# 11. TEST EVALUATION - التقييم النهائي
test_pred = evaluate_on_test(rf_model, test_final)

# 12. MODEL PERSISTENCE - حفظ النموذج
save_model(rf_model)
```

---

## الدوال الرئيسية وكيفية استخدامها

### 1. تهيئة Spark

```python
spark = initialize_spark()
```

**الوظيفة:** إنشاء Spark Session  
**المخرجات:** كائن SparkSession

---

### 2. تحميل البيانات

```python
df = load_data(spark, file_path="path/to/data.csv")
```

**المعاملات:**
- `spark`: SparkSession object
- `file_path`: مسار ملف CSV

**المخرجات:** Spark DataFrame

---

### 3. هندسة الميزات

```python
df_fe = feature_engineering(df)
```

**الوظيفة:** إنشاء ميزات تفاعلية:
- `RM_LSTAT = RM × LSTAT`
- `NOX_INDUS = NOX × INDUS`
- `DIS_RAD = DIS × RAD`

**المخرجات:** DataFrame مع الميزات الجديدة

---

### 4. معالجة القيم الشاذة

```python
# حساب الحدود
bounds = calculate_outlier_bounds(train_df, numeric_cols)

# تطبيق المعالجة
train_cap = cap_outliers(train_df, bounds)
val_cap = cap_outliers(val_df, bounds)
test_cap = cap_outliers(test_df, bounds)
```

**المنهجية:** IQR Method
- Lower: Q1 - 1.5 × IQR
- Upper: Q3 + 1.5 × IQR

**مهم:** احسب الحدود من train فقط لمنع data leakage!

---

### 5. التطبيع (Standardization)

```python
# حساب الإحصائيات
stats_row = calculate_standardization_stats(train_cap, numeric_cols)

# تطبيق التطبيع
train_final = standardize(train_cap, numeric_cols, stats_row)
val_final = standardize(val_cap, numeric_cols, stats_row)
test_final = standardize(test_cap, numeric_cols, stats_row)
```

**المعادلة:** `(x - mean) / std`

**مهم:** استخدم mean و std من train فقط!

---

### 6. تدريب Random Forest مع CrossValidator

```python
rf_model, best_rf = train_random_forest_with_cv(
    train_final, 
    assembler, 
    target_column
)
```

**المعاملات المُختبرة:**
- `numTrees`: [50, 100, 200]
- `maxDepth`: [5, 8, 10]
- `minInstancesPerNode`: [1, 2, 4]

**المجموع:** 27 combination

**المخرجات:**
- `rf_model`: أفضل Pipeline model
- `best_rf`: أفضل RandomForestRegressor

---

### 7. تقييم النماذج

```python
# تقييم نموذج واحد
results = evaluate_model(train_pred, val_pred, "Model Name", target_column)

# مقارنة جميع النماذج
results_df = compare_models([lr_results, dt_results, rf_results])
```

**المقاييس:**
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

---

### 8. تحليل أهمية الميزات

```python
importance_df = analyze_feature_importance(best_rf, feature_cols)
```

**المخرجات:**
- Pandas DataFrame مع Feature و Importance
- رسم بياني للأهمية
- طباعة أهم 3 ميزات

---

### 9. حفظ النموذج

```python
save_model(rf_model, path="./my_model")
```

**لتحميل النموذج لاحقاً:**

```python
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load("./my_model")
```

---

## التخصيص للبيانات الخاصة بك

### 1. تغيير مسار البيانات

في دالة `main()`:

```python
# قبل
file_path = "Boston House Price Data.csv"

# بعد
file_path = "/path/to/your/data.csv"
```

---

### 2. تغيير العمود المستهدف

```python
# قبل
target_column = "PRICE"

# بعد
target_column = "YourTargetColumn"
```

---

### 3. تخصيص هندسة الميزات

في دالة `feature_engineering()`:

```python
def feature_engineering(df):
    df_fe = (
        df
        # أضف ميزاتك الخاصة هنا
        .withColumn("Feature1_Feature2", col("Feature1") * col("Feature2"))
        .withColumn("Feature3_squared", col("Feature3") ** 2)
    )
    
    # حدد الأعمدة المراد حذفها
    cols_to_drop = ["OldFeature1", "OldFeature2"]
    df_fe = df_fe.drop(*cols_to_drop)
    
    return df_fe
```

---

### 4. تخصيص معاملات Random Forest

في دالة `train_random_forest_with_cv()`:

```python
# قبل
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [5, 8, 10]) \
    .addGrid(rf.minInstancesPerNode, [1, 2, 4]) \
    .build()

# بعد - جرّب قيم مختلفة
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [100, 200, 300]) \
    .addGrid(rf.maxDepth, [8, 10, 15]) \
    .addGrid(rf.minInstancesPerNode, [2, 5, 10]) \
    .addGrid(rf.maxBins, [32, 64]) \  # معامل إضافي
    .build()
```

---

### 5. تغيير نسب التقسيم

في دالة `split_data()`:

```python
# قبل
train_df, val_df, test_df = split_data(df, 0.7, 0.15, 0.15)

# بعد - مثلاً 80/10/10
train_df, val_df, test_df = split_data(df, 0.8, 0.1, 0.1)
```

---

### 6. إضافة نموذج جديد

```python
def train_gradient_boosted_tree(train_final, assembler, target_column):
    """Train Gradient Boosted Tree Regressor"""
    from pyspark.ml.regression import GBTRegressor
    
    gbt = GBTRegressor(featuresCol="features", labelCol=target_column, seed=42)
    gbt_pipeline = Pipeline(stages=[assembler, gbt])
    
    print("\nTraining Gradient Boosted Tree...")
    gbt_model = gbt_pipeline.fit(train_final)
    print("GBT training completed!")
    
    return gbt_model

# في main()
gbt_model = train_gradient_boosted_tree(train_final, assembler, target_column)
gbt_train_pred = gbt_model.transform(train_final)
gbt_val_pred = gbt_model.transform(val_final)
gbt_results = evaluate_model(gbt_train_pred, gbt_val_pred, "GBT", target_column)

# أضف للمقارنة
results_df = compare_models([lr_results, dt_results, rf_results, gbt_results])
```

---

## أمثلة للاستخدام المتقدم

### 1. تشغيل الكود على Spark Cluster

```python
spark = SparkSession.builder \
    .appName("BostonHousingRegression") \
    .master("spark://master:7077") \  # عنوان الـ cluster
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()
```

---

### 2. قراءة البيانات من Parquet

```python
df = spark.read.parquet("path/to/data.parquet")
```

---

### 3. حفظ النتائج

```python
# حفظ التنبؤات
test_pred.select("PRICE", "prediction").write.csv("predictions.csv", header=True)

# حفظ جدول المقارنة
results_df.to_csv("model_comparison.csv", index=False)

# حفظ Feature Importance
importance_df.to_csv("feature_importance.csv", index=False)
```

---

## نصائح مهمة

### ✅ أفضل الممارسات

1. **دائماً استخدم seed للـ reproducibility**
   ```python
   randomSplit([0.7, 0.15, 0.15], seed=42)
   ```

2. **احسب الإحصائيات من train فقط**
   ```python
   # ✅ صح
   bounds = calculate_outlier_bounds(train_df, numeric_cols)
   
   # ❌ خطأ
   bounds = calculate_outlier_bounds(df, numeric_cols)  # data leakage!
   ```

3. **استخدم cache() للبيانات المستخدمة بكثرة**
   ```python
   train_final.cache()
   val_final.cache()
   ```

4. **استخدم Pipeline دائماً**
   ```python
   pipeline = Pipeline(stages=[assembler, model])
   ```

---

### ⚠️ أخطاء شائعة

1. **نسيان تحديث مسار البيانات**
   - تأكد من تحديث `file_path` في `main()`

2. **عدم مطابقة أسماء الأعمدة**
   - تأكد أن `target_column` موجود في البيانات

3. **تشغيل على بيانات كبيرة بدون cluster**
   - للبيانات الكبيرة، استخدم Spark cluster

4. **نسيان stop() للـ Spark session**
   - دائماً أنهِ بـ `spark.stop()`

---

## الأسئلة الشائعة

### س: كيف أغير عدد الـ folds في CrossValidator؟

```python
crossval = CrossValidator(
    estimator=rf_pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5,  # غيّر هنا (الافتراضي 3)
    seed=42
)
```

---

### س: كيف أستخدم metric مختلف للتقييم؟

```python
# للتقييم بـ MAE بدلاً من RMSE
evaluator = RegressionEvaluator(
    labelCol=target_column,
    predictionCol="prediction",
    metricName="mae"  # أو "r2"
)
```

---

### س: كيف أحفظ الرسوم البيانية؟

```python
# بعد plt.show()
plt.savefig("comparison_plot.png", dpi=300, bbox_inches='tight')
```

---

### س: الكود بطيء جداً، كيف أسرّعه؟

1. استخدم `cache()` أكثر
2. قلل عدد الـ hyperparameters في ParamGrid
3. قلل `numFolds` في CrossValidator
4. استخدم Spark cluster

---

## الخلاصة

هذا الكود جاهز للاستخدام مباشرة ويمكن تخصيصه بسهولة لأي dataset regression. 

**للمزيد من المعلومات:**
- راجع [README.md](README.md) للنظرة العامة
- راجع [improvements_summary.md](improvements_summary.md) لتفاصيل التحسينات
- راجع [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) للتقرير الشامل

**رابط المشروع:** https://github.com/tahamohmadf19-dev/spark-ml-boston-housing

---

**بالتوفيق! 🚀**
