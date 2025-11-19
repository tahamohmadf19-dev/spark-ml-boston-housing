# Spark ML Regression - Boston Housing Dataset

## نظرة عامة

هذا المشروع يوضح تطبيق نماذج الانحدار (Regression) باستخدام **PySpark ML** على مجموعة بيانات Boston Housing، مع استخدام أفضل الممارسات في Spark ML.

## المحتويات

- **Spark_ML_Regression_Boston_Housing.ipynb**: النوتبوك الرئيسي للمشروع
- **README.md**: هذا الملف
- **requirements.txt**: المتطلبات البرمجية

## المتطلبات

```
pyspark>=3.0.0
pandas>=1.0.0
matplotlib>=3.0.0
numpy>=1.18.0
```

## كيفية الاستخدام

### 1. تثبيت المتطلبات

```bash
pip install -r requirements.txt
```

### 2. تحضير البيانات

تأكد من وجود ملف `Boston House Price Data.csv` أو قم بتحديث المسار في الكود.

### 3. تشغيل النوتبوك

#### باستخدام Jupyter Notebook:
```bash
jupyter notebook Spark_ML_Regression_Boston_Housing.ipynb
```

#### باستخدام Google Colab:
1. ارفع النوتبوك إلى Google Drive
2. افتحه باستخدام Google Colab
3. ارفع ملف البيانات إلى `/content/`
4. شغّل الـ cells

## المنهجية

### تقسيم البيانات
- **70%** للتدريب (Training)
- **15%** للتحقق (Validation)
- **15%** للاختبار (Test)

### معالجة البيانات
- فحص القيم المفقودة
- معالجة القيم الشاذة (Outliers) باستخدام IQR
- التطبيع (Standardization) باستخدام Z-score

### هندسة الميزات
إنشاء ميزات تفاعلية:
- **RM_LSTAT**: التفاعل بين متوسط عدد الغرف ونسبة السكان ذوي الوضع المنخفض
- **NOX_INDUS**: التفاعل بين تركيز أكسيد النيتريك والنسبة الصناعية
- **DIS_RAD**: التفاعل بين المسافة لمراكز العمل وإمكانية الوصول للطرق السريعة

### النماذج المستخدمة
1. **Linear Regression**: نموذج خطي بسيط
2. **Decision Tree Regressor**: نموذج شجرة قرار
3. **Random Forest Regressor**: نموذج الغابة العشوائية مع Pipeline و CrossValidator

## الميزات الرئيسية

- ✅ استخدام **Pipeline** لتنظيم خطوات المعالجة والنمذجة
- ✅ **CrossValidator** مع **ParamGridBuilder** لضبط المعاملات تلقائياً
- ✅ معالجة منهجية للقيم الشاذة والتطبيع
- ✅ تقييم شامل للنماذج باستخدام RMSE, MAE, R²
- ✅ تحليل أهمية الميزات (Feature Importance)
- ✅ حفظ النموذج النهائي

## البنية

```
boston_housing_spark/
├── Spark_ML_Regression_Boston_Housing.ipynb  # النوتبوك الرئيسي
├── README.md                                  # هذا الملف
└── requirements.txt                           # المتطلبات
```

## النتائج المتوقعة

النماذج تحقق أداءً جيداً على مجموعة بيانات Boston Housing:
- **Linear Regression**: RMSE ~4.0
- **Decision Tree**: RMSE ~5.0
- **Random Forest (Tuned)**: RMSE ~4.5, R² ~0.78

*القيم الفعلية قد تختلف قليلاً حسب التقسيم العشوائي*

## التطبيق على بيانات أخرى

يمكن تطبيق نفس المنهجية على مجموعات بيانات regression أخرى:
1. قم بتحميل البيانات الجديدة
2. حدد العمود المستهدف (target column)
3. عدّل هندسة الميزات حسب البيانات
4. شغّل نفس الخطوات

## المراجع

- [Apache Spark ML Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
- [PySpark ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html)

## المؤلف

تم تطوير هذا المشروع كجزء من مشروع في تحليلات البيانات الكبيرة باستخدام Apache Spark.

## الترخيص

هذا المشروع مفتوح المصدر ومتاح للاستخدام التعليمي والبحثي.

---

**رابط المشروع:** https://github.com/tahamohmadf19-dev/spark-ml-boston-housing
