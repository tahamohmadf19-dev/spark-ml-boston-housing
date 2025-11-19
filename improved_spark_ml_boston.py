"""
Spark ML Regression - Boston Housing (Improved Version)

This script demonstrates a production-ready approach to regression using PySpark ML
on the Boston Housing dataset with Pipeline, CrossValidator, and comprehensive analysis.

Author: Spark ML Project
Date: November 2025
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, count, isnan, mean as _mean, stddev as _stddev
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# 1. INITIALIZATION
# ============================================================================

def initialize_spark():
    """Initialize Spark Session"""
    spark = SparkSession.builder \
        .appName("BostonHousingRegression_Improved") \
        .getOrCreate()
    
    print("=" * 80)
    print("Spark Session Created Successfully!")
    print(f"Spark Version: {spark.version}")
    print("=" * 80)
    return spark


# ============================================================================
# 2. DATA LOADING AND EXPLORATION
# ============================================================================

def load_data(spark, file_path):
    """Load the Boston Housing dataset"""
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    print("\nDataset Schema:")
    df.printSchema()
    
    print("\nFirst 5 rows:")
    df.show(5)
    
    print(f"\nTotal records: {df.count()}")
    
    return df


def check_missing_values(df):
    """Check for missing values in the dataset"""
    missing_per_col = df.select([
        count(when(col(c).isNull() | isnan(c), c)).alias(c)
        for c in df.columns
    ])
    
    print("\nMissing Values per Column:")
    missing_per_col.show()
    
    return missing_per_col


def visualize_target(df, target_column="PRICE"):
    """Visualize the distribution of the target variable"""
    price_data = df.select(target_column).toPandas()
    
    plt.figure(figsize=(10, 6))
    plt.hist(price_data[target_column], bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {target_column}', fontsize=14, fontweight='bold')
    plt.xlabel(target_column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    plt.show()
    
    print(f"\nTarget Variable Statistics:")
    print(f"Mean: {price_data[target_column].mean():.2f}")
    print(f"Median: {price_data[target_column].median():.2f}")
    print(f"Std Dev: {price_data[target_column].std():.2f}")


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):
    """
    Create interaction features to capture complex relationships
    
    Features created:
    - RM_LSTAT: Interaction between average rooms and lower status population
    - NOX_INDUS: Interaction between nitric oxide and industrial proportion
    - DIS_RAD: Interaction between distance to employment and highway accessibility
    """
    df_fe = (
        df
        .withColumn("RM_LSTAT", col("RM") * col("LSTAT"))
        .withColumn("NOX_INDUS", col("NOX") * col("INDUS"))
        .withColumn("DIS_RAD", col("DIS") * col("RAD"))
    )
    
    # Drop original features used in interactions
    cols_to_drop = ["RM", "LSTAT", "NOX", "INDUS", "DIS", "RAD"]
    df_fe = df_fe.drop(*cols_to_drop)
    
    print("\nDataset after Feature Engineering:")
    df_fe.printSchema()
    df_fe.show(5, truncate=False)
    
    return df_fe


# ============================================================================
# 4. DATA SPLITTING
# ============================================================================

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        train_df, val_df, test_df
    """
    train_df, val_df, test_df = df.randomSplit([train_ratio, val_ratio, test_ratio], seed=seed)
    
    print("\nData Split Summary:")
    print(f"Train count      : {train_df.count()}")
    print(f"Validation count : {val_df.count()}")
    print(f"Test count       : {test_df.count()}")
    
    # Cache datasets for better performance
    train_df.cache()
    val_df.cache()
    test_df.cache()
    
    print("Datasets cached successfully!")
    
    return train_df, val_df, test_df


# ============================================================================
# 5. OUTLIER HANDLING
# ============================================================================

def calculate_outlier_bounds(train_df, numeric_cols):
    """
    Calculate outlier bounds using IQR method
    
    Method: IQR (Interquartile Range)
    - Lower bound: Q1 - 1.5 × IQR
    - Upper bound: Q3 + 1.5 × IQR
    
    Args:
        train_df: Training DataFrame (to prevent data leakage)
        numeric_cols: List of numeric column names
    
    Returns:
        Dictionary with column names as keys and (lower, upper) tuples as values
    """
    bounds = {}
    
    print("\nCalculating outlier bounds for each feature...\n")
    print(f"{'Feature':<12} {'Lower Bound':>12} {'Upper Bound':>12} {'IQR':>12}")
    print("-" * 50)
    
    for c in numeric_cols:
        q1, q3 = train_df.approxQuantile(c, [0.25, 0.75], 0.05)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds[c] = (lower, upper)
        print(f"{c:<12} {lower:12.3f} {upper:12.3f} {iqr:12.3f}")
    
    return bounds


def cap_outliers(df, bounds_dict):
    """
    Cap outliers in the dataframe based on pre-calculated bounds
    
    Args:
        df: Input Spark DataFrame
        bounds_dict: Dictionary with column names as keys and (lower, upper) tuples
    
    Returns:
        DataFrame with outliers capped
    """
    df_out = df
    for c, (lower, upper) in bounds_dict.items():
        df_out = df_out.withColumn(
            c,
            when(col(c) < lower, lower)
            .when(col(c) > upper, upper)
            .otherwise(col(c))
        )
    return df_out


# ============================================================================
# 6. STANDARDIZATION
# ============================================================================

def calculate_standardization_stats(train_cap, numeric_cols):
    """
    Calculate mean and standard deviation from training data
    
    Args:
        train_cap: Training DataFrame after outlier capping
        numeric_cols: List of numeric column names
    
    Returns:
        Row object containing mean and std for each column
    """
    stats_row = train_cap.select(
        *[_mean(c).alias(f"{c}_mean") for c in numeric_cols],
        *[_stddev(c).alias(f"{c}_std") for c in numeric_cols]
    ).collect()[0]
    
    return stats_row


def standardize(df, numeric_cols, stats):
    """
    Standardize numeric columns using pre-calculated mean and std
    
    Formula: (x - mean) / std
    
    Args:
        df: Input Spark DataFrame
        numeric_cols: List of column names to standardize
        stats: Row object containing mean and std for each column
    
    Returns:
        Standardized DataFrame
    """
    df_std = df
    for c in numeric_cols:
        mean_val = stats[f"{c}_mean"]
        std_val = stats[f"{c}_std"]
        
        # Skip if std is 0 or None (constant column)
        if std_val is None or std_val == 0:
            print(f"Warning: Skipping {c} (std = {std_val})")
            continue
        
        df_std = df_std.withColumn(
            c,
            (col(c) - mean_val) / std_val
        )
    return df_std


# ============================================================================
# 7. MODEL TRAINING
# ============================================================================

def train_linear_regression(train_final, feature_cols, target_column):
    """Train Linear Regression model with Pipeline"""
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol=target_column, maxIter=100)
    lr_pipeline = Pipeline(stages=[assembler, lr])
    
    print("\nTraining Linear Regression...")
    lr_model = lr_pipeline.fit(train_final)
    print("Linear Regression training completed!")
    
    return lr_model, assembler


def train_decision_tree(train_final, assembler, target_column):
    """Train Decision Tree Regressor with Pipeline"""
    dt = DecisionTreeRegressor(featuresCol="features", labelCol=target_column, seed=42)
    dt_pipeline = Pipeline(stages=[assembler, dt])
    
    print("\nTraining Decision Tree...")
    dt_model = dt_pipeline.fit(train_final)
    print("Decision Tree training completed!")
    
    return dt_model


def train_random_forest_with_cv(train_final, assembler, target_column):
    """
    Train Random Forest with CrossValidator for hyperparameter tuning
    
    Hyperparameters tested:
    - numTrees: [50, 100, 200]
    - maxDepth: [5, 8, 10]
    - minInstancesPerNode: [1, 2, 4]
    
    Total combinations: 27
    """
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=target_column,
        seed=42
    )
    
    rf_pipeline = Pipeline(stages=[assembler, rf])
    
    # Define parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(rf.maxDepth, [5, 8, 10]) \
        .addGrid(rf.minInstancesPerNode, [1, 2, 4]) \
        .build()
    
    # Create evaluator
    evaluator = RegressionEvaluator(
        labelCol=target_column,
        predictionCol="prediction",
        metricName="rmse"
    )
    
    # Create CrossValidator
    crossval = CrossValidator(
        estimator=rf_pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        seed=42
    )
    
    print("\nStarting Cross-Validation for Random Forest...")
    print(f"Total combinations to test: {len(paramGrid)}")
    print("This may take a few minutes...\n")
    
    # Train with cross-validation
    cv_model = crossval.fit(train_final)
    
    # Get best model
    best_rf_model = cv_model.bestModel
    best_rf = best_rf_model.stages[-1]
    
    print("\nBest Random Forest Parameters:")
    print(f"Number of Trees: {best_rf.getNumTrees}")
    print(f"Max Depth: {best_rf.getMaxDepth()}")
    print(f"Min Instances Per Node: {best_rf.getMinInstancesPerNode()}")
    print("\nRandom Forest training with CV completed!")
    
    return best_rf_model, best_rf


# ============================================================================
# 8. MODEL EVALUATION
# ============================================================================

def evaluate_model(train_pred, val_pred, model_name, target_column):
    """
    Evaluate model performance on train and validation sets
    
    Metrics:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - R²: Coefficient of Determination
    
    Args:
        train_pred: Training predictions DataFrame
        val_pred: Validation predictions DataFrame
        model_name: Name of the model
        target_column: Name of the target column
    
    Returns:
        Dictionary with evaluation metrics
    """
    rmse_eval = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")
    mae_eval = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mae")
    r2_eval = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2")
    
    results = {
        'Model': model_name,
        'Train_RMSE': rmse_eval.evaluate(train_pred),
        'Val_RMSE': rmse_eval.evaluate(val_pred),
        'Train_MAE': mae_eval.evaluate(train_pred),
        'Val_MAE': mae_eval.evaluate(val_pred),
        'Train_R2': r2_eval.evaluate(train_pred),
        'Val_R2': r2_eval.evaluate(val_pred)
    }
    
    return results


def compare_models(results_list):
    """Create and display comparison table for all models"""
    results_df = pd.DataFrame(results_list)
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - TRAIN & VALIDATION PERFORMANCE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    
    return results_df


def visualize_comparison(results_df):
    """Visualize model comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['RMSE', 'MAE', 'R2']
    for idx, metric in enumerate(metrics):
        train_col = f'Train_{metric}'
        val_col = f'Val_{metric}'
        
        x = range(len(results_df))
        width = 0.35
        
        axes[idx].bar([i - width/2 for i in x], results_df[train_col], width, label='Train', alpha=0.8)
        axes[idx].bar([i + width/2 for i in x], results_df[val_col], width, label='Validation', alpha=0.8)
        
        axes[idx].set_xlabel('Model', fontsize=11)
        axes[idx].set_ylabel(metric, fontsize=11)
        axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(results_df['Model'], rotation=15, ha='right')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(best_rf, feature_cols):
    """
    Extract and visualize feature importances from Random Forest
    
    Args:
        best_rf: Best Random Forest model
        feature_cols: List of feature column names
    
    Returns:
        DataFrame with feature importances
    """
    feature_importances = best_rf.featureImportances.toArray()
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    print(importance_df.to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], alpha=0.8)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nTop 3 Most Important Features:")
    for idx, row in importance_df.head(3).iterrows():
        print(f"  {row['Feature']:12s}: {row['Importance']:.4f}")
    
    return importance_df


# ============================================================================
# 10. TEST SET EVALUATION
# ============================================================================

def evaluate_on_test(best_model, test_final, target_column, val_rmse):
    """Evaluate best model on test set"""
    rmse_eval = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")
    mae_eval = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mae")
    r2_eval = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2")
    
    test_pred = best_model.transform(test_final)
    
    test_rmse = rmse_eval.evaluate(test_pred)
    test_mae = mae_eval.evaluate(test_pred)
    test_r2 = r2_eval.evaluate(test_pred)
    
    print("\n" + "=" * 60)
    print("FINAL TEST SET PERFORMANCE - Random Forest (Best Model)")
    print("=" * 60)
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE : {test_mae:.4f}")
    print(f"Test R²  : {test_r2:.4f}")
    print("=" * 60)
    
    print(f"\nValidation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE:       {test_rmse:.4f}")
    print(f"Difference:      {abs(test_rmse - val_rmse):.4f}")
    
    if abs(test_rmse - val_rmse) < 0.5:
        print("\n✓ Model generalizes well! Test performance is close to validation.")
    else:
        print("\n⚠ Significant difference between validation and test performance.")
    
    return test_pred


# ============================================================================
# 11. MODEL PERSISTENCE
# ============================================================================

def save_model(model, path="./best_rf_model"):
    """Save the best model"""
    model.write().overwrite().save(path)
    print(f"\nBest model saved to: {path}")
    print("To load later: from pyspark.ml import PipelineModel; model = PipelineModel.load(path)")


# ============================================================================
# 12. MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # 1. Initialize Spark
    spark = initialize_spark()
    
    # 2. Load data (update path as needed)
    file_path = "Boston House Price Data.csv"
    df = load_data(spark, file_path)
    
    # 3. Check missing values
    check_missing_values(df)
    
    # 4. Visualize target
    visualize_target(df, target_column="PRICE")
    
    # 5. Feature engineering
    df_fe = feature_engineering(df)
    
    # 6. Split data
    train_df, val_df, test_df = split_data(df_fe)
    
    # 7. Outlier handling
    target_column = "PRICE"
    numeric_cols = [c for c in train_df.columns if c != target_column]
    
    bounds = calculate_outlier_bounds(train_df, numeric_cols)
    train_cap = cap_outliers(train_df, bounds)
    val_cap = cap_outliers(val_df, bounds)
    test_cap = cap_outliers(test_df, bounds)
    
    # 8. Standardization
    stats_row = calculate_standardization_stats(train_cap, numeric_cols)
    train_final = standardize(train_cap, numeric_cols, stats_row)
    val_final = standardize(val_cap, numeric_cols, stats_row)
    test_final = standardize(test_cap, numeric_cols, stats_row)
    
    # Cache final datasets
    train_final.cache()
    val_final.cache()
    test_final.cache()
    
    # 9. Train models
    feature_cols = [c for c in train_final.columns if c != target_column]
    
    lr_model, assembler = train_linear_regression(train_final, feature_cols, target_column)
    dt_model = train_decision_tree(train_final, assembler, target_column)
    rf_model, best_rf = train_random_forest_with_cv(train_final, assembler, target_column)
    
    # 10. Evaluate models
    lr_train_pred = lr_model.transform(train_final)
    lr_val_pred = lr_model.transform(val_final)
    
    dt_train_pred = dt_model.transform(train_final)
    dt_val_pred = dt_model.transform(val_final)
    
    rf_train_pred = rf_model.transform(train_final)
    rf_val_pred = rf_model.transform(val_final)
    
    lr_results = evaluate_model(lr_train_pred, lr_val_pred, "Linear Regression", target_column)
    dt_results = evaluate_model(dt_train_pred, dt_val_pred, "Decision Tree", target_column)
    rf_results = evaluate_model(rf_train_pred, rf_val_pred, "Random Forest (Tuned)", target_column)
    
    # 11. Compare models
    results_df = compare_models([lr_results, dt_results, rf_results])
    visualize_comparison(results_df)
    
    # 12. Feature importance
    importance_df = analyze_feature_importance(best_rf, feature_cols)
    
    # 13. Test set evaluation
    test_pred = evaluate_on_test(rf_model, test_final, target_column, rf_results['Val_RMSE'])
    
    # 14. Save model
    save_model(rf_model)
    
    # 15. Stop Spark
    spark.stop()
    print("\nSpark session stopped.")
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
