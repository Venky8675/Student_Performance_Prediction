# Student Performance Prediction System

A machine learning project that estimates whether a student is likely to **Pass** or **Fail** using academic and background information. I trained and compared classification models in a Jupyter notebook, then built a Streamlit interface to try predictions with different student profiles.

> **Project status:** Runs locally in Streamlit. This is an educational prototype, not a validated decision-making tool for schools.

## What the app does

- Accepts seven inputs: weekly study hours, attendance rate, previous exam scores, gender, parental education, internet access at home, and extracurricular activities.
- Displays a Pass/Fail prediction and estimated probabilities from a saved scikit-learn pipeline.
- Applies a **0.50 Pass-probability threshold**: at least 50% = Pass; otherwise = Fail.
- Shows a probability gauge and risk band: **Low** (75% or above), **Moderate** (50% to below 75%), or **High** (below 50%).
- Provides simple **rule-based** study suggestions, not AI-generated recommendations.
- Includes **Predict**, **Data Insights**, and **About** tabs.

## Dataset and limitations

The final model was trained on an expanded **10,000-row** student dataset. The initial 5,000 records came from [this Kaggle dataset](https://www.kaggle.com/datasets/amrmaree/student-performance-prediction); another 5,000 records were generated with ChatGPT following the dataset structure. This mixture of original and synthetic records limits how far the evaluation results can be generalized.

The model uses seven predictors. `Student_ID` is excluded because it is an identifier, and `Final_Exam_Score` is excluded because it is closely tied to the target and would introduce target leakage.

**The separate `student_performance_data_30000` file shown in the project folder was not used to train the final model.**

## Training and evaluation

The notebook uses an 80/20 stratified train/test split (8,000 training rows and 2,000 test rows), preprocessing pipelines, and comparison of seven classification algorithms against a dummy baseline. The selected model is **Random Forest** (`n_estimators=100`, `random_state=0`). Model selection considered cross-validation performance, particularly recall for students who may fail.

| Metric | Reported result |
|---|---:|
| Final test accuracy | 87.95% |
| Actual Fail predicted Fail | 900 |
| Actual Fail predicted Pass | 138 |
| Actual Pass predicted Fail | 103 |
| Actual Pass predicted Pass | 859 |

These figures are from the supplied notebook/report on the expanded dataset, **not** independent real-world validation. The notebook also experiments with alternative decision thresholds, but the Streamlit app currently uses **0.50**.

## Tools

Python, Jupyter Notebook, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, and Streamlit.

## Project files

```text
Student-Performance-Prediction/
├── app.py
├── student_performance.ipynb
├── student_pass_fail_model.pkl
├── student_performance_data.csv
├── student_performance_cleaned.csv       # optional EDA output
├── images/                               # screenshots and charts (optional)
├── docs/                                 # detailed project report (optional)
├── README.md
├── requirements.txt
└── .gitignore
```

The paths above are a suggested GitHub layout. **Keep the model and dataset CSV alongside `app.py` unless you also update the paths in the code.** The app loads the model from its own directory, and the Data Insights tab looks for the raw CSV there (or falls back to the cleaned CSV).

## Run locally

1. Install a compatible Python version. **Python 3.10 is a practical choice** if using the notebook's reported scikit-learn 1.3.2 model export.
2. Open a terminal in the project folder and create an environment:

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```powershell
   .venv\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Make sure `student_pass_fail_model.pkl` is in the same folder as `app.py`.
5. Start the application:

   ```bash
   streamlit run app.py
   ```

6. Open **http://localhost:8501** in your browser.

If the saved model cannot be loaded, check the Python and scikit-learn versions used to export it. Only load `.pkl`/Joblib model files from trusted sources.

## Example predictions

| Example | Study hours/week | Attendance | Past exam score | Predicted result | P(Pass) | Risk |
|---|---:|---:|---:|---|---:|---|
| 1 | 25 | 80% | 75 | Pass | 63% | Moderate |
| 2 | 25 | 60% | 63 | Fail | 11% | High |

These are application screenshots/examples, **not** measured model accuracy. The other input fields differ between examples; see the screenshots for the complete profiles.

To show the screenshots on GitHub, place them in `images/pass_prediction.png` and `images/fail_prediction.png`, then uncomment these lines:

<!-- ![Pass prediction](images/pass_prediction.png) -->
<!-- ![Fail prediction](images/fail_prediction.png) -->

## Known issues and future improvements

- Some hard-coded model comparison and feature-importance figures in `app.py` do **not** match the notebook results. Update them from a single verified source before publishing the dashboard as an accurate comparison.
- The app's Data Insights fallback text also has an incorrect Pass/Fail class summary; verify it against the notebook.
- Threshold selection should use validation data rather than tuning against the held-out test set.
- Evaluate on an independent, real student dataset before drawing practical conclusions.
- Add automated tests, requirements version locking, and deployment configuration.
- Consider privacy, fairness, and potential bias before any use involving real students.

## Dataset credit

Kaggle: [Student Performance Prediction — amrmaree](https://www.kaggle.com/datasets/amrmaree/student-performance-prediction). Check the source's license/usage terms before redistributing the original records. The added 5,000 records were AI-generated.
