# 📊 Student Performance Simulation

We aimed to **model how a student's final grade** might be influenced by various factors, including academic history, habits, and even motivational treats! 🍫✨

Here is the [link to our app](https://dataanalysis3-pcz2gvtwq9ptei9hre3b2v.streamlit.app/).

## 🎯 Objective
To simulate and analyze how the **final grade** depends on:
- Previous grades 📈  
- Study habits 📚  
- Lifestyle choices (sleep, internet usage)💤🌐  
- And, of course, chocolates from Gabor! 🍫💡  

## 🧪 Data Simulation

We generated **random data** for the following explanatory variables:

| Variable                      | Description                                 |
|-------------------------------|---------------------------------------------|
| 🏅 **G1**                     | First Period Grade                         |
| 🏅 **G2**                     | Second Period Grade                        |
| ⏳ **Study Time**             | Hours studied per week                     |
| 🍫 **Chocolates from Gabor**  | Motivational chocolates received           |
| 🏫 **Attendance (%)**        | Attendance percentage                      |
| 😴 **Sleep Hours**           | Average sleep per night (hours)            |
| 🌐 **Internet Usage**        | Non-study internet usage (hours/day)       |

The **dependent variable** is the **Final Grade (G3)**, calculated using a weighted sum of these features plus random noise.

## 🤖 Models Compared

1. **Dummy Regressor**  
   - Predicts the mean of the target variable.  
   - **High bias**, **zero variance** — ignores data patterns entirely.

2. **Simple Linear Regression**  
   - Fits a linear relationship between the final grade and explanatory variables.  
   - **Moderate bias** and **moderate variance** — balances simplicity and data fit.

3. **Polynomial Regression (Degree 2)**  
   - Captures non-linear relationships by introducing polynomial terms.  
   - **Lower bias** but **higher variance** — risks overfitting on smaller samples.

## 📏 Bias-Variance Tradeoff

We analyzed how these models perform across different sample sizes — **100**, **500**, and **1000** students — observing how **MSE**, **bias**, and **variance** shift with data complexity and volume.

---

💡 *"Because sometimes, a chocolate from Gabor is all it takes to ace the finals!"* 🍫🎉

---

## Real Data Experiment

We extended our analysis using the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance), which includes real student data. The dependent variable, G3 (final grade), was modeled using four approaches:

1. **Dummy Regressor** – Predicts the mean of the target, ignoring all input features.  
2. **Single Feature Model** – Based solely on G1 (First Period Grade).  
3. **Multi-Feature Model** – Using G1, G2, Study Time, and Failures.  
4. **Full Model** – Incorporating all explanatory variables available in the dataset.

This multi-model approach allowed for a detailed exploration of the bias-variance tradeoff, comparing simple and complex models on real-world student performance data.

🤖 Note: usage of AI on all stages of the work is confirmed.
This project utilized AI assistance for brainstorming ideas and coding.
