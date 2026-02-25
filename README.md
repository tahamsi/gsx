# GSX: Gumbel-Sigmoid eXplanator for Explainable Fault Prediction

> TensorFlow implementation of **GSX (Gumbel-Sigmoid eXplanator)** for **instance-wise feature selection** and **explainable fault prediction** on IoT vibration time-series data.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Task](https://img.shields.io/badge/Task-Fault%20Prediction-green)
![XAI](https://img.shields.io/badge/XAI-Instance--wise%20Feature%20Selection-purple)

---

## Description

This repository contains the implementation of the GSX framework proposed for **fault prediction using IoT sensor data** with **instance-wise explanations**.

The core idea is to train a predictive model (e.g., LSTM/CNN-based) and then learn a **differentiable selector** that highlights which time steps (features) are most responsible for a prediction. GSX uses a **Gumbel-Sigmoid** selection mechanism to support gradient-based optimization while enabling **sparse**, **instance-specific** explanations.

The repository also includes comparison methods such as **L2X** and **INVASE**, together with multiple deep learning baselines for prediction.

---

## Paper

This repository accompanies the following paper:

**T. Mansouri and S. Vadera**,  
**“A Deep Explainable Model for Fault Prediction Using IoT Sensors”**,  
*IEEE Access*, vol. 10, pp. 66933-66942, 2022.  
DOI: **10.1109/ACCESS.2022.3184693**

---

## Citation

If you use this repository in your research, please cite:

### BibTeX
```bibtex
@article{mansouri2022deepExplainableFaultPrediction,
  author  = {Taha Mansouri and Sunil Vadera},
  title   = {A Deep Explainable Model for Fault Prediction Using IoT Sensors},
  journal = {IEEE Access},
  volume  = {10},
  pages   = {66933--66942},
  year    = {2022},
  doi     = {10.1109/ACCESS.2022.3184693}
}
