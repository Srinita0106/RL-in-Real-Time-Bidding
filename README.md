

# 🎯 Real-Time Bidding Simulator - Reinforcement Learning

## 📌 Overview

This project implements an **AI-powered Real-Time Bidding (RTB) simulation system** using multiple reinforcement learning algorithms. In online advertising, RTB determines how advertisers bid for ad slots in real-time, and efficient strategies can maximize rewards (clicks, conversions) while minimizing costs (budget).

This project provides an **interactive Streamlit application** that allows users to experiment with different algorithms and strategies, visualize performance, and compare outcomes in real-time.

---

## 🛠 Tech Stack

* **Frontend & Visualization:**

  * [Streamlit](https://streamlit.io/) → interactive UI, leaderboard, real-time charts
  * Matplotlib & Seaborn → performance visualizations

* **Backend & Core Logic:**

  * **Python 3**
  * **NumPy, Pandas** → data handling & stats tracking
  * **Reinforcement Learning Algorithms (PyTorch):**

    * Multi-Armed Bandit
    * Q-Learning
    * Deep Q-Network (DQN)
    * Actor-Critic

* **Deep Learning Framework:**

  * [PyTorch](https://pytorch.org/) → neural network models for DQN & Actor-Critic

---

## 🔑 Features

✅ **Multiple RL Algorithms:** Compare Bandit, Q-Learning, DQN, and Actor-Critic
✅ **Custom Strategies:** Standard, Random, Conservative, Aggressive bidding
✅ **Performance Tracking:**

* Real-time budget usage
* Win/loss tracking
* Total reward accumulation
  ✅ **Visualization:**
* Budget over time
* Bidding history vs competitors
* Q-Table heatmaps (Q-Learning)
  ✅ **Leaderboard:** Track top-performing algorithms and strategies
  ✅ **Algorithm Comparison:** Visual + tabular comparison of rewards, win rate, and efficiency

---

## 📊 Algorithms Implemented

### 1. **Multi-Armed Bandit**

* Simple exploration–exploitation strategy
* Learns action values (Q-values) based on rewards

### 2. **Q-Learning**

* Tabular RL approach
* Discretizes budget states for learning
* Updates Q-table with Bellman equation

### 3. **Deep Q-Network (DQN)**

* Neural network approximates Q-values
* Uses **experience replay** and **target network updates**
* Handles larger state spaces effectively

### 4. **Actor-Critic (A2C style)**

* Combines policy-based (Actor) and value-based (Critic) learning
* Encourages exploration via entropy regularization
* More stable convergence than pure policy gradient

---

## 📈 Example Visuals (from the app)

* **Budget Over Time** → line chart showing budget depletion and efficiency
* **Bidding Comparison** → player vs competitor bid history
* **Q-Learning Heatmaps** → visualize Q-values for different budget states
* **Algorithm Performance** → bar/line plots comparing total rewards and win rates

