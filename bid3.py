import streamlit as st
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import pandas as pd
import base64

# Function to encode image as base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return encoded

# Background Image (Ensure the image is in the same directory)
bg_image_path = "bid1.png"
bg_image_base64 = get_base64(bg_image_path)

# Apply Background with black text in sidebar and white text in input forms
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
[data-testid="stSidebar"] {{
    background-color: rgba(255,255,255,0.8);
}}
/* Force white text in dropdown/input fields */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {{
    color: white !important;
}}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] input {{
    color: white !important;
    background-color: rgba(0,0,0,0.3) !important;  /* Optional: Darker background for contrast */
}}
/* For other input types (text, sliders, etc.) */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSlider input {{
    color: white !important;
    background-color: rgba(0,0,0,0.3) !important;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define algorithms
class MultiArmedBandit:
    def __init__(self, action_size):
        self.action_size = action_size
        self.q_values = np.zeros(action_size)
        self.action_counts = np.zeros(action_size)
        self.epsilon = 0.1
        self.convergence_threshold = 0.01
        self.last_q_values = np.zeros(action_size)
        
    def act(self, state, strategy):
        if strategy == 'Random':
            return random.randrange(self.action_size)
        elif strategy == 'Conservative':
            return min(1, self.action_size - 1)
        elif strategy == 'Aggressive':
            return self.action_size - 1
        
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_values)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
    
    def has_converged(self):
        if np.all(self.action_counts > 5):
            change = np.max(np.abs(self.q_values - self.last_q_values))
            self.last_q_values = self.q_values.copy()
            return change < self.convergence_threshold
        return False

class QLearning:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.convergence_threshold = 0.01
        self.last_q_table = np.zeros((state_size, action_size))
        
    def act(self, state, strategy):
        if strategy == 'Random':
            return random.randrange(self.action_size)
        elif strategy == 'Conservative':
            return min(1, self.action_size - 1)
        elif strategy == 'Aggressive':
            return self.action_size - 1
        
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state[0], :])
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state[0], action]
        max_next_q = np.max(self.q_table[next_state[0], :])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q * (1 - done) - current_q)
        self.q_table[state[0], action] = new_q
    
    def has_converged(self):
        change = np.max(np.abs(self.q_table - self.last_q_table))
        self.last_q_table = self.q_table.copy()
        return change < self.convergence_threshold

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_update = 10
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_history = []
        self.convergence_threshold = 0.1
        self.steps = 0
    
    def act(self, state, strategy):
        if strategy == 'Random':
            return random.randrange(self.action_size)
        elif strategy == 'Conservative':
            return min(1, self.action_size - 1)
        elif strategy == 'Aggressive':
            return self.action_size - 1
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return np.argmax(action_values.numpy())
    
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = self.memory.sample(batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.loss_history.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def has_converged(self):
        if len(self.loss_history) < 20:
            return False
        recent_losses = self.loss_history[-20:]
        return np.std(recent_losses) < self.convergence_threshold

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU()
        )
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(24, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Linear(24, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.gamma = 0.99
        self.entropy_coef = 0.01
    
    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value

class ACAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.99
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.loss_history = []
        self.convergence_threshold = 0.1
        self.entropy_coef = 0.01
    
    def act(self, state, strategy):
        if strategy == 'Random':
            return random.randrange(self.action_size)
        elif strategy == 'Conservative':
            return min(1, self.action_size - 1)
        elif strategy == 'Aggressive':
            return self.action_size - 1
        
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.model(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        self.log_probs.append(log_prob)
        self.values.append(state_value)
        self.entropies.append(entropy)
        
        return action.item()
    
    def update(self, reward):
        self.rewards.append(reward)
    
    def train(self):
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        policy_loss = []
        value_loss = []
        for log_prob, value, reward in zip(self.log_probs, self.values, rewards):
            advantage = reward - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.mse_loss(value.squeeze(), reward))
        
        self.optimizer.zero_grad()
        loss = (torch.stack(policy_loss).sum() + 
               torch.stack(value_loss).sum() - 
               self.entropy_coef * torch.stack(self.entropies).sum())
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
    
    def has_converged(self):
        if len(self.loss_history) < 20:
            return False
        recent_losses = self.loss_history[-20:]
        return np.std(recent_losses) < self.convergence_threshold

# Streamlit UI Setup
st.title("🎯 Real-Time Bidding ")

# Simulation Parameters
state_size = 3  # Budget, Competitor's bid, Ad Quality
action_size = 5  # Possible bid values [1, 2, 3, 4, 5]

# Initialize Session State Variables
if "budget" not in st.session_state:
    st.session_state.budget = 10
if "round" not in st.session_state:
    st.session_state.round = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "strategy" not in st.session_state:
    st.session_state.strategy = "Standard"
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []
if "total_reward" not in st.session_state:
    st.session_state.total_reward = 0
if "win_count" not in st.session_state:
    st.session_state.win_count = 0
if "loss_count" not in st.session_state:
    st.session_state.loss_count = 0
if "bid_history" not in st.session_state:
    st.session_state.bid_history = []
if "competitor_bid_history" not in st.session_state:
    st.session_state.competitor_bid_history = []
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "budget_history" not in st.session_state:
    st.session_state.budget_history = [10]
if "algorithm" not in st.session_state:
    st.session_state.algorithm = "DQN"
if "agent" not in st.session_state:
    st.session_state.agent = None
if "comparison_data" not in st.session_state:
    st.session_state.comparison_data = []
if "auto_bid" not in st.session_state:
    st.session_state.auto_bid = False
if "converged" not in st.session_state:
    st.session_state.converged = False

st.sidebar.header("📊 Game Settings")
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["Multi-Armed Bandit", "Q-Learning", "DQN", "Actor-Critic"],
    key="algorithm"
)

strategy = st.sidebar.selectbox(
    "Select Bidding Strategy",
    ["Standard", "Random", "Conservative", "Aggressive"],
    key="strategy"
)

max_rounds = st.sidebar.slider("Maximum Rounds", 10, 200, 50)
st.session_state.auto_bid = st.sidebar.checkbox("Auto Bid", value=False)

# Initialize selected algorithm
if st.session_state.agent is None or not isinstance(st.session_state.agent, {
    "Multi-Armed Bandit": MultiArmedBandit,
    "Q-Learning": QLearning,
    "DQN": DQNAgent,
    "Actor-Critic": ACAgent
}[st.session_state.algorithm]):
    
    if st.session_state.algorithm == "Multi-Armed Bandit":
        st.session_state.agent = MultiArmedBandit(action_size)
    elif st.session_state.algorithm == "Q-Learning":
        st.session_state.agent = QLearning(10, action_size)  # 10 discrete budget states
    elif st.session_state.algorithm == "DQN":
        st.session_state.agent = DQNAgent(state_size, action_size)
    elif st.session_state.algorithm == "Actor-Critic":
        st.session_state.agent = ACAgent(state_size, action_size)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 Current Budget", f"${st.session_state.budget}")
with col2:
    st.metric("🏆 Total Reward", st.session_state.total_reward)
with col3:
    st.metric("📊 Win/Loss", f"{st.session_state.win_count}/{st.session_state.loss_count}")

if hasattr(st.session_state.agent, 'epsilon'):
    st.sidebar.write(f"📉 Current Epsilon: {st.session_state.agent.epsilon:.3f}")

# Game control buttons
col1, col2 = st.columns(2)
with col1:
    bid_clicked = st.button("🚀 Bid Now!")
with col2:
    if st.session_state.game_over or st.session_state.converged:
        if st.button("🔄 Start New Game"):
            st.session_state.budget = 10
            st.session_state.total_reward = 0
            st.session_state.win_count = 0
            st.session_state.loss_count = 0
            st.session_state.bid_history = []
            st.session_state.competitor_bid_history = []
            st.session_state.budget_history = [10]
            st.session_state.game_over = False
            st.session_state.converged = False
            if hasattr(st.session_state.agent, 'epsilon'):
                st.session_state.agent.epsilon = 1.0
            if isinstance(st.session_state.agent, MultiArmedBandit):
                st.session_state.agent.q_values = np.zeros(action_size)
                st.session_state.agent.action_counts = np.zeros(action_size)
            elif isinstance(st.session_state.agent, QLearning):
                st.session_state.agent.q_table = np.zeros((10, action_size))
            elif isinstance(st.session_state.agent, DQNAgent):
                st.session_state.agent.memory = ReplayMemory(1000)
                st.session_state.agent.model = DQN(state_size, action_size)
                st.session_state.agent.target_model = DQN(state_size, action_size)
                st.session_state.agent.target_model.load_state_dict(st.session_state.agent.model.state_dict())
            elif isinstance(st.session_state.agent, ACAgent):
                st.session_state.agent.model = ActorCritic(state_size, action_size)

# Main game logic
if (bid_clicked or st.session_state.auto_bid) and not st.session_state.game_over and not st.session_state.converged:
    if st.session_state.budget > 0 and len(st.session_state.bid_history) < max_rounds:
        # Ensure bids are between 1 and action_size (not zero)
        competitor_bid = random.randint(1, action_size)
        ad_quality = random.random()
        
        # For Q-Learning, discretize the state
        if isinstance(st.session_state.agent, QLearning):
            discrete_budget = min(int(st.session_state.budget), 9)
            state = [discrete_budget, competitor_bid-1, int(ad_quality * 10)]  # Adjust for zero-based index
        else:
            state = [st.session_state.budget, competitor_bid, ad_quality]
        
        bid = st.session_state.agent.act(state, strategy)
        bid = max(1, bid)  # Ensure bid is at least 1
        
        reward = 10 if bid > competitor_bid else -5
        
        next_budget = max(st.session_state.budget - bid, 0)
        
        # Update the agent based on algorithm
        if isinstance(st.session_state.agent, MultiArmedBandit):
            st.session_state.agent.update(bid-1, reward)  # Adjust for zero-based index
        elif isinstance(st.session_state.agent, QLearning):
            next_discrete_budget = min(int(next_budget), 9)
            next_state = [next_discrete_budget, random.randint(0, action_size-1), int(random.random() * 10)]
            done = next_budget <= 0
            st.session_state.agent.update(state, bid-1, reward, next_state, done)  # Adjust for zero-based index
        elif isinstance(st.session_state.agent, DQNAgent):
            next_state = [next_budget, random.randint(1, action_size), random.random()]
            done = next_budget <= 0
            st.session_state.agent.memory.push((state, bid-1, reward, next_state, done))  # Adjust for zero-based index
            st.session_state.agent.train(32)
        elif isinstance(st.session_state.agent, ACAgent):
            st.session_state.agent.update(reward)
            st.session_state.agent.train()
        
        st.session_state.budget -= bid
        st.session_state.total_reward += reward
        
        # Update histories
        st.session_state.bid_history.append(bid)
        st.session_state.competitor_bid_history.append(competitor_bid)
        st.session_state.budget_history.append(st.session_state.budget)
        
        # Display results
        st.write(f"Round {len(st.session_state.bid_history)}: Your Bid: {bid} | Competitor Bid: {competitor_bid} | Reward: {reward}")
        
        if reward > 0:
            st.session_state.win_count += 1
        else:
            st.session_state.loss_count += 1
            
        # Check stopping conditions
        if st.session_state.budget <= 0:
            st.session_state.game_over = True
            st.error("Game Over! Budget Depleted!")
        elif len(st.session_state.bid_history) >= max_rounds:
            st.session_state.converged = True
            st.success(f"Completed maximum {max_rounds} rounds!")
        elif hasattr(st.session_state.agent, 'has_converged') and st.session_state.agent.has_converged():
            st.session_state.converged = True
            st.success("Algorithm has converged!")
        
        # Update leaderboard when game ends
        if st.session_state.game_over or st.session_state.converged:
            st.session_state.leaderboard.append({
                "Algorithm": st.session_state.algorithm,
                "Strategy": strategy,
                "Rounds": len(st.session_state.bid_history),
                "Total Reward": st.session_state.total_reward,
                "Win Rate": st.session_state.win_count / len(st.session_state.bid_history) if len(st.session_state.bid_history) > 0 else 0
            })
            
            # Add to comparison data
            st.session_state.comparison_data.append({
                "Algorithm": st.session_state.algorithm,
                "Strategy": strategy,
                "Total Reward": st.session_state.total_reward,
                "Win Rate": st.session_state.win_count / len(st.session_state.bid_history) if len(st.session_state.bid_history) > 0 else 0,
                "Budget Used": 10 - st.session_state.budget,
                "Rounds": len(st.session_state.bid_history)
            })
    else:
        st.session_state.game_over = True
        st.error("Game Over! Budget Depleted!")

# Display charts only if game is not over and we have data
if not st.session_state.game_over and not st.session_state.converged and len(st.session_state.bid_history) > 0:
    # Budget Over Time Chart
    st.subheader("📊 Budget Over Time")
    st.line_chart(st.session_state.budget_history)
    
    # Bidding Comparison Chart
    st.subheader("📊 Bidding Comparison")
    df_bids = pd.DataFrame({
        "Round": range(1, len(st.session_state.bid_history) + 1),
        "Your Bids": st.session_state.bid_history,
        "Competitor Bids": st.session_state.competitor_bid_history,
    })
    st.line_chart(df_bids.set_index("Round"))
    
    # Algorithm-specific visualizations
    
    
    if isinstance(st.session_state.agent, QLearning):
        st.subheader("🤖 Q-Learning Q-Table (Budget States)")
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i in range(10):  # For each budget state (0-9)
            if i < len(axes):
                ax = axes[i]
                sns.heatmap(st.session_state.agent.q_table[i:i+1], 
                            annot=True, fmt=".1f", ax=ax,
                            cbar=False, cmap="YlGnBu")
                ax.set_title(f"Budget State {i}")
                ax.set_xlabel("Bid Amount")
                ax.set_ylabel("")
        
        plt.tight_layout()
        st.pyplot(fig)

# Algorithm Comparison Section
st.sidebar.header("🔍 Algorithm Comparison")
if st.sidebar.button("Compare Algorithms"):
    if len(st.session_state.comparison_data) > 0:
        comparison_df = pd.DataFrame(st.session_state.comparison_data)
        
        st.subheader("📊 Algorithm Performance Comparison")
        
        # Plot total reward comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in comparison_df['Algorithm'].unique():
            algo_data = comparison_df[comparison_df['Algorithm'] == algo]
            ax.plot(algo_data['Strategy'], algo_data['Total Reward'], 'o-', label=algo)
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.set_title("Total Reward by Algorithm and Strategy")
        st.pyplot(fig)
        
        # Plot win rate comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=comparison_df, x="Algorithm", y="Win Rate", hue="Strategy", ax=ax)
        ax.set_title("Win Rate by Algorithm and Strategy")
        st.pyplot(fig)
        
        # Show comparison table
        st.dataframe(comparison_df.sort_values("Total Reward", ascending=False))
    else:
        st.sidebar.warning("No comparison data available. Play games to generate data.")

# Leaderboard Display
st.subheader("🏆 Leaderboard")
if len(st.session_state.leaderboard) > 0:
    leaderboard_df = pd.DataFrame(st.session_state.leaderboard,
                                columns=["Algorithm", "Strategy", "Rounds", "Total Reward", "Win Rate"])
    st.dataframe(leaderboard_df.sort_values("Total Reward", ascending=False).style.format({
        "Win Rate": "{:.2%}"
    }))
else:
    st.write("No games completed yet. Play a game to see leaderboard entries.")

# Auto-bid logic
if st.session_state.auto_bid and not st.session_state.game_over and not st.session_state.converged:
    if len(st.session_state.bid_history) < max_rounds and st.session_state.budget > 0:
        st.experimental_rerun()