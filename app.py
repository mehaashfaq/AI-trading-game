import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import random

# 1. Page Configuration
st.set_page_config(
    page_title="AI Market Trader Game",
    page_icon="🎮",
    layout="centered"
)

# 2. Initialize Game State Variables (Prevents app resetting on every click)
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'history' not in st.session_state:
    st.session_state.history = [100.0, 102.5, 99.0, 101.2, 105.0, 103.8, 106.0]
if 'game_msg' not in st.session_state:
    st.session_state.game_msg = "Welcome to the Trading Floor! Look at the chart and make your prediction."
if 'msg_type' not in st.session_state:
    st.session_state.msg_type = "info"

# 3. Load Light Neural Network for Trend Generation
@st.cache_resource
def load_game_engine():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7, 1)), # Takes last 7 days of prices
        tf.keras.layers.LSTM(16, activation='relu', return_sequences=False),
        tf.keras.layers.Dense(1) # Outputs a weight matrix modifier
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

game_ai = load_game_engine()

# 4. Main Game Interface UI
st.title("🎮 AI Market Trader: Higher or Lower?")
st.markdown("Can you outsmart the neural network? Analyze the asset chart below and predict if the next price point will go up or down!")

# Display Scoreboard Cards
sc1, sc2 = st.columns(2)
with sc1:
    st.metric(label="Current Capital Score", value=f"${st.session_state.score:,}")
with sc2:
    st.metric(label="Current Win Streak", value=f"🔥 {st.session_state.streak} Games")

st.write("---")

# 5. Render the Game Chart
st.subheader("📈 Current Asset Price History")
chart_data = pd.DataFrame({
    'Timeline': [f"Day {i+1}" for i in range(len(st.session_state.history))],
    'Price ($)': st.session_state.history
})
st.line_chart(chart_data.set_index('Timeline'))

# Show feedback from the last round played
if st.session_state.msg_type == "success":
    st.success(st.session_state.game_msg)
elif st.session_state.msg_type == "error":
    st.error(st.session_state.game_msg)
else:
    st.info(st.session_state.game_msg)

# 6. Game Logic and Player Action Buttons
st.write("### 🔮 Make Your Move:")
col1, col2, col3 = st.columns([1, 1, 1])

# Current market value benchmark
current_price = st.session_state.history[-1]

def play_round(player_guess):
    # Prepare data for TensorFlow LSTM layer format: (batch, timesteps, features)
    input_data = np.array(st.session_state.history[-7:]).reshape(1, 7, 1)
    
    # Run AI inference pass to calculate trend momentum
    ai_bias = game_ai.predict(input_data)[0][0]
    
    # Generate next price using AI bias combined with a random market noise fluctuation
    market_fluctuation = random.uniform(-5.0, 5.0)
    price_change = (ai_bias * 0.1) + market_fluctuation
    
    # Ensure it doesn't stay perfectly flat at 0 change
    if abs(price_change) < 0.2:
        price_change = 1.5 if random.choice([True, False]) else -1.5
        
    next_price = round(max(10.0, current_price + price_change), 2)
    
    # Determine actual outcome
    actual_movement = "Higher" if next_price > current_price else "Lower"
    
    # Check if user guessed correctly
    if player_guess == actual_movement:
        st.session_state.score += 500
        st.session_state.streak += 1
        st.session_state.game_msg = f"🎯 CORRECT! The price moved from ${current_price} to ${next_price} ({actual_movement}). You earned +$500!"
        st.session_state.msg_type = "success"
    else:
        st.session_state.streak = 0
        st.session_state.game_msg = f"❌ WRONG! The price moved from ${current_price} to ${next_price} ({actual_movement}). Win streak broken!"
        st.session_state.msg_type = "error"
        
    # Update timeline history array (Remove oldest day, append new day)
    st.session_state.history.append(next_price)
    if len(st.session_state.history) > 12:  # Keep window size clean
        st.session_state.history.pop(0)

# Button Actions
with col1:
    if st.button("🟢 GO HIGHER", use_container_width=True):
        play_round("Higher")
        st.rerun()

with col2:
    if st.button("🔴 GO LOWER", use_container_width=True):
        play_round("Lower")
        st.rerun()

with col3:
    if st.button("🔄 Reset Game", use_container_width=True):
        st.session_state.score = 0
        st.session_state.streak = 0
        st.session_state.history = [100.0, 102.5, 99.0, 101.2, 105.0, 103.8, 106.0]
        st.session_state.game_msg = "Game reset successfully! Welcome back to the market."
        st.session_state.msg_type = "info"
        st.rerun()

st.write("---")
st.caption("🎮 Mechanics: The price ticker updates dynamically using a lightweight TensorFlow LSTM layer model combined with real-time market noise variables.")
