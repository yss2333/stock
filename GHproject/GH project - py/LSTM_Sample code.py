import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm_cell(prev_c, prev_h, x):
    input_size = prev_h.shape[0] + x.shape[0]
    hidden_size = prev_h.shape[0]
    
    # LSTM의 가중치와 바이어스
    Wf = np.random.randn(hidden_size, input_size)
    Wi = np.random.randn(hidden_size, input_size)
    Wc = np.random.randn(hidden_size, input_size)
    Wo = np.random.randn(hidden_size, input_size)
    
    bf = np.zeros((hidden_size, 1))
    bi = np.zeros((hidden_size, 1))
    bc = np.zeros((hidden_size, 1))
    bo = np.zeros((hidden_size, 1))
    
    concat_input = np.concatenate((prev_h, x), axis=0)

    # Forget Gate 계산
    f = sigmoid(np.dot(Wf, concat_input) + bf)
    
    # Input Gate 계산
    i = sigmoid(np.dot(Wi, concat_input) + bi)
    
    # Candidate Cell State 계산
    c_candidate = np.tanh(np.dot(Wc, concat_input) + bc)
    
    # 새로운 Cell State 계산
    c = f * prev_c + i * c_candidate
    
    # Output Gate 계산
    o = sigmoid(np.dot(Wo, concat_input) + bo)
    
    # Hidden State 계산
    h = o * np.tanh(c)
    
    return c, h

# 입력 시퀀스와 초기 상태 설정
sequence = [np.random.randn(6, 1) for _ in range(5)]
initial_c = np.zeros((4, 1))
initial_h = np.zeros((4, 1))

# LSTM 실행
current_c = initial_c
current_h = initial_h
for x in sequence:
    current_c, current_h = lstm_cell(current_c, current_h, x)
    print("Cell State:", current_c)
    print("Hidden State:", current_h)
