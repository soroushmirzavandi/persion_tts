class Hyperparams:
    seed = 42
    
    sr = 22050
    n_fft = 2048
    n_stft = int((n_fft//2) + 1)    
    frame_shift = 0.0125 # seconds
    hop_length = int(n_fft/8.0) 
    frame_length = 0.05 # seconds
    win_length = int(n_fft/2.0) 
    mel_freq = 129
    max_mel_time = 1024 
    max_db = 100
    scale_db = 10
    ref = 4.0
    power = 2.0
    norm_db = 10
    ampl_multiplier = 10.0
    ampl_amin = 1e-10
    db_multiplier = 1.0
    ampl_ref = 1.0
    ampl_power = 1.0    
    # Model params
    
    embedding_size = 256
    encoder_embedding_size = 512    
    dim_feedforward = 1024
    postnet_embedding_size = 1024   
    encoder_kernel_size = 3
    postnet_kernel_size = 5 
    # Other
    n_epochs = 1000
    batch_size = 1
    grad_clip = 1.0
    lr = 2.0 * 1e-4
    r_gate = 1.0    
    step_print = 1000
    step_test = 8000
    step_save = 8000

hp = Hyperparams()