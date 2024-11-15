import math

def get_lr(it, 
           lr_decay_iters, 
           max_lr, 
           min_lr, 
           decay_rate=0.05):  # Adjust decay_rate for desired decay speed

    # 1) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # 2) Use exponential decay down to min learning rate
    decay_lr = max_lr * math.exp(-decay_rate * it)
    
    # Ensure the learning rate does not fall below min_lr
    return max(min_lr, decay_lr)
