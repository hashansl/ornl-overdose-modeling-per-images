











reload(Train)

# log options
log_path = '../logs/log.csv'
chckpnt_path = '../checkpoints/checkpoint_{0}.pt'

# lr options
warmup_iters = 1000
lr_decay_iters = 90000
max_lr = 1e-3
min_lr = 1e-5
max_iters = 150000

Train.train_model(model = unet,
                  loss_function = loss_function,
                  optimizer = optimizer,
                  train_generator = train_generator,
                  val_generator = val_generator,
                  log_path = log_path,
                  chckpnt_path = chckpnt_path,
                  model_kwargs = model_kwargs,
                  train_idx = train_idx,
                  val_idx = val_idx,
                  device = device,
                  warmup_iters = 1000,
                  lr_decay_iters = 90000,
                  max_lr = 1e-3,
                  min_lr = 1e-5,
                  max_iters = 150000
                  )


reload(Plot)
Plot.plot_loss(log_path='../logs/log.csv',
               warmup_iters=warmup_iters,
               lr_decay_iters=90000,
               max_lr=max_lr,
               min_lr=min_lr)