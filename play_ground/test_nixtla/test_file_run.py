# Author: ray
# Date: 5/28/24
# Description:

import pandas as pd

from datasetsforecast.long_horizon import LongHorizon

from neuralforecast.core import NeuralForecast
from neuralforecast.models import Autoformer

# Change this to your own data to try the model
Y_df, _, _ = LongHorizon.load(directory='./', group='ETTm2')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])

n_time = len(Y_df.ds.unique())
val_size = int(.2 * n_time)
test_size = int(.2 * n_time)

Y_df.groupby('unique_id').head(2)

horizon = 96 # 24hrs = 4 * 15 min.
models = [#Informer(h=horizon,                 # Forecasting horizon
                # input_size=horizon,           # Input size
                # max_steps=1000,               # Number of training iterations
                # val_check_steps=100,          # Compute validation loss every 100 steps
                # early_stop_patience_steps=3), # Stop training if validation loss does not improve
          Autoformer(h=horizon,
                input_size=horizon,
                max_steps=1000,
                val_check_steps=100,
                early_stop_patience_steps=3),
          # PatchTST(h=horizon,
          #       input_size=horizon,
          #       max_steps=1000,
          #       val_check_steps=100,
          #       early_stop_patience_steps=3),
         ]

nf = NeuralForecast(
    models=models,
    freq='15min')

Y_hat_df = nf.cross_validation(df=Y_df,
                               val_size=val_size,
                               test_size=test_size,
                               n_windows=None)

from neuralforecast.losses.numpy import mae

mae_autoformer = mae(Y_hat_df['y'], Y_hat_df['Autoformer'])
print(f'Autoformer: {mae_autoformer:.3f}')