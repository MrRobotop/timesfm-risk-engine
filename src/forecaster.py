import numpy as np
import timesfm

class RiskForecaster:
    def __init__(self):
        # Load the model
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        # Compile the model with specified config
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=512,
                max_horizon=256,
                use_continuous_quantile_head=True,
                return_backcast=True,
                normalize_inputs=True
            )
        )

    def predict_dynamic_macro(self, primary_vol: np.ndarray, macro_covariates: dict, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Two-Stage Forecast:
        Stage A: Forecast the macro covariates themselves using TimesFM.
        Stage B: Use the projected macro paths as future covariates for the primary asset.
        """
        context_len = len(primary_vol)
        max_context = self.model.forecast_config.max_context
        
        # 1. Prepare Primary History (Truncate/Pad)
        if context_len > max_context:
            primary_vol = primary_vol[-max_context:]
        elif context_len < max_context:
            pad_len = max_context - context_len
            primary_vol = np.pad(primary_vol, (pad_len, 0), mode='constant', constant_values=primary_vol.mean())

        # 2. Stage A: Forecast each macro covariate
        dynamic_covariates = {}
        for key, array in macro_covariates.items():
            # Truncate/Pad macro history to match primary context
            if context_len > max_context:
                m_target = array[-max_context:]
            elif context_len < max_context:
                m_target = np.pad(array, (max_context - context_len, 0), mode='constant', constant_values=array.mean())
            else:
                m_target = array
            
            # Forecast the future path of this macro signal
            m_point, _ = self.model.forecast(horizon=horizon, inputs=[m_target])
            # Only take the last 'horizon' elements
            m_future = m_point[0][-horizon:]
            
            # Form the full path: [context, forecast]
            full_path = np.concatenate([m_target, m_future])
            dynamic_covariates[key] = [full_path]

        # 3. Stage B: Forecast primary asset using the projected macro paths
        point_forecast, quantile_forecast = self.model.forecast_with_covariates(
            inputs=[primary_vol],
            dynamic_numerical_covariates=dynamic_covariates
        )
        
        # Return both median path and all quantiles
        return np.array(point_forecast[0][-horizon:]), np.array(quantile_forecast)

    def predict_with_macro(self, primary_vol: np.ndarray, macro_covariates: dict, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        context_len = len(primary_vol)
        max_context = self.model.forecast_config.max_context
        
        # 1. Uniformly enforce max_context length (truncation or padding)
        if context_len > max_context:
            primary_vol = primary_vol[-max_context:]
        elif context_len < max_context:
            pad_len = max_context - context_len
            primary_vol = np.pad(primary_vol, (pad_len, 0), mode='constant', constant_values=primary_vol.mean())
        
        # 2. Process all macro covariates to match exactly max_context + horizon
        padded_covariates = {}
        for key, array in macro_covariates.items():
            if context_len > max_context:
                target_array = array[-max_context:]
            elif context_len < max_context:
                target_array = np.pad(array, (max_context - context_len, 0), mode='constant', constant_values=array.mean())
            else:
                target_array = array
            
            last_value = target_array[-1]
            padded_macro = np.concatenate([target_array, np.full(horizon, last_value)])
            padded_covariates[key] = [padded_macro]
            
        # Forecast using inputs and padded covariates
        point_forecast, quantile_forecast = self.model.forecast_with_covariates(
            inputs=[primary_vol],
            dynamic_numerical_covariates=padded_covariates
        )
        
        return np.array(point_forecast[0][-horizon:]), np.array(quantile_forecast)
